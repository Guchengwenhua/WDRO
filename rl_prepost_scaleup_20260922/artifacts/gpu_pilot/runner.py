"""Continuous, hash-bound pre/post budget experiment with RNG-neutral evaluation."""
import argparse,copy,fcntl,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
import engine as e
import env_geometry as geo

ROOT=Path(__file__).resolve().parent
CODE=['runner.py','engine.py','policies.py','env_geometry.py']

def protocol():return json.loads((ROOT/'protocol.json').read_text())
def task_by_id(rid):return next(t for t in protocol()['tasks'] if t['id']==rid)
def parent_path(parent):
    if parent['source']=='legacy':return ROOT/'lineage'/parent['run']/'latest.pt'
    return ROOT/'artifacts/runs'/parent['run']/'checkpoints'/f"step_{parent['step']}.pt"
def binding(task):
    return dict(protocol_sha256=geo.sha256(ROOT/'protocol.json'),code_sha256={str(ROOT/n):geo.sha256(ROOT/n) for n in CODE},
                dependencies=geo.dependencies(),parents={str(parent_path(v)):geo.sha256(parent_path(v)) for v in task['parents'].values()},task=task)

class Run(e.Run):
    def __init__(self,args):
        self.a=args;self.task=task_by_id(args.task_id);self.p=protocol();self.kind=self.task['kind'];self.method=self.task['method']
        args.kind=self.kind;args.method=self.method;args.seed=self.task['seed'];args.smoke=False
        self.run_id=self.task['id'];self.out=ROOT/'artifacts/runs'/self.run_id;self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device.type!='cuda' and not args.allow_cpu:raise RuntimeError('GPU required')
        if self.device.type=='cuda':torch.cuda.set_per_process_memory_fraction(self.p['resources']['memory_fraction'],0)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        self.actor=geo.build_actor(self.p['degree'],self.device);self.value=e.Value().to(self.device)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=self.p['actor_lr']);self.vo=torch.optim.Adam(self.value.parameters(),lr=self.p['baseline_lr'])
        self.attacker=None;self.history=[];self.opponent=None
        if self.method not in ('erm','random','es'):self.attacker=e.LearnedAttacker(self.method,self.p['degree'],self.device,self.p['attacker'])
        self.step=self.task['start'];self.lam=self.p['fixed_lambda'] if self.kind=='fixed' else self.p['lambda_init']
        self.calls={k:0 for k in ['search_greedy_scores','evaluation_greedy_scores','sampled_actor_scores','search_padded_forward_rows','evaluation_padded_forward_rows','oracle_calls']}
        self.started=time.monotonic();self.prior_elapsed=0.;self.throttle_seconds=0.
        self.binding=binding(self.task)
        self.ident=dict(e.identity(),device=str(self.device),physical_gpu_index=args.physical_gpu_index,gpu_uuid=args.gpu_uuid,
                        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),duty_fraction=args.duty_fraction)
        latest=self.out/'latest.pt'
        if latest.exists():
            c=torch.load(latest,map_location=self.device,weights_only=False)
            if c['binding']!=self.binding:raise RuntimeError('Resume binding mismatch')
            self.restore(c)
        else:
            role='resume' if 'resume' in self.task['parents'] else 'warmup'
            spec=self.task['parents'][role];c=torch.load(parent_path(spec),map_location=self.device,weights_only=False)
            if c['step']!=spec['step']:raise RuntimeError('Parent checkpoint step mismatch')
            self.actor.load_state_dict(c['actor']);self.value.load_state_dict(c['value'])
            self.ao.load_state_dict(c['actor_optimizer']);self.vo.load_state_dict(c['value_optimizer'])
            if role=='resume':
                self.lam=c['lambda'];self.history=c['history']
                if self.attacker:self.attacker.load_state_dict(c['attacker'])
                e.restore_rng(c['rng'])
            elif 'attacker' in self.task['parents']:
                ac=torch.load(parent_path(self.task['parents']['attacker']),map_location=self.device,weights_only=False)
                if ac['step']!=self.task['parents']['attacker']['step']:raise RuntimeError('Attacker step mismatch')
                self.attacker.load_state_dict(ac['attacker'])
            geo.atomic_json(self.out/'inherited_budget.json',dict(parent=str(parent_path(spec)),parent_step=c['step'],
                parent_committed_calls=c['calls'],parent_elapsed_seconds=c['elapsed_seconds'],
                note='Copied history is separate from this run additional counters; full ancestry is in binding.parents'))
        for name in ['progress.jsonl','actual_calls.jsonl']:(self.out/name).touch(exist_ok=True)
        geo.atomic_json(self.out/'run.json',dict(run_id=self.run_id,task=self.task,binding=self.binding,identity=self.ident))
        self.heartbeat('running')

    def data(self,n,d,namespace,seed=0):
        prefix=self.p['data_namespace'] if '/train/' in namespace else self.p['evaluation_namespace']
        return geo.make_data(n,self.p['degree'],d,prefix+'/'+namespace,seed)

    def evaluate(self):
        saved=e.rng_state();arrays={};before=self.calls['evaluation_greedy_scores'];began=time.monotonic()
        self.heartbeat('evaluating')
        try:
            if self.kind=='fixed':
                n=self.p['fixed_test_per_distribution']
                for d in self.p['distributions']:
                    x=self.data(n,d,'fixed/test/'+d);arrays[d+'__pins']=x
                    for lam in self.p['test_lambdas']:
                        for method in (self.method,'random','es'):
                            rng=np.random.default_rng(e.seeded('fixed_eval',self.a.seed,d,lam,method));chunks=[]
                            for start in range(0,n,32):chunks.append(self.search(method,x[start:start+32],lam,rng,k=self.p['candidates']))
                            for result in chunks:
                                row=np.arange(len(result['index']));idx=result['index']
                                result['gain']=result['objective'][row,idx]-result['lengths'][:,0]
                                result['length']=result['lengths'][row,idx];result['center_length']=result['lengths'][:,0]
                            prefix=f'{d}__lambda{lam:g}__{method}__'
                            for key in ['gain','length','selected_cost','index','center_length','selected']:
                                arrays[prefix+('cost' if key=='selected_cost' else key)]=np.concatenate([r[key] for r in chunks])
                            for key in ['candidates','lengths','objective']:arrays[prefix+'all_'+key]=np.concatenate([r[key] for r in chunks])
            else:
                n=self.p['test_per_distribution'];na=self.p['attack_per_distribution']
                for d in self.p['distributions']:
                    x=self.data(n,d,'coev/test/'+d);lens,tops=self.score(x,topology=True)
                    arrays[d+'__nominal_pins']=x;arrays[d+'__nominal_length']=lens;arrays[d+'__nominal_topology']=tops
                    if self.kind=='warmup':continue
                    attack=self.data(na,d,'coev/attack/'+d);arrays[d+'__attack_pins']=attack
                    for method in ['random','es']:
                        rng=np.random.default_rng(e.seeded('independent_attack',d,method));chunks=[]
                        for start in range(0,na,32):chunks.append(self.search(method,attack[start:start+32],0.,rng,k=self.p['attack_candidates']))
                        prefix=d+'__'+method+'32_'
                        for key in ['length','cost','selected','all_candidates','all_lengths','index']:
                            if key=='length':v=[r['lengths'][np.arange(len(r['index'])),r['index']] for r in chunks]
                            else:v=[r[{'cost':'selected_cost','all_candidates':'candidates','all_lengths':'lengths'}.get(key,key)] for r in chunks]
                            arrays[prefix+key]=np.concatenate(v)
            folder=self.out/'evaluations';folder.mkdir(exist_ok=True)
            dest=folder/f'step_{self.step}.npz';geo.atomic_npz(dest,**arrays)
            receipt=dict(step=self.step,kind=self.kind,method=self.method,seed=self.a.seed,sha256=geo.sha256(dest),
                         greedy_scores=self.calls['evaluation_greedy_scores']-before,elapsed_seconds=time.monotonic()-began)
            geo.atomic_json(folder/f'step_{self.step}.json',receipt)
        finally:e.restore_rng(saved)

    def publish_milestone(self):
        self.checkpoint()
        folder=self.out/'checkpoints';folder.mkdir(exist_ok=True)
        path=folder/f'step_{self.step}.pt';tmp=path.with_suffix('.partial')
        import shutil
        shutil.copyfile(self.out/'latest.pt',tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
        geo.atomic_json(folder/f'step_{self.step}.json',dict(step=self.step,sha256=geo.sha256(path),binding=self.binding))

    def execute(self):
        for milestone in self.task['milestones']:
            if self.step>milestone:continue
            self.p[self.kind+'_updates']=milestone
            if not super().train():return 75
            if e.STOP:return 75
            # Receipts may survive an interrupted checkpoint commit. The saved
            # checkpoint decides whether evaluation is committed; re-evaluate
            # when its counters are not in the latest saved state.
            marker=self.out/'checkpoints'/f'step_{milestone}.json'
            if not marker.exists():
                self.evaluate();self.publish_milestone()
        if self.step!=self.task['end']:raise RuntimeError('Incomplete final step')
        files=['latest.pt','run.json','progress.jsonl','actual_calls.jsonl','inherited_budget.json']
        files += [str(f.relative_to(self.out)) for sub in ['checkpoints','evaluations'] for f in sorted((self.out/sub).glob('*')) if f.suffix in ['.pt','.npz','.json']]
        geo.atomic_json(self.out/'complete.json',dict(status='complete',run_id=self.run_id,kind=self.kind,method=self.method,seed=self.a.seed,
            initial_step=self.task['start'],step=self.step,task=self.task,identity=self.ident,binding=self.binding,device=str(self.device),
            calls=self.calls,elapsed_seconds=self.elapsed(),peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device.type=='cuda' else 0,
            artifacts_sha256={f:geo.sha256(self.out/f) for f in files}))
        self.heartbeat('complete');return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task-id',required=True);parser.add_argument('--physical-gpu-index',type=int)
    parser.add_argument('--gpu-uuid');parser.add_argument('--allow-cpu',action='store_true');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--duty-fraction',type=float,default=1.)
    args=parser.parse_args()
    if not 0<args.duty_fraction<=1:parser.error('Invalid duty fraction')
    geo.install_oracle_guard();signal.signal(signal.SIGTERM,e.stop);signal.signal(signal.SIGINT,e.stop)
    run=Run(args);return run.execute()

if __name__=='__main__':raise SystemExit(main())
