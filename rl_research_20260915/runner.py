"""Isolated pure-RL mechanism study. Every score is an Actor's own action length."""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import env_geometry as geo
from policies import LearnedAttacker

ROOT=Path(__file__).resolve().parent
STOP=False
def stop(*_):
    global STOP
    STOP=True
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def identity():
    return {'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'time':now()}
def event(path,obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(v):
    random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch'].cpu())
    if v['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in v['cuda']])
def seeded(*parts): return geo.derived_seed('new-research',*parts)

class Value(nn.Module):
    def __init__(self):
        super().__init__();self.point=nn.Sequential(nn.Linear(2,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,x):
        h=self.point(x);return self.head(torch.cat((h.mean(1),h.amax(1)),-1)).squeeze(-1)

def baseline_search(method,centers,lam,scorer,rng,k,epsilon=.08,sigma=.04):
    b,n,_=centers.shape
    def legal(d):return geo.legalize(centers,np.clip(d,-epsilon,epsilon))
    if method=='random':
        z=np.concatenate((centers[:,None],legal(rng.normal(0,sigma,(b,k-1,n,2)))),1)
        lens=scorer(z.reshape(-1,n,2)).reshape(b,k)
    elif method=='es':
        # center + paired probes + one true-scored proposal, exactly k scores.
        pairs=(k-2)//2
        u=rng.normal(size=(b,pairs,n,2));u/=np.maximum(np.sqrt(np.mean(u*u,axis=(-1,-2),keepdims=True)),1e-12)
        probes=legal(sigma*np.stack((u,-u),2).reshape(b,2*pairs,n,2))
        first=np.concatenate((centers[:,None],probes),1)
        lens=scorer(first.reshape(-1,n,2)).reshape(b,1+2*pairs)
        f=lens-lam*geo.costs(centers,first)
        v=f[:,1:].reshape(b,pairs,2)
        grad=np.mean((v[:,:,0]-v[:,:,1])[:,:,None,None]*u,1)
        grad/=np.maximum(np.sqrt(np.mean(grad*grad,axis=(-1,-2),keepdims=True)),1e-12)
        last=legal(sigma*grad[:,None])
        lens=np.concatenate((lens,scorer(last[:,0])[:,None]),1)
        z=np.concatenate((first,last),1)
        if z.shape[1]!=k:raise ValueError('ES needs even k')
    else:raise ValueError(method)
    cost=geo.costs(centers,z);f=lens-lam*cost;idx=f.argmax(1);rows=np.arange(b)
    return dict(selected=z[rows,idx],selected_cost=cost[rows,idx],candidates=z,lengths=lens,
                objective=f,index=idx,diagnostics={})

class Run:
    def __init__(self,args):
        self.a=args;self.p=json.loads((ROOT/'protocol.json').read_text());self.kind=args.kind;self.method=args.method
        self.run_id=f'{args.kind}_{args.method}_s{args.seed}' if args.kind!='warmup' else f'warmup_s{args.seed}'
        self.out=ROOT/'artifacts'/('smoke' if args.smoke else 'runs')/self.run_id;self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device.type!='cuda' and not args.allow_cpu:raise RuntimeError('GPU required')
        if self.device.type=='cuda':torch.cuda.set_per_process_memory_fraction(self.p['resources']['memory_fraction'],0)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        self.actor=geo.build_actor(self.p['degree'],self.device)
        self.value=Value().to(self.device)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=self.p['actor_lr'])
        self.vo=torch.optim.Adam(self.value.parameters(),lr=self.p['baseline_lr'])
        self.attacker=None;self.history=[];self.opponent=None
        if self.method not in ('erm','random','es'):
            method='ppo' if self.method=='population_ppo' else self.method
            self.attacker=LearnedAttacker(method,self.p['degree'],self.device,self.p['attacker'])
        self.step=0;self.lam=self.p['fixed_lambda'] if self.kind=='fixed' else self.p['lambda_init']
        self.calls=dict(search_greedy_scores=0,evaluation_greedy_scores=0,sampled_actor_scores=0,
                        search_padded_forward_rows=0,evaluation_padded_forward_rows=0,oracle_calls=0)
        self.started=time.monotonic();self.prior_elapsed=0.
        self.throttle_seconds=0.
        self.binding={'protocol_sha256':geo.sha256(ROOT/'protocol.json'),
            'code_sha256':{str(ROOT/x):geo.sha256(ROOT/x) for x in ('runner.py','policies.py','env_geometry.py')},
            'dependencies':geo.dependencies(),'parents':{}}
        self.ident={**identity(),'device':str(self.device),'physical_gpu_index':args.physical_gpu_index,
                    'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'duty_fraction':args.duty_fraction}
        if self.kind=='coev' and self.method=='population_ppo':
            self.opponent=geo.build_actor(self.p['degree'],self.device)
        latest=self.out/'latest.pt'
        if latest.exists():
            c=torch.load(latest,map_location=self.device,weights_only=False)
            expected={k:v for k,v in self.binding.items() if k!='parents'}
            if any(c['binding'][k]!=v for k,v in expected.items()):raise RuntimeError('Resume binding mismatch')
            for path,digest in c['binding']['parents'].items():
                if geo.sha256(path)!=digest:raise RuntimeError('Resume parent hash mismatch')
            self.binding=c['binding'];self.restore(c)
        elif self.kind!='warmup':
            warm=self.out.parent/f'warmup_s{args.seed}'/'latest.pt'
            c=self.load_parent(warm,'warmup')
            self.actor.load_state_dict(c['actor']);self.value.load_state_dict(c['value'])
            self.ao.load_state_dict(c['actor_optimizer']);self.vo.load_state_dict(c['value_optimizer'])
            self.binding['parents'][str(warm)]=geo.sha256(warm)
            if self.kind=='coev' and self.attacker is not None:
                parent_method='ppo' if self.method=='population_ppo' else self.method
                fixed=self.out.parent/f'fixed_{parent_method}_s{args.seed}'/'latest.pt'
                c=self.load_parent(fixed,'fixed')
                self.attacker.load_state_dict(c['attacker']);self.binding['parents'][str(fixed)]=geo.sha256(fixed)
        geo.atomic_json(self.out/'run.json',dict(run_id=self.run_id,binding=self.binding,identity=self.ident,smoke=args.smoke))
        self.heartbeat('running')
    def load_parent(self,path,kind):
        complete=json.loads((path.parent/'complete.json').read_text())
        if complete['status']!='complete' or complete['kind']!=kind or complete['seed']!=self.a.seed:
            raise RuntimeError('Invalid parent completion')
        if complete['artifacts_sha256']['latest.pt']!=geo.sha256(path):raise RuntimeError('Parent checkpoint hash mismatch')
        c=torch.load(path,map_location=self.device,weights_only=False)
        for key in ('protocol_sha256','code_sha256','dependencies'):
            if c['binding'][key]!=self.binding[key]:raise RuntimeError('Parent source/protocol binding mismatch')
        if c['step']!=(2 if self.a.smoke else self.p[kind+'_updates']):raise RuntimeError('Incomplete parent checkpoint')
        return c
    def elapsed(self):return self.prior_elapsed+time.monotonic()-self.started
    def pace(self,started,already_slept=0.):
        active=max(0.,time.monotonic()-started-already_slept)
        delay=max(0.,active*(1/self.a.duty_fraction-1)-already_slept)
        began=time.monotonic()
        while not STOP and time.monotonic()-began<delay:
            time.sleep(max(0.,min(.5,delay-(time.monotonic()-began))))
        self.throttle_seconds+=time.monotonic()-began
    def heartbeat(self,status):
        geo.atomic_json(self.out/'heartbeat.json',dict(**self.ident,status=status,step=self.step,elapsed_seconds=self.elapsed(),calls=self.calls))
    def restore(self,c):
        self.actor.load_state_dict(c['actor']);self.value.load_state_dict(c['value'])
        self.ao.load_state_dict(c['actor_optimizer']);self.vo.load_state_dict(c['value_optimizer'])
        if self.attacker:self.attacker.load_state_dict(c['attacker'])
        self.step=c['step'];self.lam=c['lambda'];self.calls=c['calls'];self.history=c['history']
        self.prior_elapsed=c['elapsed_seconds'];restore_rng(c['rng'])
        p=self.out/'progress.jsonl'
        if p.exists():
            with p.open('r+b') as f:f.truncate(c['progress_bytes'])
    def checkpoint(self):
        p=self.out/'progress.jsonl'
        geo.atomic_torch(self.out/'latest.pt',dict(binding=self.binding,step=self.step,lambda_=self.lam,**{'lambda':self.lam},
            actor=self.actor.state_dict(),value=self.value.state_dict(),actor_optimizer=self.ao.state_dict(),
            value_optimizer=self.vo.state_dict(),attacker=self.attacker.state_dict() if self.attacker else None,
            history=self.history,rng=rng_state(),calls=self.calls,elapsed_seconds=self.elapsed(),progress_bytes=p.stat().st_size if p.exists() else 0))
        self.heartbeat('running')
    def data(self,n,d,namespace,seed=0):return geo.make_data(n,self.p['degree'],d,'rl-research-v1/'+namespace,seed)
    def score(self,pins,phase='evaluation',actor=None,topology=False):
        began=time.monotonic()
        actor=actor or self.actor;train=actor.training;actor.eval();lens=[];tops=[];micro=self.p['greedy_microbatch']
        tag=f'{phase}_greedy_scores';pad=f'{phase}_padded_forward_rows';cid=time.time_ns()
        event(self.out/'actual_calls.jsonl',dict(id=cid,event='start',phase=phase,count=len(pins),padded=math.ceil(len(pins)/micro)*micro))
        with torch.no_grad():
            for s in range(0,len(pins),micro):
                x=pins[s:s+micro];keep=len(x)
                if keep<micro:x=np.concatenate((x,np.repeat(x[-1:],micro-keep,0)))
                t,_=actor(x,True);t=t[:keep].cpu().numpy()
                lens.append(geo.lengths(x[:keep],t));tops.append(t)
        actor.train(train);self.calls[tag]+=len(pins);self.calls[pad]+=math.ceil(len(pins)/micro)*micro
        event(self.out/'actual_calls.jsonl',dict(id=cid,event='complete',phase=phase,count=len(pins),padded=math.ceil(len(pins)/micro)*micro))
        self.pace(began)
        return (np.concatenate(lens),np.concatenate(tops)) if topology else np.concatenate(lens)
    def search(self,method,x,lam,rng,train=False,k=8,actor=None):
        phase='search' if train else 'evaluation';scorer=lambda z:self.score(z,phase,actor)
        if method in ('random','es'):return baseline_search(method,x,lam,scorer,rng,k,self.p['epsilon'])
        return self.attacker.search(x,lam,scorer,rng,train=train,k=k)
    def update_actor(self,x):
        action_seed=seeded('router',self.a.seed,self.kind,self.step+1)
        torch.manual_seed(action_seed);torch.cuda.manual_seed_all(action_seed);self.actor.train()
        t,lp=self.actor(x,False);target=torch.as_tensor(geo.lengths(x,t.detach().cpu().numpy()),dtype=torch.float32,device=self.device)
        pred=self.value(torch.as_tensor(x,device=self.device));loss=((target-pred).detach()*lp).mean()
        vl=(pred-target.detach()).square().mean()
        if not torch.isfinite(loss+vl):raise RuntimeError('Nonfinite router loss')
        self.ao.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1.);self.ao.step()
        self.vo.zero_grad(set_to_none=True);vl.backward();nn.utils.clip_grad_norm_(self.value.parameters(),1.);self.vo.step()
        self.calls['sampled_actor_scores']+=len(x)
        event(self.out/'actual_calls.jsonl',dict(event='sampled_complete',phase='sampled',count=len(x)))
        return dict(sampled_length=target.mean().item(),actor_loss=loss.item(),value_loss=vl.item())
    def train(self):
        updates=2 if self.a.smoke else self.p[self.kind+'_updates'];batch=4 if self.a.smoke else self.p['batch_size']
        while self.step<updates:
            if STOP:self.checkpoint();return False
            if self.a.stop_after is not None and self.step>=self.a.stop_after:self.checkpoint();return False
            began=time.monotonic();sleep_before=self.throttle_seconds
            x=self.data(batch,'uniform',f'{self.kind}/train/{self.step+1}',self.a.seed)
            rec={'step':self.step+1,'lambda_before':self.lam};opponent=self.actor
            if self.kind=='warmup' or self.method=='erm':rec.update(self.update_actor(x))
            else:
                rng=np.random.default_rng(seeded('train_attack',self.a.seed,self.kind,self.step+1))
                if self.method=='population_ppo':
                    which=int(rng.integers(len(self.history)+1));rec['opponent_history_index']=which-1
                    if which:
                        self.opponent.load_state_dict(self.history[which-1]);opponent=self.opponent
                if self.kind=='coev' and self.method=='maddpg_bandit':
                    # Old rewards refer to different router/lambda versions.
                    self.attacker.replay=None
                r=self.search(self.method,x,self.lam,rng,train=True,k=self.p['candidates'],actor=opponent)
                idx=np.arange(len(x));gain=r['objective'][idx,r['index']]-r['lengths'][:,0]
                rec.update(gain=float(gain.mean()),selected_cost=float(r['selected_cost'].mean()),
                           no_move_fraction=float((r['index']==0).mean()),diagnostics=r['diagnostics'])
                if self.kind=='coev':
                    rec.update(self.update_actor(r['selected']))
                    self.lam=max(0.,self.lam+self.p['lambda_lr']*(float(r['selected_cost'].mean())-self.p['rho']))
                    if self.method=='population_ppo' and (self.step+1)%self.p['population_snapshot_every']==0:
                        self.history.append(cpu_state(self.actor));self.history=self.history[-self.p['population_history']:]
            self.pace(began,self.throttle_seconds-sleep_before)
            self.step+=1;rec.update(lambda_after=self.lam,time=now(),elapsed_seconds=self.elapsed(),throttle_seconds=self.throttle_seconds)
            event(self.out/'progress.jsonl',rec);self.heartbeat('running')
            if self.step%self.p['checkpoint_every']==0 or self.step==updates:self.checkpoint();print(json.dumps(rec),flush=True)
        return True
    def evaluate(self):
        arrays={};self.heartbeat('evaluating')
        if self.kind=='fixed':
            n=4 if self.a.smoke else self.p['fixed_test_per_distribution']
            for d in self.p['distributions']:
                x=self.data(n,d,'fixed/test/'+d);arrays[d+'__pins']=x
                for lam in self.p['test_lambdas']:
                    for method in (self.method,'random','es'):
                        rng=np.random.default_rng(seeded('fixed_eval',self.a.seed,d,lam,method))
                        chunks=[]
                        for s in range(0,n,32):chunks.append(self.search(method,x[s:s+32],lam,rng,k=self.p['candidates']))
                        for r in chunks:
                            row=np.arange(len(r['index']));r['gain']=r['objective'][row,r['index']]-r['lengths'][:,0]
                            r['length']=r['lengths'][row,r['index']];r['center_length']=r['lengths'][:,0]
                        prefix=f'{d}__lambda{lam:g}__{method}__'
                        for key in ('gain','length','selected_cost','index','center_length','selected'):
                            dest='cost' if key=='selected_cost' else key
                            arrays[prefix+dest]=np.concatenate([r[key] for r in chunks])
                        for key in ('candidates','lengths','objective'):
                            arrays[prefix+'all_'+key]=np.concatenate([r[key] for r in chunks])
        elif self.kind=='coev':
            n=4 if self.a.smoke else self.p['test_per_distribution'];na=4 if self.a.smoke else self.p['attack_per_distribution']
            for d in self.p['distributions']:
                x=self.data(n,d,'coev/test/'+d);lens,tops=self.score(x,topology=True)
                arrays[d+'__nominal_pins']=x;arrays[d+'__nominal_length']=lens;arrays[d+'__nominal_topology']=tops
                attack=self.data(na,d,'coev/attack/'+d);arrays[d+'__attack_pins']=attack
                for method in ('random','es'):
                    rng=np.random.default_rng(seeded('independent_attack',d,method));rs=[]
                    for s in range(0,na,32):rs.append(self.search(method,attack[s:s+32],0.,rng,k=self.p['attack_candidates']))
                    prefix=d+'__'+method+'32_'
                    arrays[prefix+'length']=np.concatenate([r['lengths'][np.arange(len(r['index'])),r['index']] for r in rs])
                    arrays[prefix+'cost']=np.concatenate([r['selected_cost'] for r in rs])
                    arrays[prefix+'selected']=np.concatenate([r['selected'] for r in rs])
                    arrays[prefix+'all_candidates']=np.concatenate([r['candidates'] for r in rs])
                    arrays[prefix+'all_lengths']=np.concatenate([r['lengths'] for r in rs])
                    arrays[prefix+'index']=np.concatenate([r['index'] for r in rs])
        if arrays:geo.atomic_npz(self.out/'results.npz',**arrays)
        files=['latest.pt','run.json','progress.jsonl','actual_calls.jsonl']+(['results.npz'] if arrays else [])
        geo.atomic_json(self.out/'complete.json',dict(status='complete',run_id=self.run_id,kind=self.kind,method=self.method,
            seed=self.a.seed,step=self.step,device=str(self.device),identity=self.ident,smoke=self.a.smoke,calls=self.calls,
            elapsed_seconds=self.elapsed(),peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device.type=='cuda' else 0,
            binding=self.binding,artifacts_sha256={x:geo.sha256(self.out/x) for x in files}))
        self.heartbeat('complete')

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['warmup','fixed','coev'],required=True)
    p.add_argument('--method',default='erm');p.add_argument('--seed',type=int,required=True);p.add_argument('--physical-gpu-index',type=int)
    p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--stop-after',type=int);p.add_argument('--duty-fraction',type=float,default=1.);args=p.parse_args()
    if not 0<args.duty_fraction<=1:p.error('duty fraction must be in (0,1]')
    rid=f'{args.kind}_{args.method}_s{args.seed}' if args.kind!='warmup' else f'warmup_s{args.seed}'
    complete=ROOT/'artifacts'/('smoke' if args.smoke else 'runs')/rid/'complete.json'
    if complete.exists():
        print('Already complete:',complete);return
    geo.install_oracle_guard();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    run=Run(args)
    if (run.out/'complete.json').exists():return
    if run.train():run.evaluate()
    else:raise SystemExit(75)
if __name__=='__main__':main()
