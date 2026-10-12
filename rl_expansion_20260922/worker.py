"""Resumable task worker. Training never loads test files; evaluations require global freeze."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
import torch
import geometry as G
import engine as E
from common import (ROOT,GROUPS,now,read,event,identity,digest,seed_all,rng_state,set_rng,
                    check_binding,configure,load_data,summary,verify_complete,repair_tail)

STOP=False
def stop(*_):
    global STOP
    STOP=True

class Worker:
    def __init__(self,a):
        self.a=a; self.p=read(ROOT/'protocol.json'); self.task_path=Path(a.task_json).resolve()
        self.task_path.relative_to(ROOT); self.t=read(self.task_path); self.c=self.t['config']
        self.device=configure(a.physical_gpu_index,a.cpu_threads)
        if self.device.type!='cuda' and not a.allow_cpu: raise RuntimeError('GPU required')
        self.out=(ROOT/self.t['outdir']).resolve(); self.out.relative_to(ROOT); self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+'); fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.lock.seek(0); self.lock.truncate(); json.dump(identity(),self.lock); self.lock.flush()
        self.b=E.task_binding(self.task_path)
        self.ident=dict(identity(),device=str(self.device),physical_gpu_index=a.physical_gpu_index,
                        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,started=now())
        self.start_clock=time.monotonic(); self.elapsed_prior=0.; self.update_seconds=0.; self.update_count=0
        self.step=self.t['start']; self.start_step=self.step; self.target=self.t['target']
        self.best_metric=float('inf'); self.best_step=None; self.parent_mean=None; self.parent=None
        self.ledger={k:0 for k in ('sampled_rows','rollout_rows','calibration_rows','validation_rows','test_rows','backward_updates','oracle_calls')}
        self.data_chain=''; self.last_validation=None; self.models={}; self.adv_variance=1.
        self.logr=torch.full((4,),-math.log(4),dtype=torch.float64,device=self.device); self.anchors=None
        seed_all(self.t['seed'])
        self.actor=G.build_actor(30,self.device,self.c['width'],self.c['depth'],self.c['inner'])
        self.initial_hash=digest(self.actor)
        self.optimizer=self.make_optimizer()
        self.validation=None
        if self.t['mode']=='train':
            self.validation=load_data('validation')
            if a.diagnostic: self.validation={g:x[:64] for g,x in self.validation.items()}
            self.initialize()
        if not (self.out/'run.json').exists():
            G.atomic_json(self.out/'run.json',dict(binding=self.b,identity=self.ident,task=self.t,
                initial_actor_sha256=self.initial_hash,parent=self.parent,diagnostic=a.diagnostic,
                parameter_count=sum(x.numel() for x in self.actor.parameters())))
        event(self.out/'segments.jsonl',dict(event='start',time=now(),identity=self.ident,step=self.step))
        repair_tail(self.out/'actual_calls.jsonl')
        if self.t['mode']=='eval' and a.resume:
            self.restore_eval_ledger()
        self.heartbeat('running')

    def restore_eval_ledger(self):
        path=self.out/'actual_calls.jsonl'
        if not path.exists(): return
        rows=[json.loads(x) for x in path.read_text().splitlines()]; done={r['id'] for r in rows if r['event']=='complete'}
        for r in rows:
            if r['event']=='complete': self.ledger[r['phase']+'_rows']+=r['rows']
        incomplete=[r for r in rows if r['id'] not in done]
        if incomplete:
            G.atomic_json(self.out/f'incomplete_eval_calls_{time.time_ns()}.json',incomplete)
            tmp=path.with_suffix('.jsonl.partial')
            tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['id'] in done)); os.replace(tmp,path)

    def make_optimizer(self):
        cls=torch.optim.AdamW if self.c['optimizer']=='adamw' else torch.optim.Adam
        return cls(self.actor.parameters(),lr=self.c['lr'],weight_decay=self.c['weight_decay'])

    def optimizer_load(self,state):
        self.optimizer.load_state_dict(state)
        # AdamW inherits moments, not the parent's zero weight-decay setting.
        for group in self.optimizer.param_groups: group['weight_decay']=self.c['weight_decay']

    def initialize(self):
        path=self.out/'latest.pt'
        if path.exists():
            if not self.a.resume: raise RuntimeError('Use --resume')
            c=torch.load(path,map_location=self.device,weights_only=False)
            if c['binding']!=self.b or c['diagnostic']!=self.a.diagnostic: raise RuntimeError('Resume binding mismatch')
            self.actor.load_state_dict(c['actor']); self.optimizer_load(c['optimizer'])
            for k in ('step','best_metric','best_step','parent_mean','ledger','update_seconds','update_count','data_chain',
                      'last_validation','models','parent','adv_variance'):
                setattr(self,k,c[k])
            self.logr=c['logr'].to(self.device); self.anchors=c['anchors'].to(self.device)
            self.elapsed_prior=c['elapsed_seconds']; set_rng(c['rng'])
            if not (self.out/'complete.json').exists():
                with (self.out/'progress.jsonl').open('r+b') as f: f.truncate(c['progress_bytes'])
                calls=self.out/'actual_calls.jsonl'
                if calls.exists() and calls.stat().st_size>c['calls_bytes']:
                    with calls.open('rb') as f:
                        f.seek(c['calls_bytes']); tail=f.read()
                    (self.out/f'replayed_calls_{time.time_ns()}.jsonl').write_bytes(tail)
                    with calls.open('r+b') as f: f.truncate(c['calls_bytes'])
                if c['best_checkpoint_bytes'] is not None:
                    (self.out/'best.pt.partial').write_bytes(c['best_checkpoint_bytes'])
                    os.replace(self.out/'best.pt.partial',self.out/'best.pt')
            return
        if self.t.get('parent'):
            spec=self.t['parent']; path=(ROOT/spec['path']).resolve(); path.relative_to(ROOT)
            if G.sha256(path)!=spec['sha256']: raise RuntimeError('Parent changed')
            c=torch.load(path,map_location=self.device,weights_only=False)
            if c['step']!=self.step: raise RuntimeError('Wrong parent step')
            self.actor.load_state_dict(c['actor']); self.optimizer_load(c['optimizer']); set_rng(c['rng'])
            if self.t.get('inherit_training'):
                self.logr=c['logr'].to(self.device); self.anchors=c['anchors'].to(self.device); self.adv_variance=c['adv_variance']
            self.parent=dict(spec,actor_sha256=digest(self.actor),step=c['step'],optimizer_moments_inherited=True,
                rng_inherited=True,normalization_buffers_inherited=True,objective_state_inherited=self.t.get('inherit_training',False))
        elif self.step!=0: raise RuntimeError('Missing parent')

    def elapsed(self): return self.elapsed_prior+time.monotonic()-self.start_clock

    def heartbeat(self,status,**extra):
        rate=self.update_seconds/self.update_count if self.update_count else None
        G.atomic_json(self.out/'heartbeat.json',dict(self.ident,status=status,updated=now(),step=self.step,
            updates_completed=self.update_count,updates_target=self.target-self.start_step,
            seconds_per_update=rate,estimated_remaining_seconds=rate*(self.target-self.step) if rate else None,
            elapsed_seconds=self.elapsed(),peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device.type=='cuda' else 0,**extra))

    def call(self,phase,rows,fn):
        token=f'{os.getpid()}-{time.time_ns()}'
        event(self.out/'actual_calls.jsonl',dict(id=token,event='start',phase=phase,rows=rows,step=self.step,time=now()))
        result=fn()
        event(self.out/'actual_calls.jsonl',dict(id=token,event='complete',phase=phase,rows=rows,step=self.step,time=now()))
        return result

    def score(self,pins,phase):
        def forward():
            mode=self.actor.training; self.actor.eval(); ls=[]; ts=[]
            with torch.no_grad():
                for start in range(0,len(pins),64):
                    b=pins[start:start+64]; keep=len(b)
                    if keep<64: b=np.concatenate([b,np.repeat(b[-1:],64-keep,axis=0)])
                    topology,_=self.actor(b,True); topology=topology[:keep].cpu().numpy()
                    ts.append(topology); ls.append(G.lengths(b[:keep],topology))
            self.actor.train(mode); self.ledger[phase+'_rows']+=len(pins)
            return np.concatenate(ls),np.concatenate(ts)
        return self.call(phase,len(pins),forward)

    def evaluate(self,data,phase):
        arrays={}; lengths={}
        for g,pins in data.items():
            l,t=self.score(pins,phase); arrays[g+'__length']=l; arrays[g+'__topology']=t; lengths[g]=l
            self.heartbeat(phase,group=g)
        return arrays,summary(lengths)

    def validate(self):
        arrays,metrics=self.evaluate(self.validation,'validation')
        if self.parent_mean is None: self.parent_mean=metrics['mean']
        eligible=metrics['mean']<=self.parent_mean*(1+self.p['selection_mean_tolerance'])
        if eligible and metrics['worst']<self.best_metric:
            self.best_metric=metrics['worst']; self.best_step=self.step
            G.atomic_torch(self.out/'best.pt',dict(actor=self.actor.state_dict(),step=self.step,binding=self.b,
                          config=self.c,validation=metrics,actor_sha256=digest(self.actor)))
        self.last_validation=metrics
        row=dict(type='validation',time=now(),step=self.step,metrics=metrics,eligible=eligible,
                 best_step=self.best_step,best_metric=self.best_metric,elapsed_seconds=self.elapsed())
        event(self.out/'progress.jsonl',row); print(json.dumps(row),flush=True)
        return arrays

    def state(self):
        return dict(binding=self.b,diagnostic=self.a.diagnostic,config=self.c,actor=self.actor.state_dict(),optimizer=self.optimizer.state_dict(),
            step=self.step,best_metric=self.best_metric,best_step=self.best_step,parent_mean=self.parent_mean,
            logr=self.logr,anchors=self.anchors,adv_variance=self.adv_variance,ledger=self.ledger,
            update_seconds=self.update_seconds,update_count=self.update_count,data_chain=self.data_chain,
            last_validation=self.last_validation,models=self.models,parent=self.parent,rng=rng_state(),
            elapsed_seconds=self.elapsed(),initial_actor_sha256=self.initial_hash,
            calls_bytes=(self.out/'actual_calls.jsonl').stat().st_size,
            best_checkpoint_bytes=(self.out/'best.pt').read_bytes() if (self.out/'best.pt').exists() else None,
            progress_bytes=(self.out/'progress.jsonl').stat().st_size)

    def checkpoint(self,endpoint=False):
        if endpoint:
            name=f'step{self.step}.pt'; G.atomic_torch(self.out/name,self.state())
            self.models[f'step{self.step}']=dict(file=name,sha256=G.sha256(self.out/name),step=self.step,actor_sha256=digest(self.actor))
        G.atomic_torch(self.out/'latest.pt',self.state()); self.heartbeat('running')

    def update(self):
        start=time.monotonic(); step=self.step+1
        pins,data_info=E.batch_for(self.c,self.t['seed'],step,self.p)
        rate=E.learning_rate(self.c,step,self.p)
        for group in self.optimizer.param_groups: group['lr']=rate
        baseline,_=self.score(pins,'rollout')
        seed_all(G.derived_seed('actor_action',self.t['seed'],step)); self.actor.train()
        if self.c['freeze_bn']:
            for module in self.actor.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
        def sample():
            topology,logprob=self.actor(pins,False)
            costs=G.lengths(pins,topology.detach().cpu().numpy())
            return logprob,torch.as_tensor(costs,dtype=torch.float32,device=self.device)
        logprob,cost=self.call('sampled',len(pins),sample)
        advantage=(cost-torch.as_tensor(baseline,dtype=torch.float32,device=self.device)).detach()
        loss,q,scale=E.losses(advantage,logprob,self.c,self.logr,self.anchors,self.adv_variance,self.p)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.p['gradient_clip'])
        if not torch.isfinite(grad): raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        risks=cost.detach().reshape(4,-1).mean(1).to(torch.float64)
        objective_risks=risks/self.anchors if self.c['normalize_risk'] else risks
        if self.c['objective']=='group': self.logr=E.next_logr(self.logr,objective_risks,self.c)
        if self.c['advantage_ema']:
            beta=self.p['advantage_ema_beta']; self.adv_variance=beta*self.adv_variance+(1-beta)*float(advantage.var(unbiased=False))
        self.step=step; self.update_count+=1; self.update_seconds+=time.monotonic()-start
        self.ledger['sampled_rows']+=len(pins); self.ledger['backward_updates']+=1
        data_hash=hashlib.sha256(pins.tobytes()).hexdigest()
        self.data_chain=hashlib.sha256((self.data_chain+data_hash).encode()).digest().hex()
        event(self.out/'progress.jsonl',dict(type='update',time=now(),step=step,loss=float(loss.detach()),
            sampled_mean=float(cost.mean()),baseline_mean=float(baseline.mean()),gradient_norm=float(grad),
            learning_rate=rate,q_used=q.cpu().tolist(),q_next=E.group_weights(self.logr,self.c).cpu().tolist(),
            group_risks=risks.cpu().tolist(),objective_risks=objective_risks.cpu().tolist(),
            advantage_scale_used=scale,advantage_variance_next=self.adv_variance,
            data_sha256=data_hash,data_info=data_info,elapsed_seconds=self.elapsed(),update_seconds=self.update_seconds))

    def guard(self):
        if shutil.disk_usage(ROOT).free<self.p['min_disk_gib']*2**30: return 'disk_capacity'
        usage=ROOT/'artifacts/operations/branch_usage.json'
        if usage.exists() and read(usage)['bytes']>self.p['max_branch_gib']*2**30: return 'branch_disk_budget'
        if self.device.type=='cuda' and torch.cuda.mem_get_info()[0]<2*2**30: return 'gpu_memory'
        return None

    def train(self):
        if (self.out/'complete.json').exists(): verify_complete(self.out/'complete.json'); return 0
        if not (self.out/'latest.pt').exists():
            (self.out/'progress.jsonl').touch()
            if self.anchors is None:
                calibration=load_data('calibration')
                if self.a.diagnostic: calibration={g:x[:64] for g,x in calibration.items()}
                _,m=self.evaluate(calibration,'calibration')
                self.anchors=torch.tensor([m['groups'][g]['mean'] for g in GROUPS],dtype=torch.float64,device=self.device)
            self.validate(); self.checkpoint(endpoint=True)
        while self.step<self.target:
            reason=self.guard() if self.step%32==0 else None
            if STOP or reason:
                self.checkpoint(); self.heartbeat('paused',reason=reason or 'signal'); return 75
            self.update()
            if self.step%self.p['validation_every']==0 or self.step==self.target: self.validate()
            if self.step%self.p['checkpoint_every']==0 or self.step in self.t['endpoints']:
                self.checkpoint(endpoint=self.step in self.t['endpoints'])
            if self.a.stop_after is not None and self.step>=self.a.stop_after:
                self.checkpoint(); self.heartbeat('diagnostic_paused'); return 0
        self.checkpoint(endpoint=True)
        endpoint_metrics=self.last_validation
        best=torch.load(self.out/'best.pt',map_location=self.device,weights_only=False); self.actor.load_state_dict(best['actor'])
        arrays,metrics=self.evaluate(self.validation,'validation'); G.atomic_npz(self.out/'results.npz',**arrays)
        self.models['best']=dict(file='best.pt',sha256=G.sha256(self.out/'best.pt'),step=self.best_step,actor_sha256=digest(self.actor))
        self.finish('training_complete',dict(models=self.models,endpoint_validation=endpoint_metrics,selected_validation=metrics))
        return 0

    def eval(self):
        if (self.out/'complete.json').exists(): verify_complete(self.out/'complete.json'); return 0
        selection_path=ROOT/'artifacts/selection.json'; frozen=read(selection_path)
        if G.sha256(selection_path)!=self.t['selection_sha256']: raise RuntimeError('Global freeze changed')
        if frozen['status']!='all_training_frozen': raise RuntimeError('Test gate closed')
        # Bind every completed run before opening test arrays, including runs not selected.
        for path,h in frozen['training_completions'].items():
            if G.sha256(ROOT/path)!=h: raise RuntimeError('Training changed after freeze')
        degree=self.t.get('degree',30); self.actor.degree=degree
        data=load_data('test' if degree==30 else f'test_degree{degree}')
        G.atomic_json(self.out/'selection.json',dict(global_sha256=G.sha256(selection_path),models=self.t['models']))
        summaries={}
        for label,info in self.t['models'].items():
            dest=self.out/f'{label}.npz'; receipt=self.out/f'{label}.json'
            if receipt.exists():
                c=read(receipt)
                if c['model_sha256']!=info['sha256'] or c['prediction_sha256']!=G.sha256(dest): raise RuntimeError('Eval shard mismatch')
                summaries[label]=c; continue
            source=ROOT/info['path']
            if G.sha256(source)!=info['sha256']: raise RuntimeError('Frozen model changed')
            ck=torch.load(source,map_location=self.device,weights_only=False); self.actor.load_state_dict(ck['actor']); self.step=info['step']
            if digest(self.actor)!=info['actor_sha256']: raise RuntimeError('Actor mismatch')
            arrays,metrics=self.evaluate(data,'test'); G.atomic_npz(dest,**arrays)
            row=dict(model_sha256=info['sha256'],actor_sha256=info['actor_sha256'],step=self.step,degree=degree,
                     prediction_sha256=G.sha256(dest),metrics=metrics)
            G.atomic_json(receipt,row); summaries[label]=row
            if STOP: self.heartbeat('paused'); return 75
        self.finish('evaluation_complete',dict(models=self.t['models'],summaries=summaries)); return 0

    def finish(self,acceptance,extra):
        if self.a.diagnostic: raise RuntimeError('Diagnostic cannot publish formal completion')
        check_binding(self.b)
        files=[x for x in self.out.iterdir() if x.suffix in ('.pt','.npz','.jsonl','.json') and
               x.name not in ('complete.json','heartbeat.json','FAILED.json','segments.jsonl')]
        G.atomic_json(self.out/'complete.json',dict(status='complete',acceptance=acceptance,binding=self.b,identity=self.ident,
            step=self.step,start_step=self.start_step,best_step=self.best_step,parent=self.parent,ledger=self.ledger,
            elapsed_seconds=self.elapsed(),update_seconds=self.update_seconds,new_updates=self.update_count,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device.type=='cuda' else 0,finished=now(),
            artifacts={x.name:G.sha256(x) for x in files},**extra))
        self.heartbeat('complete')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--task-json',required=True); p.add_argument('--physical-gpu-index',type=int)
    p.add_argument('--cpu-threads',type=int,default=3); p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-cpu',action='store_true'); p.add_argument('--diagnostic',action='store_true'); p.add_argument('--stop-after',type=int)
    a=p.parse_args(); os.environ['RL_BUDGET_ROLE']=read(a.task_json)['mode']
    G.install_oracle_guard(); signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop); w=None
    try:
        w=Worker(a); return w.train() if w.t['mode']=='train' else w.eval()
    except Exception as exc:
        traceback.print_exc()
        if w is not None:
            G.atomic_json(w.out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=now(),step=w.step)); w.heartbeat('failed')
        return 1
    finally:
        if w is not None:
            event(w.out/'segments.jsonl',dict(event='exit',time=now(),identity=w.ident,step=w.step))
            fcntl.flock(w.lock,fcntl.LOCK_UN); w.lock.close()

if __name__=='__main__': raise SystemExit(main())
