"""Resumable pure-RL scaling training and sealed-test evaluation."""
import argparse
import fcntl
import hashlib
import io
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
from common import (ROOT,GROUPS,now,read,event,identity,digest,seed_all,rng_state,set_rng,
                    training_batch,binding,check_binding,configure,load_data,summary,verify_complete,repair_tail)

STOP=False
def stop(*_):
    global STOP
    STOP=True

def frozen_all_training():
    p=read(ROOT/'protocol.json'); records={}
    for arm in p['arms']:
        for seed in p['seeds']:
            path=ROOT/'artifacts/runs'/f'{arm}_s{seed}'/'complete.json'
            c=verify_complete(path)
            if c['acceptance']!='training_complete' or c['step']!=p['targets'][arm]:
                raise RuntimeError('Test gate: unexpected training completion')
            records[f'{arm}_s{seed}']=dict(completion_sha256=G.sha256(path),models=c['models'])
    return records

class Worker:
    def __init__(self,a):
        self.a=a; self.p=read(ROOT/'protocol.json'); self.device=configure(a.physical_gpu_index,a.cpu_threads)
        if self.device.type!='cuda' and not a.allow_cpu: raise RuntimeError('GPU required')
        self.out=Path(a.outdir).resolve()
        self.out.relative_to(ROOT)
        self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.lock.seek(0); self.lock.truncate(); json.dump(identity(),self.lock); self.lock.flush()
        self.b=binding(a.arm,a.seed)
        self.ident=dict(identity(),device=str(self.device),physical_gpu_index=a.physical_gpu_index,
                        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,
                        numpy_version=np.__version__,started=now())
        self.start_clock=time.monotonic(); self.elapsed_prior=0.; self.update_seconds=0.; self.update_count=0
        self.step=0; self.start_step=self.p['start_steps'][a.arm]; self.target=self.p['targets'][a.arm]
        self.best_metric=float('inf'); self.best_step=None; self.parent_mean=None
        self.ledger=dict(sampled_rows=0,rollout_rows=0,validation_rows=0,test_rows=0,backward_updates=0,oracle_calls=0)
        self.data_chain=''; self.last_validation=None; self.models={}; self.parent=None
        self.logq=torch.full((4,),-math.log(4),dtype=torch.float64,device=self.device)
        seed_all(a.seed); self.actor=G.build_actor(30,self.device)
        self.initial_hash=digest(self.actor)
        line=read(ROOT/'artifacts/inputs/lineage.json')[str(a.seed)]
        if self.initial_hash!=line['initial_actor_sha256']: raise RuntimeError('Initial weights do not match historical seed')
        self.optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.p['learning_rate_early'])
        self.validation=None
        if a.mode=='train':
            self.validation=load_data('validation')
            if a.diagnostic: self.validation={g:x[:64] for g,x in self.validation.items()}
            self.initialize_training()
        if not (self.out/'run.json').exists():
            G.atomic_json(self.out/'run.json',dict(binding=self.b,identity=self.ident,initial_actor_sha256=self.initial_hash,
                parent=self.parent,start_step=self.start_step,target=self.target,diagnostic=a.diagnostic))
        event(self.out/'segments.jsonl',dict(event='start',time=now(),identity=self.ident,step=self.step))
        repair_tail(self.out/'actual_calls.jsonl')
        self.heartbeat('running')

    def elapsed(self): return self.elapsed_prior+time.monotonic()-self.start_clock

    def heartbeat(self,status,**extra):
        rate=self.update_seconds/self.update_count if self.update_count else None
        G.atomic_json(self.out/'heartbeat.json',dict(self.ident,status=status,updated=now(),step=self.step,
            updates_completed=self.step,updates_target=self.target,new_updates=self.update_count,
            seconds_per_update=rate,estimated_remaining_seconds=rate*(self.target-self.step) if rate else None,
            elapsed_seconds=self.elapsed(),peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device.type=='cuda' else 0,**extra))

    def initialize_training(self):
        latest=self.out/'latest.pt'
        if latest.exists():
            if not self.a.resume: raise RuntimeError('Use --resume')
            c=torch.load(latest,map_location=self.device,weights_only=False)
            if c['binding']!=self.b or c['diagnostic']!=self.a.diagnostic: raise RuntimeError('Resume binding mismatch')
            self.actor.load_state_dict(c['actor']); self.optimizer.load_state_dict(c['optimizer'])
            for k in ('step','best_metric','best_step','parent_mean','ledger','update_seconds','update_count','data_chain',
                      'last_validation','models','parent'):
                setattr(self,k,c[k])
            self.logq=c['logq'].to(self.device); self.elapsed_prior=c['elapsed_seconds']
            set_rng(c['rng'])
            if not (self.out/'complete.json').exists():
                with (self.out/'progress.jsonl').open('r+b') as f: f.truncate(c['progress_bytes'])
                if c['best_checkpoint_bytes'] is not None:
                    (self.out/'best.pt.partial').write_bytes(c['best_checkpoint_bytes'])
                    os.replace(self.out/'best.pt.partial',self.out/'best.pt')
            return
        parent_path=None
        if self.a.arm=='U': parent_path=ROOT/'artifacts/inputs'/f's{self.a.seed}'/'latest.pt'
        elif self.a.arm in ('ME','MG','SG'):
            parent_step=self.start_step
            parent_path=ROOT/'artifacts/runs'/f'M_s{self.a.seed}'/f'step{parent_step}.pt'
            # Both source and completion are verified before starting a dependent branch.
            c=verify_complete(parent_path.parent/'complete.json')
            if G.sha256(parent_path)!=c['artifacts'][parent_path.name]: raise RuntimeError('Parent hash mismatch')
        if self.a.parent_override:
            if not self.a.diagnostic: raise RuntimeError('Parent override is diagnostic-only')
            parent_path=Path(self.a.parent_override).resolve(); parent_path.relative_to(ROOT)
        if parent_path is not None:
            h=G.sha256(parent_path)
            if self.a.arm=='U' and not self.a.parent_override and h!=read(ROOT/'artifacts/inputs/lineage.json')[str(self.a.seed)]['parent_sha256']:
                raise RuntimeError('Historical parent hash mismatch')
            c=torch.load(parent_path,map_location=self.device,weights_only=False)
            old='networks' in c
            self.actor.load_state_dict(c['networks']['actor'] if old else c['actor'])
            self.optimizer.load_state_dict(c['optimizers']['actor'] if old else c['optimizer'])
            self.step=c['step']; set_rng(c['rng'])
            if not self.a.diagnostic and self.step!=self.start_step: raise RuntimeError('Wrong parent endpoint')
            if self.a.diagnostic: self.start_step=self.step
            self.parent=dict(path=str(parent_path),sha256=h,step=self.step,actor_sha256=digest(self.actor),
                optimizer_state_inherited=True,batchnorm_state_inherited=True,rng_inherited=True)

    def call(self,phase,rows,fn):
        token=f'{os.getpid()}-{time.time_ns()}'
        event(self.out/'actual_calls.jsonl',dict(id=token,event='start',phase=phase,rows=rows,step=self.step,time=now()))
        result=fn()
        event(self.out/'actual_calls.jsonl',dict(id=token,event='complete',phase=phase,rows=rows,step=self.step,time=now()))
        return result

    def score(self,pins,phase):
        def forward():
            was_training=self.actor.training; self.actor.eval(); ls=[]; ts=[]
            with torch.no_grad():
                for start in range(0,len(pins),64):
                    b=pins[start:start+64]; keep=len(b)
                    if keep<64: b=np.concatenate([b,np.repeat(b[-1:],64-keep,axis=0)])
                    topology,_=self.actor(b,True); topology=topology[:keep].cpu().numpy()
                    ts.append(topology); ls.append(G.lengths(b[:keep],topology))
            self.actor.train(was_training)
            self.ledger[phase+'_rows']+=len(pins)
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
        if self.parent_mean is None and self.a.arm in ('ME','MG','SG'): self.parent_mean=metrics['mean']
        eligible=self.parent_mean is None or metrics['mean']<=self.parent_mean*(1+self.p['selection_mean_tolerance'])
        if eligible and metrics['worst']<self.best_metric:
            self.best_metric=metrics['worst']; self.best_step=self.step
            G.atomic_torch(self.out/'best.pt',dict(actor=self.actor.state_dict(),step=self.step,binding=self.b,
                                                  validation=metrics,actor_sha256=digest(self.actor)))
        self.last_validation=metrics
        row=dict(type='validation',time=now(),step=self.step,metrics=metrics,eligible=eligible,
                 best_step=self.best_step,best_metric=self.best_metric,elapsed_seconds=self.elapsed())
        event(self.out/'progress.jsonl',row); print(json.dumps(row),flush=True)
        return arrays

    def state(self):
        return dict(binding=self.b,diagnostic=self.a.diagnostic,actor=self.actor.state_dict(),optimizer=self.optimizer.state_dict(),
            step=self.step,best_metric=self.best_metric,best_step=self.best_step,parent_mean=self.parent_mean,
            logq=self.logq,ledger=self.ledger,update_seconds=self.update_seconds,update_count=self.update_count,
            data_chain=self.data_chain,last_validation=self.last_validation,models=self.models,parent=self.parent,
            rng=rng_state(),elapsed_seconds=self.elapsed(),initial_actor_sha256=self.initial_hash,
            best_checkpoint_bytes=(self.out/'best.pt').read_bytes() if (self.out/'best.pt').exists() else None,
            progress_bytes=(self.out/'progress.jsonl').stat().st_size)

    def checkpoint(self,endpoint=False):
        if endpoint:
            name=f'step{self.step}.pt'
            G.atomic_torch(self.out/name,self.state())
            self.models[f'step{self.step}']=dict(file=name,sha256=G.sha256(self.out/name),step=self.step,actor_sha256=digest(self.actor))
        G.atomic_torch(self.out/'latest.pt',self.state())
        self.heartbeat('running')

    def update(self):
        start=time.monotonic(); step=self.step+1
        pins,groups=training_batch(self.a.arm,self.a.seed,step)
        rate=self.p['learning_rate_early'] if step<=self.p['learning_rate_boundary'] else self.p['learning_rate_late']
        for param in self.optimizer.param_groups: param['lr']=rate
        baseline,_=self.score(pins,'rollout')
        seed_all(G.derived_seed('actor_action',self.a.seed,step))
        self.actor.train()
        def sample():
            topology,logprob=self.actor(pins,False)
            length=G.lengths(pins,topology.detach().cpu().numpy())
            return logprob,torch.as_tensor(length,dtype=torch.float32,device=self.device)
        logprob,cost=self.call('sampled',len(pins),sample)
        advantage=(cost-torch.as_tensor(baseline,dtype=torch.float32,device=self.device)).detach()
        robust=self.a.arm in ('D','MG','SG')
        old_q=self.logq.exp().clone()
        if robust:
            group_loss=(advantage*logprob).reshape(4,16).mean(1)
            loss=(old_q.to(group_loss.dtype)*group_loss).sum()
        else: loss=(advantage*logprob).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.p['gradient_clip'])
        if not torch.isfinite(grad): raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        risks=cost.detach().reshape(4,16).mean(1) if self.a.arm!='U' else None
        if robust:
            self.logq=self.logq+self.p['group_eta']*risks.to(torch.float64)
            self.logq=self.logq-torch.logsumexp(self.logq,dim=0)
        self.step=step; self.update_count+=1; self.update_seconds+=time.monotonic()-start
        self.ledger['sampled_rows']+=64; self.ledger['backward_updates']+=1
        data_hash=hashlib.sha256(pins.tobytes()).hexdigest()
        self.data_chain=hashlib.sha256((self.data_chain+data_hash).encode()).hexdigest()
        event(self.out/'progress.jsonl',dict(type='update',time=now(),step=step,loss=float(loss.detach()),
            sampled_mean=float(cost.mean()),baseline_mean=float(baseline.mean()),gradient_norm=float(grad),
            learning_rate=rate,q_used=old_q.cpu().tolist(),q_next=self.logq.exp().cpu().tolist(),
            q_entropy=float(-(self.logq.exp()*self.logq).sum()),group_risks=risks.cpu().tolist() if risks is not None else None,
            data_sha256=data_hash,elapsed_seconds=self.elapsed(),update_seconds=self.update_seconds))

    def guard_resources(self):
        if shutil.disk_usage(ROOT).free<20*2**30: return 'disk_below_20GiB'
        if self.device.type=='cuda' and torch.cuda.mem_get_info()[0]<2*2**30: return 'free_gpu_memory_below_2GiB'
        return None

    def train(self):
        if (self.out/'complete.json').exists(): verify_complete(self.out/'complete.json'); return 0
        if not (self.out/'latest.pt').exists():
            (self.out/'progress.jsonl').touch()
            if self.step>=self.p.get('validation_start',2048): self.validate()
            self.checkpoint(endpoint=self.step in self.p['endpoints'])
        while self.step<self.target:
            reason=self.guard_resources() if self.step%32==0 else None
            if STOP or reason:
                self.checkpoint(); self.heartbeat('paused',reason=reason or 'signal'); return 75
            self.update()
            if (self.step>=self.p.get('validation_start',2048) and self.step%self.p['validation_every']==0) or self.step==self.target:
                self.validate()
            if self.step%self.p['checkpoint_every']==0 or self.step in self.p['endpoints']:
                self.checkpoint(endpoint=self.step in self.p['endpoints'])
            if self.a.stop_after is not None and self.step>=self.a.stop_after:
                if not (self.out/'best.pt').exists(): self.validate()
                self.checkpoint(); self.heartbeat('diagnostic_paused' if self.a.diagnostic else 'paused',reason='stop_after'); return 0
        self.checkpoint(endpoint=True)
        best=torch.load(self.out/'best.pt',map_location=self.device,weights_only=False)
        self.actor.load_state_dict(best['actor'])
        arrays,metrics=self.evaluate(self.validation,'validation'); G.atomic_npz(self.out/'results.npz',**arrays)
        self.models['best']=dict(file='best.pt',sha256=G.sha256(self.out/'best.pt'),step=self.best_step,actor_sha256=digest(self.actor))
        self.finish('training_complete',dict(models=self.models,selected_validation=metrics))
        return 0

    def eval(self):
        if (self.out/'complete.json').exists(): verify_complete(self.out/'complete.json'); return 0
        frozen=frozen_all_training()
        train=ROOT/'artifacts/runs'/f'{self.a.arm}_s{self.a.seed}'
        models=frozen[f'{self.a.arm}_s{self.a.seed}']['models']
        selection=dict(training=frozen,selected_models=models,frozen_at=now())
        if (self.out/'selection.json').exists():
            prior=read(self.out/'selection.json')
            if prior['training']!=frozen: raise RuntimeError('Frozen training changed')
        else: G.atomic_json(self.out/'selection.json',selection)
        data=load_data('test'); summaries={}
        for label,info in models.items():
            dest=self.out/f'{label}.npz'; receipt=self.out/f'{label}.json'
            if receipt.exists():
                c=read(receipt)
                if c['model_sha256']!=info['sha256'] or c['prediction_sha256']!=G.sha256(dest): raise RuntimeError('Eval shard mismatch')
                summaries[label]=c; continue
            source=train/info['file']
            if G.sha256(source)!=info['sha256']: raise RuntimeError('Frozen model changed')
            ck=torch.load(source,map_location=self.device,weights_only=False)
            self.actor.load_state_dict(ck['actor']); self.step=info['step']
            if digest(self.actor)!=info['actor_sha256']: raise RuntimeError('Actor mismatch')
            started=time.monotonic(); arrays,metrics=self.evaluate(data,'test')
            G.atomic_npz(dest,**arrays)
            c=dict(model_sha256=info['sha256'],actor_sha256=info['actor_sha256'],step=self.step,
                   prediction_sha256=G.sha256(dest),metrics=metrics,seconds=time.monotonic()-started)
            G.atomic_json(receipt,c); summaries[label]=c
            if STOP: self.heartbeat('paused',reason='signal'); return 75
        self.finish('evaluation_complete',dict(models=models,summaries=summaries))
        return 0

    def finish(self,acceptance,extra):
        if self.a.diagnostic: raise RuntimeError('Diagnostic cannot produce formal completion')
        check_binding(self.b)
        files=[p for p in self.out.iterdir() if p.is_file() and p.suffix in ('.pt','.npz','.jsonl','.json')
               and p.name not in ('complete.json','heartbeat.json','FAILED.json','segments.jsonl')]
        G.atomic_json(self.out/'complete.json',dict(status='complete',acceptance=acceptance,binding=self.b,identity=self.ident,
            step=self.step,best_step=self.best_step,start_step=self.start_step,initial_actor_sha256=self.initial_hash,
            parent=self.parent,ledger=self.ledger,elapsed_seconds=self.elapsed(),update_seconds=self.update_seconds,
            new_updates=self.update_count,peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device.type=='cuda' else 0,
            finished=now(),artifacts={p.name:G.sha256(p) for p in files},**extra))
        self.heartbeat('complete'); print(json.dumps(dict(status='complete',out=str(self.out))),flush=True)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mode',choices=['train','eval'],default='train'); p.add_argument('--arm',required=True,choices=['U','M','D','ME','MG','SG'])
    p.add_argument('--seed',required=True,type=int); p.add_argument('--outdir',required=True)
    p.add_argument('--physical-gpu-index',type=int); p.add_argument('--cpu-threads',type=int,default=3)
    p.add_argument('--resume',action='store_true'); p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--diagnostic',action='store_true'); p.add_argument('--stop-after',type=int); p.add_argument('--parent-override')
    a=p.parse_args(); os.environ['RL_BUDGET_ROLE']=a.mode
    G.install_oracle_guard(); signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    w=None
    try:
        w=Worker(a)
        return w.train() if a.mode=='train' else w.eval()
    except Exception as exc:
        traceback.print_exc()
        if w is not None:
            G.atomic_json(w.out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=now(),step=w.step))
            w.heartbeat('failed',error=repr(exc))
        return 1
    finally:
        if w is not None:
            event(w.out/'segments.jsonl',dict(event='exit',time=now(),identity=w.ident,step=w.step))
            fcntl.flock(w.lock,fcntl.LOCK_UN); w.lock.close()

if __name__=='__main__': raise SystemExit(main())
