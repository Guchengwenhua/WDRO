"""Matched ERM / basic Random-Sobol WDRO, with exact resumable state."""
from __future__ import annotations
import argparse, copy, fcntl, json, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, kron, eye
from core import *

def save_pt(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.partial.{os.getpid()}')
    with q.open('wb') as f: torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def finite_risk(loss, cost, rho):
    """Each row is a conditional distribution; common average transport budget."""
    b,k=loss.shape
    eq=kron(eye(b,format='csr'),csr_matrix(np.ones((1,k))),format='csr')
    result=linprog(-loss.ravel()/b,A_ub=csr_matrix(cost.reshape(1,-1)/b),b_ub=[rho],A_eq=eq,b_eq=np.ones(b),bounds=(0,None),method='highs',options={'threads':1})
    if not result.success: raise RuntimeError('Finite risk LP failed: '+result.message)
    mass=result.x.reshape(b,k);avg_cost=float((mass*cost).sum()/b)
    err=float(np.max(np.abs(mass.sum(1)-1)))
    if err>1e-7 or avg_cost>rho+1e-7 or mass.min()<-1e-8: raise ValueError('infeasible LP witness')
    return {'risk':float((mass*loss).sum()/b),'average_cost':avg_cost,'row_error':err,
            'nominal':float(loss[:,0].mean()),'unconstrained_max':float(loss.max(1).mean())},mass

class Experiment:
    def __init__(self,args):
        self.args=args;self.p=read_protocol();self.smoke=args.smoke
        if self.smoke:
            self.p=copy.deepcopy(self.p);self.p.update(updates=6,batch_size=8,validation_every=2,checkpoint_every=2,validation_count=8,test_count=8,attack_count=4,greedy_microbatch=8)
            if args.smoke_batch_size:
                size=args.smoke_batch_size
                self.p.update(batch_size=size,validation_count=size,test_count=size,attack_count=min(32,size),greedy_microbatch=size)
        self.run_id=args.run_id or (('smoke_' if self.smoke else '')+f'{args.family}_{args.method}_s{args.seed}')
        self.out=ROOT/'artifacts/runs'/self.run_id;self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.device=torch.device(args.device)
        set_active_device(self.device)
        self.binding=digest(self.p) if self.smoke else verify_frozen()[0]
        self.identity={'family':args.family,'method':args.method,'seed':args.seed,'protocol_sha256':sha(ROOT/'protocol.json'),
            'frozen_sha256':self.binding,'scope':'smoke' if self.smoke else 'degree10_pilot','updates':self.p['updates'],'run_id':self.run_id}
        self.attempt=f'{time.time_ns()}_{os.getpid()}';self.stopped=False;self.current=0;self.active_update=0
        signal.signal(signal.SIGTERM,self.request_stop);signal.signal(signal.SIGINT,self.request_stop)
        torch.set_num_threads(int(os.environ.get('PAPERS_WDRO_CPU_THREADS','1')))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        seed_all(args.seed)
        params=dict(self.p['parameters'][args.family],cache_path=str(self.out/'references.sqlite'))
        self.adapter=make_adapter(args.family,self.p['degree'],self.device,params)
        self.critic=Baseline(self.device)
        self.optimizer=torch.optim.Adam(list(self.adapter.parameters())+list(self.critic.parameters()),lr=self.p['learning_rate'])
        self.price=self.p['lambda_initial'];self.best={'update':-1,'objective':1e100};self.calls={};self.elapsed=0.0
        self.training_chain=hashlib.sha256(b'').hexdigest();self.loaded_shard=None;self.shard_data=None
        calibration=json.loads((ROOT/'artifacts/data/calibration.json').read_text())
        self.rho=calibration['unrooted' if args.family=='rest' else 'rooted']
        self.identity['rho']=self.rho
        if (self.out/'latest.pt').exists():
            state=torch.load(self.out/'latest.pt',map_location=self.device,weights_only=False)
            if state['identity']!=self.identity: raise ValueError('resume identity mismatch')
            self.adapter.load_state_dict(state['adapter']);self.critic.load_state_dict(state['critic']);self.optimizer.load_state_dict(state['optimizer'])
            self.current=state['update'];self.price=state['lambda'];self.best=state['best'];self.calls=state['calls'];self.elapsed=state['elapsed']
            self.training_chain=state['training_chain'];restore_rng(state['rng'])
        else:
            if (self.out/'run.json').exists():
                old=json.loads((self.out/'run.json').read_text())
                if old['identity']!=self.identity: raise ValueError('existing run identity mismatch')
            atomic_json(self.out/'run.json',{'identity':self.identity,'created_at':time.time(),'params':params,'initialization':getattr(self.adapter,'initialization','fresh_same_seed_within_pair')})
            save_pt(self.out/'initial.pt',{'adapter':self.adapter.state_dict(),'identity':self.identity})
        self.start_time=time.time()
        self.heartbeat('initialized')

    def request_stop(self,*args): self.stopped=True

    def heartbeat(self,phase):
        atomic_json(self.out/'heartbeat.json',{'time':time.time(),'timestamp':time.time(),'phase':phase,'update':self.current,'total_updates':self.p['updates'],
            'identity':proc_identity(),'attempt':self.attempt,'elapsed_seconds':self.elapsed+time.time()-getattr(self,'start_time',time.time()),'device':str(self.device),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'lambda':self.price,'rho':self.rho})

    def state(self):
        return {'identity':self.identity,'adapter':self.adapter.state_dict(),'critic':self.critic.state_dict(),'optimizer':self.optimizer.state_dict(),
            'rng':rng_state(),'update':self.current,'lambda':self.price,'rho':self.rho,'best':self.best,'calls':self.calls,'training_chain':self.training_chain,
            'elapsed':self.elapsed+time.time()-self.start_time}

    def checkpoint(self):
        for name in ['progress.jsonl','actual_calls.jsonl','validation.jsonl']:
            path=self.out/name
            if path.exists():
                with path.open('rb') as handle:os.fsync(handle.fileno())
        save_pt(self.out/'latest.pt',self.state())

    def score(self,pins,greedy,category):
        self.heartbeat(category)
        requested=len(pins);self.calls[category]=self.calls.get(category,0)+requested
        append(self.out/'actual_calls.jsonl',{'event':'started','attempt':self.attempt,'completed_update':self.current,'target_update':self.active_update,'category':category,'nets':requested,'time':time.time()})
        cost,logp,metrics=self.adapter.evaluate(pins,greedy)
        if cost.shape!=(requested,) or logp.shape!=(requested,): raise ValueError('adapter outputs must have shape [batch]')
        if not torch.isfinite(cost).all() or not torch.isfinite(logp).all(): raise ValueError('nonfinite cost/logprob')
        clean={k:np.asarray(v,dtype=np.float64) for k,v in metrics.items()}
        if any(v.shape!=(requested,) or not np.isfinite(v).all() for v in clean.values()): raise ValueError('invalid adapter metrics')
        append(self.out/'actual_calls.jsonl',{'event':'completed','attempt':self.attempt,'completed_update':self.current,'target_update':self.active_update,'category':category,'nets':requested,'time':time.time()})
        return cost,logp,clean

    def greedy(self,pins,category):
        mode=self.adapter.training;self.adapter.eval();outs=[];metrics={};step=self.p['greedy_microbatch']
        with torch.no_grad():
            for i in range(0,len(pins),step):
                part=pins[i:i+step];n=len(part)
                # Fix inference batch shape, including the final partial batch.
                if n<step: part=np.concatenate([part,np.repeat(part[-1:],step-n,axis=0)])
                c,_,m=self.score(part,True,category);outs.append(c.cpu().numpy()[:n])
                for key,value in m.items():metrics.setdefault(key,[]).append(value[:n])
        self.adapter.train(mode)
        return np.concatenate(outs),{k:np.concatenate(v) for k,v in metrics.items()}

    def train_batch(self,u):
        p=self.p
        if self.smoke:
            batch=dataset(p,f'smoke_train_u{u:06d}','uniform',self.args.seed,p['batch_size'])
            sup=generate_support(batch,training_seed=self.args.seed,update=u,search=p['search']);ids,_=shortlist_support(sup,4)
            chosen=sup.pins[np.arange(len(batch.pins))[:,None],ids]
            return batch.pins,chosen,costs(batch.pins,chosen,self.args.family!='rest')
        first=1+((u-1)//p['shard_updates'])*p['shard_updates']
        if self.loaded_shard!=first:
            path=ROOT/f'artifacts/data/train_s{self.args.seed}_{first:04d}.npz'
            with np.load(path,allow_pickle=False) as data:self.shard_data={k:data[k] for k in data.files}
            self.loaded_shard=first
        d=self.shard_data;idx=u-first
        return d['pins'][idx],d['shortlist'][idx],d['cost_unrooted' if self.args.family=='rest' else 'cost_rooted'][idx]

    def eval_dataset(self,split,dist):
        if self.smoke:
            count=self.p[split+'_count'];batch=dataset(self.p,'smoke_'+split,dist,9901,count)
            arrays={'pins':batch.pins}
            if split=='attack':
                support=generate_support(batch,training_seed=9902,update=1,search=self.p['search']);arrays['candidates']=support.pins
                arrays['cost_rooted']=costs(batch.pins,support.pins,True);arrays['cost_unrooted']=costs(batch.pins,support.pins,False)
            return arrays
        with np.load(ROOT/f'artifacts/data/{split}_{dist}.npz',allow_pickle=False) as data:return {k:data[k] for k in data.files}

    def evaluate(self,split,label):
        data={};summary={}
        for dist in self.p['distributions']:
            pins=self.eval_dataset(split,dist)['pins'];loss,metrics=self.greedy(pins,label+'_'+dist)
            metrics={'cost':loss,**metrics}
            for key,v in metrics.items():data[dist+'__'+key]=v
            summary[dist]={key:{'mean':float(v.mean()),'p95':float(np.quantile(v,.95))} for key,v in metrics.items()}
        return data,summary

    def validation(self):
        saved=rng_state();_,summary=self.evaluate('validation','validation')
        objective=max(summary[g]['cost']['mean'] for g in self.p['distributions'] if g!='uniform')
        if objective<self.best['objective']:
            self.best={'update':self.current,'objective':objective,'nominal':summary['uniform']['cost']['mean']}
            save_pt(self.out/'best.pt',{'adapter':self.adapter.state_dict(),'identity':self.identity,'best':self.best,'lambda':self.price})
        append(self.out/'validation.jsonl',{'update':self.current,'summary':summary,'best':self.best,'attempt':self.attempt})
        restore_rng(saved)

    def run(self):
        if self.current==0 and self.best['update']<0:self.validation();self.checkpoint()
        end=min(self.args.stop_after or self.p['updates'],self.p['updates'])
        for u in range(self.current+1,end+1):
            self.active_update=u
            t=time.time();pins,short,c=self.train_batch(u);mean_cost=0.0;selected=None
            if self.args.method=='wdro':
                flat=short.reshape(-1,self.p['degree'],2);true,_=self.greedy(flat,'search_true');true=true.reshape(c.shape)
                selected=np.argmax(true-self.price*c,axis=1);rows=np.arange(len(pins));pins=short[rows,selected]
                mean_cost=float(c[rows,selected].mean())
                self.price=max(0.0,self.price+self.p['lambda_lr']*(mean_cost-self.rho))
                if not np.isfinite(self.price) or self.price>1e6:raise ValueError('dual price divergence')
            self.adapter.train();self.critic.train();self.optimizer.zero_grad(set_to_none=True)
            true,logp,metrics=self.score(pins,False,'train_sampled');base=self.critic(pins)
            actor_loss=((true.detach()-base.detach())*logp).mean();critic_loss=(base-true.detach()).square().mean()
            loss=actor_loss+critic_loss;loss.backward()
            params=list(self.adapter.parameters())+list(self.critic.parameters())
            norm=torch.nn.utils.clip_grad_norm_(params,self.p['gradient_clip'],error_if_nonfinite=True)
            self.optimizer.step();self.current=u
            previous_chain=self.training_chain
            input_bytes=np.ascontiguousarray(pins).tobytes()
            self.training_chain=hashlib.sha256(bytes.fromhex(self.training_chain)+input_bytes).hexdigest()
            append(self.out/'progress.jsonl',{'update':u,'attempt':self.attempt,'mean_cost':float(true.mean()),'actor_loss':float(actor_loss.detach()),
                'critic_loss':float(critic_loss.detach()),'grad_norm':float(norm),'lambda':self.price,'rho':self.rho,'transport_mean':mean_cost,
                'seconds':time.time()-t,'time':time.time(),'input_sha256':hashlib.sha256(input_bytes).hexdigest(),
                'previous_training_chain':previous_chain,'training_chain':self.training_chain,
                'selected_columns':selected.tolist() if selected is not None else None})
            if u%self.p['validation_every']==0:self.validation()
            if u%self.p['checkpoint_every']==0 or self.stopped or u==end:self.checkpoint()
            self.heartbeat('training')
            if self.stopped: self.heartbeat('capacity_paused');return 75
        if self.current<self.p['updates']:self.heartbeat('requested_checkpoint_stop');return 75
        # Selection is finished before either test set is evaluated.
        selected=torch.load(self.out/'best.pt',map_location=self.device,weights_only=False)
        self.adapter.load_state_dict(selected['adapter']);best_data,best_summary=self.evaluate('test','test_best')
        atomic_npz(self.out/'test_best.npz',**best_data)
        attack_data={};attack_summary={}
        for dist in self.p['distributions']:
            a=self.eval_dataset('attack',dist);shape=a['candidates'].shape
            loss,_=self.greedy(a['candidates'].reshape(-1,self.p['degree'],2),'attack_'+dist);loss=loss.reshape(shape[:2])
            cost=a['cost_unrooted' if self.args.family=='rest' else 'cost_rooted']
            summary,mass=finite_risk(loss,cost,self.rho);attack_summary[dist]=summary
            attack_data[dist+'__loss']=loss;attack_data[dist+'__cost']=cost;attack_data[dist+'__mass']=mass
        atomic_npz(self.out/'attack.npz',**attack_data)
        init=torch.load(self.out/'initial.pt',map_location=self.device,weights_only=False)
        self.adapter.load_state_dict(init['adapter']);initial_data,initial_summary=self.evaluate('test','test_initial')
        atomic_npz(self.out/'test_initial.npz',**initial_data)
        # latest remains the resumable final training state, best remains selected weights.
        summary={'best':self.best,'test_best':best_summary,'test_initial':initial_summary,'attack':attack_summary,
            'calls_effective_checkpoint_trajectory':self.calls,'actual_calls_log':'actual_calls.jsonl',
            'cuda_max_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if self.device.type=='cuda' else None,
            'cuda_max_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if self.device.type=='cuda' else None,
            'seconds':self.elapsed+time.time()-self.start_time,'identity':self.identity}
        if self.args.family=='rest':
            row=self.adapter.cache.db.execute('SELECT COALESCE(SUM(requested),0), COALESCE(SUM(misses_started),0), COALESCE(SUM(misses_completed),0) FROM calls').fetchone()
            summary['geosteiner_cache_actual_receipts']={'requested':int(row[0]),'misses_started':int(row[1]),'misses_completed':int(row[2]),'includes_retries':True}
        if self.args.family=='hub':
            summary['mspd_internal_pd_sweeps_per_completed_score']=11
        atomic_json(self.out/'metrics.json',summary)
        artifacts={name:sha(self.out/name) for name in ['initial.pt','latest.pt','best.pt','test_best.npz','test_initial.npz','attack.npz','metrics.json','run.json']}
        if self.stopped:self.heartbeat('capacity_paused_after_eval');return 75
        atomic_json(self.out/'complete.json',{'status':'complete','scope':self.identity['scope'],'protocol_sha256':self.identity['protocol_sha256'],
            'frozen_sha256':self.binding,'completed_updates':self.current,'identity':self.identity,'artifacts':artifacts,'time':time.time()})
        self.heartbeat('complete');return 0

    def close(self):
        if hasattr(self.adapter,'close'):self.adapter.close()
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['rest','timing','hub','buffer'],required=True);ap.add_argument('--method',choices=['erm','wdro'],required=True)
    ap.add_argument('--seed',type=int,default=101);ap.add_argument('--device',default='cuda');ap.add_argument('--run-id');ap.add_argument('--smoke',action='store_true');ap.add_argument('--stop-after',type=int)
    ap.add_argument('--smoke-batch-size',type=int)
    args=ap.parse_args();e=Experiment(args)
    try:code=e.run()
    except torch.cuda.OutOfMemoryError as error:
        # Keep the last complete update. A failed forward may already have
        # changed batch-normalization buffers/RNG and must not be checkpointed.
        atomic_json(e.out/f'capacity_oom_{e.attempt}.json',{'status':'capacity_pause','error':str(error),'time':time.time(),'resume_from':'latest.pt'})
        code=75
    finally:e.close()
    atomic_json(e.out/f'exit_{e.attempt}.json',{'exit_code':code,'identity':proc_identity(),'time':time.time(),'scope':e.identity['scope']})
    return code

if __name__=='__main__':sys.exit(main())
