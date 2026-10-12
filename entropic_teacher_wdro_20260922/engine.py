from __future__ import annotations
import argparse,copy,fcntl,json,math,os,random,signal,time,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
import geometry as geo
from transport import solve,distribution,sample,tail_loss

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
P=json.loads((ROOT/'protocol.json').read_text())
STOP=False
def stop(*_):
    global STOP
    STOP=True
def identity(pid=None):
    pid=pid or os.getpid()
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X'):return None
        return dict(pid=pid,start_ticks=fields[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None
def event(path,v):
    with Path(path).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush()
def seed(*parts):return geo.derived_seed(P['namespace'],*parts)
def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def binding():
    return {str(p.relative_to(ROOT)):geo.sha256(p) for p in sorted(list(ROOT.glob('*.py'))+[ROOT/'protocol.json']+list((ROOT/'vendor/models').glob('*.py')))}
def data(n,degree,dist,namespace,s=0):return geo.make_data(n,degree,dist,P['namespace']+'/'+namespace,s)
def candidates(x,rng,k):
    delta=np.clip(rng.normal(0,P['sigma'],(len(x),k-1,x.shape[1],2)),-P['epsilon'],P['epsilon'])
    return np.concatenate((x[:,None],geo.legalize(x,delta)),axis=1)
class Value(nn.Module):
    def __init__(self):
        super().__init__();self.point=nn.Sequential(nn.Linear(2,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,x):
        h=self.point(x);return self.head(torch.cat((h.mean(1),h.amax(1)),-1)).squeeze(-1)

class Run:
    def __init__(self,t,device='cuda:0',base=ART/'runs'):
        self.t=t;self.device=torch.device(device);self.out=Path(base)/t['id'];self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.bind=binding();self.start=time.monotonic();self.prior=0.;self.step=0
        self.calls=dict(search=0,teacher=0,validation=0,test=0,sampled=0,padded=0)
        self.lam=P['lambda_init'];self.eta=1.;self.best_metric=float('inf');self.parent_nominal=None
        self.latest_val=None;self.parent_hash=None;self.best_step=0
        random.seed(t['seed']);np.random.seed(t['seed']);torch.manual_seed(t['seed'])
        self.actor=geo.build_actor(t['degree'],self.device);self.value=Value().to(self.device)
        self.teacher=None
        if t.get('parent'):
            parent=Path(base)/t['parent'];receipt=json.loads((parent/'complete.json').read_text())
            self.parent_hash=geo.sha256(parent/'best.pt')
            if receipt['best_sha256']!=self.parent_hash:raise RuntimeError('parent hash mismatch')
            c=torch.load(parent/'best.pt',map_location=self.device,weights_only=False)
            self.actor.load_state_dict(c['actor']);self.value.load_state_dict(c['value'])
            if t['method']=='excess':
                self.teacher=copy.deepcopy(self.actor).eval()
                for v in self.teacher.parameters():v.requires_grad_(False)
        lr=P['pre_lr'] if t['stage']=='pre' else P['post_lr']
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=lr);self.vo=torch.optim.Adam(self.value.parameters(),lr=lr)
        latest=self.out/'latest.pt'
        if latest.exists():
            c=torch.load(latest,map_location=self.device,weights_only=False)
            if c['binding']!=self.bind or c['task']!=t or c['parent_sha256']!=self.parent_hash:raise RuntimeError('resume binding mismatch')
            self.actor.load_state_dict(c['actor']);self.value.load_state_dict(c['value'])
            self.ao.load_state_dict(c['actor_optimizer']);self.vo.load_state_dict(c['value_optimizer'])
            for key in ('step','lam','eta','calls','best_metric','parent_nominal','latest_val','best_step'):setattr(self,key,c[key])
            self.prior=c['elapsed_seconds']
            torch.set_rng_state(c['rng'].cpu())
            if c['cuda_rng'] and self.device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in c['cuda_rng']])
            progress=self.out/'progress.jsonl'
            if progress.exists():
                with progress.open('r+b') as f:f.truncate(c['progress_bytes'])
        geo.atomic_json(self.out/'run.json',dict(task=t,binding=self.bind,parent_sha256=self.parent_hash,identity=identity(),device=str(self.device),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))
        self.heartbeat('running')
    def elapsed(self):return self.prior+time.monotonic()-self.start
    def heartbeat(self,status):
        geo.atomic_json(self.out/'heartbeat.json',dict(**identity(),time=time.time(),status=status,step=self.step,target=self.t['updates'],elapsed_seconds=self.elapsed(),calls=self.calls,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_mib=torch.cuda.max_memory_allocated()/2**20 if self.device.type=='cuda' else 0))
    def checkpoint(self):
        p=self.out/'progress.jsonl'
        c=dict(task=self.t,binding=self.bind,parent_sha256=self.parent_hash,actor=self.actor.state_dict(),value=self.value.state_dict(),actor_optimizer=self.ao.state_dict(),value_optimizer=self.vo.state_dict(),elapsed_seconds=self.elapsed(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],progress_bytes=p.stat().st_size if p.exists() else 0)
        for key in ('step','lam','eta','calls','best_metric','parent_nominal','latest_val','best_step'):c[key]=getattr(self,key)
        geo.atomic_torch(self.out/'latest.pt',c);self.heartbeat('running')
    def score(self,x,phase='search',actor=None):
        actor=actor or self.actor;mode=actor.training;actor.eval();out=[];micro=P['microbatch']
        event(self.out/'actual_calls.jsonl',dict(event='start',phase=phase,rows=len(x),time=time.time()))
        with torch.no_grad():
            for i in range(0,len(x),micro):
                v=x[i:i+micro];n=len(v)
                if n<micro:v=np.concatenate((v,np.repeat(v[-1:],micro-n,0)))
                top,_=actor(v,True);out.append(geo.lengths(v[:n],top[:n].cpu().numpy()))
                self.calls['padded']+=micro
        self.calls[phase]+=len(x);actor.train(mode)
        event(self.out/'actual_calls.jsonl',dict(event='complete',phase=phase,rows=len(x),time=time.time()))
        return np.concatenate(out)
    def validate(self):
        self.heartbeat('validation');values={};arrays=[]
        for d in P['distributions']:
            namespace='tune-validation' if self.t.get('cohort')=='pilot' else 'validation'
            v=self.score(data(self.t.get('val_n',P['validation_n']),self.t['degree'],d,namespace),phase='validation')
            values[d]=float(v.mean());arrays.append(v)
        if self.parent_nominal is None:
            self.parent_nominal=values['uniform'];self.eta=float(np.quantile(np.concatenate(arrays)/math.sqrt(self.t['degree']),P['alpha']))
        eligible=self.t['stage']=='pre' or values['uniform']<=self.parent_nominal*(1+P['nominal_tolerance'])
        metric=values['uniform'] if self.t['stage']=='pre' else max(values[d] for d in P['distributions'][1:])
        if eligible and metric<self.best_metric:
            self.best_metric=metric;self.best_step=self.step
            geo.atomic_torch(self.out/'best.pt',dict(actor=self.actor.state_dict(),value=self.value.state_dict(),step=self.step,validation=values,binding=self.bind))
        self.latest_val=values
        event(self.out/'validation.jsonl',dict(step=self.step,values=values,eligible=eligible,metric=metric,best_step=self.best_step))
        self.heartbeat('running')
    def update(self,x,denominator=None):
        torch.manual_seed(seed('action',self.t['degree'],self.t['seed'],self.t['stage'],self.step))
        self.actor.train();top,logp=self.actor(x,False)
        raw=geo.lengths(x,top.detach().cpu().numpy());norm=raw/math.sqrt(self.t['degree'])
        target=raw/denominator-1 if denominator is not None else norm
        if self.t['method']=='tail':target=tail_loss(target,self.eta,P['alpha'],P['beta'])
        target=torch.as_tensor(target,dtype=torch.float32,device=self.device)
        pred=self.value(torch.as_tensor(x,device=self.device));loss=((target-pred).detach()*logp).mean();vl=(target.detach()-pred).square().mean()
        if not torch.isfinite(loss+vl):raise RuntimeError('nonfinite update')
        self.ao.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1);self.ao.step()
        self.vo.zero_grad(set_to_none=True);vl.backward();nn.utils.clip_grad_norm_(self.value.parameters(),1);self.vo.step()
        if self.t['method']=='tail':self.eta=max(0.,self.eta-P['eta_lr']*(1-float((norm>self.eta).mean())/(1-P['alpha'])))
        self.calls['sampled']+=len(x)
        event(self.out/'actual_calls.jsonl',dict(event='complete',phase='sampled',rows=len(x),time=time.time()))
        return dict(sampled_length=float(raw.mean()),actor_loss=loss.item(),value_loss=vl.item(),eta=self.eta)
    def train(self,stop_after=None):
        if self.latest_val is None:self.validate();self.checkpoint()
        while self.step<self.t['updates']:
            if STOP or (ART/'operations/PAUSE').exists() or (stop_after is not None and self.step>=stop_after):
                self.checkpoint();return False
            if self.step%16==0:
                if shutil.disk_usage(ART).free<P['resources']['disk_reserve_gib']*2**30 or (self.device.type=='cuda' and torch.cuda.mem_get_info()[0]<2*2**30):
                    self.checkpoint();self.heartbeat('waiting_resources');return False
            start=time.monotonic();t=self.t;method=t['method'];b=t.get('batch',P['batch']);k=t.get('k',P['candidates'])
            x=data(b,t['degree'],'uniform',f"train/{t['stage']}/{self.step}",t['seed']);rec=dict(step=self.step+1);den=None
            if method!='erm':
                rng=np.random.default_rng(seed('candidate',t['degree'],t['seed'],t['stage'],self.step));z=candidates(x,rng,k)
                length=self.score(z.reshape(-1,t['degree'],2)).reshape(b,k);cost=geo.costs(x,z)
                ell=length/math.sqrt(t['degree']);teacher=None
                if method=='excess':
                    teacher=self.score(z.reshape(-1,t['degree'],2),phase='teacher',actor=self.teacher).reshape(b,k)
                    ell=length/teacher-1
                if method=='tail':ell=tail_loss(ell,self.eta,P['alpha'],P['beta'])
                tau=t.get('tau',P['tau_default'])
                if method=='anneal':tau*=4**(1-2*self.step/max(1,t['updates']-1))
                if method=='hard':
                    idx=(ell-self.lam*cost).argmax(1);probs=np.eye(k)[idx]
                    info=dict(expected_cost=float(cost[np.arange(b),idx].mean()),lambda_=self.lam)
                elif method=='soft_dual':
                    probs=distribution(ell,cost,tau,self.lam);idx=sample(probs,rng)
                    info=dict(expected_cost=float((probs*cost).sum(1).mean()),lambda_=self.lam)
                else:
                    probs,info=solve(ell,cost,tau,P['rho']);idx=sample(probs,rng);self.lam=info['lambda_']
                selected_cost=float(cost[np.arange(b),idx].mean());x=z[np.arange(b),idx]
                if teacher is not None:den=teacher[np.arange(b),idx]
                if method in ('hard','soft_dual'):self.lam=max(0.,self.lam+P['lambda_lr']*(selected_cost-P['rho']))
                rec.update(info,tau=tau,selected_cost=selected_cost,no_move=float((idx==0).mean()))
                if self.step%256==0:geo.atomic_npz(self.out/'candidate_audit.npz',centers=z[:,0],candidates=z,length=length,cost=cost,probability=probs,index=idx)
            rec.update(self.update(x,den));self.step+=1;rec.update(seconds=time.monotonic()-start,elapsed_seconds=self.elapsed())
            event(self.out/'progress.jsonl',rec)
            if self.step%8==0:self.heartbeat('running')
            if self.step%self.t.get('val_every',P['validation_every'])==0 or self.step==t['updates']:self.validate()
            if self.step%P['checkpoint_every']==0 or self.step==t['updates']:self.checkpoint()
        self.heartbeat('complete')
        geo.atomic_json(self.out/'complete.json',dict(status='complete',task=self.t,step=self.step,best_step=self.best_step,best_sha256=geo.sha256(self.out/'best.pt'),latest_sha256=geo.sha256(self.out/'latest.pt'),parent_sha256=self.parent_hash,elapsed_seconds=self.elapsed(),calls=self.calls,validation=self.latest_val,time=time.time(),binding=self.bind))
        return True
    def close(self):
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()

def configure(device):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(P['resources']['memory_fraction'])
    geo.install_oracle_guard()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--base',default=str(ART/'runs'));p.add_argument('--stop-after',type=int)
    a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);configure(a.device)
    run=Run(json.loads(Path(a.task).read_text()),a.device,Path(a.base))
    try:ok=run.train(a.stop_after)
    finally:run.close()
    raise SystemExit(0 if ok else 75)
