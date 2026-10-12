"""Resumable deterministic inference for a frozen checkpoint across every corpus."""
import argparse
import fcntl
import importlib.util
import signal
import sys
import time
import traceback
import types
import numpy as np
import torch
from common import *

STOP=False
def stop(*_):
    global STOP
    STOP=True

def import_rest():
    prefix='_cross_rest_models'
    folder=PROJECT/'rl_expansion_20260922/vendor/models'
    package=types.ModuleType(prefix);package.__path__=[str(folder)];sys.modules[prefix]=package
    for name in ('utils','self_attn','actor_critic'):
        path=folder/(name+'.py');spec=importlib.util.spec_from_file_location(prefix+'.'+name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        source=path.read_text().replace('from models.','from '+prefix+'.')
        exec(compile(source,str(path),'exec'),module.__dict__)
    return sys.modules[prefix+'.actor_critic'].Actor

def lengths(pins,topology):
    # The project's ordered REST action representation: extend vertical at u,
    # horizontal at v. This is the evaluation metric, not an extra audit pass.
    x=np.asarray(pins,np.float64);t=np.asarray(topology);b,n,_=x.shape
    xl=x[:,:,0].copy();xr=xl.copy();yl=x[:,:,1].copy();yh=yl.copy()
    rows=np.repeat(np.arange(b),n-1);u=t[:,::2].reshape(-1);v=t[:,1::2].reshape(-1)
    np.minimum.at(yl,(rows,u),x[rows,v,1]);np.maximum.at(yh,(rows,u),x[rows,v,1])
    np.minimum.at(xl,(rows,v),x[rows,u,0]);np.maximum.at(xr,(rows,v),x[rows,u,0])
    return (yh-yl+xr-xl).sum(-1)

def metric(values):
    return dict(mean=float(np.mean(values)),p95=float(np.quantile(values,.95)),n=len(values))

class Evaluation:
    def __init__(self,model,gpu,batch):
        self.m=model;self.gpu=gpu;self.batch=batch;self.start=time.time();self.invocation_n=0
        self.out=ART/'results'/model['id'];self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.device=torch.device('cuda:0');torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_per_process_memory_fraction(.15)
        self.cells=read(self.out/'cells.json') if (self.out/'cells.json').exists() else {}
        self.packs=read(ART/'packs.json')
        self.ready_n=sum(p['count'] for p in self.packs if (self.out/f'd{p["degree"]}/complete.json').exists())
        self.target=sum(p['count'] for p in self.packs)
        p=Path(model['path'])
        if p.stat().st_size!=model['bytes'] or p.stat().st_mtime_ns!=model['mtime_ns']:
            raise RuntimeError('checkpoint changed since model catalog; refusing mixed evaluation')
        try:cp=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
        except RuntimeError:cp=torch.load(p,map_location='cpu',weights_only=False)
        self.state=extract_state(cp,model)
        self.family=model['family'];self.net=None
        if self.family=='rest':
            cls=import_rest();self.net=cls(10,self.device,width=model['width'],depth=model['depth'],inner=model['inner'])
            self.net.load_state_dict(self.state,strict=True);self.net.eval();self.state=None
        else:
            sys.path.insert(0,str(PROJECT/'papers_wdro_20260915'))
        self.beat('loaded',0)

    def beat(self,phase,n,**extra):
        elapsed=time.time()-self.start
        atomic(self.out/'heartbeat.json',dict(**identity(),model=self.m['id'],gpu=self.gpu,phase=phase,
                    committed_n=self.ready_n+n,invocation_n=self.invocation_n,target=self.target,
                    elapsed_seconds=elapsed,rows_per_second=self.invocation_n/elapsed,
                    batch_size=self.batch,precision='float32',**extra))

    def setup_degree(self,d):
        if self.family=='rest':self.net.degree=d;return
        import importlib
        params=read(PROJECT/'papers_wdro_20260915/protocol.json')['parameters'][self.family]
        cls=importlib.import_module('adapter_'+self.family).Adapter
        self.net=cls(d,self.device,params)
        self.net.load_state_dict(self.state,strict=True);self.net.eval()

    def predict(self,x):
        keep=len(x)
        if keep<self.batch:x=np.concatenate((x,np.repeat(x[-1:],self.batch-keep,axis=0)))
        with torch.inference_mode():
            if self.family=='rest':
                topology,_=self.net(x,True)
                length=lengths(x[:keep],topology[:keep].cpu().numpy());objective=length.copy()
            else:
                objective,_,details=self.net.evaluate(x,True)
                objective=objective[:keep].cpu().numpy().astype(np.float64)
                key='wirelength' if 'wirelength' in details else 'length'
                length=np.asarray(details[key][:keep],np.float64)
        if not np.isfinite(length).all() or not np.isfinite(objective).all():
            raise RuntimeError('nonfinite evaluation result')
        self.invocation_n+=keep
        return length,objective

    def run(self):
        # Degree 30 first gives the largest method overlap in early progress.
        for pack in sorted(self.packs,key=lambda p:(p['degree']!=30,p['degree'])):
            d=pack['degree'];out=self.out/f'd{d}';out.mkdir(exist_ok=True)
            if (out/'complete.json').exists():continue
            self.setup_degree(d)
            x=np.load(pack['path'],mmap_mode='r');ref=np.load(pack['reference'],mmap_mode='r')
            committed=read(out/'cursor.json') if (out/'cursor.json').exists() else dict(cursor=0,shards=[])
            cursor=committed['cursor'];chunk=8192
            while cursor<len(x):
                if STOP:self.beat('paused',cursor,degree=d);return 75
                end=min(cursor+chunk,len(x));ls=[];obs=[]
                for start in range(cursor,end,self.batch):
                    l,o=self.predict(np.array(x[start:min(start+self.batch,end)],copy=True));ls.append(l);obs.append(o)
                name=f'{cursor:09d}_{end:09d}.npz'
                npz(out/name,length=np.concatenate(ls),objective=np.concatenate(obs))
                committed['shards'].append(dict(file=name,start=cursor,end=end));committed['cursor']=end
                atomic(out/'cursor.json',committed);cursor=end
                self.beat('evaluating',cursor,degree=d)
                print(json.dumps(dict(model=self.m['id'],degree=d,committed=cursor,total=len(x))),flush=True)
            lengths_all=[];objectives=[]
            for row in committed['shards']:
                with np.load(out/row['file'],allow_pickle=False) as z:
                    lengths_all.append(z['length']);objectives.append(z['objective'])
            length=np.concatenate(lengths_all);objective=np.concatenate(objectives)
            for column in pack['columns']:
                sl=slice(column['start'],column['end']);v=length[sl];r=ref[sl]
                result=dict(length=metric(v),objective=metric(objective[sl]),family=self.family,degree=d,
                            cross_degree=d!=self.m['train_degree'],root_convention='pin0 artificial root' if self.family!='rest' else 'unrooted',
                            reference_count=int(np.isfinite(r).sum()))
                if np.isfinite(r).all() and (r>0).all():
                    regret=v/r-1
                    if regret.min()>=-1e-5:result['regret_percent']=metric(100*regret)
                    result['reference_note']='known precision_v2 references' if regret.min()>=-1e-5 else 'negative values: different routing/rounding convention; do not rank as optimality regret'
                self.cells[column['id']]=result
            atomic(self.out/'cells.json',self.cells)
            atomic(out/'complete.json',dict(status='complete',count=len(x),degree=d,columns=len(pack['columns']),batch_size=self.batch,gpu=self.gpu))
            self.ready_n+=len(x);self.beat('degree_complete',0,degree=d)
            if self.family!='rest':del self.net;self.net=None;torch.cuda.empty_cache()
        atomic(self.out/'complete.json',dict(status='complete',model=self.m['id'],cells=len(self.cells),nets=self.target,batch_size=self.batch,gpu=self.gpu,elapsed_seconds=time.time()-self.start))
        self.beat('complete',0)
        return 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--gpu',required=True);ap.add_argument('--batch',type=int,default=256);args=ap.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    m=next(m for m in read(ART/'models.json') if m['id']==args.model)
    try:return Evaluation(m,args.gpu,args.batch).run()
    except Exception as exc:
        atomic(ART/'results'/m['id']/'failure.json',dict(model=m['id'],error=repr(exc),traceback=traceback.format_exc(),**identity()))
        raise

if __name__=='__main__':sys.exit(main())
