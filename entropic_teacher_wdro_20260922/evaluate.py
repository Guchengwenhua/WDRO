import argparse,fcntl,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import engine as e
import geometry as geo

class Evaluation:
    score=e.Run.score
    def __init__(self,task,device):
        self.t=task;self.device=torch.device(device);self.out=e.ART/'evaluation'/task['id'];self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'run.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.calls=dict(search=0,teacher=0,validation=0,test=0,sampled=0,padded=0)
        self.model=e.ART/'runs'/task['run_id']/task['checkpoint']
        freeze=json.loads((e.ART/'operations/model_freeze.json').read_text())
        if geo.sha256(self.model)!=freeze['models'][str(self.model)]:raise RuntimeError('model freeze mismatch')
        self.actor=geo.build_actor(task['degree'],self.device)
        self.actor.load_state_dict(torch.load(self.model,map_location=device,weights_only=False)['actor'])
        self.start=time.time()
    def heartbeat(self,step):
        geo.atomic_json(self.out/'heartbeat.json',dict(**e.identity(),time=time.time(),status='evaluating',step=step,target=12,elapsed_seconds=time.time()-self.start))
    def attack(self,x,method,dist):
        # identical centers and perturbation random numbers for all models
        rng=np.random.default_rng(e.seed('independent_attack',self.t['degree'],dist,method));k=e.P['attack_k'];n=x.shape[1];b=len(x)
        if method=='random':
            z=e.candidates(x,rng,k);v=self.score(z.reshape(-1,n,2),'test').reshape(b,k)
        else:
            pairs=(k-2)//2;u=rng.normal(size=(b,pairs,n,2));u/=np.maximum(np.sqrt((u*u).mean((-1,-2),keepdims=True)),1e-12)
            d=np.clip(e.P['sigma']*np.stack((u,-u),2).reshape(b,2*pairs,n,2),-e.P['epsilon'],e.P['epsilon'])
            z=np.concatenate((x[:,None],geo.legalize(x,d)),1);v=self.score(z.reshape(-1,n,2),'test').reshape(b,k-1)
            pair=v[:,1:].reshape(b,pairs,2);g=((pair[:,:,0]-pair[:,:,1])[:,:,None,None]*u).mean(1)
            g/=np.maximum(np.sqrt((g*g).mean((-1,-2),keepdims=True)),1e-12)
            last=geo.legalize(x,np.clip(e.P['sigma']*g,-e.P['epsilon'],e.P['epsilon']))
            v=np.concatenate((v,self.score(last,'test')[:,None]),1)
        return v.max(1),v[:,0]
    def run(self):
        arrays={};step=0
        # Reuse only validated completed shards from this frozen model.
        for d in e.P['distributions']:
            for suite in ('natural','random64','es64'):
                path=self.out/f'{d}_{suite}.npz';self.heartbeat(step)
                if path.exists():
                    with np.load(path,allow_pickle=False) as v:arr=v['length'];original=v['original']
                else:
                    count=e.P['test_n'] if suite=='natural' else e.P['attack_n']
                    x=e.data(count,self.t['degree'],d,'test' if suite=='natural' else 'attack')
                    if suite=='natural':arr=self.score(x,'test');original=arr
                    else:arr,original=self.attack(x,'random' if suite=='random64' else 'es',d)
                    geo.atomic_npz(path,length=arr,original=original)
                if not np.isfinite(arr).all() or (arr<=0).any():raise RuntimeError('invalid evaluation')
                arrays[f'{d}_{suite}']=arr;step+=1
        metrics={}
        for key,v in arrays.items():
            q=float(np.quantile(v,.9));metrics[key]=dict(mean=float(v.mean()),p95=float(np.quantile(v,.95)),cvar90=float(q+np.maximum(v-q,0).mean()/.1),n=len(v))
        geo.atomic_json(self.out/'complete.json',dict(status='complete',task=self.t,model_sha256=geo.sha256(self.model),metrics=metrics,calls=self.calls,elapsed_seconds=time.time()-self.start,time=time.time(),shards={p.name:geo.sha256(p) for p in self.out.glob('*.npz')}))
        self.heartbeat(12)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    e.configure(a.device);v=Evaluation(json.loads(Path(a.task).read_text()),a.device)
    v.run()
