"""Conditional local deformation VAE; all tuning uses A_train/A_val only."""
import argparse
import time
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from exp_common import *
from geometry_local import legalize,normalize_torch

class LocalCVAE(nn.Module):
    def __init__(self,latent):
        super().__init__();self.latent=latent
        self.point=nn.Sequential(nn.Linear(2,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(128,128),nn.SiLU())
        self.prior_head=nn.Linear(128,2*latent)
        self.pair=nn.Sequential(nn.Linear(4,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.posterior_head=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,2*latent))
        self.decoder=nn.Sequential(nn.Linear(2+64+128+latent,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))

    def condition(self,x):
        point=self.point(x);context=self.context(torch.cat([point.mean(1),point.amax(1)],-1))
        mu,lv=self.prior_head(context).chunk(2,-1)
        return point,context,mu,lv.clamp(-6,3)

    def posterior(self,x,y,context):
        p=self.pair(torch.cat([x,y-x],-1));mu,lv=self.posterior_head(torch.cat([p.mean(1),p.amax(1),context],-1)).chunk(2,-1)
        return mu,lv.clamp(-6,3)

    def delta(self,x,z,point=None,context=None):
        if point is None:point,context,_,_=self.condition(x)
        h=torch.cat([x,point,context[:,None].expand(-1,30,-1),z[:,None].expand(-1,30,-1)],-1)
        return CFG['cvae']['delta_max']*torch.tanh(self.decoder(h))

    def decode(self,x,z,point=None,context=None):return normalize_torch(x+self.delta(x,z,point,context))

def align(source,target):
    d=np.abs(source.astype(np.float64)[:,None]-target.astype(np.float64)[None]).sum(-1)
    _,cols=linear_sum_assignment(d);return target[cols]

def matched_loss(pred,target):
    matrix=(pred[:,:,None]-target[:,None]).square().sum(-1)
    columns=np.stack([linear_sum_assignment(d)[1] for d in matrix.detach().cpu().numpy()])
    selected=target[torch.arange(len(target),device=target.device)[:,None],torch.as_tensor(columns,device=target.device)]
    return (pred-selected).square().sum(-1).mean()

def l1_costs(pred,target):
    result=[]
    for a,b in zip(pred,target,strict=True):
        d=np.abs(a.astype(np.float64)[:,None]-b.astype(np.float64)[None]).sum(-1)
        result.append(float(d[linear_sum_assignment(d)].mean()))
    return np.array(result)

def prepare_pairs(data):
    path=ART/'cvae'/'neighbors.npz'
    if path.exists():return np.load(path)['neighbors']
    train=data['pins'][data['split']=='A_train'];tree=cKDTree(train.reshape(len(train),-1))
    _,candidates=tree.query(train.reshape(len(train),-1),k=17);neighbors=np.full((len(train),4),-1,dtype=np.int64)
    for i,x in enumerate(train):
        scored=[]
        for j in candidates[i]:
            if i==j:continue
            y=align(x,train[j]);d=np.abs(y.astype(np.float64)-x)
            cost=float(d.sum(-1).mean())
            if 1e-8<cost<=.12 and d.max()<=.16:scored.append((cost,int(j)))
        scored.sort();ids=[j for _,j in scored[:4]];neighbors[i,:len(ids)]=ids
    atomic_npz(path,neighbors=neighbors)
    atomic_json(ART/'cvae'/'pairs.json',dict(centers=len(train),with_real_neighbors=int((neighbors>=0).any(1).sum()),
        real_neighbor_slots=int((neighbors>=0).sum()),source='A_train only',max_mean_l1=.12,max_coordinate_displacement=.16))
    return neighbors

def denoising_pairs(pins,seed):
    rng=np.random.default_rng(seed);x=[];y=[]
    for target in pins:
        amplitude=float(rng.choice(CFG['cvae']['corruption_amplitudes']))
        source=legalize(target+rng.normal(0,amplitude,target.shape))[0]
        x.append(source);y.append(align(source,target))
    return np.stack(x),np.stack(y)

def training_batch(train,neighbors,indices,rng):
    xx=[];yy=[];neighbor_count=0
    for i in indices:
        valid=neighbors[i][neighbors[i]>=0]
        if len(valid) and rng.random()<CFG['cvae']['neighbor_fraction']:
            x=train[i];y=align(x,train[int(rng.choice(valid))]);neighbor_count+=1
        else:
            target=train[i];amp=float(rng.choice(CFG['cvae']['corruption_amplitudes']))
            x=legalize(target+rng.normal(0,amp,target.shape))[0];y=align(x,target)
        xx.append(x);yy.append(y)
    return np.stack(xx),np.stack(yy),neighbor_count

@torch.no_grad()
def validate(model,x,y,device):
    model.eval();posterior=[];prior=[];kls=[];diversities=[]
    noise=np.random.default_rng(2026092201).normal(size=(len(x),8,32)).astype(np.float32)[...,:model.latent]
    for b in range(0,len(x),128):
        xx=torch.as_tensor(x[b:b+128],device=device);yy=torch.as_tensor(y[b:b+128],device=device)
        point,context,pm,pl=model.condition(xx);qm,ql=model.posterior(xx,yy,context)
        pred=model.decode(xx,qm,point,context).cpu().numpy();posterior.extend(l1_costs(pred,yy.cpu().numpy()))
        n=len(xx);eps=torch.as_tensor(noise[b:b+n],device=device)
        z=(pm[:,None]+(.5*pl).exp()[:,None]*eps).reshape(n*8,-1)
        repeated=xx[:,None].expand(-1,8,-1,-1).reshape(n*8,30,2)
        generated=model.decode(repeated,z).cpu().numpy().reshape(n,8,30,2)
        targets=np.repeat(y[b:b+n],8,axis=0)
        costs=l1_costs(generated.reshape(n*8,30,2),targets).reshape(n,8);prior.extend(costs.min(1))
        diversities.extend(generated.var(1).mean((1,2)))
        kl=.5*(pl-ql+((ql.exp()+(qm-pm).square())/pl.exp())-1);kls.extend(kl.mean(-1).cpu().numpy())
    return dict(prior_best8_mean_l1=float(np.mean(prior)),prior_best8_median_l1=float(np.median(prior)),
        posterior_mean_l1=float(np.mean(posterior)),posterior_median_l1=float(np.median(posterior)),
        prior_coordinate_variance=float(np.mean(diversities)),kl=float(np.mean(kls)))

def fit(generator,spec,train,neighbors,vx,vy,device):
    latent=spec['latent'];beta_max=spec['beta_max'];tag=f'L{latent}_b{beta_max:g}'
    folder=ART/'cvae'/f'g{generator}'/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():return read_json(folder/'done.json')
    seed=CFG['generator_seeds'][generator];seed_all(seed);model=LocalCVAE(latent).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001);epoch=0;best=None;history=[];patience=0;elapsed=0.
    if (folder/'last.pt').exists():
        cp=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if cp['device_type']!=device:raise RuntimeError('resume CVAE with same backend')
        model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);set_rng(cp['rng'])
        epoch=cp['epoch'];best=cp['best'];history=cp['history'];patience=cp['patience'];elapsed=cp['seconds']
    wall=time.monotonic()
    while epoch<CFG['cvae']['max_epochs']:
        epoch+=1;model.train();rng=np.random.default_rng(seed+epoch*10007);order=rng.permutation(len(train));losses=[];recs=[];kls=[];real_neighbors=0
        beta=beta_max*min(1.,epoch/30)
        for b in range(0,len(train),128):
            xx,yy,count=training_batch(train,neighbors,order[b:b+128],rng);real_neighbors+=count
            x=torch.as_tensor(xx,device=device);y=torch.as_tensor(yy,device=device)
            point,context,pm,pl=model.condition(x);qm,ql=model.posterior(x,y,context)
            z=qm+(.5*ql).exp()*torch.randn_like(qm);pred=model.decode(x,z,point,context)
            rec=matched_loss(pred,y);kl=(.5*(pl-ql+(ql.exp()+(qm-pm).square())/pl.exp()-1)).mean();loss=rec+beta*kl
            if not torch.isfinite(loss):raise RuntimeError('nonfinite CVAE loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('nonfinite CVAE gradient')
            opt.step();losses.append(float(loss.detach()));recs.append(float(rec.detach()));kls.append(float(kl.detach()))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),reconstruction_l2=float(np.mean(recs)),kl=float(np.mean(kls)),beta=beta,
            real_neighbor_pairs=real_neighbors,denoising_pairs=len(train)-real_neighbors,seconds=elapsed+time.monotonic()-wall)
        improved=False
        if epoch%5==0:
            val=validate(model,vx,vy,device);row['validation']=val
            if best is None or val['prior_best8_mean_l1']<best['prior_best8_mean_l1']:
                best=dict(epoch=epoch,**val);patience=0;improved=True
            else:patience+=1
        history.append(row)
        cp=dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=generator,seed=seed,spec=spec,
            device_type=device,epoch=epoch,best=best,history=history,patience=patience,seconds=elapsed+time.monotonic()-wall,rng=rng_state())
        atomic_torch(folder/'last.pt',cp)
        if improved:atomic_torch(folder/'best.pt',cp)
        (folder/'curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        progress(f'cvae_g{generator}',spec=spec,epoch=epoch,best=best,seconds=cp['seconds'],device=device)
        if epoch%5==0:print('CVAE',generator,tag,epoch,best,flush=True)
        if epoch>=50 and (patience>=10 or cp['seconds']>=CFG['cvae']['soft_minutes']*60):break
    result=dict(generator=generator,seed=seed,spec=spec,epochs=epoch,best=best,seconds=elapsed+time.monotonic()-wall,
        path=str(folder/'best.pt'),device_type=device)
    atomic_json(folder/'done.json',result);return result

def load_model(generator,device='cpu'):
    selection=read_json(ART/'cvae'/f'g{generator}'/'selected.json')
    cp=torch.load(selection['path'],map_location='cpu',weights_only=False)
    model=LocalCVAE(cp['spec']['latent']).to(device);model.load_state_dict(cp['model'],strict=True);model.eval()
    return model,selection

def main(generator,device):
    if (ART/'done'/f'cvae_g{generator}.json').exists():return
    data=load_data();train=data['pins'][data['split']=='A_train'];neighbors=prepare_pairs(data)
    vx,vy=denoising_pairs(data['pins'][data['split']=='A_val'],2026092202)
    if generator==0:grid=CFG['cvae']['tuning_grid']
    else:grid=[read_json(ART/'cvae'/'g0'/'selected.json')['spec']]
    fits=[fit(generator,spec,train,neighbors,vx,vy,device) for spec in grid]
    chosen=min(fits,key=lambda x:x['best']['prior_best8_mean_l1'])
    selection=dict(**chosen,all_fits=fits,selection='A_val fixed denoising pairs; conditional prior best-of-8 Hungarian L1',test_access=False)
    atomic_json(ART/'cvae'/f'g{generator}'/'selected.json',selection)
    mark_done(f'cvae_g{generator}',seconds=sum(f['seconds'] for f in fits),selection=selection)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generator',type=int,default=0);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();main(a.generator,a.device)
