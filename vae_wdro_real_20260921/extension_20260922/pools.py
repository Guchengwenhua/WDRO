"""Calibrated conditional BALL/SHELL pools and fixed-budget mixed pools."""
import argparse
import time
import numpy as np
import torch
from exp_common import *
from geometry_local import legalize,hungarian_l1_mean
from cvae import load_model,denoising_pairs

@torch.no_grad()
def parameters(model,pins):
    mu=[];lv=[]
    for b in range(0,len(pins),256):
        _,_,m,v=model.condition(torch.from_numpy(pins[b:b+256]));mu.append(m.numpy());lv.append(v.numpy())
    return np.concatenate(mu),np.concatenate(lv)

@torch.no_grad()
def deltas(model,pins,z):
    k=z.shape[1];x=np.repeat(pins,k,axis=0);flat=z.reshape(-1,z.shape[-1]).astype(np.float32);out=[]
    for b in range(0,len(x),256):out.append(model.delta(torch.from_numpy(x[b:b+256]),torch.from_numpy(flat[b:b+256])).numpy())
    return np.concatenate(out).reshape(len(pins),k,30,2)

def build_geometry(pins,delta,alpha):
    n,k=len(pins),delta.shape[1];bank=np.empty((n,k+1,30,2),np.float32);bank[:,0]=pins
    cost=np.zeros((n,k+1));repaired=np.zeros((n,k+1),bool);displacement=np.zeros((n,k+1))
    for i,x in enumerate(pins):
        for slot in range(k):
            y,r,d=legalize(x.astype(np.float64)+alpha*delta[i,slot].astype(np.float64))
            bank[i,slot+1]=y;cost[i,slot+1]=hungarian_l1_mean(x,y);repaired[i,slot+1]=r;displacement[i,slot+1]=d
    return bank,cost,repaired,displacement

def project(raw,rmin,rmax):
    radii=np.linalg.norm(raw,axis=-1);direction=raw/np.maximum(radii[...,None],1e-30)
    zero=radii<1e-30
    if zero.any():direction[zero]=np.eye(1,raw.shape[-1])[0]
    return direction*np.clip(radii,rmin,rmax)[...,None]

def geometry(generator):
    if (ART/'done'/f'geometry_g{generator}.json').exists():return
    start=time.monotonic();seed_all(CFG['generator_seeds'][generator]);data=load_data();model,selection=load_model(generator)
    cal=data['pins'][data['split']=='A_cal'];cx,cy=denoising_pairs(cal,2026092203)
    rng=np.random.default_rng(CFG['generator_seeds'][generator]+30000);rad=[]
    with torch.no_grad():
        for b in range(0,len(cx),128):
            x=torch.from_numpy(cx[b:b+128]);y=torch.from_numpy(cy[b:b+128]);_,context,pm,pl=model.condition(x);qm,ql=model.posterior(x,y,context)
            eps=torch.from_numpy(rng.normal(size=qm.shape).astype(np.float32))
            z=qm+(.5*ql).exp()*eps;u=(z-pm)/(.5*pl).exp();rad.extend(u.norm(dim=-1).numpy())
    rmin,rmax=np.quantile(rad,[.025,.975])
    if not np.isfinite(rad).all() or not 0<rmin<rmax:raise ValueError('invalid conditional radius')
    tau=np.array(CFG['candidates']['latent_local_tau'])[None,:,None]
    pm,pl=parameters(model,cal);rng=np.random.default_rng(CFG['generator_seeds'][generator]+40000)
    u=tau*rng.normal(size=(len(cal),7,model.latent));z=pm[:,None]+np.exp(.5*pl)[:,None]*project(u,0,rmax)
    delta=deltas(model,cal,z);target=float(np.median(np.load(ART/'calibration_raw_costs.npz')['cost'][:,1:]))
    scores=[]
    for alpha in CFG['cvae']['alpha_grid']:
        _,cost,_,_=build_geometry(cal,delta,alpha);median=float(np.median(cost[:,1:]))
        scores.append(dict(alpha=alpha,median_cost=median,target_raw_median=target,absolute_gap=abs(median-target)))
        progress(f'geometry_g{generator}',stage='alpha_calibration',alpha=alpha,median_cost=median,target=target)
    alpha=min(scores,key=lambda x:(x['absolute_gap'],x['alpha']))['alpha']
    calibration=dict(generator=generator,r_min=float(rmin),r_max=float(rmax),alpha=alpha,alpha_grid=scores,
        radius_source='A_cal denoising posterior samples whitened by conditional prior',calibration_nets=len(cal),
        latent=model.latent,model_path=selection['path'],test_access=False)
    atomic_json(ART/'cvae'/f'g{generator}'/'calibration.json',calibration)
    indices=np.load(ART/'training_centers.npz')['indices'];train=data['pins'][indices]
    pm,pl=parameters(model,train);rng=np.random.default_rng(CFG['generator_seeds'][generator]+50000)
    raw=tau*rng.normal(size=(len(train),7,model.latent));norm=np.linalg.norm(raw,axis=-1)
    inside=norm<rmin;outside=norm>rmax;arrays={}
    for method,lower in [('local_ball',0.),('local_shell',rmin)]:
        z=pm[:,None]+np.exp(.5*pl)[:,None]*project(raw,lower,rmax)
        arrays[method]=list(build_geometry(train,deltas(model,train,z),alpha))
    retries=0
    for i,x in enumerate(train):
        seen={m:{x.tobytes()} for m in arrays}
        for slot in range(1,8):
            for retry in range(16):
                if all(arrays[m][0][i,slot].tobytes() not in seen[m] for m in arrays):break
                retries+=1;u=tau[0,slot-1]*rng.normal(size=model.latent);r=float(np.linalg.norm(u));inside[i,slot-1]=r<rmin;outside[i,slot-1]=r>rmax
                for method,lower in [('local_ball',0.),('local_shell',rmin)]:
                    z=pm[i]+np.exp(.5*pl[i])*project(u[None,None],lower,rmax)[0,0]
                    d=deltas(model,x[None],z[None,None])[0,0]
                    y,rep,movement=legalize(x.astype(np.float64)+alpha*d.astype(np.float64));arrays[method][0][i,slot]=y
                    arrays[method][1][i,slot]=hungarian_l1_mean(x,y);arrays[method][2][i,slot]=rep;arrays[method][3][i,slot]=movement
            for method in arrays:seen[method].add(arrays[method][0][i,slot].tobytes())
    equal=float(np.all(arrays['local_ball'][0][:,1:]==arrays['local_shell'][0][:,1:],axis=(2,3)).mean());rho=read_json(ART/'rho.json')['rho']
    for method,(bank,cost,repair,disp) in arrays.items():
        name=f'{method}_g{generator}';unique=np.array([len({x.tobytes() for x in group}) for group in bank])
        atomic_npz(ART/'pools'/f'{name}.geometry.npz',pins=bank,cost=cost,center_index=indices,
            slot=np.tile(np.arange(8),(len(bank),1)),repaired=repair,repair_max_displacement=disp,unique_K=unique)
        atomic_json(ART/'pools'/f'{name}.geometry.json',dict(method=method,centers=len(bank),count=8,
            mean_cost=float(cost[:,1:].mean()),median_cost=float(np.median(cost[:,1:])),
            cost_le_rho_fraction=float((cost[:,1:]<=rho).mean()),centers_with_cost_le_rho=float((cost[:,1:]<=rho).any(1).mean()),
            nominal_equal_non_nominal_fraction=float(np.all(bank[:,1:]==bank[:,:1],axis=(2,3)).mean()),
            mean_unique_K=float(unique.mean()),minimum_unique_K=int(unique.min()),repair_fraction=float(repair[:,1:].mean()),
            max_repair_displacement=float(disp.max()),inner_fraction=float(inside.mean()) if method=='local_shell' else 0.,
            outer_fraction=float(outside.mean()),ball_shell_exact_equal_fraction=equal,duplicate_retries=retries,**calibration))
    # Same fixed validation examples for a readable local-generation illustration.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=data['pins'][data['split']=='A_val'][:4];ep,el=parameters(model,examples)
    eu=tau*np.random.default_rng(2026092204).normal(size=(4,7,model.latent));fig,axes=plt.subplots(4,3,figsize=(9,12))
    for col,lower in [(1,0),(2,rmin)]:
        ez=ep[:,None]+np.exp(.5*el)[:,None]*project(eu,lower,rmax);gen=build_geometry(examples,deltas(model,examples,ez),alpha)[0]
        for row,x in enumerate(examples):
            for y in gen[row,1:]:axes[row,col].scatter(y[:,0],y[:,1],s=7,alpha=.2,color='tab:orange')
    for row,x in enumerate(examples):
        for col in range(3):
            axes[row,col].scatter(x[:,0],x[:,1],s=12,marker='x',c='black');axes[row,col].set(xlim=(.05,.95),ylim=(.05,.95),aspect='equal')
    for col,title in enumerate(['Validation input','Conditional BALL','Conditional SHELL']):axes[0,col].set_title(title)
    fig.tight_layout();fig.savefig(ART/'cvae'/f'g{generator}'/'validation_generation.png',dpi=140);plt.close(fig)
    mark_done(f'geometry_g{generator}',seconds=time.monotonic()-start,calibration=calibration)

def mix(kind,generator):
    name='mix_legacy' if kind=='legacy' else f'mix_local_g{generator}';marker=name
    if (ART/'done'/f'{marker}.json').exists():return
    left=np.load(ART/'pools'/'raw_wdro.npz');right=np.load(ART/'pools'/('legacy_shell.npz' if kind=='legacy' else f'local_shell_g{generator}.npz'))
    if not np.array_equal(left['center_index'],right['center_index']):raise ValueError('mixed center mismatch')
    a={};left_slots=[0,1,3,5,6];right_slots=[2,3,6]
    for key in ['pins','cost','lstar','valid','repaired','repair_max_displacement']:
        a[key]=np.concatenate([left[key][:,left_slots],right[key][:,right_slots]],axis=1)
    a['center_index']=left['center_index'];a['slot']=np.tile(np.arange(8),(len(a['pins']),1))
    a['unique_K']=np.array([len({x.tobytes() for x in group}) for group in a['pins']])
    atomic_npz(ART/'pools'/f'{name}.npz',**a)
    atomic_json(ART/'pools'/f'{name}.geometry.json',dict(method=name,raw_slots=left_slots,vae_slots=right_slots,
        mean_unique_K=float(a['unique_K'].mean()),mean_cost=float(a['cost'][:,1:].mean()),new_reference_requests=0))
    mark_done(marker,method=name,new_reference_requests=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generator',type=int,default=0);p.add_argument('--mix',choices=['legacy','local']);a=p.parse_args()
    if a.mix:mix(a.mix,a.generator)
    else:geometry(a.generator)
