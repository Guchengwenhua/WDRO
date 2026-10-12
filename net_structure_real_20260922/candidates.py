"""Fixed RAW/BALL/SHELL pools; no actor is used for calibration or generation."""
import argparse
import time
import numpy as np
from exp_common import *
from src.geometry import project_normalized_net
from src.costs import hungarian_l1_mean

def legalize(x):
    x=np.asarray(x,dtype=np.float64);x=x-x.min(0)
    span=x.max()
    if not np.isfinite(span) or span<=1e-12: raise ValueError('degenerate generated net')
    norm=np.clip(.1+.8*x/span,.1,.9).astype(np.float32)
    pair=np.abs(norm[:,None,:]-norm[None,:,:]).sum(-1);np.fill_diagonal(pair,np.inf)
    repaired=pair.min()<1.1e-6
    out=project_normalized_net(norm,minimum_l1_spacing=2e-6).astype(np.float32) if repaired else norm
    check_pins(out)
    return canonical(out),bool(repaired),float(np.abs(out-norm).max())

def perturb(x,family,amplitude,rng):
    n=len(x)
    if family=='jitter': delta=rng.normal(0,amplitude,(n,2))
    elif family=='sparse':
        delta=np.zeros((n,2));ids=rng.choice(n,max(1,n//5),replace=False);delta[ids]=rng.normal(0,amplitude,(len(ids),2))
    elif family=='correlated':
        delta=(x-x.mean(0))@rng.normal(0,amplitude,(2,2))+rng.normal(0,amplitude*.25,(n,2))
    elif family=='axis_biased':
        scales=np.array([amplitude,amplitude*.2]);
        if rng.integers(2): scales=scales[::-1]
        delta=rng.normal(size=(n,2))*scales
    else: raise ValueError(family)
    return legalize(x+delta)

def raw_bank(pins,seed,label):
    rng=np.random.default_rng(seed);bank=np.empty((len(pins),8,30,2),np.float32);cost=np.zeros((len(pins),8))
    repair=np.zeros((len(pins),8),bool);displacement=np.zeros((len(pins),8));unique=[];retries=0
    for i,x in enumerate(pins):
        bank[i,0]=x;seen={x.tobytes()}
        for slot,(family,amp) in enumerate(CFG['candidates']['raw_family_amplitude'],1):
            for attempt in range(CFG['candidates']['duplicate_retry_limit']+1):
                y,r,d=perturb(x,family,amp,rng)
                if y.tobytes() not in seen: break
                if attempt<CFG['candidates']['duplicate_retry_limit']: retries+=1
            bank[i,slot]=y;cost[i,slot]=hungarian_l1_mean(x,y);repair[i,slot]=r;displacement[i,slot]=d;seen.add(y.tobytes())
        unique.append(len(seen))
        if i%128==0: progress(label,stage='geometry',completed=i,total=len(pins))
    return bank,cost,repair,displacement,np.array(unique),retries

def save_geometry(name,bank,cost,repair,displacement,unique,indices,**stats):
    atomic_npz(ART/'pools'/f'{name}.geometry.npz',pins=bank,cost=cost,center_index=indices,
        slot=np.tile(np.arange(8),(len(bank),1)),repaired=repair,repair_max_displacement=displacement,unique_K=unique)
    atomic_json(ART/'pools'/f'{name}.geometry.json',dict(method=name,centers=len(bank),candidates=len(bank)*8,
        repair_fraction=float(repair[:,1:].mean()),max_repair_displacement=float(displacement.max()),
        mean_unique_K=float(unique.mean()),min_unique_K=int(unique.min()),
        cost_mean=float(cost[:,1:].mean()),cost_q50=float(np.median(cost[:,1:])),cost_q95=float(np.quantile(cost[:,1:],.95)),**stats))

def build_raw():
    if (ART/'done'/'raw_geometry.json').exists(): return
    start=time.monotonic();data=load_data()
    if not (ART/'rho.json').exists():
        bank,cost,repair,displacement,unique,retries=raw_bank(data['pins'][data['split']=='A_cal'],CFG['candidates']['rho_calibration_seed'],'raw_calibration')
        rho=float(np.quantile(cost[:,1:],CFG['candidates']['rho_quantile']))
        if not np.isfinite(rho) or rho<=0: raise ValueError('invalid rho')
        atomic_json(ART/'rho.json',dict(rho=rho,nets=len(bank),candidate_costs=cost[:,1:].size,quantile=.75,
            calibration='A_cal geometry only, no Actor',seconds=time.monotonic()-start))
        atomic_npz(ART/'calibration_raw_costs.npz',cost=cost)
        print('COMMON RHO',rho,flush=True)
    indices=np.load(ART/'training_centers.npz')['indices']
    bank,cost,repair,displacement,unique,retries=raw_bank(data['pins'][indices],CFG['candidates']['seed'],'raw_geometry')
    save_geometry('raw_wdro',bank,cost,repair,displacement,unique,indices,retries=retries,
        raw_formulas=dict(jitter='iid Gaussian amplitude',sparse='6 of 30 pins iid Gaussian amplitude',
            correlated='centered pins @ Gaussian 2x2 amplitude + iid Gaussian amplitude/4',axis_biased='random axis std amplitude, other axis amplitude/5'))
    mark_done('raw_geometry',seconds=time.monotonic()-start)

def build_latent():
    if (ART/'done'/'latent_geometry.json').exists(): return
    import torch
    from vae import load_selected
    start=time.monotonic();seed_all(CFG['candidates']['seed']);model,selected=load_selected(device='cpu');data=load_data()
    indices=np.load(ART/'training_centers.npz')['indices'];train=data['pins'][indices]
    def encode(pins):
        out=[]
        with torch.no_grad():
            for b in range(0,len(pins),256): out.append(model.encode(torch.as_tensor(pins[b:b+256],device='cpu'))[0].cpu().numpy())
        return np.concatenate(out)
    all_train_mu=encode(data['pins'][data['split']=='A_train']);mu=encode(train);cal=encode(data['pins'][data['split']=='A_cal'])
    radii=np.linalg.norm(cal,axis=1);rmin,rmax=np.quantile(radii,CFG['candidates']['radius_quantiles'])
    scales=np.maximum(all_train_mu.std(0),CFG['candidates']['latent_scale_floor'])
    if not 0 <= rmin <= rmax or rmax<=0: raise ValueError('invalid latent radius')
    atomic_json(ART/'vae'/'calibration.json',dict(r_min=rmin,r_max=rmax,scale=scales,
        train_mu_variance=all_train_mu.var(0),calibration_count=len(cal),latent=mu.shape[1]))
    rng=np.random.default_rng(CFG['candidates']['seed']);tau=np.array(CFG['candidates']['latent_local_tau'])[None,:,None]
    zraw=mu[:,None,:]+tau*scales[None,None,:]*rng.normal(size=(len(mu),7,mu.shape[1]))
    norm=np.linalg.norm(zraw,axis=-1);direction=zraw/np.maximum(norm[...,None],1e-30)
    zeros=norm<1e-30
    if zeros.any():
        fallback=rng.normal(size=(int(zeros.sum()),mu.shape[1]));fallback/=np.linalg.norm(fallback,axis=1,keepdims=True);direction[zeros]=fallback
    arrays={};out_masks=norm>rmax;in_masks=norm<rmin
    for name,lower in [('vae_ball_wdro',0.),('vae_shell_wdro',rmin)]:
        z=(direction*np.clip(norm,lower,rmax)[...,None]).astype(np.float32)
        decoded=[]
        with torch.no_grad():
            flat=z.reshape(-1,z.shape[-1])
            for b in range(0,len(flat),256):decoded.append(model.decode(torch.as_tensor(flat[b:b+256],device='cpu')).cpu().numpy())
        decoded=np.concatenate(decoded).reshape(len(train),7,30,2)
        bank=np.empty((len(train),8,30,2),np.float32);bank[:,0]=train;cost=np.zeros((len(train),8));repair=np.zeros_like(cost,dtype=bool);disp=np.zeros_like(cost)
        for i in range(len(train)):
            for slot in range(1,8):
                y,r,d=legalize(decoded[i,slot-1]);bank[i,slot]=y;repair[i,slot]=r;disp[i,slot]=d
        arrays[name]=[bank,cost,repair,disp]
    # Shared proposals for both methods, including any duplicate retries.
    retries=0
    for i,x in enumerate(train):
        seen={name:{x.tobytes()} for name in arrays}
        for slot in range(1,8):
            for attempt in range(CFG['candidates']['duplicate_retry_limit']):
                if all(arrays[name][0][i,slot].tobytes() not in seen[name] for name in arrays): break
                retries+=1
                z=mu[i]+tau[0,slot-1,0]*scales*rng.normal(size=mu.shape[1]);r=float(np.linalg.norm(z))
                direction=z/max(r,1e-30)
                if r<1e-30: direction=np.eye(1,mu.shape[1])[0]
                out_masks[i,slot-1]=r>rmax;in_masks[i,slot-1]=r<rmin
                for name,lower in [('vae_ball_wdro',0.),('vae_shell_wdro',rmin)]:
                    latent=(direction*np.clip(r,lower,rmax)).astype(np.float32)
                    with torch.no_grad():decoded=model.decode(torch.as_tensor(latent[None],device='cpu')).cpu().numpy()[0]
                    y,repaired,d=legalize(decoded);arrays[name][0][i,slot]=y;arrays[name][2][i,slot]=repaired;arrays[name][3][i,slot]=d
            for name,(bank,cost,repair,disp) in arrays.items():
                seen[name].add(bank[i,slot].tobytes());cost[i,slot]=hungarian_l1_mean(x,bank[i,slot])
        if i%128==0: progress('latent_geometry',completed=i,total=len(train))
    equal=float(np.all(arrays['vae_ball_wdro'][0][:,1:]==arrays['vae_shell_wdro'][0][:,1:],axis=(2,3)).mean())
    for name,(bank,cost,repair,disp) in arrays.items():
        unique=np.array([len({x.tobytes() for x in group}) for group in bank])
        save_geometry(name,bank,cost,repair,disp,unique,indices,retries=retries,
            inner_clipping_fraction=float(in_masks.mean()) if name=='vae_shell_wdro' else 0.,
            outer_clipping_fraction=float(out_masks.mean()),ball_shell_exact_equal_fraction=equal)
    mark_done('latent_geometry',seconds=time.monotonic()-start,ball_shell_exact_equal_fraction=equal)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['raw','latent']);args=parser.parse_args()
    build_raw() if args.stage=='raw' else build_latent()
