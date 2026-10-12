"""Held-out denoising and geometry stress diagnostics; no downstream selection."""
from exp_common import *
from generator import load_model
from pools import gate_fit
from structure import features
import torch

def main():
    if (ART/'done/generator_diagnostics.json').exists():return
    seed_all(666);data=load_data();rows=[];gate=gate_fit();stress=[]
    for split in ['B_test','C_external']:
        ids=np.flatnonzero(data['split']==split);x=data['pins'][ids];ds=data['design_ids'][ids]
        for kind in ['set','graph']:
            for seed in [301,302,303]:
                model,_=load_model(kind,seed);rng=np.random.default_rng(66492)
                for sigma in CFG['generator']['sigmas']:
                    noise=rng.normal(size=x.shape).astype(np.float32)*sigma;y=x+noise;out=[]
                    with torch.no_grad():
                        for b in range(0,len(x),128):out.append(model(torch.tensor(y[b:b+128],device='cuda'),torch.full((len(y[b:b+128]),),sigma,device='cuda')).cpu().numpy())
                    rec=np.concatenate(out);err=((rec-x)**2).mean((1,2))/sigma**2;identity=(noise**2).mean((1,2))/sigma**2
                    for d in np.unique(ds):rows.append(dict(kind=kind,seed=seed,split=split,sigma=sigma,design=d,nets=int((ds==d).sum()),normalized_mse=float(err[ds==d].mean()),identity_mse=float(identity[ds==d].mean())))
                    progress('generator_diagnostics',kind=kind,seed=seed,split=split,sigma=sigma)
    for i in np.flatnonzero(data['split']=='B_test')[::20]:
        x=data['pins'][i]
        variants={'real':x,'cluster_collapse':x.mean(0)+.05*(x-x.mean(0)),'single_outlier':x.copy(),'axis_collapse':x.copy()}
        variants['single_outlier'][0]=np.array([.9,.9]);variants['axis_collapse'][:,1]=x[:,1].mean()+.005*(x[:,1]-x[:,1].mean())
        for name,z in variants.items():
            score=float(np.max(np.abs((features(z)-gate['center'])/gate['scale'])))
            stress.append(dict(net_id=data['net_ids'][i],design=data['design_ids'][i],variant=name,score=score,rejected=score>gate['threshold'],label_scope='synthetic geometry stress, not actual circuit illegality'))
    write_csv(ROOT/'tables/generator_test.csv',rows);write_csv(ROOT/'tables/geometry_stress.csv',stress);mark_done('generator_diagnostics',generator_rows=len(rows),stress_rows=len(stress))
if __name__=='__main__':main()
