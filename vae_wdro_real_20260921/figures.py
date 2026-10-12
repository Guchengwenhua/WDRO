"""Fixed validation-only reconstruction/generation illustrations and fit curves."""
import numpy as np
import torch
from exp_common import *
from candidates import perturb,legalize
from vae import load_selected

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed_all(CFG['candidates']['seed']);data=load_data();pins=data['pins'][data['split']=='A_val'][:4]
    model,_=load_selected('cpu');cal=read_json(ART/'vae'/'calibration.json');rng=np.random.default_rng(20260921)
    with torch.no_grad():mu=model.encode(torch.as_tensor(pins))[0].numpy()
    tau=np.array(CFG['candidates']['latent_local_tau']);scale=np.array(cal['scale'])
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    for row,x in enumerate(pins):
        raw=np.stack([perturb(x,f,a,rng)[0] for f,a in CFG['candidates']['raw_family_amplitude']])
        z=mu[row]+tau[:,None]*scale*rng.normal(size=(7,len(scale)))
        radius=np.linalg.norm(z,axis=1);direction=z/np.maximum(radius[:,None],1e-30)
        decoded=[]
        for lower in [0,cal['r_min']]:
            latent=(direction*np.clip(radius,lower,cal['r_max'])[:,None]).astype(np.float32)
            with torch.no_grad():y=model.decode(torch.as_tensor(latent)).numpy()
            decoded.append(np.stack([legalize(v)[0] for v in y]))
        groups=[x[None],raw,*decoded]
        for col,group in enumerate(groups):
            ax=axes[row,col]
            for y in group:ax.scatter(y[:,0],y[:,1],s=8,alpha=.25,color='tab:orange')
            ax.scatter(x[:,0],x[:,1],s=15,c='black',marker='x')
            ax.set(xlim=(.05,.95),ylim=(.05,.95),aspect='equal')
            if row==0:ax.set_title(['Validation input','RAW proposals','VAE BALL proposals','VAE SHELL proposals'][col])
    fig.suptitle('Same four validation inputs; black = real pins, orange = candidates',fontsize=13)
    fig.tight_layout();fig.savefig(ART/'vae'/'validation_generation.png',dpi=160);fig.savefig(ART/'vae'/'validation_generation.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for latent in [16,32]:
        path=ART/'vae'/f'L{latent}'/'curve.jsonl'
        if not path.exists():continue
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        axes[0].plot([r['epoch'] for r in rows],[r['reconstruction_squared_l2'] for r in rows],label=f'L={latent}')
        val=[r for r in rows if 'validation' in r]
        axes[1].plot([r['epoch'] for r in val],[r['validation']['mean_l1'] for r in val],label=f'L={latent}')
    axes[0].set(ylabel='Train Hungarian squared L2',xlabel='Epoch');axes[1].set(ylabel='Validation Hungarian L1',xlabel='Epoch')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(ART/'vae'/'fit_curves.png',dpi=160);plt.close(fig)
    atomic_json(ART/'vae'/'illustration.json',dict(validation_indices=[0,1,2,3],seed=20260921,
        note='Illustration-only shared validation inputs; no oracle calls, parameter selection or training use.'))

if __name__=='__main__':main()
