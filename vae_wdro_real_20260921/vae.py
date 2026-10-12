"""Set VAE with detached Hungarian assignment and differentiable matched loss."""
import time
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from exp_common import *

def normalize_torch(x):
    shifted = x - x.amin(dim=-2, keepdim=True)
    span = shifted.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return .1 + .8 * shifted / span

class SetVAE(nn.Module):
    def __init__(self, latent):
        super().__init__(); self.latent = latent
        self.point = nn.Sequential(nn.Linear(2,64), nn.SiLU(), nn.Linear(64,128), nn.SiLU())
        self.pool = nn.Sequential(nn.Linear(256,128), nn.SiLU())
        self.mu = nn.Linear(128,latent); self.logvar = nn.Linear(128,latent)
        self.decoder = nn.Sequential(nn.Linear(latent,128), nn.SiLU(), nn.Linear(128,256), nn.SiLU(), nn.Linear(256,60))

    def encode(self, x):
        h = self.point(x); h = self.pool(torch.cat([h.mean(1), h.amax(1)], -1))
        return self.mu(h), self.logvar(h).clamp(-10,5)

    def decode(self, z):
        return normalize_torch(.1 + .8 * torch.sigmoid(self.decoder(z).reshape(-1,30,2)))

    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(mu + torch.randn_like(mu) * (.5*lv).exp()), mu, lv

def reconstruction_loss(decoded, target):
    pairwise = (decoded[:,:,None,:] - target[:,None,:,:]).square().sum(-1)
    columns = np.stack([linear_sum_assignment(c)[1] for c in pairwise.detach().cpu().numpy()])
    columns = torch.as_tensor(columns, device=target.device)
    chosen = target[torch.arange(len(target),device=target.device)[:,None], columns]
    return (decoded - chosen).square().sum(-1).mean()

@torch.no_grad()
def validate(model, pins):
    model.eval(); costs=[]; mus=[]; kls=[]
    for begin in range(0,len(pins),128):
        x = torch.as_tensor(pins[begin:begin+128],device='cuda')
        mu, lv = model.encode(x); out=model.decode(mu)
        pair = (out[:,:,None,:] - x[:,None,:,:]).abs().sum(-1).cpu().numpy()
        costs.extend(float(m[linear_sum_assignment(m)].mean()) for m in pair)
        mus.append(mu.cpu().numpy()); kls.append((-.5*(1+lv-mu.square()-lv.exp())).mean(-1).cpu().numpy())
    mus=np.concatenate(mus); kls=np.concatenate(kls)
    return dict(mean_l1=float(np.mean(costs)), median_l1=float(np.median(costs)), p95_l1=float(np.quantile(costs,.95)),
                latent_mean_variance=float(mus.var(0).mean()), mean_kl=float(kls.mean()))

def fit(latent, train, val, budget_minutes):
    folder=ART/'vae'/f'L{latent}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists(): return read_json(folder/'done.json')
    seed_all(CFG['vae']['seed'])
    model=SetVAE(latent).cuda(); opt=torch.optim.Adam(model.parameters(),lr=CFG['vae']['learning_rate'])
    epoch=0; best=None; patience=0; elapsed=0.; history=[]
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True); opt.load_state_dict(state['optimizer'])
        epoch=state['epoch']; best=state['best']; patience=state['patience']; elapsed=state['elapsed_seconds']; history=state['history']
        set_rng(state['rng'])
    wall=time.monotonic()
    while epoch < CFG['vae']['max_epochs']:
        epoch+=1; model.train(); order=torch.randperm(len(train)); losses=[]; recons=[]; kls=[]
        beta=CFG['vae']['beta_max']*min(1.,epoch/CFG['vae']['beta_warmup_epochs'])
        for begin in range(0,len(train),CFG['vae']['batch_size']):
            x=torch.as_tensor(train[order[begin:begin+128].numpy()],device='cuda')
            # Independent within-net shuffling leaves the set encoder unchanged.
            perm=torch.rand(len(x),30,device='cuda').argsort(1)
            x=x[torch.arange(len(x),device='cuda')[:,None],perm]
            decoded,mu,lv=model(x); rec=reconstruction_loss(decoded,x)
            kl=(-.5*(1+lv-mu.square()-lv.exp())).mean(); loss=rec+beta*kl
            if not torch.isfinite(loss): raise RuntimeError('nonfinite VAE loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise RuntimeError('nonfinite VAE gradient')
            opt.step();losses.append(float(loss.detach()));recons.append(float(rec.detach()));kls.append(float(kl.detach()))
        record=dict(epoch=epoch,loss=float(np.mean(losses)),reconstruction_squared_l2=float(np.mean(recons)),kl=float(np.mean(kls)),beta=beta)
        improved=False
        if epoch % CFG['vae']['validation_interval_epochs'] == 0:
            metrics=validate(model,val);record['validation']=metrics
            if best is None or metrics['mean_l1'] < best['mean_l1']:
                best=dict(epoch=epoch,**metrics);patience=0;improved=True
            else: patience+=1
        record['elapsed_seconds']=elapsed+time.monotonic()-wall;history.append(record)
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),latent=latent,epoch=epoch,best=best,
                   patience=patience,elapsed_seconds=record['elapsed_seconds'],rng=rng_state(),history=history)
        atomic_torch(folder/'last.pt',state)
        if improved: atomic_torch(folder/'best.pt',state)
        # Epoch history is committed with the resumable checkpoint.
        (folder/'curve.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
        progress('vae',latent=latent,epoch=epoch,best=best,elapsed_seconds=record['elapsed_seconds'])
        if epoch % 5 == 0: print('VAE',latent,record,flush=True)
        if epoch >= CFG['vae']['min_epochs'] and (patience >= CFG['vae']['early_stop_validation_patience'] or record['elapsed_seconds'] >= budget_minutes*60): break
    result=dict(latent=latent,epochs=epoch,best=best,seconds=elapsed+time.monotonic()-wall,path=str(folder/'best.pt'))
    atomic_json(folder/'done.json',result);return result

def load_selected(device='cuda'):
    selected=read_json(ART/'vae'/'selected.json')
    state=torch.load(selected['path'],map_location='cpu',weights_only=False)
    model=SetVAE(state['latent']).to(device);model.load_state_dict(state['model'],strict=True);model.eval()
    return model,selected

def main():
    if (ART/'done'/'vae.json').exists(): return
    start=time.monotonic();data=load_data()
    train=data['pins'][data['split']=='A_train'];val=data['pins'][data['split']=='A_val']
    results=[fit(CFG['vae']['latent_dim'],train,val,CFG['vae']['soft_time_minutes'])]
    while not (ART/'rho.json').exists():
        progress('vae',status='waiting_for_geometry_only_rho');time.sleep(5)
    rho=read_json(ART/'rho.json')['rho']
    if results[0]['best']['median_l1'] > rho:
        results.append(fit(CFG['vae']['fallback_latent_dim'],train,val,CFG['vae']['fallback_soft_time_minutes']))
    selected=min(results,key=lambda x:x['best']['mean_l1'])
    atomic_json(ART/'vae'/'selected.json',dict(**selected,common_rho=rho,fallback_triggered=len(results)>1,
        all_fits=results,selection_data='A_val only',fallback_still_poor=selected['best']['median_l1']>rho))
    model,_=load_selected()
    # Fixed validation examples only; no test-driven generator selection.
    with torch.no_grad():
        v=torch.as_tensor(val[:8],device='cuda');out=model.decode(model.encode(v)[0]).cpu().numpy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for ax,x,y in zip(axes.flat,val[:8],out,strict=True):
        ax.scatter(x[:,0],x[:,1],s=12,label='Validation input');ax.scatter(y[:,0],y[:,1],s=12,marker='x',label='VAE reconstruction')
        ax.set(xlim=(.05,.95),ylim=(.05,.95),aspect='equal')
    axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ART/'vae'/'validation_reconstruction.png',dpi=160);plt.close(fig)
    mark_done('vae',seconds=sum(x['seconds'] for x in results),selected=selected)

if __name__=='__main__': main()
