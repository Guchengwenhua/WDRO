"""Fixed-budget multidegree pretraining, d30 posttraining, validation-only selection."""
from exp_common import *
import torch, argparse
from net_models import Denoiser,losses

def configs():
    return [dict(kind=k,width=w,eta=e,key=f'{k}_w{w}_e{e:g}') for k in ['set','graph'] for w in [96,160] for e in ([0.] if k=='set' else [.1,1.])]

def validation(model,device):
    f=np.load(ART/'multidegree/d30_A_val.npz');x=f['pins'];ds=f['design'];rng=np.random.default_rng(99881);rows=[]
    model.eval()
    with torch.no_grad():
        for sigma in CFG['generator']['sigmas']:
            noise=rng.normal(size=x.shape).astype(np.float32)*sigma;y=x+noise;outs=[]
            for b in range(0,len(x),128):outs.append(model(torch.tensor(y[b:b+128],device=device),torch.full((len(y[b:b+128]),),sigma,device=device)).cpu().numpy())
            out=np.concatenate(outs);err=((out-x)**2).mean((1,2))/sigma**2;identity=(noise**2).mean((1,2))/sigma**2
            for design in np.unique(ds):rows.append(dict(design=design,sigma=sigma,normalized_mse=float(err[ds==design].mean()),identity_mse=float(identity[ds==design].mean())))
    return dict(score=float(np.mean([r['normalized_mse'] for r in rows])),rows=rows)

def train(key,seed,phase):
    spec=next(c for c in configs() if c['key']==key);gc=CFG['generator'];name=f'gen_{phase}_{key}_s{seed}';folder=ART/'generators'/name
    if (folder/'done.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);device=torch.device('cuda');seed_all(seed);torch.cuda.set_per_process_memory_fraction(CFG['runtime']['gpu_memory_fraction'])
    model=Denoiser(spec['kind'],spec['width']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=gc['lr'],weight_decay=1e-4)
    pre=gc['tune_pre' if phase=='tune' else 'formal_pre'];post=gc['tune_post' if phase=='tune' else 'formal_post'];total=pre+post
    arrays={};groups={}
    for d in gc['degrees']:
        f=np.load(ART/'multidegree'/f'd{d}_A_train.npz');arrays[d]=f['pins'];groups[d]=[np.flatnonzero(f['design']==v) for v in np.unique(f['design'])]
    step=0;best=None;history=[];elapsed=0.
    if (folder/'latest.pt').exists():
        cp=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        assert cp['spec']==spec and cp['seed']==seed and cp['total']==total
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);set_rng(cp['rng']);step=cp['step'];best=cp['best'];history=cp['history'];elapsed=cp['seconds']
    start=time.monotonic()
    def save(path):
        atomic_torch(path,dict(spec=spec,seed=seed,phase=phase,step=step,total=total,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),best=best,history=history,seconds=elapsed+time.monotonic()-start))
    acc=[]
    while step<total:
        tick=time.monotonic();d=int(np.random.choice(gc['degrees'])) if step<pre else 30
        design=np.random.randint(len(groups[d]),size=gc['batch']);idx=np.array([np.random.choice(groups[d][j]) for j in design])
        x=torch.tensor(arrays[d][idx],device=device);sigma=torch.tensor(np.random.choice(gc['sigmas'],len(x)),dtype=torch.float32,device=device)
        # Training augmentation preserves axis geometry, with no arbitrary-angle rotations.
        if np.random.rand()<.5:x=x.flip(-1)
        if np.random.rand()<.5:x=1-x
        y=x+sigma[:,None,None]*torch.randn_like(x);model.train();out=model(y,sigma);loss,rec,st=losses(out,x,sigma,spec['eta'])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite denoiser loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite denoiser gradient')
        opt.step();step+=1;acc.append((float(loss.detach()),float(rec.detach()),time.monotonic()-tick))
        if step%50==0:
            progress(name,stage='pretrain' if step<=pre else 'posttrain',step=step,total=total,seconds=elapsed+time.monotonic()-start,seconds_per_step=float(np.mean([a[2] for a in acc])),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if step%gc['validation_every']==0 or step==total:
            v=validation(model,device);record=dict(step=step,stage='pre' if step<=pre else 'post',loss=float(np.mean([a[0] for a in acc])),rec=float(np.mean([a[1] for a in acc])),validation=v,seconds=elapsed+time.monotonic()-start);history.append(record);acc=[]
            # Select only within posttraining; pretraining best separately retained.
            if step<=pre:
                if best is None or v['score']<best['score']:best=dict(step=step,**v);save(folder/'pre_best.pt')
                if step==pre:save(folder/'pre_last.pt');best=None
            else:
                if best is None or v['score']<best['score']:best=dict(step=step,**v);save(folder/'best.pt')
            save(folder/'latest.pt');atomic_json(folder/'history.json',history)
            print(name,step,total,'validation',v['score'],flush=True)
    result=dict(name=name,key=key,spec=spec,seed=seed,phase=phase,total=total,best=best,seconds=elapsed+time.monotonic()-start,checkpoint=str(folder/'best.pt'))
    atomic_json(folder/'done.json',result);mark_done(name,**{k:v for k,v in result.items() if k!='name'})

def select():
    rows=[];selected={}
    for c in configs():
        p=ART/'generators'/f"gen_tune_{c['key']}_s301"/'done.json';r=read_json(p);rows.append(r)
    for kind in ['set','graph']:
        selected[kind]=min([r for r in rows if r['spec']['kind']==kind],key=lambda r:r['best']['score'])
    atomic_json(ART/'generator_selected.json',dict(selected=selected,selection='A_val fixed-noise d30 design-macro normalized MSE only',trials=rows,time=utc()))
    mark_done('generator_selection')

def load_model(kind,seed,device='cuda'):
    selected=read_json(ART/'generator_selected.json')['selected'][kind];key=selected['key'];folder=ART/'generators'/f'gen_formal_{key}_s{seed}'
    cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);m=Denoiser(kind,cp['spec']['width']).to(device);m.load_state_dict(cp['model']);m.eval();return m,cp

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--kind',choices=['set','graph']);p.add_argument('--seed',type=int,default=301);p.add_argument('--phase',choices=['tune','formal']);p.add_argument('--select',action='store_true');a=p.parse_args()
    if a.select:select()
    else:
        key=a.key or read_json(ART/'generator_selected.json')['selected'][a.kind]['key']
        train(key,a.seed,a.phase)
