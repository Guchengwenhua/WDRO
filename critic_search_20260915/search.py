"""Budget-matched trajectory ablations. Frozen parent utilities remain untouched."""
import bootstrap
import numpy as np
import torch
from adv_v3.common import legalize, costs, GEOMETRY_LEDGER
from adv_v3.critic import matched_l1
from adv_v3.search import _check_pins

BASE = dict(steps=8, step=.01, starts=63, init='gaussian', mode='gradient',
            select='endpoint', update='rms', geometry='post', max_motion=.08)

def configurations():
    configs = {'baseline':dict(BASE), 'no_move':dict(BASE,steps=0),
               'random_direction':dict(BASE,mode='random')}
    for t in (4,8,16):
        for a in (.0025,.005,.01):
            configs[f'step_{a:g}_{t}'] = dict(BASE,steps=t,step=a)
    configs.update({
        'decay16':dict(BASE,steps=16,update='decay'),
        'adaptive16':dict(BASE,steps=16,update='adaptive'),
        'adam16':dict(BASE,steps=16,update='adam'),
        'mixed_scales':dict(BASE,init='mixed'),
        'structured_starts':dict(BASE,init='structured'),
        'wide126_short4':dict(BASE,starts=126,steps=4,select='diverse_end'),
        'narrow21_long24':dict(BASE,starts=21,steps=24,select='checkpoints'),
        'trajectory21_8':dict(BASE,starts=21,steps=8,select='checkpoints'),
        'trajectory_diverse':dict(BASE,select='diverse_history'),
        'differentiable_normalization':dict(BASE,geometry='differentiable'),
        'remove_rigid':dict(BASE,geometry='remove_rigid'),
    })
    return configs

FAMILIES = {
    'direction3':[x for x in configurations() if x.startswith('step_')] + ['decay16','adaptive16','adam16'],
    'direction4':['mixed_scales','structured_starts','wide126_short4','narrow21_long24'],
    'direction5':['trajectory21_8','narrow21_long24','trajectory_diverse'],
    'direction6':['differentiable_normalization','remove_rigid','adaptive16'],
}

def normalized_map(x, u):
    raw = (x + u).clamp(.1,.9)
    low = raw.amin(dim=1,keepdim=True)
    span = (raw.amax(dim=1,keepdim=True)-low).amax(dim=2,keepdim=True)
    return .1 + .8*(raw-low)/span.clamp_min(1e-12)

def _diverse_pick(points, values, count):
    # Restrict eligibility to top 4*budget by proxy, then greedily spread geometry.
    order = np.argsort(-values,kind='stable')[:min(len(values),count*4)]
    pool = points[order].reshape(len(order),-1).astype(np.float64)
    selected=[0]; dist=np.square(pool-pool[0]).mean(1)
    dist[0]=-np.inf
    for _ in range(1,count):
        k=int(np.argmax(dist)); selected.append(k)
        dist=np.minimum(dist,np.square(pool-pool[k]).mean(1));dist[selected]=-np.inf
    return order[np.asarray(selected)]

def propose(model, centers, price, seed, config):
    cfg=dict(BASE,**config); centers=np.asarray(centers,dtype=np.float32)
    b,n,_=centers.shape; k=cfg['starts']; steps=cfg['steps']
    orders=np.argsort(centers[:,:,0]+centers[:,:,1]*1e-7,axis=1,kind='stable')
    # Lexicographic canonicalization agrees with the existing implementation.
    orders=np.asarray([np.lexsort((x[:,1],x[:,0])) for x in centers])
    canon=np.stack([x[o] for x,o in zip(centers,orders)])
    x_np=np.repeat(canon,k,axis=0)
    rng=np.random.default_rng(seed)
    initial=rng.normal(size=(b,k,n,2))
    scale=np.full(k,.02)
    if cfg['init']=='mixed':scale=np.resize([.005,.02,.05],k)
    initial*=scale[None,:,None,None]
    if cfg['init']=='structured':
        # Shared Gaussian starts in even positions; odd starts perturb a sparse axis.
        for j in range(1,k,2):
            initial[:,j,:,j%2]=0
            mask=np.arange(n)%4 != (j//2)%4
            initial[:,j,mask,:]=0
    initial=np.clip(initial.reshape(-1,n,2),-cfg['max_motion'],cfg['max_motion'])
    before=dict(GEOMETRY_LEDGER)
    candidate=legalize(x_np,initial)
    history=[candidate.copy()]
    diag=dict(forward_samples=0,backward_samples=0,clipped_coordinates=0,total_coordinates=0,
              cosine_sum=0.,cosine_count=0)
    p=next(model.parameters());device=p.device;was_training=model.training
    flags=[v.requires_grad for v in model.parameters()]
    model.eval()
    for v in model.parameters():v.requires_grad_(False)
    lr=np.full((len(x_np),1,1),cfg['step'],dtype=np.float32)
    moment=np.zeros_like(x_np); variance=np.zeros_like(x_np)
    latent=initial.copy()
    try:
        for t in range(steps):
            next_points=np.empty_like(candidate)
            # Generate random direction for the full population, independent of chunking.
            random_dir=rng.normal(size=candidate.shape).astype(np.float32) if cfg['mode']=='random' else None
            for start in range(0,len(x_np),128):
                sl=slice(start,start+128); xx=x_np[sl]; zz=candidate[sl]
                x=torch.as_tensor(xx,device=device,dtype=p.dtype)
                if cfg['geometry']=='differentiable':
                    variable=torch.tensor(latent[sl],device=device,dtype=p.dtype,requires_grad=True)
                    z=normalized_map(x,variable)
                else:
                    variable=torch.tensor(zz,device=device,dtype=p.dtype,requires_grad=True);z=variable
                score=model(z)-float(price)*matched_l1(x,z)
                grad=torch.autograd.grad(score.sum(),variable)[0].detach().cpu().numpy()
                diag['forward_samples']+=len(xx);diag['backward_samples']+=len(xx)
                if not np.isfinite(grad).all():raise RuntimeError('nonfinite search gradient')
                if random_dir is not None:grad=random_dir[sl]
                if cfg['geometry']=='remove_rigid':
                    grad=grad-grad.mean(axis=1,keepdims=True)
                    radial=zz-zz.mean(axis=1,keepdims=True)
                    coeff=(grad*radial).sum((1,2),keepdims=True)/np.maximum((radial**2).sum((1,2),keepdims=True),1e-12)
                    grad-=coeff*radial
                if cfg['update']=='adam':
                    moment[sl]=.9*moment[sl]+.1*grad;variance[sl]=.999*variance[sl]+.001*grad**2
                    grad=(moment[sl]/(1-.9**(t+1)))/(np.sqrt(variance[sl]/(1-.999**(t+1)))+1e-8)
                direction=grad/np.maximum(np.sqrt(np.mean(grad**2,axis=(1,2),keepdims=True)),1e-12)
                alpha=lr[sl]
                if cfg['update']=='decay':alpha=alpha*(1-.75*t/max(steps-1,1))
                raw_delta=(latent[sl] if cfg['geometry']=='differentiable' else zz-xx)+alpha*direction
                diag['clipped_coordinates']+=int((np.abs(raw_delta)>=cfg['max_motion']).sum())
                diag['total_coordinates']+=raw_delta.size
                delta=np.clip(raw_delta,-cfg['max_motion'],cfg['max_motion'])
                trial=legalize(xx,delta)
                actual=trial-zz; expected=alpha*direction
                denominator=np.linalg.norm(actual.reshape(len(xx),-1),axis=1)*np.linalg.norm(expected.reshape(len(xx),-1),axis=1)
                valid=denominator>1e-15
                diag['cosine_sum']+=float(((actual*expected).sum((1,2))[valid]/denominator[valid]).sum())
                diag['cosine_count']+=int(valid.sum())
                if cfg['update']=='adaptive':
                    with torch.no_grad():
                        znew=torch.as_tensor(trial,device=device,dtype=p.dtype)
                        newscore=model(znew)-float(price)*matched_l1(x,znew)
                    diag['forward_samples']+=len(xx)
                    better=(newscore>=score.detach()).cpu().numpy()
                    trial[~better]=zz[~better]
                    lr[sl]=np.where(better[:,None,None],alpha,np.maximum(alpha*.5,.000625))
                next_points[sl]=trial;latent[sl]=delta
            candidate=next_points;history.append(candidate.copy())
        hist=np.stack(history,axis=1).reshape(b,k,steps+1,n,2)
        if cfg['select']=='checkpoints':
            indices=[max(1,steps//4),steps//2,steps] if steps==8 else [steps//3,2*steps//3,steps]
            chosen=hist[:,:,indices].reshape(b,k*3,n,2)
        elif cfg['select'] in ('diverse_history','diverse_end'):
            pool=hist[:,:,1:].reshape(b,-1,n,2) if cfg['select']=='diverse_history' else hist[:,:,-1]
            chosen=[]
            for center,pts in zip(canon,pool):
                vals=[]
                with torch.no_grad():
                    for j in range(0,len(pts),128):
                        zz=torch.as_tensor(pts[j:j+128],device=device,dtype=p.dtype)
                        xx=torch.as_tensor(np.repeat(center[None],len(zz),axis=0),device=device,dtype=p.dtype)
                        vals.extend((model(zz)-float(price)*matched_l1(xx,zz)).cpu().tolist())
                        diag['forward_samples']+=len(zz)
                chosen.append(pts[_diverse_pick(pts,np.asarray(vals),63)])
            chosen=np.asarray(chosen)
        else:chosen=hist[:,:,-1]
        if chosen.shape!=(b,63,n,2):raise AssertionError(chosen.shape)
        restored=np.empty_like(chosen)
        for i,o in enumerate(orders):restored[i][:,o]=chosen[i]
        for field in GEOMETRY_LEDGER:diag[field]=GEOMETRY_LEDGER[field]-before[field]
        diag['mean_movement_cosine']=diag['cosine_sum']/max(1,diag['cosine_count'])
        diag['clipped_fraction']=diag['clipped_coordinates']/max(1,diag['total_coordinates'])
        diag['duplicate_fraction']=float(np.mean([1-len(np.unique(row.reshape(63,-1),axis=0))/63 for row in restored]))
        return restored,diag
    finally:
        for v,flag in zip(model.parameters(),flags):v.requires_grad_(flag)
        model.train(was_training)

def evaluate_candidates(centers, proposals, oracle, price):
    pins=np.concatenate((centers[:,None],proposals),axis=1)
    _check_pins(pins,centers.shape[1]);b,k,n,_=pins.shape
    if k!=64:raise AssertionError('64 real scores required')
    truth=oracle.evaluate(pins.reshape(-1,n,2))
    regret=np.asarray(truth['regret']).reshape(b,k);cost=costs(centers,pins)
    score=regret-float(price)*cost; selected=score.argmax(1);rows=np.arange(b)
    if np.any(score[rows,selected]<score[:,0]):raise AssertionError('lost nominal candidate')
    return dict(centers=centers,support=pins,all_regret=regret,all_cost=cost,all_score=score,
                all_lstar=np.asarray(truth['lstar']).reshape(b,k),all_length=np.asarray(truth['length']).reshape(b,k),
                selected_index=selected,regret=regret[rows,selected],cost=cost[rows,selected],
                score=score[rows,selected],original_regret=regret[:,0],calls=np.full(b,64),price=np.full(b,price))
