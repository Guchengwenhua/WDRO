"""Pure-RL expansion formulas, paired streams and immutable task bindings."""
import hashlib
import math
import numpy as np
import torch
import geometry as G
from common import ROOT, GROUPS, read, training_batch

def sources():
    return sorted([*ROOT.glob('*.py'), *ROOT.glob('*.sh'), *(ROOT/'vendor/models').glob('*.py'),
                   ROOT/'docs/THEORY_PROTOCOL.md'])

def task_binding(task_path):
    t=read(task_path)
    return dict(arm=t['arm'],seed=t['seed'],task_id=t['id'],
                protocol_sha256=G.sha256(ROOT/'protocol.json'),
                data_manifest_sha256=G.sha256(ROOT/'artifacts/data/manifest.json'),
                code_sha256={str(x):G.sha256(x) for x in sources()},
                dependencies=G.dependencies(),task_sha256=G.sha256(task_path))

def learning_rate(c,step,p):
    if c['schedule']=='original': return 1e-4 if step<=8192 else 5e-5
    if c['schedule']=='cosine':
        fraction=np.clip((step-p['cosine_start'])/(p['cosine_end']-p['cosine_start']),0,1)
        return p['cosine_min_lr']+(c['lr']-p['cosine_min_lr'])*.5*(1+math.cos(math.pi*fraction))
    return c['lr']

def batch_for(c,seed,step,p):
    chunks=c['batch']//64
    first=step if chunks==1 else p['parent_step']+(step-p['parent_step']-1)*chunks+1
    raw=[training_batch('M',seed,first+i)[0].reshape(4,16,30,2) for i in range(chunks)]
    pins=np.concatenate(raw,axis=1).reshape(c['batch'],30,2)
    original_hash=hashlib.sha256(pins.tobytes()).hexdigest()
    if c['augmentation']:
        rng=np.random.default_rng(G.derived_seed('expansion-d4',seed,step))
        swap=rng.integers(2,size=len(pins)).astype(bool)
        pins[swap]=pins[swap,:,::-1]
        flip=rng.integers(2,size=(len(pins),1,2)).astype(bool)
        pins=np.where(flip,1-pins,pins).astype(np.float32)
    return pins,dict(first_base_step=first,chunks=chunks,unaugmented_sha256=original_hash)

def group_weights(logr,c):
    return (1-c['uniform_mix'])*logr.exp()+c['uniform_mix']/4

def next_logr(logr,risks,c):
    out=logr+c['eta']*risks.to(torch.float64)
    return out-torch.logsumexp(out,dim=0)

def losses(advantage,logprob,c,logr,anchors,previous_variance,p):
    scale=max(math.sqrt(previous_variance),p['advantage_scale_floor']) if c['advantage_ema'] else 1.
    group_loss=(advantage.detach()*logprob/scale).reshape(4,-1).mean(1)
    if c['normalize_risk']: group_loss=group_loss/anchors.to(group_loss.dtype)
    q=group_weights(logr,c) if c['objective']=='group' else torch.full_like(logr,.25)
    loss=(advantage.detach()*logprob/scale).mean() if c['objective']=='erm' else (q.to(group_loss.dtype)*group_loss).sum()
    return loss,q,scale

def select_variants(records,p):
    """Only fixed-endpoint validation, never test; missing candidate is not an empty success."""
    base=records['E50']; selected={}
    for metric,other in [('mean','worst'),('worst','mean')]:
        eligible=[]
        for arm,rows in records.items():
            if arm in ('E50','B256'): continue
            a=np.asarray([x[metric] for x in rows]); b=np.asarray([x[metric] for x in base])
            av=np.mean([x[other] for x in rows]); bv=np.mean([x[other] for x in base])
            gain=100*(b.mean()-a.mean())/b.mean()
            if gain>=p['promotion_threshold_percent'] and (a<b).sum()>=2 and av<=bv*1.01:
                eligible.append((float(a.mean()),arm))
        selected[metric]=min(eligible)[1] if eligible else None
    return selected

def select_architecture(records,p):
    b=records['BASE']; candidates=[]
    for arm in ('WIDE','DEEP'):
        a=records[arm]
        x=np.asarray([v['worst'] for v in a]); y=np.asarray([v['worst'] for v in b])
        gain=100*(y.mean()-x.mean())/y.mean()
        if gain>=p['promotion_threshold_percent'] and (x<y).sum()>=2 and np.mean([v['mean'] for v in a])<=np.mean([v['mean'] for v in b])*1.01:
            candidates.append((float(x.mean()),arm))
    return min(candidates)[1] if candidates else None
