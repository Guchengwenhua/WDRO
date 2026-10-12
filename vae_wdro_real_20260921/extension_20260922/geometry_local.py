"""Shared canonical normalization, safe separation and raw perturbations."""
import numpy as np
from exp_common import *
from src.geometry import project_normalized_net
from src.costs import hungarian_l1_mean

def legalize(x):
    x=np.asarray(x,dtype=np.float64);x=x-x.min(0);span=float(x.max())
    if not np.isfinite(span) or span<=1e-12:raise ValueError('degenerate candidate')
    base=np.clip(.1+.8*x/span,.1,.9).astype(np.float32)
    dist=np.abs(base[:,None]-base[None,:]).sum(-1);np.fill_diagonal(dist,np.inf)
    repaired=bool(dist.min()<1.1e-6)
    out=project_normalized_net(base,minimum_l1_spacing=2e-6).astype(np.float32) if repaired else base
    check_pins(out);return canonical(out),repaired,float(np.abs(out-base).max())

def perturb(x,family,amplitude,rng):
    n=len(x)
    if family=='jitter':delta=rng.normal(0,amplitude,(n,2))
    elif family=='sparse':
        delta=np.zeros((n,2));idx=rng.choice(n,max(1,n//5),replace=False);delta[idx]=rng.normal(0,amplitude,(len(idx),2))
    elif family=='correlated':delta=(x-x.mean(0))@rng.normal(0,amplitude,(2,2))+rng.normal(0,amplitude/4,(n,2))
    elif family=='axis_biased':
        scale=np.array([amplitude,amplitude/5]);scale=scale[::-1] if rng.integers(2) else scale;delta=rng.normal(size=(n,2))*scale
    else:raise ValueError(family)
    return legalize(x+delta)

def normalize_torch(x):
    shifted=x-x.amin(-2,keepdim=True)
    return .1+.8*shifted/shifted.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-8)
