"""Auditable net geometry priors, preserving pin correspondence until canonicalization."""
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def features(x):
    w=x.max(0)-x.min(0);hp=max(w.sum(),1e-8);dist=np.abs(x[:,None]-x[None,:]).sum(-1)
    mst=float(minimum_spanning_tree(dist).sum());np.fill_diagonal(dist,np.inf);nn=dist.min(1)
    delta=np.abs(x[:,None]-x[None,:]);off=~np.eye(len(x),dtype=bool)
    axis=(delta[off]<.005).mean(0)
    q=(x>x.mean(0)).astype(int);quad=np.bincount(2*q[:,0]+q[:,1],minlength=4)/len(x)
    return np.r_[w,hp,np.log((w.max()+1e-6)/(w.min()+1e-6)),mst/hp,np.quantile(nn,[0,.25,.5,.95]),axis,quad]

def graph_noise(x,noise,tau=2.):
    x=np.asarray(x,dtype=np.float64)
    d=np.abs(x[:,None]-x[None,:]).sum(-1);np.fill_diagonal(d,np.inf);order=np.lexsort((x[:,1],x[:,0]));ii=order[np.argsort(d[:,order],axis=1,kind="stable")[:,:4]]
    a=np.zeros_like(d);rows=np.arange(len(x))[:,None];s=max(float(np.median(d[rows,ii])),.005)
    a[rows,ii]=np.exp(-d[rows,ii]/s);a=np.maximum(a,a.T);lap=np.diag(a.sum(1))-a
    out=np.linalg.solve(np.eye(len(x))+tau*lap,noise);out-=out.mean(0)
    return out*(np.sqrt((noise**2).mean())/max(np.sqrt((out**2).mean()),1e-12))

def tile_project(x,y,lower,upper):
    """Keep original pin identity, bbox and routing tile; backtrack as one displacement."""
    # Offset bounds toward the interior; original points on tile boundaries are retained when fixed.
    z=np.clip(y,np.maximum(lower+1e-7,x.min(0)),np.minimum(upper-1e-7,x.max(0)))
    fixed=(x==x.min(0))|(x==x.max(0));z[fixed]=x[fixed]
    for fraction in [1.,.5,.25,.125,0.]:
        out=(x+fraction*(z-x)).astype(np.float32)
        dist=np.abs(out[:,None]-out[None,:]).sum(-1);np.fill_diagonal(dist,np.inf)
        if dist.min()>=1.01e-6 and np.all(out>=lower-6e-8) and np.all(out<=upper+6e-8):
            return out,fraction
    raise RuntimeError('even nominal net failed tile constraint')
