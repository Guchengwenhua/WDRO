"""Finite-support entropy-regularized transport; numpy float64, no oracle."""
import numpy as np

def distribution(loss, cost, tau, lam):
    a=(np.asarray(loss,dtype=np.float64)-lam*np.asarray(cost,dtype=np.float64))/tau
    a-=a.max(axis=1,keepdims=True)
    e=np.exp(a)
    return e/e.sum(axis=1,keepdims=True)

def solve(loss,cost,tau,rho):
    loss=np.asarray(loss,dtype=np.float64);cost=np.asarray(cost,dtype=np.float64)
    if loss.ndim!=2 or cost.shape!=loss.shape or tau<=0 or rho<=0:
        raise ValueError('invalid finite transport problem')
    if not np.isfinite(loss).all() or not np.isfinite(cost).all() or (cost<0).any() or (cost[:,0]!=0).any():
        raise ValueError('nonfinite inputs or missing zero-cost anchor')
    def expectation(lam):
        p=distribution(loss,cost,tau,lam)
        return p,float((p*cost).sum(1).mean())
    p,ec=expectation(0);lam=0.
    if ec>rho:
        lo,hi=0.,1.
        for _ in range(80):
            if expectation(hi)[1]<=rho:break
            hi*=2
        else:raise RuntimeError('transport bracket failed')
        for _ in range(60):
            mid=(lo+hi)/2
            if expectation(mid)[1]>rho:lo=mid
            else:hi=mid
        lam=hi;p,ec=expectation(lam)
    kl=(p*np.log(np.maximum(p,1e-300)*loss.shape[1])).sum(1).mean()
    primal=float((p*loss).sum(1).mean()-tau*kl)
    a=(loss-lam*cost)/tau;m=a.max(1)
    dual=float(lam*rho+tau*(m+np.log(np.exp(a-m[:,None]).mean(1))).mean())
    tolerance=1e-9*max(1.,abs(primal),abs(dual))
    info=dict(lambda_=lam,expected_cost=ec,primal=primal,dual=dual,gap=dual-primal,gap_tolerance=tolerance,
              entropy=float(-(p*np.log(np.maximum(p,1e-300))).sum(1).mean()))
    if ec>rho+1e-10 or abs(dual-primal)>tolerance:
        raise RuntimeError(f'finite transport certificate failed: {info}')
    return p,info

def sample(p,rng):
    return (rng.random(len(p))[:,None]>np.cumsum(p,axis=1)).sum(1).clip(max=p.shape[1]-1)

def tail_loss(x,eta,alpha=.9,beta=.5):
    return (1-beta)*x+beta*(eta+np.maximum(x-eta,0)/(1-alpha))
