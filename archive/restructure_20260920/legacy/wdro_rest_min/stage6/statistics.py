"""Paired, stratified net bootstrap conditional on the three fixed seeds."""
import numpy as np

def bootstrap_cell(values, *, samples=10000, seed=606, chunk=16, heartbeat=None):
    """values is seed x net; identical net draws across seeds and methods.

    Return per-seed mean and P95 replicates. Degrees/distributions use distinct
    random streams; comparison methods use the same stream for exact pairing.
    """
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]==0 or not np.isfinite(values).all():raise ValueError('seed x net finite values required')
    rng=np.random.default_rng(seed);means=np.empty((samples,len(values)));tails=np.empty_like(means)
    for start in range(0,samples,chunk):
        count=min(chunk,samples-start);indices=rng.integers(0,values.shape[1],size=(count,values.shape[1]))
        for s in range(len(values)):
            draws=values[s,indices];means[start:start+count,s]=draws.mean(axis=1)
            tails[start:start+count,s]=np.quantile(draws,.95,axis=1)
        if heartbeat and start%160==0:heartbeat(start)
    return dict(mean=means,p95=tails)

def ci(draws):
    a=np.asarray(draws);return dict(ci_low=float(np.quantile(a,.025)),ci_high=float(np.quantile(a,.975)))

def metrics(values):
    a=np.asarray(values);return dict(mean=float(a.mean()),p95=float(np.quantile(a,.95,axis=1).mean()))
