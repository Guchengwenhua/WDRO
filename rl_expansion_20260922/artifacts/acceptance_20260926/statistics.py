"""Recompute frozen inference without importing original report/statistical code."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
GROUPS=('uniform','mix_gauss','edge_bias','grid_cluster')

def get(stage,arm,seeds,step):
    values=[]
    for seed in seeds:
        with np.load(ROOT/f'artifacts/eval/{stage}_{arm}_s{seed}_d30/step{step}.npz') as f:
            values.append(np.stack([f[g+'__length'] for g in GROUPS]))
    return np.stack(values)

def measure(a,m):
    groups=a.mean(axis=2)
    return groups.mean(axis=1) if m=='mean' else groups.max(axis=1)

def boot(job):
    name,b,c,m,reps,seed=job;started=time.time()
    bv=measure(b,m);cv=measure(c,m);delta=bv-cv
    rng=np.random.default_rng(seed);draws=np.empty(reps)
    for k in range(reps):
        si=rng.integers(b.shape[0],size=b.shape[0])
        ni=rng.integers(b.shape[2],size=(b.shape[1],b.shape[2]))
        # Draw paired nets once per group, then resample the trained model seeds.
        bgroup=np.take_along_axis(b,ni[None,:,:],axis=2).mean(2)
        cgroup=np.take_along_axis(c,ni[None,:,:],axis=2).mean(2)
        if m=='mean':dd=bgroup.mean(1)-cgroup.mean(1)
        else:dd=bgroup.max(1)-cgroup.max(1)
        draws[k]=dd[si].mean()
    point=float(bv.mean()-cv.mean());pct=100*point/bv.mean()
    ci95=np.quantile(draws,[.025,.975]);ci99=np.quantile(draws,[.005,.995])
    deterioration=100*(measure(c,'mean').mean()/measure(b,'mean').mean()-1)
    result=dict(metric=m,seeds=len(b),baseline_value=float(bv.mean()),candidate_value=float(cv.mean()),
        improvement=point,improvement_percent=float(pct),improved_seeds=int((delta>0).sum()),
        seed_improvement=delta.tolist(),seed_improvement_percent=(100*delta/bv).tolist(),
        ci95=ci95.tolist(),simultaneous_ci99=ci99.tolist(),bootstrap_replicates=reps,bootstrap_seed=seed,
        support=bool(pct>=.5 and (delta>0).sum()>len(b)/2 and ci99[0]>0 and (m!='worst' or deterioration<=1)),
        inference_scope='frozen_primary' if name.startswith('budget_') else 'posthoc_descriptive_no_familywise_claim',
        elapsed_seconds=time.time()-started)
    if name.startswith('budget_'):
        original=json.loads((ROOT/'artifacts/report/statistics.json').read_text())[name]
        for key in ('baseline_value','candidate_value','improvement','improvement_percent','seed_improvement','ci95','simultaneous_ci99'):
            np.testing.assert_allclose(result[key],original[key],rtol=0,atol=1e-10)
        assert result['support']==original['support'] and result['improved_seeds']==original['improved_seeds']
    np.save(OUT/(name+'_bootstrap.npy'),draws)
    print(name,pct,ci99.tolist(),flush=True)
    return name,result

def main():
    p=json.loads((ROOT/'protocol.json').read_text());seeds=p['screening_seeds']+p['new_seeds']
    old=np.concatenate([get('origin','ME',p['screening_seeds'],12288),get('parents','ME',p['new_seeds'],12288)])
    final=get('final','E50',seeds,49152);jobs=[]
    for i,m in enumerate(('mean','worst')):
        jobs.append(('budget_'+m,old,final,m,p['bootstrap_replicates'],p['bootstrap_seed']+i))
    reference=get('screen','E50',p['screening_seeds'],20480)
    for j,(label,step) in enumerate([('batch_sample_matched',14336),('batch_update_matched',20480)]):
        candidate=get('screen','B256',p['screening_seeds'],step)
        for i,m in enumerate(('mean','worst')):jobs.append((label+'_'+m,reference,candidate,m,5000,9262026+2*j+i))
    with ProcessPoolExecutor(max_workers=6) as pool:stats=dict(pool.map(boot,jobs))
    for slot in p['primary_slots']:stats.setdefault(slot,dict(status='not_selected_no_claim'))
    (OUT/'statistics_independent.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
