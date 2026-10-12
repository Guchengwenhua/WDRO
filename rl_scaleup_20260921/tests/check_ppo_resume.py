"""CPU engineering test: both PPO co-training checkpoints preserve exact continuation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ppo_resume_audit'

def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def launch(kind,method,extra=()):
    cmd=['bash',str(ROOT/'run_env.sh'),str(ROOT/'runner.py'),'--kind',kind,'--method',method,'--seed','997','--smoke','--allow-cpu',*extra]
    with (OUT/'console.log').open('a') as log:
        return subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode

def main():
    if OUT.exists():raise RuntimeError('Use fresh audit output')
    OUT.mkdir(parents=True)
    assert launch('warmup','erm')==0
    results={}
    for method in ['bestof_ppo','sequential_ppo']:
        assert launch('fixed',method)==0
        assert launch('coev',method)==0
        folder=ROOT/'artifacts/smoke'/f'coev_{method}_s997'
        continuous=OUT/method;shutil.move(folder,continuous)
        assert launch('coev',method,['--stop-after','1'])==75
        assert launch('coev',method)==0
        a=torch.load(continuous/'latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        fields=['actor','value','actor_optimizer','value_optimizer','attacker','rng','calls','step','lambda','history']
        checks={k:same(a[k],b[k]) for k in fields}
        with np.load(continuous/'results.npz') as x,np.load(folder/'results.npz') as y:
            checks['evaluation_arrays']=x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
        results[method]=checks
    passed=all(all(v.values()) for v in results.values())
    (OUT/'audit.json').write_text(json.dumps(dict(status='PASS' if passed else 'FAIL',checks=results,scope='CPU engineering, not GPU scientific results'),indent=2))
    print(json.dumps(results));assert passed

if __name__=='__main__':main()
