"""Subprocess checkpoint equivalence audit, separate from scientific GPU results."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/resume_audit';OUT.mkdir(parents=True,exist_ok=True)
def launch(extra):
    cmd=['bash',str(ROOT/'run_env.sh'),str(ROOT/'runner.py'),'--kind','warmup','--seed','996','--smoke','--allow-cpu',*extra]
    r=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    with (OUT/'console.log').open('a') as f:f.write(r.stdout+r.stderr)
    return r.returncode
def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
path=ROOT/'artifacts/smoke/warmup_s996'
if path.exists() or (OUT/'continuous').exists():raise RuntimeError('Use only a fresh audit output')
assert launch([])==0
shutil.move(path,OUT/'continuous')
assert launch(['--stop-after','1'])==75
assert launch([])==0
a=torch.load(OUT/'continuous/latest.pt',weights_only=False,map_location='cpu')
b=torch.load(path/'latest.pt',weights_only=False,map_location='cpu')
fields=['actor','value','actor_optimizer','value_optimizer','rng','calls','step','lambda','history']
checks={name:same(a[name],b[name]) for name in fields}
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'scope':'CPU engineering recovery, not scientific acceptance'}
(OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert all(checks.values())
