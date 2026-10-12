"""Engineering checks only; isolated data/runs, no scientific model selection."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import engine as e
import geometry as geo

def compare(a,b,path=''):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(f'tensor mismatch {path}')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(f'keys mismatch {path}')
        for k in a:compare(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError(path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif a!=b:raise AssertionError(f'{path}: {a} != {b}')

def run_task(t,device,base,stop=None):
    r=e.Run(t,device,base)
    try:
        ok=r.train(stop)
        result=dict(task=t['id'],finished=ok,step=r.step,seconds=r.elapsed(),calls=r.calls,peak_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0)
    finally:r.close()
    return result

def gpu():
    base=e.ART/'tests/gpu';base.mkdir(parents=True,exist_ok=True);rows=[]
    for degree in (30,20,40):
        t=dict(id=f'pre_d{degree}',stage='pre',degree=degree,seed=17,method='erm',updates=8,val_n=16,batch=128,val_every=8)
        rows.append(run_task(t,'cuda:0',base))
        methods=e.P['methods'] if degree==30 else ['soft_budget','excess']
        for method in methods:
            torch.cuda.reset_peak_memory_stats()
            task=dict(id=f'post_d{degree}_{method}',stage='post',degree=degree,seed=17,method=method,updates=8,val_n=16,batch=128,k=16,val_every=8,parent=t['id'],tau=.02)
            rows.append(run_task(task,'cuda:0',base))
    for row in rows:
        path=base/row['task']/'progress.jsonl';steps=[json.loads(l) for l in path.read_text().splitlines()]
        row['median_step_seconds']=float(np.median([s['seconds'] for s in steps[2:]]))
        task=json.loads((base/row['task']/'complete.json').read_text())['task']
        assert row['calls']['sampled']==8*128
        assert row['calls']['search']==(0 if task['method']=='erm' else 8*128*16)
        assert row['calls']['teacher']==(8*128*16 if task['method']=='excess' else 0)
    geo.atomic_json(e.ART/'tests/gpu_calibration.json',dict(status='PASS',rows=rows,time=time.time(),binding=e.binding()))
    # Complete real GPU evaluator with reduced independent data, isolated freeze.
    from evaluate import Evaluation
    original=e.ART;old={k:e.P[k] for k in ('test_n','attack_n','attack_k')}
    e.ART=original/'tests/evaluation_e2e';(e.ART/'operations').mkdir(parents=True,exist_ok=True)
    (e.ART/'runs').symlink_to(base.resolve(),target_is_directory=True)
    e.P.update(test_n=8,attack_n=4,attack_k=8)
    models={str(e.ART/'runs'/'post_d30_tail'/name):geo.sha256(base/'post_d30_tail'/name) for name in ('best.pt','latest.pt')}
    geo.atomic_json(e.ART/'operations/model_freeze.json',dict(models=models,time=time.time()))
    for name in ('best.pt','latest.pt'):
        task=dict(id='test_'+name,run_id='post_d30_tail',checkpoint=name,degree=30,seed=17,method='tail')
        v=Evaluation(task,'cuda:0');v.run();v.lock.close()
        assert v.calls['test']==4*(8+2*4*8)
    e.ART=original;e.P.update(old)
    geo.atomic_json(e.ART/'tests/evaluation_e2e.json',dict(status='PASS',time=time.time(),binding=e.binding()))

def resume():
    base=e.ART/'tests/resume';base.mkdir(parents=True,exist_ok=True)
    parent=dict(id='parent',stage='pre',degree=5,seed=37,method='erm',updates=2,val_n=4,batch=4,val_every=2)
    a=base/'continuous';b=base/'resumed';run_task(parent,'cpu',a);b.mkdir(exist_ok=True);shutil.copytree(a/'parent',b/'parent',dirs_exist_ok=True)
    rows=[]
    for method in e.P['methods']:
        t=dict(id=method,stage='post',degree=5,seed=37,method=method,updates=4,val_n=4,batch=4,k=4,val_every=2,parent='parent',tau=.02)
        run_task(t,'cpu',a);run_task(t,'cpu',b,2);run_task(t,'cpu',b)
        x=torch.load(a/method/'latest.pt',weights_only=False);y=torch.load(b/method/'latest.pt',weights_only=False)
        for key in ['actor','value','actor_optimizer','value_optimizer','step','lam','eta','calls','best_metric','parent_nominal','latest_val','best_step','rng']:compare(x[key],y[key],key)
        rows.append(dict(method=method,status='PASS'))
    geo.atomic_json(e.ART/'tests/resume.json',dict(status='PASS',rows=rows,time=time.time(),binding=e.binding()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['gpu','resume']);a=p.parse_args()
    e.configure('cuda:0' if a.mode=='gpu' else 'cpu')
    (gpu if a.mode=='gpu' else resume)()
