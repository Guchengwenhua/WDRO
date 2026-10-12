"""Real-GPU gate and current-load throughput measurement; no scientific claims."""
import json
import os
import subprocess
import threading
import time
import traceback
import numpy as np
from pathlib import Path
from resources import preflight
from common import (ROOT,ART,atomic_json,artifact,build_initial,ExactOracle,data,
    derived_seed,dual_settings,_configure,PROTOCOL,load_parent)

def equal_tree(a,b):
    import torch
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b,(a,b)

def main():
    record=preflight()
    if not record['admitted']:raise RuntimeError('no current GPU passes admission')
    gpu=record['admitted'][0];os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    import torch
    from worker import train,screen,evaluation,Pause
    from common import sampled_update
    from search import attack
    from src.geosteiner import GeoSteinerEvaluator
    import src.geosteiner as gs
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.1)
    measurements=[];state={};pause=threading.Event()
    # Small deterministic real-objective resume; both branches use identical seed/data.
    base=dict(kind='train',phase='post',degree=30,seed=919,method='es',updates=6,batch=8,lr=1e-5,parent=None,
        mu=.02,search_step=.01,nominal_fraction=.25,validation_count=4,validation_every=3,checkpoint_every=3)
    for name,stop_after in [('full',None),('resume',3),('resume',None)]:
        task=dict(base,id=f'engineering_resume_{name}');out=ART/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        try:train(task,out,device,state,pause,stop_after=stop_after,workers=2)
        except Pause:pass
    full=torch.load(ART/'runs/engineering_resume_full/latest.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(ART/'runs/engineering_resume_resume/latest.pt',map_location='cpu',weights_only=False)
    for key in ['actor_state_dict','critic_state_dict','optimizer_state_dict','price','rng','step']:
        equal_tree(full[key],resumed[key])
    atomic_json(ART/'tests/resume.json',dict(status='PASS',steps=6,interrupted_after=3,compared=['Actor','Critic','Adam','lambda','RNG'],artifacts=[artifact(ART/'runs/engineering_resume_full/latest.pt'),artifact(ART/'runs/engineering_resume_resume/latest.pt')]))
    # Production shapes and degrees: true scores, sampled update, memory and timings.
    for degree in (20,30,40):
        actor,critic,_=build_initial(degree,911,device)
        optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=3e-5)
        oracle=ExactOracle(actor,device,ART/'runs'/f'engineering_profile_d{degree}'/'oracle',2)
        price,rho=dual_settings(degree)
        try:
            for method,batch,budget in [('erm',128,0),('random',64,64),('es',64,64)]+([('es',64,256)] if degree==30 else []):
                x=data(degree,batch,'uniform',f'engineering-profile-{method}{budget}')
                tick=time.monotonic();before=dict(oracle.ledger)
                if method=='erm':
                    reference,_=oracle.cache.lengths(x,oracle.geosteiner);pins=x
                else:
                    r=attack(method,x,oracle,price,991,mu=.02,step=.01,budget=budget)
                    pins,reference=r['pins'],r['lstar']
                    assert oracle.ledger['actor_completed']-before.get('actor_completed',0)==batch*budget
                    np.testing.assert_allclose(r['regret'],r['length']/r['lstar']-1,rtol=1e-12,atol=1e-12)
                sampled_update(actor,critic,optimizer,oracle,pins,reference)
                elapsed=time.monotonic()-tick
                measurements.append(dict(degree=degree,method=method,budget=budget,batch=batch,seconds=elapsed,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    ledger_delta={k:oracle.ledger.get(k,0)-before.get(k,0) for k in oracle.ledger}))
                atomic_json(ART/'tests/profile_progress.json',dict(measurements=measurements,gpu=gpu))
                print(json.dumps(measurements[-1]),flush=True)
            # Fixed greedy microbatch makes the nominal result independent of request batch size.
            x=data(degree,8,'uniform','engineering-greedy')
            a=oracle.evaluate(x[:2]);b=oracle.evaluate(x)
            np.testing.assert_array_equal(a['length'],b['length'][:2])
            solver=GeoSteinerEvaluator()
            standard=solver.lengths(x[:2]);old=gs.SCALE
            try:
                gs.SCALE=2**28;double=solver.lengths(x[:2])
            finally:gs.SCALE=old
            np.testing.assert_array_equal(standard,double)
        finally:oracle.close()
        del actor,critic,optimizer;torch.cuda.empty_cache()
    # Exercise separate fixed-search and frozen-model evaluation entrypoints with real artifacts.
    screen_task=dict(id='engineering_screen',kind='screen',method='es',degree=30,seed=101,count=2,budget=64)
    out=ART/'runs'/screen_task['id'];out.mkdir(parents=True,exist_ok=True)
    _configure(101);screen(screen_task,out,device,state,pause,workers=2)
    eval_task=dict(id='engineering_eval',kind='eval',degree=30,seed=919,method='es',test_count=8,pressure_count=2,data_prefix='engineering-',
        model=artifact(ART/'runs/engineering_resume_full/best.pt'))
    out=ART/'runs'/eval_task['id'];out.mkdir(parents=True,exist_ok=True)
    evaluation(eval_task,out,device,state,pause,workers=2)
    gate=dict(status='PASS',at=time.time(),gpu=gpu,measurements=measurements,
        peak_reserved_mib=max(x['peak_reserved_mib'] for x in measurements),resume=artifact(ART/'tests/resume.json'),
        checks=['exact_resume','production_shapes_d20_d30_d40','64_256_exact_calls','real_sampled_updates','greedy_batch_consistency','geosteiner_2^27_2^28','screen_entrypoint','evaluation_entrypoint'])
    atomic_json(ART/'tests/gpu_gate.json',gate)
    print(json.dumps(gate),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(ART/'tests/gpu_failure.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
