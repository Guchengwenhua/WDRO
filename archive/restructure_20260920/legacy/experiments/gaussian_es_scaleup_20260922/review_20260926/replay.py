"""Inference-only replay of all selected validation scores and sampled test outputs."""
from pathlib import Path
import os
import json
import sys
import time
import traceback
import numpy as np
ROOT=Path('/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922')
ART=ROOT/'artifacts';OUT=ART/'review_20260926'
sys.path.insert(0,str(ROOT))

def write(name,v):
    p=OUT/name;q=p.with_name(p.name+'.partial');q.write_text(json.dumps(v,indent=2,ensure_ascii=False));q.replace(p)

def main():
    from resources import snapshot,admissible
    samples=[]
    for i in range(3):
        samples.append(snapshot())
        if i<2:time.sleep(1)
    available=admissible(samples)
    if not available:raise RuntimeError('No GPU meets live admission for inference replay')
    gpu=available[0];write('replay_preflight.json',dict(samples=samples,selected=gpu))
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    import torch
    from common import build_actor,data,_configure,RestTreeEvaluator,pin_set_sha256,sha256
    from src.geosteiner import GeoSteinerEvaluator
    import src.geosteiner as gs
    _configure(20260926);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.10)
    rest=RestTreeEvaluator();solver=GeoSteinerEvaluator();refs={};references_27=0;references_28=0;predictions=0
    status=json.loads((ART/'status.json').read_text());tasks=[]
    for row in status['tasks']:
        r=json.loads((ART/'runs'/row['id']/'complete.json').read_text())
        if r['task']['kind']=='train':tasks.append(r)
    def exact(pins,scale=2**27):
        nonlocal references_27,references_28
        output=[];old=gs.SCALE
        try:
            gs.SCALE=scale
            for p in pins:
                key=(pin_set_sha256(p),scale)
                if key not in refs:
                    refs[key]=solver.solve(p,require_integral=True)
                    if scale==2**27:references_27+=1
                    else:references_28+=1
                output.append(refs[key])
        finally:gs.SCALE=old
        return np.asarray(output)
    def score(actor,pins):
        nonlocal predictions
        lengths=[]
        actor.eval()
        with torch.no_grad():
            for off in range(0,len(pins),256):
                x=pins[off:off+256];n=len(x)
                if n<256:x=np.concatenate([x,np.repeat(x[-1:],256-n,axis=0)])
                topology,_=actor(x,True)
                lengths.append(rest.lengths(x[:n],topology[:n].cpu().numpy()));predictions+=n
        length=np.concatenate(lengths);reference=exact(pins)
        return length,reference,length/reference-1
    checks=[];validation_max_error=0.0;test_max_error=0.0;test_selected=0;started=time.time()
    D=['uniform','mix_gauss','edge_bias','grid_cluster']
    for i,r in enumerate(tasks):
        t=r['task'];d=t['degree'];out=ART/'runs'/t['id']
        assert sha256(out/'best.pt')==r['best']['sha256']
        cp=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        actor=build_actor(d,device);actor.load_state_dict(cp['actor_state_dict']);actor.eval()
        split='pilot-validation' if t['phase']=='pilot' else 'model-validation'
        actual={}
        for dist in D:
            pins=data(d,256,dist,split);length,reference,regret=score(actor,pins)
            actual[dist]=dict(mean=float(regret.mean()),p95=float(np.quantile(regret,.95)),n=256)
            for metric in ('mean','p95'):
                err=abs(actual[dist][metric]-r['best_validation'][dist][metric]);validation_max_error=max(validation_max_error,err)
                assert err<1e-12,(t['id'],dist,metric,err)
        if t['phase']=='post':
            evaldir=ART/'runs'/('eval_'+t['id']);points=[];targets_l=[];targets_ref=[]
            for dist in D:
                for kind in ('nominal','pressure'):
                    files=sorted(evaldir.glob(f'{dist}_{kind}_*.npz'))
                    for path in (files[0],files[-1]):
                        with np.load(path) as f:
                            ix=[0,len(f['pins'])-1]
                            points.extend(f['pins'][ix]);targets_l.extend(f['length'][ix]);targets_ref.extend(f['lstar'][ix])
            pins=np.asarray(points);length,reference,regret=score(actor,pins)
            np.testing.assert_allclose(length,targets_l,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(reference,np.asarray(targets_ref))
            double=exact(pins,2**28);np.testing.assert_array_equal(reference,double)
            test_max_error=max(test_max_error,float(np.max(np.abs(length-targets_l))));test_selected+=len(pins)
        checks.append(dict(task=t['id'],best_step=cp['step'],validation=actual,model_sha256=r['best']['sha256']))
        write('replay_progress.json',dict(at=time.time(),done=i+1,total=len(tasks),gpu=gpu['index'],elapsed=time.time()-started))
        print(f'replayed {i+1}/{len(tasks)} {t["id"]}',flush=True)
        del actor,cp;torch.cuda.empty_cache()
    write('replay.json',dict(status='PASS',gpu=gpu,models=len(checks),validation_predictions=49*1024,
        sampled_test_predictions=test_selected,predictions=predictions,reference_solves_2p27=references_27,reference_solves_2p28=references_28,
        max_validation_metric_error=validation_max_error,max_test_length_error=test_max_error,elapsed_seconds=time.time()-started,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,checks=checks))

if __name__=='__main__':
    try:main()
    except BaseException:
        write('replay_failure.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
