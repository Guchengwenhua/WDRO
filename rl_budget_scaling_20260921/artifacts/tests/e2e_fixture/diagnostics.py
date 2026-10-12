"""Exact restart checks and measured batch64 GPU pilot, separate from formal runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import geometry as G
from common import ROOT,read,now
from scheduler import resource_snapshot

OUT=ROOT/'artifacts/tests'

def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def run(device,arm,name,end,resume=False):
    out=OUT/name
    cmd=['bash','run_env.sh','worker.py','--mode','train','--arm',arm,'--seed','101','--diagnostic',
         '--outdir',str(out),'--stop-after',str(end),'--physical-gpu-index',str(device),'--cpu-threads','3']
    if device<0: cmd.append('--allow-cpu')
    if resume: cmd.append('--resume')
    with (OUT/(name+('_resume' if resume else '')+'.log')).open('w') as log:
        r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'Diagnostic {name} failed: exit {r.returncode}')
    return torch.load(out/'latest.pt',map_location='cpu',weights_only=False)

def pair(device,arm,prefix):
    start=2048 if arm=='U' else 0
    a=run(device,arm,prefix+'_continuous',start+2)
    run(device,arm,prefix+'_restart',start+1)
    b=run(device,arm,prefix+'_restart',start+2,True)
    fields=('actor','optimizer','logq','rng','data_chain','step','initial_actor_sha256')
    checks={k:equal(a[k],b[k]) for k in fields}
    if not all(checks.values()): raise RuntimeError('Resume mismatch '+str(checks))
    row=dict(device=device,arm=arm,checks=checks,status='PASS')
    G.atomic_json(OUT/(prefix+'_comparison.json'),row); print(json.dumps(row),flush=True)
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['cpu','gpu'],required=True); a=p.parse_args()
    if a.phase=='cpu':
        rows=[pair(-1,arm,'cpu_'+arm) for arm in ('U','D')]
        G.atomic_json(OUT/'cpu_resume.json',dict(status='PASS',rows=rows,time=now())); return
    samples=[]
    for attempt in range(360):
        snap=resource_snapshot(); samples.append(snap); samples=samples[-3:]
        G.atomic_json(OUT/'pilot_resource_latest.json',snap)
        eligible=[]
        if len(samples)==3 and snap['disk_free_bytes']>=20*2**30:
            for gpu in snap['gpus']:
                hist=[next(x for x in s['gpus'] if x['index']==gpu['index']) for s in samples]
                if all(x['free_mib']>=8192 and x['utilization']<=35 for x in hist): eligible.append(gpu)
        if eligible: break
        time.sleep(5)
    else: raise RuntimeError('No admitted GPU for pilot within 30 minutes')
    gpu=sorted(eligible,key=lambda g:(g['utilization'],-g['free_mib']))[0]['index']
    G.atomic_json(OUT/'pilot_admission.json',dict(gpu=gpu,samples=samples,time=now()))
    rows=[pair(gpu,arm,'gpu_'+arm) for arm in ('U','D')]
    pilots=[]
    for arm,start in [('U',2048),('M',0)]:
        c=run(gpu,arm,'pilot_'+arm,start+32)
        heart=read(OUT/f'pilot_{arm}/heartbeat.json')
        pilots.append(dict(arm=arm,new_updates=c['update_count'],seconds_per_update=c['update_seconds']/c['update_count'],
                           peak_reserved_bytes=heart['peak_reserved_bytes'],ledger=c['ledger']))
    G.atomic_json(OUT/'gpu_resume_pilot.json',dict(status='PASS',rows=rows,pilots=pilots,gpu=gpu,time=now()))
    print(json.dumps(pilots),flush=True)

if __name__=='__main__': main()
