"""Exact restart and measured resource pilots; only local diagnostic tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time
import numpy as np
import torch
import geometry as G
import pipeline as P
from common import ROOT,read,now
from scheduler import resource_snapshot

def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def run(gpu,config,name,steps,resume=False,scratch=False):
    out=ROOT/'artifacts/tests'/name; out.mkdir(parents=True,exist_ok=True)
    p=read(ROOT/'protocol.json'); line=read(ROOT/'artifacts/inputs/lineage.json')['101']
    start=0 if scratch else p['parent_step']; par=None if scratch else {k:line[k] for k in ('path','sha256')}
    t=P.task('diagnostic',name,101,config,start,start+1024,par); t['outdir']=str((out/'run').relative_to(ROOT))
    path=out/'task.json'; P.publish(path.relative_to(ROOT),t)
    cmd=['bash','run_env.sh','worker.py','--task-json',str(path),'--physical-gpu-index',str(gpu),
         '--cpu-threads','3','--diagnostic','--stop-after',str(start+steps)]
    if gpu<0: cmd.append('--allow-cpu')
    if resume: cmd.append('--resume')
    with (out/('resume.log' if resume else 'run.log')).open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{name} diagnostic failed: {result.returncode}; see {out}')
    return torch.load(out/'run/latest.pt',map_location='cpu',weights_only=False),read(out/'run/heartbeat.json')

def pair(gpu,prefix):
    p=read(ROOT/'protocol.json'); c=dict(p['arms']['GDNF'],advantage_ema=True)
    a,_=run(gpu,c,prefix+'_continuous',3)
    run(gpu,c,prefix+'_restart',1)
    b,_=run(gpu,c,prefix+'_restart',3,True)
    fields=('actor','optimizer','logr','anchors','adv_variance','rng','step','data_chain','ledger')
    checks={k:equal(a[k],b[k]) for k in fields}
    if not all(checks.values()): raise RuntimeError('Exact resume failed: '+str(checks))
    return dict(status='PASS',gpu=gpu,checks=checks)

def admitted():
    snaps=[]
    for _ in range(180):
        snap=resource_snapshot(); snaps.append(snap); snaps=snaps[-3:]
        eligible=[]
        if len(snaps)==3 and snap['disk_free_bytes']>20*2**30:
            for g in snap['gpus']:
                hist=[next(x for x in s['gpus'] if x['index']==g['index']) for s in snaps]
                if all(x['free_mib']>=12288 and x['utilization']<=60 for x in hist): eligible.append(g['index'])
        if eligible:
            G.atomic_json(ROOT/'artifacts/tests/pilot_admission.json',dict(gpus=eligible,snapshots=snaps,time=now()))
            return eligible
        time.sleep(5)
    raise RuntimeError('No admitted GPU within 15 minutes')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['cpu','gpu'],required=True); ap.add_argument('--prefix',default='final'); a=ap.parse_args()
    p=read(ROOT/'protocol.json')
    if a.phase=='cpu':
        row=pair(-1,a.prefix+'_cpu'); G.atomic_json(ROOT/f'artifacts/tests/{a.prefix}_cpu.json',row); return
    gpus=admitted(); row=pair(gpus[0],a.prefix+'_gpu')
    configs=[('E50',p['arms']['E50'],False),('B256',p['arms']['B256'],False),('WIDE',p['architectures']['WIDE'],True),('DEEP',p['architectures']['DEEP'],True)]
    def pilot(item):
        i,(name,c,scratch)=item; ck,h=run(gpus[i%len(gpus)],c,a.prefix+'_pilot_'+name,32,scratch=scratch)
        if ck['ledger']['test_rows'] or ck['ledger']['oracle_calls']: raise RuntimeError('Pilot isolation')
        return dict(name=name,config=c,new_updates=ck['update_count'],seconds_per_update=ck['update_seconds']/ck['update_count'],
                    peak_reserved_bytes=h['peak_reserved_bytes'],ledger=ck['ledger'])
    with ThreadPoolExecutor(max_workers=min(4,len(gpus))) as pool: profiles=list(pool.map(pilot,enumerate(configs)))
    # Exercise every other optimizer/objective path on CUDA once before freezing code.
    for i,(name,c) in enumerate(p['arms'].items()):
        if name not in ('E50','B256'): run(gpus[0],c,a.prefix+'_smoke_'+name,2)
    G.atomic_json(ROOT/f'artifacts/tests/{a.prefix}_gpu.json',dict(status='PASS',resume=row,profiles=profiles,time=now()))
    print(json.dumps(profiles,indent=2),flush=True)

if __name__=='__main__': main()
