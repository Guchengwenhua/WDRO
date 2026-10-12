"""Exercise all six branches, sealed-test gate, and final report in an isolated tiny fixture."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import geometry as G
from common import ROOT,read,now
from scheduler import resource_snapshot

def main():
    dest=ROOT/'artifacts/tests/e2e_fixture'
    if dest.exists(): raise RuntimeError('Use a fresh fixture directory')
    dest.mkdir(parents=True)
    for path in ROOT.glob('*.py'): shutil.copyfile(path,dest/path.name)
    shutil.copyfile(ROOT/'run_env.sh',dest/'run_env.sh'); shutil.copytree(ROOT/'vendor',dest/'vendor')
    for folder in ('artifacts/inputs','artifacts/data','artifacts/operations','artifacts/tests'):
        (dest/folder).mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'artifacts/inputs',dest/'artifacts/inputs',dirs_exist_ok=True)
    p=read(ROOT/'protocol.json')
    p.update(targets=dict(U=2050,M=2,D=4,ME=4,MG=4,SG=4),start_steps=dict(U=2048,M=0,D=0,ME=2,MG=2,SG=1),
        endpoints=[1,2,3,4,2048,2050],validation_start=1,validation_every=1,checkpoint_every=1,
        val_per_group=64,test_per_group=64,bootstrap_replicates=50,train_new_updates_total=45,
        supplemental_comparison=dict(name='posttraining_scale_exploratory',baseline='SG/step3',candidate='SG/step4',metric='worst'))
    replacements={'U/step2048':'U/step2048','U/step8192':'U/step2050','M/step8192':'M/step2',
                  'MG/step12288':'MG/step4','ME/step12288':'ME/step4','D/step12288':'D/step4','SG/step12288':'SG/step4'}
    for c in p['primary_comparisons']:
        c['baseline']=replacements[c['baseline']]; c['candidate']=replacements[c['candidate']]
    G.atomic_json(dest/'protocol.json',p)
    import numpy as np
    hashes={}
    for split in ('validation','test'):
        with np.load(ROOT/'artifacts/data'/f'{split}.npz') as z:
            G.atomic_npz(dest/'artifacts/data'/f'{split}.npz',**{g:z[g][:64] for g in z.files})
        hashes[f'{split}.npz']=G.sha256(dest/'artifacts/data'/f'{split}.npz')
    G.atomic_json(dest/'artifacts/data/manifest.json',dict(artifacts=hashes,fixture_only=True))
    histories=[]
    for _ in range(180):
        snap=resource_snapshot(); histories.append(snap); histories=histories[-3:]
        good=[]
        if len(histories)==3:
            for g in snap['gpus']:
                if all(next(v for v in s['gpus'] if v['index']==g['index'])['utilization']<=35 for s in histories) and g['free_mib']>=8192:
                    good.append(g)
        if good: break
        time.sleep(5)
    else: raise RuntimeError('No GPU for e2e fixture')
    gpu=min(good,key=lambda x:x['utilization'])['index']
    G.atomic_json(ROOT/'artifacts/tests/e2e_admission.json',dict(gpu=gpu,samples=histories))
    def run(args,log):
        with (dest/'artifacts/tests'/log).open('w') as f:
            r=subprocess.run(['bash','run_env.sh',*args],cwd=dest,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode: raise RuntimeError(f'Fixture failed: {log}')
    for mode in ('train','eval'):
        for arm in p['arms']:
            for seed in p['seeds']:
                key=f'{arm}_s{seed}'; run(['worker.py','--mode',mode,'--arm',arm,'--seed',str(seed),
                    '--outdir',f'artifacts/{"runs" if mode=="train" else "eval"}/{key}','--physical-gpu-index',str(gpu)],f'{mode}_{key}.log')
                print(mode,key,'PASS',flush=True)
    run(['report.py'],'report.log')
    report=read(dest/'artifacts/report/complete.json')
    if report['status']!='complete': raise RuntimeError('Fixture report incomplete')
    G.atomic_json(ROOT/'artifacts/tests/e2e_result.json',dict(status='PASS',fixture_only=True,time=now(),gpu=gpu,
        report_sha256=G.sha256(dest/'artifacts/report/complete.json'),train_tasks=18,eval_tasks=18,
        source_sha256={n:G.sha256(ROOT/n) for n in ('worker.py','common.py','geometry.py','report.py')}))

if __name__=='__main__': main()
