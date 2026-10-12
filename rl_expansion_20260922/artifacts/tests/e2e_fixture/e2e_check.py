"""Small separate-data full lifecycle fixture, never formal experimental evidence."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
import geometry as G
from common import ROOT,read,now
import engine as E
import pipeline as P

def run_fixture(gpu):
    if not (ROOT/'DIAGNOSTIC_FIXTURE').exists(): raise RuntimeError('Fixture marker missing')
    def stage(name,ts):
        for t in ts:
            log=ROOT/f"artifacts/operations/{t['id']}.log"
            cmd=['bash','run_env.sh','worker.py','--task-json',f"artifacts/tasks/{t['id']}.json",'--physical-gpu-index',str(gpu),'--resume']
            with log.open('w') as f: subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    stage('screen',P.screen()); P.promote(); stage('parents',P.parents()); stage('final',P.final())
    stage('evaluation',P.freeze_and_evaluations())
    subprocess.run(['bash','run_env.sh','report.py'],cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int); ap.add_argument('--run-fixture',action='store_true'); a=ap.parse_args()
    if a.run_fixture: run_fixture(a.gpu); return
    from diagnostics import admitted
    gpu=admitted()[0] if a.gpu is None else a.gpu
    dest=ROOT/'artifacts/tests/e2e_fixture'
    if dest.exists(): raise RuntimeError('Fixture already exists; preserve evidence, choose a new fixture path')
    dest.mkdir(); (dest/'DIAGNOSTIC_FIXTURE').write_text('Tiny synthetic lifecycle only. No scientific acceptance.\n')
    for path in E.sources():
        target=dest/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    for path in ('artifacts/data','artifacts/inputs','artifacts/operations','artifacts/tests'): (dest/path).mkdir(parents=True,exist_ok=True)
    p=read(ROOT/'protocol.json'); p.update(screening_seeds=[101,102],new_seeds=[104],architecture_seeds=[201,202],
        parent_step=2,screen_target=4,final_target=6,architecture_final_target=6,validation_every=2,checkpoint_every=1,
        bootstrap_replicates=20,secondary_degrees=[20,40],cosine_start=2,cosine_end=6)
    p['arms']={k:p['arms'][k] for k in ('E50','GDF')}
    G.atomic_json(dest/'protocol.json',p); line={}
    for seed in p['screening_seeds']:
        ck=torch.load(ROOT/f'artifacts/inputs/ME_s{seed}.pt',map_location='cpu',weights_only=False)
        ck['step']=2; ck['fixture_only_parent_step_alias']=True
        path=dest/f'artifacts/inputs/ME_s{seed}.pt'; G.atomic_torch(path,ck)
        line[str(seed)]=dict(path=str(path.relative_to(dest)),sha256=G.sha256(path))
    G.atomic_json(dest/'artifacts/inputs/lineage.json',line)
    hashes={}
    for split,degree in [('calibration',30),('validation',30),('test',30),('test_degree20',20),('test_degree40',40)]:
        arrays={g:G.make_data(16,degree,g,'expansion-lifecycle-fixture/'+split,0,True) for g in G.PARAMETERS}
        path=dest/f'artifacts/data/{split}.npz'; G.atomic_npz(path,**arrays); hashes[path.name]=G.sha256(path)
    G.atomic_json(dest/'artifacts/data/manifest.json',dict(artifacts=hashes,fixture=True))
    with (ROOT/'artifacts/tests/e2e.log').open('w') as log:
        result=subprocess.run(['bash','run_env.sh','e2e_check.py','--run-fixture','--gpu',str(gpu)],cwd=dest,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('End-to-end fixture failed; see artifacts/tests/e2e.log')
    report=read(dest/'artifacts/report/complete.json')
    if report['status']!='complete': raise RuntimeError('Fixture closeout failed')
    G.atomic_json(ROOT/'artifacts/tests/e2e.json',dict(status='PASS',time=now(),gpu=gpu,
        source_sha256={str(x.relative_to(ROOT)):G.sha256(x) for x in E.sources()},
        fixture_report_sha256=G.sha256(dest/'artifacts/report/complete.json'),scope='engineering tiny-data fixture only'))
    print('PASS: isolated end-to-end lifecycle',flush=True)

if __name__=='__main__': main()
