"""Server-side complete experiment lifecycle; restart with launch.sh after a host outage."""
import fcntl
import os
import subprocess
import sys
import traceback
import geometry as G
from common import ROOT,read,now,identity
import pipeline as P

def run_stage(stage):
    path=ROOT/f'artifacts/operations/{stage}/exit.json'
    if path.exists() and read(path)['status']=='complete': return
    G.atomic_json(ROOT/'artifacts/lifecycle.json',dict(status='running',stage=stage,time=now(),identity=identity()))
    env=dict(os.environ,RL_SWEEP_STAGE=stage)
    cmd=['bash','run_env.sh','scheduler.py','--manifest',f'artifacts/queues/{stage}.json',
         '--poll-seconds','10','--admission-samples','3','--max-gpu-util','60',
         '--min-free-mib','8192','--memory-margin-mib','2048','--min-disk-gib','20',
         '--max-workers','8','--max-retries','2']
    result=subprocess.run(cmd,cwd=ROOT,env=env)
    if result.returncode or not path.exists() or read(path)['status']!='complete': raise RuntimeError('Stage failed: '+stage)

def main():
    lock=(ROOT/'artifacts/operations/pipeline.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0); lock.truncate(); import json; json.dump(identity(),lock); lock.flush()
    try:
        P.screen(); run_stage('screen'); P.promote()
        P.parents(); run_stage('parents'); P.final(); run_stage('final')
        P.freeze_and_evaluations(); run_stage('evaluation')
        G.atomic_json(ROOT/'artifacts/lifecycle.json',dict(status='reporting',time=now(),identity=identity()))
        subprocess.run(['bash','run_env.sh','report.py'],cwd=ROOT,check=True)
        G.atomic_json(ROOT/'artifacts/lifecycle.json',dict(status='complete',time=now(),identity=identity(),
            report_complete_sha256=G.sha256(ROOT/'artifacts/report/complete.json')))
    except BaseException:
        G.atomic_json(ROOT/'artifacts/lifecycle.json',dict(status='stopped',time=now(),identity=identity(),error=traceback.format_exc()))
        raise
    finally: fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__': main()
