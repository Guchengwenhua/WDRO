"""Production-batch calibration, isolated from all formal runs and test data."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import scheduler as s

ROOT=Path(__file__).resolve().parent
PILOT=ROOT/'artifacts/pilot'

def main():
    if PILOT.exists():raise RuntimeError('Pilot output exists; inspect it instead of overwriting')
    PILOT.mkdir(parents=True)
    p=json.loads((ROOT/'protocol.json').read_text())
    p.update(experiment='engineering_calibration_only',seeds=[9001],warmup_updates=16,fixed_updates=16,coev_updates=16,
             fixed_test_per_distribution=64,test_per_distribution=64,attack_per_distribution=64,checkpoint_every=4,
             data_namespace='rl-scaleup-20260921-pilot-only')
    (PILOT/'protocol.json').write_text(json.dumps(p,indent=2))
    for name in ['runner.py','policies.py','env_geometry.py']:
        shutil.copy2(ROOT/name,PILOT/name)
    (PILOT/'run_env.sh').write_text((ROOT/'run_env.sh').read_text().replace(str(ROOT),str(PILOT)))
    tasks=s.tasks(p);done=set();running={};history=[]
    while len(done)<len(tasks):
        for key in list(running):
            item=running[key];code=item['process'].poll()
            if code is None:continue
            item['stream'].close()
            if code:raise RuntimeError(f'Pilot failed: {key}, exit {code}')
            done.add(key);del running[key]
        snap=s.snapshot();history=(history+[snap])[-3:]
        resources=dict(p['resources'],max_workers=4)
        for task in s.ready_tasks(tasks,done,running,{}):
            fresh=s.snapshot();cap,cards,reasons=s.admission(fresh,history,running,resources)
            if cap<=0 or not cards:break
            gpu=cards[0];out=PILOT/'artifacts/runs'/task['id'];out.mkdir(parents=True)
            cmd=['nice','-n','10','bash',str(PILOT/'run_env.sh'),str(PILOT/'runner.py'),'--kind',task['kind'],
                 '--method',task['method'],'--seed',str(task['seed']),'--physical-gpu-index',str(gpu['index'])]
            s.atomic(out/'preflight.json',dict(snapshot=fresh,gpu=gpu,command=cmd,role='engineering_pilot_only'))
            log=(out/'console.log').open('w')
            proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index'])),stdout=log,stderr=subprocess.STDOUT,cwd=PILOT)
            running[task['id']]=dict(process=proc,stream=log,gpu=gpu['index'],pid=proc.pid)
        s.atomic(PILOT/'state.json',dict(complete=sorted(done),total=len(tasks),running={k:{'pid':v['pid'],'gpu':v['gpu']} for k,v in running.items()},time=time.time()))
        time.sleep(3)
    rates={};formal=json.loads((ROOT/'protocol.json').read_text())
    for task in tasks:
        out=PILOT/'artifacts/runs'/task['id'];m=json.loads((out/'complete.json').read_text())
        rows=[json.loads(l) for l in (out/'progress.jsonl').read_text().splitlines()]
        elapsed=rows[-1]['elapsed_seconds']-rows[3]['elapsed_seconds']
        sec=elapsed/(rows[-1]['step']-rows[3]['step'])
        evaluation=m['elapsed_seconds']-rows[-1]['elapsed_seconds']
        expected_eval=0
        if task['kind']=='fixed':expected_eval=formal['fixed_test_per_distribution']/p['fixed_test_per_distribution']
        if task['kind']=='coev':expected_eval=(formal['test_per_distribution']+formal['attack_per_distribution']*64)/(p['test_per_distribution']+p['attack_per_distribution']*64)
        rates[task['kind']+'_'+task['method']]=dict(seconds_per_update=sec,peak_allocated_mib=m['peak_allocated_bytes']/2**20,
             estimated_formal_evaluation_seconds=evaluation*expected_eval,pilot_elapsed_seconds=m['elapsed_seconds'],
             pilot_completion_sha256=s.sha256(out/'complete.json'))
    s.atomic(ROOT/'artifacts/operations/calibration.json',rates)
    s.atomic(PILOT/'acceptance.json',dict(status='PASS',scope='Engineering production-batch calibration, not scientific results',
        tasks=len(tasks),protocol_sha256=s.sha256(PILOT/'protocol.json'),source_sha256={n:s.sha256(PILOT/n) for n in ['runner.py','policies.py','env_geometry.py']},rates=rates))
    print(json.dumps(rates,indent=2))

if __name__=='__main__':main()
