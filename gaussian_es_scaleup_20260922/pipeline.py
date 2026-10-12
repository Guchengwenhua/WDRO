"""tmux-owned adaptive queue with live admission, exit records and bounded retry."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from common import ROOT, ART, PROTOCOL, atomic_json, append_json, proc_ticks, verify_freeze, checked
from resources import snapshot, admissible
from design import initial_tasks, expand, done

class Adopted:
    def __init__(self,record):self.pid=record['pid'];self.ticks=record['start_ticks'];self.ident=record['task']
    def poll(self):
        if proc_ticks(self.pid)==self.ticks:return None
        if done(self.ident):return 0
        p=ART/'runs'/self.ident/'exit.json'
        if p.exists():
            r=json.loads(p.read_text())
            if r.get('start_ticks')==self.ticks and r.get('status')=='paused':return 75
        return 1
    def send_signal(self,sig):
        if proc_ticks(self.pid)==self.ticks:os.kill(self.pid,sig)

def status(tasks,running,failed,samples,phase):
    entries=[]
    for t in tasks:
        entry=dict(id=t['id'],kind=t['kind'],status='complete' if done(t['id']) else ('running' if t['id'] in running else ('failed' if t['id'] in failed else 'pending')))
        hb=ART/'runs'/t['id']/'heartbeat.json'
        if hb.exists():entry['heartbeat']=json.loads(hb.read_text())
        entries.append(entry)
    atomic_json(ART/'status.json',dict(at=time.time(),phase=phase,pid=os.getpid(),start_ticks=proc_ticks(os.getpid()),tasks=entries,failed=failed,resources=samples[-1] if samples else None))

def main():
    lock=(ART/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_freeze()
    gate=json.loads((ART/'tests/gpu_gate.json').read_text())
    if gate['status']!='PASS':raise RuntimeError('GPU engineering gate not passed')
    tasks=initial_tasks();running={};failed={};samples=[];stop=False
    attempts_path=ART/'attempts.json'
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else {}
    # Reattach only a launch recorded by this controller, with exact PID start ticks.
    events=ART/'events.jsonl'
    if events.exists():
        for line in events.read_text().splitlines():
            record=json.loads(line)
            if record.get('event')=='launch' and record.get('start_ticks') and proc_ticks(record['pid'])==record['start_ticks']:
                running[record['task']]=(Adopted(record),open(os.devnull,'w'),record['gpu']['index'],record['at'])
    for ident,count in attempts.items():
        if count>PROTOCOL['resources']['max_retries']:failed[ident]=dict(attempts=count)
    def on_signal(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    while True:
        tasks=expand(tasks)
        for ident,(proc,handle,gpu,launched) in list(running.items()):
            rc=proc.poll()
            if rc is None:continue
            handle.close();del running[ident]
            append_json(ART/'events.jsonl',dict(at=time.time(),event='exit',task=ident,returncode=rc,pid=proc.pid))
            if rc!=0 and rc!=75:
                attempts[ident]=attempts.get(ident,0)+1
                if attempts[ident]>PROTOCOL['resources']['max_retries']:failed[ident]=dict(returncode=rc,attempts=attempts[ident])
            if rc==0 and not done(ident):failed[ident]=dict(error='exit0 without receipt')
        atomic_json(attempts_path,attempts)
        samples=(samples+[snapshot()])[-3:]
        if stop:
            for proc,_,_,_ in running.values():
                if proc.poll() is None:proc.send_signal(signal.SIGTERM)
            status(tasks,running,failed,samples,'STOPPING')
            if not running:break
        else:
            slots=admissible(samples,[item[2] for item in running.values()]) if len(samples)==3 else []
            pending=[t for t in tasks if not done(t['id']) and t['id'] not in running and t['id'] not in failed]
            for gpu,task in zip(slots,pending):
                if len(running)>=PROTOCOL['resources']['max_workers']:break
                path=ART/'tasks'/f"{task['id']}.json";atomic_json(path,task)
                out=ART/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
                log=(out/f'console_{time.time_ns()}.log').open('w')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'])
                cmd=['bash',str(ROOT/'run_env.sh'),str(ROOT/'worker.py'),'--task',str(path),'--physical-gpu-index',str(gpu['index'])]
                launched=time.time();proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[task['id']]=(proc,log,gpu['index'],launched)
                append_json(ART/'events.jsonl',dict(at=launched,event='launch',task=task['id'],pid=proc.pid,start_ticks=proc_ticks(proc.pid),gpu=gpu,command=cmd,admission=samples))
            phase='NEEDS_ATTENTION' if failed else 'RUNNING'
            status(tasks,running,failed,samples,phase)
            # Retry timeouts only for our own worker processes; no foreign PID is signaled.
            for ident,(proc,_,_,launched) in running.items():
                hb=ART/'runs'/ident/'heartbeat.json'
                last=json.loads(hb.read_text()).get('at',launched) if hb.exists() else launched
                if time.time()-last>PROTOCOL['resources']['heartbeat_timeout']:
                    proc.send_signal(signal.SIGTERM)
                    append_json(ART/'events.jsonl',dict(at=time.time(),event='stale_heartbeat_sigterm',task=ident,pid=proc.pid))
                current=next(g for g in samples[-1]['gpus'] if g['index']==running[ident][2])
                if current['free']<1024 or samples[-1]['disk_free_gib']<PROTOCOL['resources']['min_disk_gib']:
                    proc.send_signal(signal.SIGTERM)
            evaluations=[t for t in tasks if t['kind']=='eval']
            if len(evaluations)==33 and all(done(t['id']) for t in tasks) and not running:
                from report import finalize
                finalize(tasks);status(tasks,running,failed,samples,'COMPLETE');break
            if failed and not running and not pending:break
        time.sleep(15)

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(ART/'pipeline_failure.json',dict(at=time.time(),traceback=traceback.format_exc()))
        raise
