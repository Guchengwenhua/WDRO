"""Detached finalizer. Reports failures explicitly; never treats missing work as done."""
import json,os,subprocess,time
from pathlib import Path
import fcntl
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'artifacts'; OPS=EXP/'operations'

def read(p):return json.loads(p.read_text())
def active(record):
    p=Path(f"/proc/{record['pid']}/stat")
    if not p.exists():return False
    fields=p.read_text().rsplit(')',1)[1].split()
    return fields[19]==str(record.get('start_ticks',record.get('process_start_ticks')))
def publish(status,**extra):
    p=OPS/'autofinish_status.json';tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(dict(status=status,at=time.time(),pid=os.getpid(),**extra),indent=2))
    os.replace(tmp,p)
def main():
    with (OPS/'autofinish.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish('monitoring_server_queue')
        while active(read(OPS/'launch.json')):
            q=read(OPS/'queue.json')
            publish('monitoring_server_queue',completed=len(q['completed']),running=q['running'],pending=q['pending'],failures=q['failures'])
            time.sleep(15)
        q=read(OPS/'queue.json')
        if q['pending'] or q['failures'] or q['running'] or len(set(q['completed']))!=16:
            publish('failed_incomplete_tasks',queue=q,recovery='Inspect failure logs; recover unchanged commands with bash run_env.sh scheduler.py after verifying no workers remain.')
            return 1
        audit=EXP/'audit/completion_audit.json'
        if not audit.exists() or read(audit).get('status')!='pass':
            publish('running_independent_audit')
            rc=subprocess.call(['bash',str(ROOT/'run_env.sh'),'report.py'],cwd=ROOT)
            if rc:
                publish('failed_independent_audit',exit_code=rc,log=str(OPS/'autofinish.log'))
                return rc
        publish('checking_final_process_exit')
        rc=subprocess.call(['bash',str(ROOT/'run_env.sh'),'closeout.py'],cwd=ROOT)
        if rc:
            publish('failed_final_closeout',exit_code=rc);return rc
        publish('complete',results=str(ROOT/'RESULTS.md'),delivery=str(EXP/'audit/final_delivery.json'))
        return 0
if __name__=='__main__':raise SystemExit(main())
