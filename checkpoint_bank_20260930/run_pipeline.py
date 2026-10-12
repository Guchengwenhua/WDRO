import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from summarize import main as summarize
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
lock=(ROOT/'artifacts/queue.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Another queue holds artifacts/queue.lock; refusing concurrent queue')
status={'pid':os.getpid(),'start':time.time(),'current_job':None,'child_pid':None,'finished':[],'failures':[],'status':'running'}
def save_status():
 tmp=ROOT/'artifacts/queue_status.json.partial';tmp.write_text(json.dumps(status,indent=2));tmp.replace(ROOT/'artifacts/queue_status.json')
save_status()
try:
 for seed in [101,102]:
  for arm in ['geo','k1','k3','k5','refresh3']:
   job=f'{arm}_s{seed}';log=ROOT/'artifacts'/f'launch_{job}.log';started=time.time()
   status.update(current_job=job,child_pid=None);save_status()
   with log.open('a') as stream:
    cmd=[sys.executable,str(ROOT/'run_bank.py'),'--arm',arm,'--seed',str(seed),'--steps','200']
    child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);status['child_pid']=child.pid;save_status();rc=child.wait()
   record={'arm':arm,'seed':seed,'exit_code':rc,'seconds':time.time()-started,'command':cmd}
   with (ROOT/'artifacts/queue.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   status['finished'].append(job)
   if rc:status['failures'].append(record);print('Failed arm; see',log,flush=True)
   status['child_pid']=None;save_status();summarize()
 status.update(status='complete' if not status['failures'] else 'complete_with_failures',current_job=None,end=time.time());save_status();summarize()
except BaseException as exc:
 status.update(status='interrupted_or_failed',error=repr(exc),end=time.time());save_status();raise
