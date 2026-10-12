import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
lock=(ROOT/'artifacts/queue.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Scale queue already active')
status={'pid':os.getpid(),'started':time.time(),'status':'waiting_for_rolling','current_job':None,'child_pid':None,'finished':[],'failures':[]}
def save():
 p=ROOT/'artifacts/queue_status.json';t=p.with_suffix('.json.partial');t.write_text(json.dumps(status,indent=2));os.replace(t,p)
save()
with Path('/remote-home/Guilin/DRO/Codex/checkpoint_bank_rolling_20260930/artifacts/queue.lock').open('a') as prior:fcntl.flock(prior,fcntl.LOCK_EX)
status['status']='running';save()
for seed in [101,102,103]:
 for arm in ['geo','k3','lagged3']:
  job=f'{arm}_s{seed}';status['current_job']=job;save();started=time.time()
  with (ROOT/'artifacts'/f'launch_{job}.log').open('a') as stream:
   child=subprocess.Popen([sys.executable,'-u','run_scale.py','--arm',arm,'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT);status['child_pid']=child.pid;save();rc=child.wait()
  rec={'job':job,'exit_code':rc,'seconds':time.time()-started};status['finished'].append(rec);status['child_pid']=None
  with (ROOT/'artifacts/queue.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  if rc:status['failures'].append(rec);status['status']='failed_stop';save();raise SystemExit(rc)
  save()
status.update(status='complete',current_job=None,finished_time=time.time());save()
