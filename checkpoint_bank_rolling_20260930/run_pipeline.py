import json,os,pathlib,subprocess,sys,time,fcntl
p=pathlib.Path(__file__).resolve().parent
os.chdir(p)
lock=(p/'artifacts/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'status':'running','pid':os.getpid(),'start':time.time(),'finished':[],'failures':[]}
def save():
 t=p/'artifacts/queue_status.json.partial';t.write_text(json.dumps(state,indent=2));t.replace(p/'artifacts/queue_status.json')
for seed in [101,102]:
 state['current_job']=f'lagged3_s{seed}';save();start=time.time()
 with (p/'artifacts'/f'launch_lagged3_s{seed}.log').open('a') as f:
  c=subprocess.Popen([sys.executable,str(p/'run_bank.py'),'--arm','lagged3','--seed',str(seed),'--steps','200'],stdout=f,stderr=subprocess.STDOUT)
  state['child_pid']=c.pid;save();rc=c.wait()
 record={'seed':seed,'exit_code':rc,'seconds':time.time()-start}
 state['finished'].append(record)
 if rc:state['failures'].append(record);state['status']='failed';save();raise SystemExit(rc)
state.update(status='complete',end=time.time(),current_job=None,child_pid=None);save()
