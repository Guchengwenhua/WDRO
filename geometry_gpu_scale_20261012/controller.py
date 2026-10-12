"""One owned process per GPU, persistent DAG and PID-token recovery."""
import fcntl,json,os,signal,subprocess,time,traceback,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;ART=P/'artifacts';ROOT=P.parent
CFG=json.loads((P/'policy.json').read_text());JOBS=json.loads((P/'jobs.json').read_text())['jobs'];BYID={j['id']:j for j in JOBS}
STOP=False
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial')
 with t.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 t.replace(p)
def event(kind,**kw):
 with (ART/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def token(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
 except OSError:return None
def owned(a):
 try:
  fields=Path(f"/proc/{a['pid']}/stat").read_text().split()
  return fields[2]!='Z' and fields[21]==a['token'] and Path(f"/proc/{a['pid']}/cwd").resolve()==Path(a['cwd']).resolve()
 except (OSError,KeyError,TypeError):return False
def complete(j):
 d=read(j['complete'])
 return isinstance(d,dict) and (d.get('complete') is True or d.get('status') in ['complete','succeeded','report_generated','prefix_completed_not_full_training'] or isinstance(d.get('allow'),bool))
state=read(ART/'state.json',dict(created=time.time(),jobs={},active={}))
children={};handles={}
def save(status=None):
 if status:state['status']=status
 state.update(updated=time.time(),controller_pid=os.getpid(),run_until_complete=True)
 atomic(ART/'state.json',state)
def disk():
 total=0
 for folder in [ART]:
  for base,dirs,files in os.walk(folder.resolve()):
   for name in files:
    try:total+=Path(base,name).stat().st_size
    except FileNotFoundError:pass
 return dict(root_free_gib=shutil.disk_usage(ROOT).free/2**30,attached_free_gib=shutil.disk_usage(ART).free/2**30,owned_gib=total/2**30)
def gpus(j):
 data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
 rows=[list(map(int,line.split(','))) for line in data.splitlines()]
 reserved={a['gpu'] for a in state['active'].values() if a['gpu']>=0}
 return sorted([r for r in rows if r[0] in CFG['gpus'] and r[0] not in reserved and r[0] not in state['jobs'][j['id']].get('failed_gpus',[]) and r[1]>=max(j.get('gpu_min_free_mib',CFG['gpu_min_free_mib']),r[2]*j.get('cuda_fraction',CFG['cuda_fraction'])+CFG['foreign_gpu_reserve_mib']) and r[3]<=CFG['gpu_max_utilization']],key=lambda r:(r[3],r[1] if j.get('gpu_min_free_mib',0)<44000 else -r[1]))
def launch(j,gpu,resources):
 jid=j['id'];r=state['jobs'][jid];r['attempts']+=1
 log=ART/'logs'/f"{jid}.attempt{r['attempts']}.log";log.parent.mkdir(exist_ok=True)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',RESEARCH_CUDA_FRACTION=str(j.get('cuda_fraction',CFG['cuda_fraction'])))
 if gpu>=0:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 else:env['CUDA_VISIBLE_DEVICES']=''
 cache=ART/'runtime'/jid;cache.mkdir(parents=True,exist_ok=True)
 for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:env[key]=str(cache)
 h=log.open('a');child=subprocess.Popen(j['argv'],cwd=j['cwd'],env=env,stdout=h,stderr=subprocess.STDOUT)
 a=dict(pid=child.pid,token=token(child.pid),cwd=j['cwd'],gpu=gpu,started=time.time(),log=str(log))
 state['active'][jid]=a;children[jid]=child;handles[jid]=h;r.update(status='running',**a,resources=resources)
 event('job_start',job=jid,**a);save('running')
def finish(jid,rc):
 a=state['active'].pop(jid);r=state['jobs'][jid];j=BYID[jid]
 r.update(exit_code=rc,ended=time.time(),wall_seconds_this_attempt=time.time()-a['started'])
 r['wall_seconds_total']=r.get('wall_seconds_total',0)+r['wall_seconds_this_attempt']
 if complete(j):r['status']='complete'
 elif a.get('stop_reason')=='controller_stop':r.update(status='paused',reason='controller_stop')
 elif a.get('stop_reason'):r.update(status='failed',reason=a['stop_reason'])
 else:
  tail=Path(a['log']).read_text(errors='replace')[-12000:]
  transient=any(x in tail for x in ['CUDA out of memory','Resource temporarily unavailable','Input/output error','No space left on device','CUDA error: unknown error'])
  if 'CUDA out of memory' in tail:r.setdefault('failed_gpus',[]).append(a['gpu'])
  r.update(status='pending' if transient and r['attempts']<CFG['max_attempts'] else 'failed',reason='transient_retry' if transient else 'process_error_or_missing_marker')
 event('job_end',job=jid,status=r['status'],exit_code=rc,seconds=r['wall_seconds_this_attempt'])
 if jid in handles:handles.pop(jid).close()
 children.pop(jid,None);save()
def main():
 lock=(ART/'controller.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Controller already active')
 frozen=read(ART/'frozen_jobs.json')
 if frozen is not None and frozen!=JOBS:raise RuntimeError('Frozen job commands changed')
 if frozen is None:atomic(ART/'frozen_jobs.json',JOBS)
 for j in JOBS:
  r=state['jobs'].setdefault(j['id'],dict(status='pending',attempts=0))
  a=state['active'].get(j['id'])
  if a and owned(a):continue
  if a:state['active'].pop(j['id'])
  if complete(j):r['status']='complete'
  elif r['status'] in ['paused','running']:r['status']='pending'
 event('controller_started',pid=os.getpid(),gpus=CFG['gpus']);save('running')
 last_resource=0;resources={}
 while True:
  if time.time()-last_resource>60:resources=disk();last_resource=time.time();state['resources']=resources
  disk_ok=resources['root_free_gib']>=CFG['root_min_free_gib'] and resources['attached_free_gib']>=CFG['attached_min_free_gib'] and resources['owned_gib']<=CFG['new_artifact_cap_gib']
  for jid,a in list(state['active'].items()):
   child=children.get(jid);rc=child.poll() if child else None
   alive=(rc is None) if child else owned(a)
   if not alive:finish(jid,rc);continue
   reason='controller_stop' if STOP else 'disk_limit' if not disk_ok and a['gpu']>=0 else 'bounded_job_walltime' if time.time()-a['started']>BYID[jid]['max_wall_seconds'] else None
   if reason and not a.get('stop_reason'):
    if owned(a):os.kill(a['pid'],signal.SIGTERM)
    a.update(stop_reason=reason,stop_requested=time.time());event('checkpoint_stop',job=jid,reason=reason)
   if a.get('stop_requested') and time.time()-a['stop_requested']>1800:
    save('blocked_live_child');raise RuntimeError('Owned child did not exit after checkpoint request; preserved PID, no forced termination')
  if STOP:
   save('pausing' if state['active'] else 'paused')
   if not state['active']:break
   time.sleep(2);continue
  ready=[]
  for j in JOBS:
   r=state['jobs'][j['id']]
   if r['status']!='pending':continue
   deps=[state['jobs'][d]['status'] for d in j['depends_on']]
   if any(d in ['pending','running','paused'] for d in deps):continue
   if not j.get('always_run') and any(d!='complete' for d in deps):r.update(status='skipped_dependency',reason='required_stage_not_complete');continue
   if j.get('gate_file'):
    d=read(j['gate_file'])
    if not isinstance(d,dict) or 'allow' not in d:r.update(status='failed',reason='missing_development_gate');continue
    if not d['allow']:r.update(status='skipped_gate',reason=d.get('reason','negative_development_gate'));continue
   if r['attempts']>=CFG['max_attempts']:r.update(status='failed',reason='attempt_limit');continue
   ready.append(j)
  ready.sort(key=lambda j:(j['priority'],JOBS.index(j)))
  for j in ready:
   cpu=j['kind'] in ['gate','summarize']
   if cpu:
    if sum(a['gpu']<0 for a in state['active'].values())>=CFG['max_cpu_jobs']:continue
    if min(resources['root_free_gib'],resources['attached_free_gib'])<1:continue
    launch(j,-1,resources)
   else:
    if not disk_ok or sum(a['gpu']>=0 for a in state['active'].values())>=CFG['max_gpu_jobs']:continue
    try:available=gpus(j)
    except Exception as e:state['resource_error']=repr(e);continue
    if available:launch(j,available[0][0],dict(resources,gpu_admission=available[0]))
  save('running' if state['active'] else 'waiting_resources')
  if not state['active'] and not any(r['status']=='pending' for r in state['jobs'].values()):
   save('closed');event('controller_closed',statuses={status:sum(r['status']==status for r in state['jobs'].values()) for status in {r['status'] for r in state['jobs'].values()}});break
  time.sleep(CFG['poll_seconds'])
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if not isinstance(e,SystemExit):event('controller_error',error=repr(e));save('controller_error')
  traceback.print_exc();raise
