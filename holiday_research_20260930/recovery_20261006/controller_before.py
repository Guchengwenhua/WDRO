"""Bounded, persistent research controller. Existing jobs are never restarted while alive."""
import datetime as dt,fcntl,json,os,signal,shutil,subprocess,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parent; ART=P/'artifacts'
CFG=json.loads((P/'policy.json').read_text()); PY='/remote-home/Guilin/miniconda3/envs/py311/bin/python'
DEAD=dt.datetime.fromisoformat(CFG['deadline_utc']).timestamp()
CUT=dt.datetime.fromisoformat(CFG['training_cutoff_utc']).timestamp()
STOP=False
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def event(kind,**kw):
 with (ART/'events.jsonl').open('a') as f:
  f.write(json.dumps(dict(time=time.time(),kind=kind,**kw))+chr(10));f.flush();os.fsync(f.fileno())
state=read(ART/'state.json',{'jobs':{},'dependencies':{},'created':time.time(),'active':None})
def save(status=None,**kw):
 if status:state['status']=status
 state.update(kw);state['updated']=time.time();state['controller_pid']=os.getpid();atomic(ART/'state.json',state)
def nap(seconds=None):
 end=time.time()+(seconds or CFG['poll_seconds'])
 while not STOP and time.time()<end:time.sleep(min(5,end-time.time()))
def live(pid,token=None):
 try:
  fields=Path(f'/proc/{int(pid)}/stat').read_text().split()
  return fields[2]!='Z' and (token is None or fields[21]==str(token))
 except (OSError,ValueError):return False
def token(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
 except OSError:return None
def marker(path):
 d=read(path)
 if not isinstance(d,dict):return False
 return d.get('status') in ('complete','succeeded','pass','skipped','negative_gate') or d.get('complete') is True or ('allow' in d and isinstance(d['allow'],bool))
def owned_bytes():
 total=0
 for entry in CFG['owned_artifact_dirs']:
  path=Path(entry)
  if not path.exists():continue
  for base,dirs,files in os.walk(path.resolve(),followlinks=False):
   for name in files:
    f=Path(base)/name
    if not f.is_symlink():
     try:total+=f.stat().st_size
     except FileNotFoundError:pass
 return total
def resource(preferred=None,cpu=False):
 root_free=shutil.disk_usage(ROOT).free/2**30
 attached_free=shutil.disk_usage(ART.resolve()).free/2**30
 size=owned_bytes()/2**30
 r={'root_free_gib':root_free,'attached_free_gib':attached_free,'owned_gib':size}
 if cpu:
  if root_free<1 or attached_free<1:return None,dict(r,reason='disk_limit')
  return -1,r
 if root_free<CFG['root_min_free_gib'] or attached_free<CFG['attached_min_free_gib'] or size>CFG['new_artifact_cap_gib']:return None,dict(r,reason='disk_limit')
 try:
  data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
  rows=[tuple(int(x.strip()) for x in line.split(',')) for line in data.splitlines()]
 except Exception as exc:return None,dict(r,reason='gpu_query_failed',error=type(exc).__name__)
 allowed=[preferred] if preferred is not None else CFG['gpus']
 good=[x for x in rows if x[0] in allowed and x[1]>=CFG['gpu_min_free_mib'] and x[2]<=CFG['gpu_max_utilization_before_start']]
 r['gpu_snapshot']=rows
 if not good:return None,dict(r,reason='no_low_load_gpu')
 return min(good,key=lambda x:(x[2],-x[1]))[0],r
def env_for(gpu):
 env=os.environ.copy()
 env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 if gpu>=0:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 cache=ART/'cache';cache.mkdir(exist_ok=True)
 for key in ['XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TMPDIR','MPLCONFIGDIR']:
  target=cache/key.lower();target.mkdir(exist_ok=True);env[key]=str(target)
 return env
def log_tail(path):
 try:
  with Path(path).open('rb') as f:f.seek(max(0,Path(path).stat().st_size-12000));return f.read().decode(errors='replace')
 except OSError:return ''
TRANSIENT=['CUDA out of memory','Resource temporarily unavailable','Input/output error','No space left on device','CUDA error: unknown error','CUDA error: device-side assert'] # device-side assert excluded below
def transient(text):
 return any(x in text for x in TRANSIENT[:-1])
def run_job(job,recovery=False):
 jid=job['id'];rec=state['jobs'].setdefault(jid,{'attempts':0,'status':'pending'})
 if marker(job['complete']):rec['status']='complete';save();return True
 if rec.get('status') in ('failed','skipped_gate','skipped_dependency','skipped_deadline'):return False
 cpu=job.get('kind') in ('gate','summarize')
 limit=CUT if job.get('kind')=='train' else DEAD
 estimate=job.get('estimated_seconds',3600)
 if time.time()+estimate>limit:
  rec.update(status='skipped_deadline',reason='insufficient_prespecified_time_budget');save();return False
 while not STOP and time.time()<limit:
  gpu,snapshot=resource(job.get('gpu'),cpu)
  if gpu is not None:break
  save('waiting_resources',waiting_job=jid,resources=snapshot);nap()
 else:rec['status']='paused';save();return False
 if time.time()+estimate>limit:
  rec.update(status='skipped_deadline',reason='resource_wait_consumed_remaining_budget');save();return False
 if rec['attempts']>=CFG['max_attempts']:rec.update(status='failed',reason='attempt_limit');save();return False
 rec['attempts']+=1
 path=ART/'logs'/f"{jid}.attempt{rec['attempts']}.log";path.parent.mkdir(exist_ok=True)
 argv=[str(x).replace('{gpu}',str(gpu)) for x in job['argv']]
 rec.update(status='running',started=time.time(),gpu=gpu,argv=argv,log=str(path),resources=snapshot)
 with path.open('a') as f:
  child=subprocess.Popen(argv,cwd=job['cwd'],env=env_for(gpu),stdout=f,stderr=subprocess.STDOUT)
  state['active']={'job':jid,'pid':child.pid,'token':token(child.pid),'cwd':job['cwd'],'started':time.time()}
  save('running');event('job_start',job=jid,pid=child.pid,gpu=gpu)
  reason=None;last_disk=0
  while child.poll() is None:
   if STOP:reason='controller_stop'
   elif time.time()>=limit:reason='stage_deadline'
   elif time.time()-rec['started']>job.get('max_wall_seconds',max(7200,estimate*3)):reason='bounded_job_walltime'
   if time.time()-last_disk>300:
    _,res=resource(cpu=True);last_disk=time.time()
    if job.get('kind') not in ('gate','summarize') and (res['root_free_gib']<CFG['root_min_free_gib'] or res['attached_free_gib']<CFG['attached_min_free_gib'] or res['owned_gib']>CFG['new_artifact_cap_gib']):res['reason']='disk_limit'
    if res.get('reason')=='disk_limit':reason='disk_limit'
   if reason:
    child.send_signal(signal.SIGTERM);event('own_child_checkpoint_stop',job=jid,reason=reason,pid=child.pid)
    try:child.wait(timeout=180)
    except subprocess.TimeoutExpired:
     rec.update(status='blocked_live_child',reason=reason);save('blocked_live_child')
     raise RuntimeError('Own child did not exit after checkpoint request; no force kill, no other job started')
    break
   save('running',resources=res if 'res' in locals() else snapshot);nap(CFG['child_poll_seconds'])
  rc=child.returncode
 rec.update(exit_code=rc,ended=time.time(),wall_seconds=time.time()-rec['started']);state['active']=None
 if marker(job['complete']):rec['status']='complete'
 elif reason:
  rec.update(status='paused',reason=reason)
  if reason=='stage_deadline':rec['status']='skipped_deadline'
  elif reason in ('bounded_job_walltime','disk_limit'):rec['status']='failed'
 else:rec.update(status='failed',reason='missing_completion_marker' if rc==0 else 'process_error')
 event('job_end',job=jid,status=rec['status'],exit_code=rc,seconds=rec['wall_seconds']);save()
 if rec['status']=='failed' and not reason and transient(log_tail(path)) and rec['attempts']<CFG['max_attempts']:
  rec.update(status='pending',retry_reason='one_transient_retry_from_checkpoint');save('retry_wait');nap();return run_job(job,recovery)
 return rec['status']=='complete'
def existing_dependencies():
 scale=Path(CFG['dependencies']['scale']);geom=Path(CFG['dependencies']['geometry'])
 def owns_live(pid,cwd):
  try:return live(pid) and Path(f'/proc/{int(pid)}/cwd').resolve()==cwd.resolve()
  except (OSError,ValueError):return False
 sf=(scale/'artifacts/queue.lock').open('a');locked=False;scale_ok=False
 while not STOP and time.time()<CUT:
  try:fcntl.flock(sf,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True;break
  except BlockingIOError:save('waiting_existing_scale',dependency_queue=read(scale/'artifacts/queue_status.json'));nap()
 if STOP:sf.close();return False
 if locked:
  try:
   # A parent queue could die while its worker remains alive; never duplicate that worker.
   source=read(scale/'artifacts/queue_status.json',{})
   while owns_live(source.get('child_pid',0),scale) and not STOP and time.time()<CUT:
    save('waiting_existing_scale_child',dependency_queue=source);nap()
   scale_ok=not STOP and time.time()<CUT
   for seed in [101,102,103]:
    if not scale_ok:break
    for arm in ['geo','k3','lagged3']:
     done=scale/'artifacts'/f'{arm}_s{seed}'/'complete.json'
     if marker(done):continue
     oldlog=scale/'artifacts'/f'launch_{arm}_s{seed}.log';tail=log_tail(oldlog)
     if 'Traceback' in tail and not transient(tail):
      event('existing_scale_branch_error',error_log=str(oldlog));scale_ok=False;break
     job={'id':f'recover_scale_{arm}_s{seed}','argv':[PY,'-u','-B',str(scale/'run_scale.py'),'--arm',arm,'--seed',str(seed)],'cwd':str(scale),'complete':str(done),'kind':'train','estimated_seconds':7200,'gpu':5}
     if not run_job(job,True):scale_ok=False;break
  finally:fcntl.flock(sf,fcntl.LOCK_UN);sf.close()
 else:sf.close()
 state['dependencies']['scale']='complete' if scale_ok else 'failed_or_deadline';save()
 if STOP:return False
 while not marker(geom/'artifacts/complete.json') and not STOP and time.time()<CUT:
  hb=read(geom/'artifacts/heartbeat.json',{})
  if owns_live(hb.get('pid',0),geom):
   save('waiting_existing_geometry',dependency_heartbeat=hb);nap();continue
  tail=log_tail(geom/'artifacts/run.log')
  if 'Traceback' in tail and not transient(tail):
   event('existing_geometry_branch_error',error_log=str(geom/'artifacts/run.log'));break
  job={'id':'recover_geometry_extension','argv':['bash',str(geom/'run_env.sh'),'7'],'cwd':str(geom),'complete':str(geom/'artifacts/complete.json'),'kind':'evaluate','estimated_seconds':7200,'gpu':7}
  if not run_job(job,True):break
 geometry_ok=marker(geom/'artifacts/complete.json')
 state['dependencies']['geometry']='complete' if geometry_ok else 'failed_or_deadline';save()
 return scale_ok and geometry_ok and not STOP
def snapshot_reports(jobs):
 inventory=[]
 for j in jobs:
  inventory.append({'id':j['id'],'complete_marker':j['complete'],'completed':marker(j['complete']),'state':state['jobs'].get(j['id'],{'status':'not_started'}),'source_dir':j['cwd']})
 atomic(ART/'acceptance_inventory.json',{'generated':time.time(),'deadline':CFG['deadline_utc'],'jobs':inventory})
 lines=['# Operational acceptance inventory','','Scientific results reside in each stage report; missing jobs are not successful results.','']
 for j in inventory:lines.append('- '+j['id']+': '+j['state']['status']+'; '+j['complete_marker'])
 (P/'ACCEPTANCE_INVENTORY.md').write_text(chr(10).join(lines)+chr(10))
def main():
 ART.mkdir(exist_ok=True)
 lock=(ART/'controller.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Controller already active; no duplicate launch')
 active=state.get('active')
 if active and live(active['pid'],active['token']):
  save('reattached_existing_child')
  while live(active['pid'],active['token']) and not STOP:nap()
  if STOP:return
  rec=state['jobs'][active['job']];rec['status']='pending';rec['reason']='controller_restarted_after_orphan_child';state['active']=None;save()
 save('dependency_check');event('controller_started',pid=os.getpid())
 dependencies_ok=existing_dependencies()
 if STOP:
  save('paused',pause_reason='controller stop during dependency wait');return
 jobs=[]
 for file in CFG['manifest_paths']:
  document=read(file)
  if document is None:raise RuntimeError('Frozen manifest missing: '+file)
  jobs.extend(document if isinstance(document,list) else document['jobs'])
 ids=[j['id'] for j in jobs]
 if len(ids)!=len(set(ids)):raise RuntimeError('Duplicate experiment IDs')
 for j in jobs:
  for key in ['id','argv','cwd','complete','kind']:assert key in j,(j,key)
  if not Path(j['cwd']).resolve().is_relative_to(ROOT):raise RuntimeError('job outside authorized project')
  if not Path(j['complete']).is_relative_to(ROOT):raise RuntimeError('marker outside project')
  if any(d not in ids for d in j.get('depends_on',[])):raise RuntimeError('Unknown dependency')
 frozen=read(ART/'frozen_jobs.json')
 if frozen is not None and frozen!=jobs:raise RuntimeError('Manifest differs from frozen queue; refusing silent protocol change')
 if frozen is None:atomic(ART/'frozen_jobs.json',jobs)
 for j in jobs:
  rec=state['jobs'].setdefault(j['id'],{'attempts':0,'status':'pending'})
  if marker(j['complete']):rec['status']='complete'
  elif rec['status'] in ('paused','running'):rec['status']='pending'
 if not dependencies_ok:
  for j in jobs:
   required='geometry' if 'week_geometry_' in j['cwd'] else 'scale'
   if state['dependencies'].get(required)!='complete' and not j.get('always_run') and not marker(j['complete']):
    state['jobs'][j['id']].update(status='skipped_dependency',reason='required existing dependency failed or deadline; see controller state/events')
 save('stages_ready' if dependencies_ok else 'dependency_blocked_reports_only')
 lane_last=None
 while not STOP and time.time()<DEAD:
  candidates=[];unresolved=False
  for j in jobs:
   rec=state['jobs'][j['id']]
   if rec['status']!='pending':continue
   unresolved=True;deps=[state['jobs'][d]['status'] for d in j.get('depends_on',[])]
   if not j.get('always_run') and any(s in ('failed','skipped_gate','skipped_dependency','skipped_deadline','blocked_live_child') for s in deps):
    rec.update(status='skipped_dependency',reason='required_stage_did_not_complete');continue
   if any(s in ('pending','running','paused') for s in deps):continue
   gate=j.get('gate_file')
   if gate:
    decision=read(gate)
    if not isinstance(decision,dict) or 'allow' not in decision:
     rec.update(status='failed',reason='missing_or_invalid_frozen_gate');continue
    if not decision['allow']:
     rec.update(status='skipped_gate',reason=decision.get('reason','negative_development_gate'));continue
   candidates.append(j)
  snapshot_reports(jobs);save()
  if not candidates:
   if unresolved:save('closed_gated_or_blocked')
   break
  candidates.sort(key=lambda j:(j.get('priority',50),Path(j['cwd']).name==lane_last,ids.index(j['id'])))
  job=candidates[0];lane_last=Path(job['cwd']).name
  run_job(job);snapshot_reports(jobs)
 for j in jobs:
  r=state['jobs'][j['id']]
  if r['status']=='pending' and not STOP:r.update(status='not_run',reason='deadline_or_dependency')
 snapshot_reports(jobs);save('paused' if STOP else 'closed',ended=time.time())
 event('controller_closed',status=state['status'])
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  event('controller_error',error=repr(exc));save('controller_error',error=repr(exc));traceback.print_exc();raise
