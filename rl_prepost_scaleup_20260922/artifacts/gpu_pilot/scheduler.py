"""Own queue; immutable source imports and automatic handoff after prior experiment."""
import json,os,shutil,subprocess,time
from pathlib import Path
import ops as o

ROOT=Path(__file__).resolve().parent;OPS=ROOT/'artifacts/operations'
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def config():return read(ROOT/'protocol.json')
def task_path(parent):
    return ROOT/'lineage'/parent['run']/'latest.pt' if parent['source']=='legacy' else ROOT/'artifacts/runs'/parent['run']/'checkpoints'/f"step_{parent['step']}.pt"

def import_parent(parent):
    if parent['source']!='legacy':return task_path(parent).exists()
    target=task_path(parent);receipt=target.parent/'receipt.json'
    if receipt.exists():
        r=read(receipt)
        if o.sha256(target)!=r['checkpoint_sha256']:raise RuntimeError('Imported parent changed')
        return True
    origin=Path(config()['origin_root']);p=read(origin/'protocol.json');tasks={t['id']:t for t in o.tasks(p)}
    ok,reason=o.validate_complete(tasks[parent['run']],p,root=origin,expected=o.expected_binding(origin))
    if not ok:
        if reason in ['missing_complete','worker_still_live','run_lock_held']:return False
        raise RuntimeError('Invalid origin parent '+parent['run']+': '+reason)
    src=origin/'artifacts/runs'/parent['run'];target.parent.mkdir(parents=True,exist_ok=True)
    for name in ['latest.pt','complete.json','run.json']:
        tmp=target.parent/(name+'.partial');shutil.copyfile(src/name,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        if o.sha256(tmp)!=o.sha256(src/name):raise RuntimeError('Parent copy hash mismatch')
        os.replace(tmp,target.parent/name)
    o.atomic(receipt,dict(source_directory=str(src),source_complete_sha256=o.sha256(src/'complete.json'),
        checkpoint_sha256=o.sha256(target),source_protocol_sha256=o.sha256(origin/'protocol.json'),imported_at=time.time()))
    return True

def verify_complete(task,require_gpu=True):
    out=ROOT/'artifacts/runs'/task['id'];m=read(out/'complete.json')
    if m is None:return False
    if m['task']!=task or m['step']!=task['end'] or m['status']!='complete':raise RuntimeError('Invalid completion '+task['id'])
    if o.same_identity(o.ident(m['identity']['pid']),m['identity']) or o.lock_held(out/'run.lock'):return False
    if require_gpu and (m['device']!='cuda:0' or m['peak_allocated_bytes']<=0):raise RuntimeError('Missing CUDA evidence')
    if m['binding']['protocol_sha256']!=o.sha256(ROOT/'protocol.json'):raise RuntimeError('Protocol mismatch')
    for group in ['code_sha256','dependencies','parents']:
        for path,sha in m['binding'][group].items():
            if o.sha256(path)!=sha:raise RuntimeError('Binding mismatch '+path)
    for name,sha in m['artifacts_sha256'].items():
        if o.sha256(out/name)!=sha:raise RuntimeError('Artifact mismatch '+str(out/name))
    return True

def origin_gate():
    p=config()
    if not p['resources'].get('wait_for_origin_training',True):return True,0
    src=Path(p['origin_root']);q=read(src/'artifacts/operations/queue.json',{})
    delivery=read(src/'artifacts/operations/delivery.json',{})
    if q.get('status') in ['failed','audit_failed','invalid_completion'] or delivery.get('status')=='NEEDS_REVIEW':
        raise RuntimeError('Origin experiment needs review; preserving new queue')
    return delivery.get('status')=='COMPLETE' and not o.discover_live_workers(src),len(q.get('complete',[]))

def main():
    OPS.mkdir(parents=True,exist_ok=True);lock=o.acquire_scheduler_lock(OPS/'scheduler.lock')
    p=config();tasks=p['tasks'];resources=p['resources'];running={};done=set();cooldown={}
    old=o.ensure_no_prior_workers(tasks,root=ROOT);attempts=old.get('attempts',{})
    o.atomic(OPS/'launch.json',dict(identity=o.ident(os.getpid()),time=time.time(),tasks=tasks))
    try:
        while True:
            if not o.verify_source_freeze(ROOT):raise RuntimeError('Frozen source manifest required')
            for key in list(running):
                item=running[key];code=item['process'].poll();hb=ROOT/'artifacts/runs'/key/'heartbeat.json'
                age=time.time()-(hb.stat().st_mtime if hb.exists() else item['launch'])
                if code is None and age>resources['heartbeat_timeout_seconds'] and o.same_identity(o.ident(item['pid']),item['identity']):
                    if item.get('termination_requested') and time.time()-item['termination_requested']>120:item['process'].kill()
                    elif not item.get('termination_requested'):item['process'].terminate();item['termination_requested']=time.time()
                if code is not None:
                    item['stream'].close();del running[key]
                    o.atomic(OPS/f'{key}_exit_{time.time_ns()}.json',dict(returncode=code,identity=item['identity'],time=time.time()))
                    if code==75:attempts[key]=max(0,attempts[key]-1)
                    cooldown[key]=time.time()+resources['retry_cooldown_seconds']
            for task in tasks:
                if task['id'] not in done and task['id'] not in running and verify_complete(task):done.add(task['id'])
            if len(done)==len(tasks):
                o.atomic(OPS/'queue.json',dict(status='auditing',complete=sorted(done),total=len(tasks),running={},attempts=attempts,time=time.time()))
                result=subprocess.run(['nice','-n','10','bash',str(ROOT/'run_env.sh'),str(ROOT/'report.py')],cwd=ROOT)
                o.atomic(OPS/'queue.json',dict(status='complete' if result.returncode==0 else 'audit_failed',complete=sorted(done),total=len(tasks),running={},attempts=attempts,time=time.time()))
                return result.returncode
            failed=[t['id'] for t in tasks if attempts.get(t['id'],0)>=3 and t['id'] not in done and t['id'] not in running]
            if failed and not running:raise RuntimeError('Retry limit: '+','.join(failed))
            gate,origin_done=origin_gate();ready=[]
            for t in tasks:
                if t['id'] in done or t['id'] in running or attempts.get(t['id'],0)>=3 or time.time()<cooldown.get(t['id'],0):continue
                if all((v['run'] in done if v['source']=='new' else import_parent(v)) for v in t['parents'].values()):ready.append(t)
            try:snap=o.snapshot()
            except (OSError,subprocess.SubprocessError,ValueError) as exc:
                o.atomic(OPS/'queue.json',dict(status='resource_probe_failed',error=repr(exc),complete=sorted(done),total=len(tasks),running=o.serial_running(running),attempts=attempts,time=time.time()))
                time.sleep(15);continue
            history=getattr(main,'history',[]);history=(history+[snap])[-3:];main.history=history
            o.atomic(OPS/'resources_latest.json',snap)
            with (OPS/'resources.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
            if not gate or (OPS/'PAUSE').exists():ready=[]
            for task in ready:
                try:fresh=o.snapshot()
                except (OSError,subprocess.SubprocessError,ValueError):break
                cap,cards,reasons=o.admission(fresh,history[:-1]+[fresh],running,resources)
                if cap<=0 or not cards:break
                gpu=cards[0];key=task['id'];attempts[key]=attempts.get(key,0)+1;tag=f'{key}_{time.time_ns()}'
                cmd=['nice','-n','10','bash',str(ROOT/'run_env.sh'),str(ROOT/'runner.py'),'--task-id',key,
                     '--physical-gpu-index',str(gpu['index']),'--gpu-uuid',gpu['uuid']]
                out=ROOT/'artifacts/runs'/key;out.mkdir(parents=True,exist_ok=True)
                o.atomic(OPS/(tag+'_preflight.json'),dict(snapshot=fresh,history=history,admission=gpu,command=cmd,exclusions=reasons))
                stream=(out/f'console_{tag}.log').open('a')
                process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index'])),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                running[key]=dict(process=process,stream=stream,pid=process.pid,identity=o.ident(process.pid),gpu=gpu['index'],gpu_uuid=gpu['uuid'],launch=time.time(),command=cmd)
                o.atomic(OPS/(tag+'_launch.json'),o.serial_running(running)[key])
                o.atomic(OPS/'queue.json',dict(status='running',complete=sorted(done),total=len(tasks),running=o.serial_running(running),attempts=attempts,time=time.time()))
            status='running' if running else ('waiting_origin' if not gate else 'waiting_resources')
            o.atomic(OPS/'queue.json',dict(status=status,origin_completed=origin_done,complete=sorted(done),total=len(tasks),running=o.serial_running(running),attempts=attempts,time=time.time()))
            time.sleep(15)
    except BaseException as exc:
        o.atomic(OPS/'queue.json',dict(status='failed',error=repr(exc),complete=sorted(done),total=len(tasks),running=o.serial_running(running),attempts=attempts,time=time.time()))
        raise
    finally:lock.close()

if __name__=='__main__':raise SystemExit(main())
