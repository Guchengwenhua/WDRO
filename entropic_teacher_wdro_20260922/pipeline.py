"""Single-instance opportunistic GPU queue. Never signals foreign processes."""
import collections,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
from engine import ROOT,ART,P,identity,binding
import geometry as geo

OPS=ART/'operations'
def train_task(stage,degree,seed,method,updates,parent=None,tau=None,prefix=''):
    name=f'{prefix}{stage}_d{degree}_{method}_s{seed}'
    t=dict(id=name,stage=stage,degree=degree,seed=seed,method=method,updates=updates)
    if parent:t['parent']=parent
    if tau is not None:t['tau']=tau
    if prefix:t['cohort']='pilot';t['val_every']=128
    return t
def pilot_tasks():
    p=P['pilot'];parent=train_task('pre',30,p['seed'],'erm',p['pre'],prefix='pilot_')
    return [parent]+[train_task('post',30,p['seed'],'soft_budget',p['post'],parent['id'],tau,prefix=f'pilot_t{tau}_') for tau in p['taus']]
def formal_tasks(tau):
    tasks=[]
    for degree,config in P['degrees'].items():
        for seed in config['seeds']:
            parent=train_task('pre',int(degree),seed,'erm',config['pre']);tasks.append(parent)
            for method in P['methods']:tasks.append(train_task('post',int(degree),seed,method,config['post'],parent['id'],tau))
    return tasks
def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    gpus=[]
    for line in raw.splitlines():
        i,uuid,free,util=map(str.strip,line.split(','));gpus.append(dict(index=int(i),uuid=uuid,free=int(free),util=int(util)))
    mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/2**20
    return dict(time=time.time(),gpus=gpus,load=os.getloadavg()[0],memory_available_gib=mem,disk_free_gib=shutil.disk_usage(ART).free/2**30)
def verify_source():
    frozen=json.loads((OPS/'source_freeze.json').read_text())
    if binding()!=frozen['binding']:raise RuntimeError('source or protocol modified after freeze')
def no_live_prior():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            cmd=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(str(ROOT/n) in cmd for n in ('engine.py','evaluate.py')) and identity(int(proc.name)):
                raise RuntimeError(f'Existing worker {proc.name}; wait for exit before restarting scheduler')
        except (FileNotFoundError,ProcessLookupError):pass
def result_path(t,evaluation):return ART/('evaluation' if evaluation else 'runs')/t['id']/'complete.json'
def completed(t,evaluation):
    p=result_path(t,evaluation)
    if not p.exists():return False
    r=json.loads(p.read_text())
    if r['status']!='complete' or r['task']!=t:raise RuntimeError(f'invalid completion {t["id"]}')
    return True
def estimate(tasks,running,done,evaluation):
    rates=collections.defaultdict(list)
    for t in tasks:
        base=ART/('evaluation' if evaluation else 'runs')/t['id']
        f=base/'heartbeat.json'
        if f.exists():
            h=json.loads(f.read_text())
            if h['step']>0:rates[(t['degree'],t['method'],t.get('stage','eval'))].append(h['elapsed_seconds']/h['step'])
    total=0.
    for t in tasks:
        if t['id'] in done:continue
        key=(t['degree'],t['method'],t.get('stage','eval'));fallback=(.25 if t.get('stage')=='pre' or t['method']=='erm' else 2.5)*(t['degree']/30)
        rate=float(__import__('statistics').median(rates[key])) if rates[key] else fallback
        base=ART/('evaluation' if evaluation else 'runs')/t['id'];f=base/'heartbeat.json';step=json.loads(f.read_text())['step'] if f.exists() else 0
        total+=max(0,(12 if evaluation else t['updates'])-step)*(rate if not evaluation or rates[key] else 90.)
    active=max(1,len(running));return dict(remaining_worker_hours=total/3600,remaining_hours_at_8_workers=total/3600/8,remaining_hours_at_current_concurrency=total/3600/active,active_workers=len(running),scope='current stage only; queue wait has no fixed upper bound')
def queue(tasks,label,evaluation=False):
    (ART/'tasks').mkdir(exist_ok=True);(ART/'logs').mkdir(exist_ok=True)
    for t in tasks:geo.atomic_json(ART/'tasks'/f'{t["id"]}.json',t)
    progress=OPS/f'{label}_attempts.json';attempts=json.loads(progress.read_text()) if progress.exists() else {}
    running={};history=collections.deque(maxlen=3);cooldown={};failed={}
    while True:
        verify_source();done={t['id'] for t in tasks if completed(t,evaluation)}
        for name,r in list(running.items()):
            code=r['process'].poll()
            if code is None:
                hb=ART/('evaluation' if evaluation else 'runs')/name/'heartbeat.json'
                heartbeat=json.loads(hb.read_text()) if hb.exists() else {}
                last=heartbeat.get('time',r['launched'])
                if time.time()-last>1800 and not r.get('watchdog'):
                    # Popen owns this exact child; no external PID is signalled.
                    r['process'].terminate();r['watchdog']=time.time()
                    geo.atomic_json(OPS/f'{name}_watchdog.json',dict(time=time.time(),identity=identity(r['process'].pid),reason='no heartbeat for 1800 seconds'))
                elif r.get('watchdog') and time.time()-r['watchdog']>120:r['process'].kill()
                continue
            r['log'].close();running.pop(name)
            if r.get('watchdog'):code=99
            if code==75:cooldown[name]=time.time()+120
            elif code!=0 or name not in done:
                attempts[name]=attempts.get(name,0)+1;cooldown[name]=time.time()+60
                if attempts[name]>P['resources']['max_retries']:failed[name]=f'exit {code}; retries exhausted'
                geo.atomic_json(progress,attempts)
        for name,count in attempts.items():
            if count>P['resources']['max_retries'] and name not in done:failed[name]='retries exhausted'
        if len(done)==len(tasks):return
        if failed and not running:raise RuntimeError(f'queue {label} has failed tasks: {failed}')
        snap=snapshot();history.append(snap)
        state=dict(time=time.time(),label=label,total=len(tasks),done=len(done),failed=failed,running={name:{'identity':identity(r['process'].pid),'gpu':r['gpu'],'command':r['command']} for name,r in running.items()},resources=snap,eta=estimate(tasks,running,done,evaluation))
        geo.atomic_json(OPS/'queue.json',state);geo.atomic_json(OPS/'eta.json',state['eta'])
        if len(history)==3 and not (OPS/'PAUSE').exists() and snap['disk_free_gib']>P['resources']['disk_reserve_gib'] and snap['memory_available_gib']>32 and snap['load']<P['resources']['max_load']:
            used={r['gpu']['uuid'] for r in running.values()}
            eligible=[]
            for gpu in snap['gpus']:
                samples=[next(g for g in s['gpus'] if g['uuid']==gpu['uuid']) for s in history]
                if gpu['uuid'] not in used and min(g['free'] for g in samples)>=P['resources']['min_free_mib'] and sum(g['util'] for g in samples)/3<P['resources']['max_util']:eligible.append(gpu)
            eligible.sort(key=lambda g:(g['util'],-g['free']))
            for gpu in eligible:
                if len(running)>=P['resources']['max_workers']:break
                # Pretrain parents first, then methods interleaved by seed.
                ready=[t for t in tasks if t['id'] not in done and t['id'] not in running and t['id'] not in failed and time.time()>=cooldown.get(t['id'],0) and (not t.get('parent') or (ART/'runs'/t['parent']/'complete.json').exists())]
                if not ready:break
                ready.sort(key=lambda t:(t.get('stage')!='pre',tasks.index(t)))
                t=ready[0];cmd=['bash',str(ROOT/'run_env.sh'),str(ROOT/('evaluate.py' if evaluation else 'engine.py')),'--task',str(ART/'tasks'/f'{t["id"]}.json')]
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
                log=(ART/'logs'/f'{t["id"]}.log').open('a');proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[t['id']]=dict(process=proc,log=log,gpu=gpu,command=cmd,launched=time.time())
                geo.atomic_json(OPS/f'{t["id"]}_launch.json',dict(time=time.time(),identity=identity(proc.pid),gpu=gpu,command=cmd))
        time.sleep(10)
def select_tau(pilot):
    rows=[]
    for t in pilot[1:]:
        hist=[json.loads(l) for l in (ART/'runs'/t['id']/'validation.jsonl').read_text().splitlines()]
        # Select using final pilot update, never step0 or its selected checkpoint.
        final=next(v for v in reversed(hist) if v['step']==t['updates'])
        rows.append(dict(tau=t['tau'],eligible=final['eligible'],metric=final['metric'],task=t['id']))
    eligible=[r for r in rows if r['eligible']]
    tau=min(eligible,key=lambda r:(r['metric'],r['tau']))['tau'] if eligible else P['tau_default']
    geo.atomic_json(OPS/'selection.json',dict(tau=tau,pilots=rows,fallback=not bool(eligible),rule='eligible final pilot worst-group mean; tie smaller tau',time=time.time()))
    return tau
def main():
    OPS.mkdir(exist_ok=True);lock=(OPS/'scheduler.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    no_live_prior();verify_source();geo.atomic_json(OPS/'controller.json',dict(**identity(),time=time.time()))
    pilot=pilot_tasks()
    geo.atomic_json(OPS/'planned_training.json',dict(tasks=pilot+formal_tasks(P['tau_default']),temperature_status='provisional until selection.json; task count fixed',total=116))
    # Independent main pretraining can fill idle GPUs while temperature pilots run.
    parents=[t for t in formal_tasks(P['tau_default']) if t['stage']=='pre']
    queue(pilot+parents,'calibration_and_parents');tau=select_tau(pilot)
    formal=formal_tasks(tau);alltraining=pilot+formal
    queue(formal,'formal_training')
    models={}
    for t in formal:
        for name in ('best.pt','latest.pt'):
            path=ART/'runs'/t['id']/name;models[str(path)]=geo.sha256(path)
    if not (OPS/'model_freeze.json').exists():geo.atomic_json(OPS/'model_freeze.json',dict(models=models,time=time.time(),source=geo.sha256(OPS/'source_freeze.json')))
    elif json.loads((OPS/'model_freeze.json').read_text())['models']!=models:raise RuntimeError('frozen model set changed')
    evaluation=[]
    for t in formal:
        for checkpoint,tag in [('best.pt','best'),('latest.pt','last')]:
            evaluation.append(dict(id=t['id']+'_'+tag,run_id=t['id'],checkpoint=checkpoint,degree=t['degree'],seed=t['seed'],method=t['method']))
    geo.atomic_json(OPS/'manifest.json',dict(training=alltraining,evaluation=evaluation))
    queue(evaluation,'frozen_evaluation',True)
    from report import make_report
    audit=make_report();geo.atomic_json(OPS/'delivery.json',dict(status='COMPLETE' if audit['engineering_pass'] else 'NEEDS_ATTENTION',time=time.time(),audit=audit))
    geo.atomic_json(OPS/'queue.json',dict(status='COMPLETE' if audit['engineering_pass'] else 'NEEDS_ATTENTION',time=time.time(),label='complete',total=len(alltraining)+len(evaluation),done=len(alltraining)+len(evaluation),running={}))
if __name__=='__main__':
    try:main()
    except Exception as error:
        geo.atomic_json(OPS/'delivery.json',dict(status='NEEDS_ATTENTION',time=time.time(),error=repr(error)));raise
