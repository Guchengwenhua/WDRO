"""Resource-aware, resumable long-running inference queue and final delivery."""
import fcntl
import shutil
import subprocess
import time
from collections import Counter
from common import *

def gpu_status():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=25)
    out=[]
    for line in text.splitlines():
        i,uuid,free,util=[v.strip() for v in line.split(',')]
        out.append(dict(index=int(i),uuid=uuid,free=int(free),util=int(util)))
    return out

def run():
    lock=(ART/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(ROOT/'protocol.json');config=protocol['resources']
    models=read(ART/'models.json');datasets=read(ART/'datasets.json');total_n=sum(x['count'] for x in datasets)
    priority={'wdro_precision_v2':0,'rl_expansion_20260922':1,'net_structure_real_20260922':2,'vae_wdro_real_20260921':3,'entropic_teacher_wdro_20260922':4,'gaussian_es_scaleup_20260922':5}
    ready=sorted([m for m in models if m['status']=='ready'],key=lambda m:(m['family']!='rest',m.get('train_degree')!=30,priority.get(m['branch'],9),'best' not in m['path'],m['id']))
    active={};attempts=read(ART/'attempts.json') if (ART/'attempts.json').exists() else {};last_render=0;start=time.time()
    # Model flock prevents duplicate work even after a supervisor is restarted.
    while True:
        for mid,(proc,gpu,log) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close();del active[mid]
                event(ART/'queue_events.jsonl',dict(time=time.time(),event='worker_exit',model=mid,gpu=gpu,exit_code=code))
        completed=[m['id'] for m in ready if (ART/'results'/m['id']/'complete.json').exists()]
        failed=[m['id'] for m in ready if attempts.get(m['id'],0)>config['max_retries'] and m['id'] not in completed and m['id'] not in active]
        pending=[m for m in ready if m['id'] not in completed and m['id'] not in active and m['id'] not in failed]
        if time.time()-last_render>300 or (not pending and not active):
            from render import render
            render();last_render=time.time()
        beats=[]
        for mid in active:
            p=ART/'results'/mid/'heartbeat.json'
            if p.exists():beats.append(read(p))
        speed=sum(b['rows_per_second'] for b in beats)
        partial=sum(b['committed_n'] for b in beats)
        remaining=(len(ready)-len(completed))*total_n-partial
        atomic(ART/'status.json',dict(**identity(),status='running',models=len(models),runnable=len(ready),completed=len(completed),failed=failed,
                    columns=len(datasets),nets_per_model=total_n,total_prediction_rows=len(ready)*total_n,
                    active=[dict(model=mid,pid=v[0].pid,gpu=v[1]) for mid,v in active.items()],
                    measured_rows_per_second=speed,estimated_remaining_seconds=remaining/speed if speed>0 else None,
                    estimate_note='current active worker throughput; dataset degree, native scorer and shared GPU load may change substantially',
                    elapsed_seconds=time.time()-start))
        if not pending and not active:break
        usage=Counter(v[1] for v in active.values())
        if pending and len(active)<config['max_workers'] and shutil.disk_usage(ART).free>config['disk_reserve_gib']*2**30:
            try:gpus=sorted(gpu_status(),key=lambda g:(usage[g['uuid']],g['util'],-g['free']))
            except (subprocess.SubprocessError,OSError) as exc:
                event(ART/'queue_events.jsonl',dict(time=time.time(),event='resource_query_unavailable',error=repr(exc)))
                time.sleep(config['poll_seconds']);continue
            for gpu in gpus:
                if not pending or len(active)>=config['max_workers']:break
                if usage[gpu['uuid']]>=config['max_workers_per_gpu'] or gpu['free']<config['min_free_mib']:continue
                if usage[gpu['uuid']]==0 and gpu['util']>config['foreign_util_limit']:continue
                m=pending.pop(0);mid=m['id'];folder=ART/'results'/mid;folder.mkdir(parents=True,exist_ok=True)
                # Respect an orphaned worker from an earlier supervisor.
                with (folder/'run.lock').open('a+') as probe:
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                log=(ART/'logs'/f'{mid}.log').open('a')
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
                proc=subprocess.Popen([PYTHON,'-u',str(ROOT/'evaluate.py'),'--model',mid,'--gpu',gpu['uuid'],'--batch',str(protocol['batch_size'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[mid]=(proc,gpu['uuid'],log);usage[gpu['uuid']]+=1
                attempts[mid]=attempts.get(mid,0)+1;atomic(ART/'attempts.json',attempts)
                event(ART/'queue_events.jsonl',dict(time=time.time(),event='worker_start',model=mid,pid=proc.pid,gpu=gpu,batch=protocol['batch_size']))
                print(json.dumps(dict(event='worker_start',model=mid,pid=proc.pid,gpu=gpu['index'])),flush=True)
        time.sleep(config['poll_seconds'])
    if failed or any(m['status']=='catalog_error' for m in models):
        atomic(ART/'completion.json',dict(status='INCOMPLETE',completed=len(completed),runnable=len(ready),failed=failed,
                    reason='failed or unreadable model weights; same table retains explicit gaps',time=time.time()))
        return 2
    # Required folder organization is the last authorized step after inference.
    from organize import organize
    organize(protocol['layout_after_evaluation'])
    from render import render
    result=render()
    atomic(ART/'completion.json',dict(status='COMPLETE',**result,time=time.time()))
    return 0

if __name__=='__main__':raise SystemExit(run())
