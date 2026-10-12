"""Recoverable dynamic GPU admission; never signal unrelated processes."""
from collections import Counter
import fcntl
import shlex
import signal
import socket
from common import *

class ExistingProcess:
    def __init__(self, identity):
        self.pid=identity['pid'];self.start_ticks=str(identity['start_ticks'])
    def poll(self):
        try:
            fields=Path(f'/proc/{self.pid}/stat').read_text().rsplit(') ',1)[1].split()
            return None if fields[0] not in ('Z','X') and fields[19]==self.start_ticks else 0
        except FileNotFoundError:return 0

def adopt(model):
    directory=OUT/'results'/task_id(model)
    if not (directory/'run.lock').exists():return None
    with (directory/'run.lock').open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            job=read(directory/'job.json');proc=ExistingProcess(job['process'])
            ev.require(proc.poll() is None,'locked worker lacks matching live PID/start time')
            return dict(process=proc,model=model,gpu=job['physical_gpu'],start=time.monotonic(),log=None)
        else:fcntl.flock(stream,fcntl.LOCK_UN)

def main():
    logdir=OUT/'logs/queue';logdir.mkdir(parents=True,exist_ok=True)
    with ev.Lease(logdir,-1) as lease:
        frozen=read(OUT/'models_frozen.json');inputs=read(OUT/'input_manifest.json')
        pending=list(frozen['models']);active=[];completed=[];failures=[];attempts=Counter()
        for model in pending[:]:
            adopted=adopt(model)
            if adopted:active.append(adopted);pending.remove(model)
        snapshots=[gpu_snapshot()]
        for _ in range(2):time.sleep(1);snapshots.append(gpu_snapshot())
        utilization_history={g['index']:[next(x['utilization'] for x in snapshot if x['index']==g['index'])
                            for snapshot in snapshots] for g in snapshots[-1]}
        import shutil
        preflight=dict(at=now(),gpus=snapshots,host=socket.gethostname(),cpu_count=os.cpu_count(),
            loadavg=os.getloadavg(),disk=dict(zip(('total','used','free'),shutil.disk_usage(OUT))),
            output_logical=str(OUT),output_resolved=str(OUT.resolve()),model_index=artifact(OUT/'models_frozen.json'),
            input_manifest=artifact(OUT/'input_manifest.json'),slots_per_gpu=8,max_workers=32,
            admission='Every physical GPU considered. Keep 3072 MiB free plus 1536 MiB per admitted process; up to 8 workers/card and 32 total. Skip cards with <4608 MiB free or >=85% recent utilization when none of our workers owns that card.')
        write(logdir/'preflight.json',preflight)
        history=logdir/'events.jsonl';begin=time.monotonic()
        while pending or active:
            for item in active[:]:
                code=item['process'].poll()
                if code is None:continue
                if item['log']:item['log'].close()
                model=item['model'];result_path=OUT/'results'/task_id(model)/'result.json'
                event=dict(event='exit',at=now(),task=task_id(model),gpu=item['gpu'],pid=item['process'].pid,exit_code=code)
                if isinstance(item['process'],ExistingProcess):
                    # Exit status is unknown after controller recovery: revalidate through worker.
                    pending.append(model);event['action']='idempotent_revalidation'
                elif code==0 and result_path.exists():
                    result=read(result_path)
                    ev.require(result['status']=='complete' and result['completed_nets']==inputs['count'],'incomplete successful worker')
                    ev.require(result['binding']['model']==model,'worker model differs')
                    completed.append(dict(task=task_id(model),result=artifact(result_path)))
                    event['action']='complete'
                elif attempts[task_id(model)]<3:
                    pending.append(model);event['action']='retry_committed_cursor'
                else:
                    failures.append(event);event['action']='failed_after_three_attempts'
                ev.append_event(history,event);active.remove(item)
            gpus=gpu_snapshot();counts=Counter(x['gpu'] for x in active)
            for gpu in gpus:
                utilization_history.setdefault(gpu['index'],[]).append(gpu['utilization'])
                utilization_history[gpu['index']]=utilization_history[gpu['index']][-5:]
            capacity=32-len(active)
            # Balance each wave across available devices; reserve unobserved launches.
            candidates={g['index']:dict(g,slots=(0 if counts[g['index']]==0 and max(utilization_history[g['index']])>=85
                        else min(8-counts[g['index']],max(0,(g['free_mib']-3072)//1536)))) for g in gpus}
            while pending and capacity>0:
                choices=[g for g in candidates.values() if g['slots']>0]
                if not choices:break
                gpu=min(choices,key=lambda g:(counts[g['index']],-g['free_mib']))
                idx=gpu['index'];model=pending.pop(0);tid=task_id(model);attempts[tid]+=1
                command=worker_command(model,idx);logpath=logdir/f'{tid}.{time.time_ns()}.log'
                log=logpath.open('a')
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=BASE,start_new_session=True)
                active.append(dict(process=proc,model=model,gpu=idx,start=time.monotonic(),log=log))
                ev.append_event(history,dict(event='launch',at=now(),task=tid,pid=proc.pid,gpu=idx,
                    command=shlex.join(command),attempt=attempts[tid],admission_snapshot=gpu.copy(),log=str(logpath)))
                gpu['slots']-=1;counts[idx]+=1;capacity-=1
            rows=[]
            for item in active:
                heartbeat=OUT/'results'/task_id(item['model'])/'heartbeat.json'
                progress=read(heartbeat) if heartbeat.exists() else {}
                rows.append(dict(task=task_id(item['model']),gpu=item['gpu'],pid=item['process'].pid,
                    completed_nets=progress.get('completed_nets',0),phase=progress.get('status','starting')))
            state=dict(status='running',at=now(),process=ev.pid_identity(),completed=len(completed),expected=33,
                       active=rows,pending=len(pending),failed=len(failures),gpus=gpus,wall_seconds=time.monotonic()-begin)
            write(logdir/'state.json',state);ev.append_event(logdir/'resources.jsonl',state)
            lease.update(len(completed),'running',active=len(active),pending=len(pending))
            if pending or active:time.sleep(3)
        result=dict(status='pass' if not failures else 'failed',at=now(),completed=completed,failed=failures,
                    expected_tasks=33,completed_tasks=len(completed),preflight=artifact(logdir/'preflight.json'))
        write(logdir/'completion.json',result)
        ev.require(not failures and len(completed)==33,'queue incomplete; inspect completion.json and worker logs')
        write(logdir/'state.json',dict(status='complete',at=now(),completed=33,active=[],pending=0,failed=0))
        lease.update(33,'complete')
        print('All 33 model tasks complete.',flush=True)

if __name__=='__main__':main()
