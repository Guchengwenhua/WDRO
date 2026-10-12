"""Persistent dynamic multi-GPU evaluation, preserving all frozen model tasks."""
import argparse
from collections import Counter
import shlex
import subprocess
import time

from common import *

class ExistingProcess:
    """Observe a surviving child after controller interruption, without signals."""
    def __init__(self, identity):
        self.pid=identity['pid']
        self.start_ticks=str(identity['start_ticks'])
    def poll(self):
        try:
            fields=Path(f'/proc/{self.pid}/stat').read_text().rsplit(') ',1)[1].split()
            return None if fields[0] not in ('Z','X') and fields[19]==self.start_ticks else 0
        except FileNotFoundError:
            return 0

def existing_writer(task):
    output=Path(task['output'])
    lock=output/'run.lock'
    if not lock.exists():
        return None
    with lock.open('a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            job=read(output/'job.json')
            process=ExistingProcess(job['process'])
            assert process.poll() is None, 'locked writer lacks matching live process identity'
            return dict(process=process,task=task,gpu=job['physical_gpu'],start=time.monotonic(),log=None)
        else:
            fcntl.flock(f,fcntl.LOCK_UN)
            return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--slots-per-gpu',type=int,default=4)
    args=parser.parse_args()
    with lease('queue'):
        pf, resource=preflight('new public design external evaluation; frozen 75 models; batch 256')
        inputs=read(BASE/'cache/input_manifest.json')
        refs=read(BASE/'cache/reference_manifest.json')
        assert refs['status']=='pass' and refs['input_manifest']['sha256']==sha(BASE/'cache/input_manifest.json')
        index=Path(inputs['model_index']['logical_path'])
        models=read(index)['models']
        groups={g['degree']:g for g in inputs['groups']}
        refgroups={g['degree']:g for g in refs['groups']}
        pending=[]
        completed=[]
        for m in models:
            if m['degree'] not in groups:
                continue
            task=dict(method=m['method'],degree=m['degree'],seed=m['seed'])
            task['id']=f"{m['method']}_d{m['degree']}_s{m['seed']}"
            task['output']=str(BASE/'results'/task['id'])
            done=Path(task['output'])/'result.json'
            if done.exists():
                r=read(done)
                assert r['status']=='complete'
                # evaluate.py revalidates the full binding when resumed; never trust a status alone.
            pending.append(task)
        pending.sort(key=lambda t:(-t['degree'],t['seed'],t['method']))
        write(BASE/'logs/queue/task_plan.json',dict(at=now(),input=artifact(BASE/'cache/input_manifest.json'),reference=artifact(BASE/'cache/reference_manifest.json'),
            implementation=artifact(__file__),evaluator=artifact(BASE/'evaluate.py'),preflight=artifact(pf),tasks=pending,
            slots_per_gpu=args.slots_per_gpu,admission='All GPUs sampled dynamically; skip persistently saturated external GPUs; minimum 3072MiB free plus 1536MiB per newly admitted unobserved task.'))
        active=[]
        history_path=BASE/'logs/queue/history.json'
        history=read(history_path) if history_path.exists() else []
        attempts=Counter()
        for row in history:
            if 'attempt' in row:
                attempts[row['task']['id']]=max(attempts[row['task']['id']],row['attempt'])
        for task in pending[:]:
            writer=existing_writer(task)
            if writer:
                active.append(writer)
                pending.remove(task)
                history.append(dict(task=task,pid=writer['process'].pid,gpu=writer['gpu'],adopted_at=now()))
        failed=[]
        utilization_history={g['index']:[s['gpus'][g['index']]['utilization'] for s in resource['gpus']] for g in resource['gpus'][-1]['gpus']}
        begin=time.monotonic()
        while pending or active:
            for item in active[:]:
                code=item['process'].poll()
                if code is None:
                    continue
                if item['log']:
                    item['log'].close()
                task=item['task']
                result=Path(task['output'])/'result.json'
                row=dict(task=task, gpu=item['gpu'],pid=item['process'].pid, exit_code=code,finished=now(),seconds=time.monotonic()-item['start'])
                if isinstance(item['process'], ExistingProcess):
                    row['adopted_exit_status']='unknown; revalidate with evaluator before accepting'
                    pending.append(task)
                elif code==0 and result.exists() and read(result)['status']=='complete':
                    row['result']=artifact(result)
                    completed.append(row)
                elif attempts[task['id']] < 3:
                    pending.append(task)
                else:
                    failed.append(row)
                history.append(row)
                active.remove(item)
            snapshots=gpu_snapshot()
            for g in snapshots:
                utilization_history.setdefault(g['index'], []).append(g['utilization'])
                utilization_history[g['index']]=utilization_history[g['index']][-5:]
            counts=Counter(x['gpu'] for x in active)
            launches=Counter()
            for gpu in sorted(snapshots,key=lambda g:(counts[g['index']],-g['free_mib'])):
                idx=gpu['index']
                if gpu['free_mib']<4608 or (counts[idx]==0 and max(utilization_history[idx])>=85):
                    continue
                slots=min(args.slots_per_gpu-counts[idx],max(0,(gpu['free_mib']-3072)//1536))
                for _ in range(slots):
                    if not pending:
                        break
                    task=pending.pop(0)
                    attempts[task['id']]+=1
                    g=groups[task['degree']]
                    r=refgroups[task['degree']]
                    command=['bash',str(ENV),PYTHON,str(BASE/'evaluate.py'),
                        '--input',g['input']['logical_path'],'--reference',r['reference']['logical_path'],
                        '--model-index',str(index),'--method',task['method'],'--degree',str(task['degree']),
                        '--seed',str(task['seed']),'--gpu',str(idx),'--output',task['output'],'--batch-size','256']
                    logpath=BASE/'logs/queue'/f"{task['id']}.attempt{attempts[task['id']]}.log"
                    logpath.parent.mkdir(parents=True,exist_ok=True)
                    log=logpath.open('a')
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=BASE)
                    item=dict(process=proc,task=task,gpu=idx,start=time.monotonic(),log=log)
                    active.append(item)
                    launches[idx]+=1
                    history.append(dict(task=task,gpu=idx,pid=proc.pid,started=now(),attempt=attempts[task['id']],command=shlex.join(command),log=str(logpath),admission_snapshot=gpu))
            state=dict(status='running',at=now(),pid=os.getpid(),completed=len(completed),pending=len(pending),failed=len(failed),
                active=[dict(task=x['task']['id'],gpu=x['gpu'],pid=x['process'].pid) for x in active],gpus=snapshots,
                wall_seconds=time.monotonic()-begin)
            write(BASE/'logs/queue/heartbeat.json',state)
            write(BASE/'logs/queue/history.json',history)
            with (BASE/'logs/queue/resources.jsonl').open('a') as f:
                f.write(json.dumps(state)+'\n')
            if pending or active:
                time.sleep(3)
        result=dict(status='pass' if not failed else 'failed',at=now(),completed_tasks=len(completed),expected_tasks=inputs['expected_model_tasks'],
            completed=completed,failed=failed,wall_seconds=time.monotonic()-begin,preflight=artifact(pf),history=artifact(BASE/'logs/queue/history.json'))
        assert len(completed)==inputs['expected_model_tasks'] and not failed,result
        write(BASE/'logs/queue/completion.json',result)
        write(BASE/'logs/queue/heartbeat.json',dict(status='complete',at=now(),completed=len(completed),active=[],pending=0,failed=0))
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
