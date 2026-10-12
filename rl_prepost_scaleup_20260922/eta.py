"""Dependency-aware ETA using formal prior rates and current progress."""
import argparse,heapq,json,statistics,time
from pathlib import Path
import ops as o
ROOT=Path(__file__).resolve().parent;OPS=ROOT/'artifacts/operations'
def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def rows(path):
    try:return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    except (FileNotFoundError,json.JSONDecodeError):return []
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    p=read(ROOT/'protocol.json');origin=Path(p['origin_root']);rates={};evaluations={}
    for out in (origin/'artifacts/runs').glob('*'):
        c=read(out/'complete.json');log=rows(out/'progress.jsonl')
        if not c or len(log)<100:continue
        key=c['kind']+'_'+c['method'];a,b=log[-129],log[-1]
        rates.setdefault(key,[]).append((b['elapsed_seconds']-a['elapsed_seconds'])/(b['step']-a['step']))
        evaluations.setdefault(key,[]).append(max(0,c['elapsed_seconds']-log[-1]['elapsed_seconds']))
    while True:
        queue=read(OPS/'queue.json',{});completed=set(queue.get('complete',[]));finish={};workers=[0.]*p['resources']['max_workers'];heapq.heapify(workers)
        details=[];current_rates={}
        for task in p['tasks']:
            key=task['kind']+'_'+task['method'];out=ROOT/'artifacts/runs'/task['id'];log=rows(out/'progress.jsonl')
            if len(log)>=20:
                deltas=[(b['elapsed_seconds']-a['elapsed_seconds'])/(b['step']-a['step']) for a,b in zip(log[-65:-1],log[-64:]) if a['step'] not in task['milestones'] and b['step']>a['step']]
                rate=statistics.median(deltas) if deltas else statistics.median(rates.get(key,[3.]))
            else:rate=statistics.median(rates.get(key,[.8 if task['kind']=='warmup' else 4.]))
            current_rates[key]=rate
            if task['id'] in completed:finish[task['id']]=0.;continue
            heartbeat=read(out/'heartbeat.json',{});step=heartbeat.get('step',task['start'])
            unevaluated=sum(not (out/'checkpoints'/f'step_{x}.json').exists() for x in task['milestones'])
            eval_sec=statistics.median(evaluations.get(key,[90.]))
            if task['kind']=='warmup':eval_sec=90.
            remain=max(0,task['end']-step)*rate+unevaluated*eval_sec
            parent_end=max([finish[v['run']] for v in task['parents'].values() if v['source']=='new'] or [0.])
            start=max(heapq.heappop(workers),parent_end);end=start+remain;heapq.heappush(workers,end);finish[task['id']]=end
            details.append(dict(task=task['id'],step=step,end_step=task['end'],seconds_per_update=rate,remaining_seconds=remain))
        oldq=read(origin/'artifacts/operations/queue.json',{});olde=read(origin/'artifacts/operations/eta.json',{});oldd=read(origin/'artifacts/operations/delivery.json',{})
        wait=0.
        if oldd.get('status')!='COMPLETE':
            # Prior stage has no remaining warmup/fixed dependencies. Simulate
            # the tail, including current long sequential jobs explicitly.
            remain={x['run_id']:x['estimated_remaining_seconds'] for x in olde.get('tasks',[])}
            slots=[remain.pop(rid,0.) for rid in oldq.get('running',{})]
            slots += [0.]*(8-len(slots));heapq.heapify(slots)
            for sec in sorted(remain.values(),reverse=True):heapq.heappush(slots,heapq.heappop(slots)+sec)
            wait=max(slots or [0])+1800
        left=max(workers or [0])+1800
        if len(completed)==len(p['tasks']):left=0 if queue.get('status')=='complete' else 1800
        hours=(left+wait)/3600
        result=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),queue_status=queue.get('status','not_started'),
            completed=len(completed),total=len(p['tasks']),origin_completed=len(oldq.get('complete',[])),origin_wait_hours=wait/3600,
            new_experiment_hours=left/3600,estimated_remaining_hours=hours,range_hours=[hours*.75,hours*1.6],
            rates=current_rates,tasks=details,note='Dependency and tail scheduling estimate; assumes eight available GPUs after origin acceptance. Contention and resource waits can exceed the range.')
        o.atomic(OPS/'eta.json',result)
        if not args.watch or (OPS/'delivery.json').exists():return
        time.sleep(30)
if __name__=='__main__':main()
