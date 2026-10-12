"""Compact read-only progress; no GPU/model initialization."""
import json
from pathlib import Path
import statistics
import time
ROOT=Path(__file__).resolve().parent
def read(p,default=None):
    try: return json.loads(p.read_text())
    except (OSError,json.JSONDecodeError): return default

def main():
    q=read(ROOT/'artifacts/operations/queue.json',{})
    p=read(ROOT/'protocol.json'); completed=q.get('completed',{}); rows=[]; rates=[]; remaining=0
    for arm in p['arms']:
        for seed in p['seeds']:
            key=f'{arm}_s{seed}'; h=read(ROOT/'artifacts/runs'/key/'heartbeat.json',{})
            step=h.get('step',p['start_steps'][arm])
            left=0 if key in completed else max(0,p['targets'][arm]-step); remaining+=left
            if h.get('seconds_per_update'): rates.append(h['seconds_per_update'])
            rows.append(dict(task=key,status='complete' if key in completed else h.get('status','pending'),
                             step=step,target=p['targets'][arm],gpu=h.get('physical_gpu_index'),
                             seconds_per_update=h.get('seconds_per_update'),heartbeat=h.get('updated')))
    rate=statistics.median(rates) if rates else None
    result=dict(at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status=q.get('status','not_started'),
        training_completed=sum(k in completed for k in [r['task'] for r in rows]),
        evaluation_completed=sum(k.startswith('eval_') for k in completed),workers=len(q.get('running',[])),
        remaining_training_updates=remaining,median_seconds_per_update=rate,
        remaining_gpu_training_hours=remaining*rate/3600 if rate else None,
        eta_note='GPU hours exclude validation/evaluation and resource wait; dependency critical path constrains wall time.',
        tasks=rows,failures=q.get('failures',{}),report_complete=(ROOT/'artifacts/report/complete.json').exists())
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
