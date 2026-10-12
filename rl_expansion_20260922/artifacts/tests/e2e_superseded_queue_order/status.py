"""Compact current lifecycle, progress and measured range (read only)."""
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent
def read(p,default=None):
    try: return json.loads(p.read_text())
    except FileNotFoundError: return default
def main():
    life=read(ROOT/'artifacts/lifecycle.json',{'status':'not_started'}); stage=life.get('stage')
    q=read(ROOT/f'artifacts/operations/{stage}/queue.json',{}) if stage else {}
    running=[]; rates=[]
    for row in q.get('running',[]):
        h=read(ROOT/f"artifacts/runs/{row['task_id']}/heartbeat.json",{})
        t=read(ROOT/f"artifacts/tasks/{row['task_id']}.json",{})
        running.append(dict(id=row['task_id'],gpu=row['gpu'],step=h.get('step'),target=t.get('target'),
                            seconds_per_update=h.get('seconds_per_update'),updated=h.get('updated')))
        if h.get('seconds_per_update'): rates.append(h['seconds_per_update'])
    remaining=0
    for key in q.get('pending',[]):
        t=read(ROOT/f'artifacts/tasks/{key}.json',{}); remaining+=t.get('target',0)-t.get('start',0)
    for row in running: remaining+=max(0,(row['target'] or 0)-(row['step'] or 0))
    result=dict(lifecycle=life,queue_status=q.get('status'),completed=len(q.get('completed',{})),
                pending=len(q.get('pending',[])),running=running,failures=q.get('failures',{}),
                current_stage_remaining_updates=remaining,
                stage_hours_at_current_worker_count=(remaining*sum(rates)/len(rates)/max(1,len(running))/3600) if rates else None,
                note='ETA excludes changing shared-GPU contention; later stages depend on validation-only promotion.')
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
