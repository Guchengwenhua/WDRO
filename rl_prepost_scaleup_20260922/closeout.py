import argparse,json,time
from pathlib import Path
import ops as o
import scheduler as s
ROOT=Path(__file__).resolve().parent;OPS=ROOT/'artifacts/operations'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue-exit',type=int,required=True);arg=parser.parse_args();errors=[]
    q=s.read(OPS/'queue.json',{});audit=s.read(ROOT/'artifacts/report/audit.json',{})
    if arg.queue_exit or q.get('status')!='complete':errors.append('Queue did not complete')
    if audit.get('status')!='PASS':errors.append('Independent numerical/statistical audit did not pass')
    if q.get('running') or o.discover_live_workers(ROOT):errors.append('Live worker remains')
    if o.lock_held(OPS/'scheduler.lock'):errors.append('Scheduler lock remains held')
    for t in s.config()['tasks']:
        if o.lock_held(ROOT/'artifacts/runs'/t['id']/'run.lock'):errors.append('Held worker lock '+t['id'])
    for n,digest in audit.get('artifact_sha256',{}).items():
        if o.sha256(ROOT/'artifacts/report'/n)!=digest:errors.append('Report hash mismatch '+n)
    result=dict(status='COMPLETE' if not errors else 'NEEDS_REVIEW',time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        errors=errors,queue_exit=arg.queue_exit,queue_sha256=o.sha256(OPS/'queue.json'),audit_sha256=o.sha256(ROOT/'artifacts/report/audit.json') if audit else None,
        scope='Protocol completion is separate from evidence of performance improvement')
    o.atomic(OPS/'delivery.json',result)
    (ROOT/'docs/STATUS.md').write_text('# 当前状态\n\n'+result['time_utc']+'：'+result['status']+'。\n\n详见 [交付状态](../artifacts/operations/delivery.json) 与最终 artifacts/report/REPORT.md。\n')
    print(json.dumps(result));return 0 if not errors else 1
if __name__=='__main__':raise SystemExit(main())
