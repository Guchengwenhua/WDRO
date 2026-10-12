"""Read-only process/lock verification before publishing final delivery status."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OPS=ROOT/'artifacts/operations'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def live(identity):
    try:
        x=Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        return x[0]!='Z' and x[19]==str(identity['start_ticks']) and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==identity['boot_id']
    except (FileNotFoundError,KeyError):return False
def unlocked(p):
    with p.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        except BlockingIOError:return False
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue-exit',type=int,required=True);a=parser.parse_args()
    queue=read(OPS/'queue.json');audit=read(ROOT/'artifacts/report/audit.json');errors=[]
    if a.queue_exit:errors.append(f'Queue exit {a.queue_exit}')
    if queue.get('status')!='complete':errors.append('Queue is not complete')
    if audit.get('status')!='PASS':errors.append('Independent scientific audit did not PASS')
    if queue.get('running'):errors.append('Queue still records running workers')
    for p in (ROOT/'artifacts/runs').glob('*/run.json'):
        r=read(p)
        if live(r.get('identity',{})):errors.append('Live worker: '+p.parent.name)
        if not unlocked(p.parent/'run.lock'):errors.append('Held worker lock: '+p.parent.name)
    if not unlocked(OPS/'scheduler.lock'):errors.append('Scheduler lock held')
    for name,expected in audit.get('report_artifacts_sha256',{}).items():
        p=ROOT/'artifacts/report'/name
        if not p.exists() or digest(p)!=expected:errors.append('Report hash mismatch: '+name)
    result={'status':'COMPLETE' if not errors else 'NEEDS_REVIEW','time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'queue_exit':a.queue_exit,'errors':errors,'queue_sha256':digest(OPS/'queue.json') if (OPS/'queue.json').exists() else None,
            'audit_sha256':digest(ROOT/'artifacts/report/audit.json') if (ROOT/'artifacts/report/audit.json').exists() else None,
            'scientific_scope':'exploratory degree30,3seeds; no optimality or general robustness certificate'}
    tmp=OPS/'delivery.partial';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));os.replace(tmp,OPS/'delivery.json')
    text='# 当前状态\n\n状态：'+result['status']+'。更新时间：'+result['time']+'。\n\n'
    if not errors:text+='51项GPU任务、独立统计审计和进程/锁退出核对均通过。结论见 [最终报告](artifacts/report/REPORT.md)。\n'
    else:text+='自动运行未获完整验收，不能标记DONE。详情见 [交付状态](artifacts/operations/delivery.json) 与 [队列](artifacts/operations/queue.json)。\n'
    (ROOT/'STATUS.md').write_text(text)
    print(json.dumps(result,ensure_ascii=False));return 0 if not errors else 1
if __name__=='__main__':raise SystemExit(main())
