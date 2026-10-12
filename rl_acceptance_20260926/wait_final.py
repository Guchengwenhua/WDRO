"""Server-side acceptance continuation after the existing experiment closes."""
import json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent/'rl_prepost_scaleup_20260922'
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def write(value):
    value['time_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    p=ROOT/'final_watch.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def main():
    while True:
        q=read(EXP/'artifacts/operations/queue.json');d=read(EXP/'artifacts/operations/delivery.json')
        if d.get('status')=='COMPLETE':break
        if d.get('status')=='NEEDS_REVIEW' or q.get('status') in ['failed','audit_failed','invalid_completion']:
            write(dict(status='NEEDS_REVIEW',queue=q.get('status'),delivery=d));return 1
        write(dict(status='WAITING_EXPERIMENT_COMPLETION',completed=len(q.get('complete',[])),total=104,queue=q.get('status')))
        time.sleep(30)
    for name,digest in read(ROOT/'review_source_freeze.json')['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Acceptance source changed: '+name)
    write(dict(status='RUNNING_FULL_REACCEPTANCE',completed=104,total=104))
    with (ROOT/'final_review.log').open('a') as f:
        result=subprocess.run(['nice','-n','10','bash',str(EXP/'run_env.sh'),str(ROOT/'review.py'),'new','--output-dir',str(ROOT/'final')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        write(dict(status='NEEDS_REVIEW',returncode=result.returncode));return result.returncode
    result=subprocess.run(['bash',str(EXP/'run_env.sh'),str(ROOT/'summarize.py'),'--final'],cwd=ROOT)
    verdict=read(ROOT/'STATUS.json')
    write(dict(status='COMPLETE' if verdict.get('status')=='PASS' else 'NEEDS_REVIEW',acceptance=verdict))
    return result.returncode
if __name__=='__main__':raise SystemExit(main())
