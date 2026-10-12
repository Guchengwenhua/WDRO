"""Audit the first complete ERM trajectory without claiming the batch passed."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import utc_now
from precision_replay_pipeline import cpu_preflight,progress

def main():
    queue=json.loads((EXP/'logs/stage2_queue/stage2_erm_formal_v1/queue.json').read_text())
    run=next(t['run_id'] for t in queue['tasks'] if t['status']=='complete')
    target=EXP/'logs/precision_first_erm';target.mkdir(parents=True,exist_ok=True)
    pre=cpu_preflight('first_complete_erm_full_stream')
    cmd=[sys.executable,str(ROOT/'scripts/audit_stage2_formal.py'),'--smoke-run',run,'--host-runtime-manifest',str(pre),
        '--output',str(target/'raw_audit.json'),'--table',str(target/'validation.csv')]
    with (target/'audit.log').open('a') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
    raw=json.loads((target/'raw_audit.json').read_text())
    assert raw['audit_scope']=='one_run_smoke' and len(raw['run_checks'])==1
    assert all(all(v.values()) for v in raw['run_checks'].values()),raw['run_checks']
    assert all(v for k,v in raw['checks'].items() if k!='queue_complete'),raw['checks']
    assert r.returncode in (0,2)
    output=dict(status='pass',at_utc=utc_now(),scope='one completed formal ERM; not batch acceptance',run_id=run,
        run_checks=raw['run_checks'],queue_complete=raw['checks']['queue_complete'],batch_acceptance=False,
        audit=str(target/'raw_audit.json'),audit_sha256=file_sha256(target/'raw_audit.json'),preflight=str(pre),command=cmd,
        raw_nonpass_reason='queue_complete gate is expected to remain false while the other fourteen runs execute' if not raw['checks']['queue_complete'] else None)
    atomic_write_json(target/'acceptance.json',output)
    progress('首个纠错ERM的完整训练流/选模/账本审计通过（仅1 run，整批仍待15-run验收）',str(target/'acceptance.json'),True)
    print(json.dumps(dict(status='pass',run=run,run_checks=sum(len(v) for v in raw['run_checks'].values()),batch_acceptance=False)))
if __name__=='__main__':main()
