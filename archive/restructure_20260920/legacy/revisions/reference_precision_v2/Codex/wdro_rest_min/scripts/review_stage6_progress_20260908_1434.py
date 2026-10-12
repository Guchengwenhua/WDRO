"""Re-read the seven immutable R5 acceptances and capture live continuation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from precision_stage5_completed_batch import artifact, verified, check_files
from precision_replay_pipeline import cpu_preflight, progress
from precision_queue_runtime import process_matches
import precision_stage5_capacity_pipeline as pipeline
from src.io_utils import atomic_write_json
from src.job_control import utc_now


def main():
    pipeline.gate()
    exp = ROOT.parent / 'experiments'
    job = exp / 'logs/stage6_remaining_20260908_1434'
    job.mkdir(parents=True, exist_ok=True)
    batch = exp / 'logs/stage5_completed_batches/stage6_20260908_1023'
    plan = json.loads((batch / 'plan.json').read_text())
    saved = json.loads((batch / 'result.json').read_text())
    check_files(plan['files'])
    assert saved['status'] == 'pass' and saved['plan'] == artifact(batch / 'plan.json')
    records = [verified(task) for task in plan['tasks']]
    # The completed batch records completion order; the plan records task order.
    assert len(records) == len(saved['runs']) == 7
    assert {r['run_id']: r for r in records} == {r['run_id']: r for r in saved['runs']}
    assert saved['full_audit_checks'] == 308 and saved['validation_nets'] == 56000
    preflight = cpu_preflight('stage6_20260908_1434_seven_completed_review')
    queue = json.loads((exp / 'logs/stage5_pipeline/queue.json').read_text())
    active = []
    for row in queue['active']:
        rid = row['task']['run_id']
        assert process_matches(row['pid'], rid, ROOT / 'scripts/train_stage5.py')
        heartbeat = json.loads((exp / 'logs/jobs' / rid / 'heartbeat.json').read_text())
        active.append(dict(run_id=rid, pid=row['pid'], gpu=row['gpu'], heartbeat=heartbeat,
            training_log=str(exp / 'logs/stage5' / rid / 'events.jsonl')))
    assert len(active) == 2
    proof = dict(status='pass', at_utc=utc_now(), source=artifact(Path(__file__)),
        batch=artifact(batch / 'result.json'), runs=records, preflight=artifact(preflight),
        active=active, full_audits=7, full_audit_checks=308, best_validation_runs=7,
        validation_nets=56000, stage6='DOING', test_ispd_used=False,
        research_protocol_override=False)
    output = job / ('seven_completed_review_' + str(__import__('time').time_ns()) + '.json')
    atomic_write_json(output, proof, mode=0o444)
    progress('R5已完成7/9个run的完整审计308项及56000-net best validation独立复算；两项64-call训练继续，等待其完整验收与配置冻结', str(output), True)
    print(json.dumps(dict(status='pass', proof=str(output), active=[dict(run_id=r['run_id'],
        pid=r['pid'], gpu=r['gpu'], update=r['heartbeat']['completed_updates']) for r in active])), flush=True)


if __name__ == '__main__':
    main()
