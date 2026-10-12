"""Recompute validation for already completed and fully audited Stage5 runs."""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent / 'experiments'
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import JobLease, utc_now
from src.rest import file_sha256
import precision_stage5_capacity_pipeline as pipeline
import precision_queue_runtime as runtime


def main():
    pipeline.gate()
    job = EXP / 'logs/stage6_remaining_20260908/completed_validation'
    tasks = json.loads((pipeline.JOB / 'all_tasks.json').read_text())['tasks']
    ready = []
    for t in tasks:
        rid = t['run_id']; rp = EXP / 'logs/stage5' / rid / 'result.json'
        ap = EXP / 'logs/stage5_audit/runs' / f'{rid}.json'
        if not rp.exists() or not ap.exists(): continue
        r = json.loads(rp.read_text()); a = json.loads(ap.read_text())
        if r['status'] != 'complete': continue
        assert r['completed_updates'] == 2000 and a['status'] == 'pass' and all(a['checks'].values())
        assert a['binding']['result_sha256'] == file_sha256(rp)
        ready.append(t)
    exact = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__))])
    records = []
    with JobLease(job, run_id='completed_stage5_validation', exact_command=exact, resume_command=exact,
                  monitor_command=f'cat {job}/heartbeat.json', expected_updates=len(ready),
                  physical_gpu_index=None, host_runtime_manifest=None) as lease:
        for task in ready:
            pipeline.gate(); rid = task['run_id']; output = EXP / 'logs/stage5_best_validation' / f'{rid}.json'
            if not output.exists():
                host = pipeline.preflight('completed_stage5_validation_' + rid)
                assert host, 'capacity unavailable; resume this stateless verification after a new preflight'
                h = json.loads(Path(host['manifest']).read_text())
                q = json.loads((pipeline.JOB / 'queue.json').read_text())
                active = {a['task']['run_id']: a for a in q['active']}
                blocked = {int(k): v for k, v in q['blocked_until'].items()}
                gpu = runtime.choose_gpu(h['gpu_decisions'], active, [], blocked, 4702, time.time())
                assert gpu is not None, 'capacity currently reserved by training; keep completed outputs'
                cmd = [sys.executable, str(ROOT / 'scripts/verify_stage5_best_validation.py'),
                       '--run-id', rid, '--gpu', str(gpu), '--preflight', host['manifest']]
                log_path = job / (rid + '.log')
                with log_path.open('a') as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                launch = dict(at_utc=utc_now(), run_id=rid, pid=proc.pid, gpu=gpu, command=cmd,
                    exact_command=shlex.join(cmd), preflight=host, log=str(log_path), log_resolved=str(log_path.resolve()))
                append_jsonl(job / 'launches.jsonl', launch)
                try:
                    while proc.poll() is None:
                        append_jsonl(job / 'resources.jsonl', dict(pipeline.observe(), run_id=rid, pid=proc.pid))
                        lease.update(update=len(records), status='validation_recompute', detail=launch)
                        time.sleep(5)
                    assert proc.returncode == 0, f'validation recomputation failed: {log_path}'
                except BaseException:
                    if proc.poll() is None: proc.terminate(); proc.wait()
                    j = json.loads((pipeline.JOB / 'job.json').read_text()); pid = j['pid']; p = Path(f'/proc/{pid}/cmdline')
                    if runtime.process_is_alive(pid) and p.exists() and str(ROOT / 'scripts/precision_stage5_capacity_pipeline.py').encode() in p.read_bytes().split(b'\0'):
                        os.kill(pid, signal.SIGTERM)
                    atomic_write_json(job / 'failure.json', dict(at_utc=utc_now(), launch=launch, pipeline_stop_requested=True))
                    raise
            result = json.loads(output.read_text())
            rp = EXP / 'logs/stage5' / rid / 'result.json'
            assert result['status'] == 'pass' and result['exact_metric_equality'] and result['validation_nets'] == 8000
            assert result['result_sha256'] == file_sha256(rp) and not result['test_ispd_used']
            records.append(dict(run_id=rid, logical_path=str(output), resolved_path=str(output.resolve()), sha256=file_sha256(output)))
            atomic_write_json(job / 'result.json', dict(status='pass' if len(records) == len(ready) else 'partial_pass',
                at_utc=utc_now(), runs=records, validation_nets=8000 * len(records), stage6_complete=False))
            print(json.dumps(dict(status='pass', run_id=rid, completed=len(records), total=len(ready))), flush=True)
        lease.finish(update=len(records), status='complete')


if __name__ == '__main__':
    main()
