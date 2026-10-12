"""Resume validation checks for a fixed batch of already completed R5 runs."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / 'experiments'
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import JobLease, utc_now
from src.rest import file_sha256
from precision_replay_pipeline import cpu_preflight, progress
import precision_stage5_capacity_pipeline as pipeline
import precision_queue_runtime as runtime


def artifact(path):
    path = Path(path)
    assert path.is_relative_to(CODEX)
    return dict(logical_path=str(path), resolved_path=str(path.resolve()), sha256=file_sha256(path))


def check_files(records):
    for item in records:
        path = Path(item['logical_path'])
        assert artifact(path) == item, str(path)


def inputs(task):
    check_files(task['files'])
    rid = task['run_id']
    rp = EXP / 'logs/stage5' / rid / 'result.json'
    result = json.loads(rp.read_text())
    assert result['status'] == 'complete' and result['completed_updates'] == 2000
    assert not result['test_or_ispd_used']
    return result


def audit(task):
    rid = task['run_id']
    rp = EXP / 'logs/stage5' / rid / 'result.json'
    path = EXP / 'logs/stage5_audit/runs' / (rid + '.json')
    if not path.exists():
        failure = EXP / 'logs/stage5_completed_review' / rid / 'failure.json'
        assert not failure.exists(), str(failure)
        jp = EXP / 'logs/stage5_audit/jobs' / rid / 'job.json'
        job = json.loads(jp.read_text()) if jp.exists() else {}
        assert runtime.process_matches(job.get('pid'), rid, ROOT / 'scripts/audit_stage5.py'), 'Resume interrupted original audit: ' + rid
        return None
    value = json.loads(path.read_text())
    assert value['status'] == 'pass' and all(value['checks'].values())
    assert value['completed_updates'] == 2000
    cursor = json.loads(Path(value['stream_cursor']).read_text())
    assert cursor['heldout_intersections'] == 0 and cursor['completed_updates'] == 2000
    assert value['binding']['result_sha256'] == file_sha256(rp)
    assert not value['test_ispd_model_metrics_read']
    return value


def verified(task):
    rid = task['run_id']
    result = inputs(task)
    report = audit(task)
    if report is None:
        return None
    path = EXP / 'logs/stage5_best_validation' / (rid + '.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    assert value['status'] == 'pass' and value['exact_metric_equality'] and value['validation_nets'] == 8000
    assert value['result_sha256'] == file_sha256(EXP / 'logs/stage5' / rid / 'result.json')
    assert value['checkpoint_sha256'] == result['best_checkpoint_sha256']
    assert value['selected_update'] == result['best']['update']
    assert value['validation_manifest_sha256'] == result['best']['validation_manifest_sha256']
    assert not value['test_ispd_used']
    return dict(run_id=rid, validation=artifact(path), audit=artifact(EXP / 'logs/stage5_audit/runs' / (rid + '.json')),
                full_audit_checks=len(report['checks']), validation_nets=8000)


def stop_pipeline(job, error):
    record = json.loads((pipeline.JOB / 'job.json').read_text())
    pid = record['pid']
    path = Path(f'/proc/{pid}/cmdline')
    stopped = False
    if runtime.process_is_alive(pid) and path.exists() and str(ROOT / 'scripts/precision_stage5_capacity_pipeline.py').encode() in path.read_bytes().split(b'\0'):
        os.kill(pid, signal.SIGTERM)
        stopped = True
    atomic_write_json(job / 'failure.json', dict(at_utc=utc_now(), error=str(error), pipeline_pid=pid,
        pipeline_stop_requested=stopped, recovery='Resolve the failed check, then resume the same batch and precision_wdro_replay.'))


def evaluate(task, job, lease, completed):
    rid = task['run_id']
    blocked = {}
    active_path = job / 'active.json'
    while True:
        pipeline.gate()
        inputs(task)
        assert audit(task)
        if verified(task):
            return
        old = json.loads(active_path.read_text()) if active_path.exists() else {}
        if old.get('run_id') == rid and runtime.process_matches(old.get('pid'), rid, ROOT / 'scripts/verify_stage5_best_validation.py'):
            lease.update(update=completed, status='adopting_validation_child', detail=old)
            time.sleep(5)
            continue
        host = runtime.capture('completed_batch_validation_' + rid, 1630, job)
        if host is None:
            lease.update(update=completed, status='waiting_capacity', detail=dict(run_id=rid))
            time.sleep(30)
            continue
        snapshot = json.loads(Path(host['manifest']).read_text())
        queue = json.loads((pipeline.JOB / 'queue.json').read_text())
        active = {row['task']['run_id']: row for row in queue['active']}
        for gpu, until in queue['blocked_until'].items():
            blocked[int(gpu)] = max(blocked.get(int(gpu), 0), until)
        history = []
        with (pipeline.JOB / 'resource_monitor.jsonl').open() as source:
            for line in source:
                row = json.loads(line)
                if row['time'] > time.time() - 600:
                    history.append(row)
        gpu = runtime.choose_gpu(snapshot['gpu_decisions'], active, history, blocked, 4702, time.time())
        if gpu is None:
            lease.update(update=completed, status='waiting_capacity', detail=dict(run_id=rid, preflight=host))
            time.sleep(30)
            continue
        cmd = ['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable,
               str(ROOT / 'scripts/verify_stage5_best_validation.py'), '--run-id', rid, '--gpu', str(gpu), '--preflight', host['manifest']]
        log_path = job / (rid + '_' + str(time.time_ns()) + '.log')
        with log_path.open('a') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(at_utc=utc_now(), run_id=rid, pid=proc.pid, gpu=gpu, command=cmd, exact_command=shlex.join(cmd),
                      preflight=host, log=str(log_path), log_resolved=str(log_path.resolve()))
        atomic_write_json(active_path, launch)
        append_jsonl(job / 'launches.jsonl', launch)
        constrained = False
        saturated = 0
        try:
            while proc.poll() is None:
                observed = pipeline.observe()
                append_jsonl(job / 'resources.jsonl', dict(observed, run_id=rid, pid=proc.pid))
                lease.update(update=completed, status='validation_recompute', detail=launch)
                row = next(x for x in observed['gpus'] if x['index'] == gpu)
                saturated = saturated + 1 if row['utilization_gpu_percent'] >= 95 else 0
                if row['memory_free_mib'] < 2048 or observed['disk'][1]['free_gib'] < 100 or saturated >= 6:
                    constrained = True
                    proc.terminate()
                    proc.wait()
                    break
                time.sleep(5)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        if proc.returncode == 0:
            assert verified(task)
            return
        tail = log_path.read_text()[-20000:]
        if constrained or 'CUDA out of memory' in tail:
            blocked[gpu] = time.time() + 600
            append_jsonl(job / 'capacity_retries.jsonl', dict(launch, exit_code=proc.returncode, reason='stateless validation capacity retry'))
            continue
        raise RuntimeError('Validation recomputation failed: ' + str(log_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--max-new-runs', type=int)
    args = parser.parse_args()
    assert args.job_id.replace('_', '').isalnum()
    pipeline.gate()
    job = EXP / 'logs/stage5_completed_batches' / args.job_id
    job.mkdir(parents=True, exist_ok=True)
    plan_path = job / 'plan.json'
    if args.prepare:
        assert not plan_path.exists(), 'Reuse the existing fixed batch with --check-only or the resume command.'
        tasks = []
        for task in json.loads((pipeline.JOB / 'all_tasks.json').read_text())['tasks']:
            rp = EXP / 'logs/stage5' / task['run_id'] / 'result.json'
            result = json.loads(rp.read_text())
            if result['status'] != 'complete':
                continue
            assert result['completed_updates'] == 2000
            files = [rp, Path(result['latest_checkpoint']), Path(result['best_checkpoint'])]
            tasks.append(dict(run_id=task['run_id'], files=[artifact(p) for p in files]))
        assert tasks
        atomic_write_json(plan_path, dict(at_utc=utc_now(), tasks=tasks, research_protocol_override=False,
            files=[artifact(p) for p in [Path(__file__), ROOT / 'scripts/verify_stage5_best_validation.py',
                ROOT / 'scripts/audit_stage5.py', ROOT / 'manifests/precision_queue_recovery_operations_v1.json']]), mode=0o444)
    plan = json.loads(plan_path.read_text())
    check_files(plan['files'])
    for task in plan['tasks']:
        inputs(task)
    preflight = cpu_preflight('completed_batch_' + args.job_id)
    if (job / 'result.json').exists():
        saved = json.loads((job / 'result.json').read_text())
        assert saved['status'] == 'pass' and saved['plan'] == artifact(plan_path)
        assert sorted(r['run_id'] for r in saved['runs']) == sorted(t['run_id'] for t in plan['tasks'])
        for task in plan['tasks']:
            assert verified(task) == next(r for r in saved['runs'] if r['run_id'] == task['run_id'])
        print(json.dumps(dict(status='already_complete', result=str(job / 'result.json'))), flush=True)
        return
    if args.prepare or args.check_only:
        ready = [t['run_id'] for t in plan['tasks'] if audit(t)]
        value = dict(status='pass', at_utc=utc_now(), plan=artifact(plan_path), preflight=artifact(preflight),
            completed_training_runs=len(plan['tasks']), completed_full_audits=len(ready), audited_runs=ready, model_calls=0)
        atomic_write_json(job / ('foreground_check_' + str(time.time_ns()) + '.json'), value)
        print(json.dumps(value), flush=True)
        return
    exact = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__)), '--job-id', args.job_id])
    command = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__)), *sys.argv[1:]])
    runtime.cleanup_dead_lock(job)
    with JobLease(job, run_id='completed_batch_' + args.job_id, exact_command=command, resume_command=exact,
                  monitor_command=f'cat {job}/heartbeat.json', expected_updates=len(plan['tasks']), physical_gpu_index=None,
                  host_runtime_manifest=str(preflight)) as lease:
        records = []
        newly = 0
        try:
            while len(records) < len(plan['tasks']):
                pipeline.gate()
                known = {r['run_id'] for r in records}
                for task in plan['tasks']:
                    if task['run_id'] in known or audit(task) is None:
                        continue
                    record = verified(task)
                    if record is None:
                        evaluate(task, job, lease, len(records))
                        record = verified(task)
                        newly += 1
                    records.append(record)
                    value = dict(status='pass' if len(records) == len(plan['tasks']) else 'partial_pass', at_utc=utc_now(),
                        plan=artifact(plan_path), runs=records, expected_runs=len(plan['tasks']),
                        full_audit_checks=sum(r['full_audit_checks'] for r in records), validation_nets=8000 * len(records),
                        stage6_complete=False, test_ispd_used=False)
                    atomic_write_json(job / 'cursor.json', value)
                    print(json.dumps(dict(status='pass', run_id=task['run_id'], verified=len(records), expected=len(plan['tasks']))), flush=True)
                    if args.max_new_runs and newly >= args.max_new_runs:
                        lease.finish(update=len(records), status='paused_after_foreground_smoke')
                        return
                lease.update(update=len(records), status='waiting_original_full_audits', detail=dict(plan=str(plan_path)))
                if len(records) < len(plan['tasks']):
                    time.sleep(15)
            for task in plan['tasks']:
                assert verified(task)
            atomic_write_json(job / 'result.json', value, mode=0o444)
            progress(f'R5完成项验收：{len(records)}/9个run完整审计共{value["full_audit_checks"]}项通过，best独立重算{value["validation_nets"]} validation nets精确一致；其余run依原门禁继续', str(job / 'result.json'), True)
            lease.finish(update=len(records), status='complete')
        except BaseException as error:
            stop_pipeline(job, error)
            raise


if __name__ == '__main__':
    main()
