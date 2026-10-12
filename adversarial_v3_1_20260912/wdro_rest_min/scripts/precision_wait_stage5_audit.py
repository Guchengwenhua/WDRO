"""Wait for one R5 run, then hand its unchanged full audit to the pipeline."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent / 'experiments'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import JobLease, utc_now
from precision_replay_pipeline import cpu_preflight, progress
from precision_stage5_completed_batch import artifact, check_files, audit, stop_pipeline
import precision_stage5_capacity_pipeline as pipeline
import precision_queue_runtime as runtime


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def original_audit_alive(rid):
    pid = read(EXP / 'logs/stage5_audit/jobs' / rid / 'job.json').get('pid')
    return runtime.process_matches(pid, rid, ROOT / 'scripts/audit_stage5.py')


def state(rid):
    result = read(EXP / 'logs/stage5' / rid / 'result.json')
    trainer = read(EXP / 'logs/jobs' / rid / 'job.json').get('pid')
    if result.get('status') != 'complete' or runtime.process_matches(trainer, rid, ROOT / 'scripts/train_stage5.py'):
        return 'waiting_training', None
    assert result['completed_updates'] == 2000 and not result['test_or_ispd_used']
    for kind in ('latest', 'best'):
        assert artifact(Path(result[kind + '_checkpoint']))['sha256'] == result[kind + '_checkpoint_sha256']
    report = EXP / 'logs/stage5_audit/runs' / (rid + '.json')
    if report.exists():
        assert audit(dict(run_id=rid))
        return 'complete', artifact(report)
    assert not (EXP / 'logs/stage5_completed_review' / rid / 'failure.json').exists()
    if original_audit_alive(rid):
        return 'adopting_original_audit', None
    early = read(EXP / 'logs/stage5_completed_review' / rid / 'job.json').get('pid')
    if runtime.process_matches(early, rid, ROOT / 'scripts/precision_review_completed_stage5.py'):
        return 'adopting_early_audit_wrapper', None
    return 'ready_for_audit', None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    rid = args.run_id
    pipeline.gate()
    tasks = read(pipeline.JOB / 'all_tasks.json')['tasks']
    assert rid in {task['run_id'] for task in tasks}
    job = EXP / 'logs/stage5_audit_waiters' / rid
    job.mkdir(parents=True, exist_ok=True)
    plan_path = job / 'plan.json'
    if not plan_path.exists():
        atomic_write_json(plan_path, dict(run_id=rid, task=next(t for t in tasks if t['run_id'] == rid),
            files=[artifact(p) for p in (Path(__file__), ROOT / 'scripts/precision_review_completed_stage5.py',
                ROOT / 'scripts/precision_stage5_completed_batch.py', ROOT / 'scripts/audit_stage5.py',
                ROOT / 'manifests/precision_queue_recovery_operations_v1.json')],
            research_protocol_override=False, stage6_complete=False), mode=0o444)
    plan = read(plan_path)
    assert plan['task'] == next(t for t in tasks if t['run_id'] == rid)
    check_files(plan['files'])
    preflight = cpu_preflight('wait_stage5_audit_' + rid)
    exact = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__)), '--run-id', rid])
    runtime.cleanup_dead_lock(job)
    child = None
    child_log = None
    with JobLease(job, run_id='wait_audit_' + rid, exact_command=exact + (' --once' if args.once else ''),
        resume_command=exact, monitor_command=f'cat {job}/heartbeat.json', expected_updates=1,
        physical_gpu_index=None, host_runtime_manifest=str(preflight)) as lease:
        try:
            while True:
                pipeline.gate()
                status, report = state(rid)
                lease.update(update=int(status == 'complete'), status=status, detail=dict(plan=str(plan_path)))
                if status == 'complete':
                    output = job / 'result.json'
                    if output.exists():
                        assert read(output)['audit'] == report
                    else:
                        atomic_write_json(output, dict(status='pass', at_utc=utc_now(), plan=artifact(plan_path),
                            audit=report, stage6_complete=False, test_ispd_used=False), mode=0o444)
                        progress(f'R5审计交接完成：{rid}已通过原完整审计；后续best复算/冻结仍由原流水线验收', str(output), True)
                    lease.finish(update=1, status='complete')
                    print(json.dumps(dict(status='pass', result=str(output))), flush=True)
                    return
                if args.once:
                    output = job / ('foreground_' + str(time.time_ns()) + '.json')
                    atomic_write_json(output, dict(status='pass', at_utc=utc_now(), observed=status,
                        preflight=artifact(preflight), plan=artifact(plan_path), model_calls=0), mode=0o444)
                    lease.finish(update=0, status='paused_after_foreground_smoke')
                    print(json.dumps(dict(status='pass', observed=status, evidence=str(output))), flush=True)
                    return
                if child is not None:
                    code = child.poll()
                    if code is not None:
                        if code != 0:
                            tail = child_log.read_text()[-8000:]
                            duplicate = ('live lock prevents duplicate launch' in tail or 'already locked by live PID' in tail)
                            assert duplicate and original_audit_alive(rid), 'early audit wrapper failed; see launches.jsonl'
                            append_jsonl(job / 'adoptions.jsonl', dict(at_utc=utc_now(), exit_code=code,
                                reason='original auditor acquired the run lock first', log=str(child_log)))
                        child = None
                if status == 'ready_for_audit' and child is None:
                    # Once all training finishes, the original pipeline owns launch order.
                    remaining = [t for t in tasks if read(EXP / 'logs/stage5' / t['run_id'] / 'result.json').get('status') != 'complete']
                    if remaining:
                        command = ['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable,
                            str(ROOT / 'scripts/precision_review_completed_stage5.py'), '--run-id', rid]
                        log_path = job / ('audit_' + str(time.time_ns()) + '.log')
                        with log_path.open('a') as log:
                            checked = subprocess.run([*command, '--check-only'], stdout=log, stderr=subprocess.STDOUT)
                            assert checked.returncode == 0, str(log_path)
                            # Recheck after the foreground preflight before spawning the original wrapper.
                            remaining = [t for t in tasks if read(EXP / 'logs/stage5' / t['run_id'] / 'result.json').get('status') != 'complete']
                            if remaining and state(rid)[0] == 'ready_for_audit':
                                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), start_new_session=True)
                                child_log = log_path
                                launch = dict(at_utc=utc_now(), pid=child.pid, gpu=None, command=command,
                                    exact_command=shlex.join(command), recovery_command=exact,
                                    log=str(log_path), log_resolved=str(log_path.resolve()),
                                    preflight=str(EXP / 'logs/stage5_completed_review' / rid / 'foreground_check.json'))
                                append_jsonl(job / 'launches.jsonl', launch)
                time.sleep(15)
        except BaseException as error:
            # The original wrapper owns its auditor. A restarted waiter adopts it.
            if not isinstance(error, (KeyboardInterrupt, SystemExit)):
                stop_pipeline(job, error)
            raise


if __name__ == '__main__':
    main()
