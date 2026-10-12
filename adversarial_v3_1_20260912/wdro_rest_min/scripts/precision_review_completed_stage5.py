"""Run one unchanged full audit early, stopping this queue if it fails."""
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
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import JobLease, utc_now
from precision_replay_pipeline import cpu_preflight, progress
import precision_stage5_capacity_pipeline as pipeline
import precision_queue_runtime as runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); rid = args.run_id
    tasks = json.loads((pipeline.JOB / 'all_tasks.json').read_text())['tasks']
    assert rid in {t['run_id'] for t in tasks}
    pipeline.gate()
    rp = EXP / 'logs/stage5' / rid / 'result.json'
    result = json.loads(rp.read_text())
    assert result['status'] == 'complete' and result['completed_updates'] == 2000
    assert not runtime.process_is_alive(json.loads((EXP / 'logs/jobs' / rid / 'job.json').read_text())['pid'])
    assert file_sha256(Path(result['latest_checkpoint'])) == result['latest_checkpoint_sha256']
    report = EXP / 'logs/stage5_audit/runs' / f'{rid}.json'
    job = EXP / 'logs/stage5_completed_review' / rid
    job.mkdir(parents=True, exist_ok=True)
    preflight = cpu_preflight('completed_stage5_audit_' + rid)
    if args.check_only:
        atomic_write_json(job / 'foreground_check.json', dict(status='pass', at_utc=utc_now(),
            preflight=str(preflight), result_sha256=file_sha256(rp), auditor_sha256=file_sha256(ROOT / 'scripts/audit_stage5.py'),
            prior_smoke_gate=str(EXP / 'logs/stage5_audit/smoke_gate.json'), gpu=None))
        print(json.dumps(dict(status='pass', preflight=str(preflight), job=str(job)))); return
    exact = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__)), '--run-id', rid])
    with JobLease(job, run_id='early_audit_' + rid, exact_command=exact, resume_command=exact,
                  monitor_command=f'cat {job}/heartbeat.json', expected_updates=2000,
                  physical_gpu_index=None, host_runtime_manifest=str(preflight)) as lease:
        if not report.exists():
            runtime.cleanup_dead_lock(EXP / 'logs/stage5_audit/jobs' / rid)
            cmd = ['bash', str(ROOT / 'scripts/stage3_env.sh'), 'nice', '-n', '19', sys.executable,
                   str(ROOT / 'scripts/audit_stage5.py'), '--run-id', rid, '--stability']
            with (job / 'output.log').open('a') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', WDRO_HOST_RUNTIME_MANIFEST=str(preflight)), start_new_session=True)
            launch = dict(at_utc=utc_now(), pid=child.pid, gpu=None, command=cmd, exact_command=shlex.join(cmd),
                preflight=str(preflight), log=str(job / 'output.log'), log_resolved=str((job / 'output.log').resolve()),
                recovery_command=exact, monitor_command=f'cat {job}/heartbeat.json')
            atomic_write_json(job / 'launch.json', launch)
            try:
                while child.poll() is None:
                    cursor = EXP / 'logs/stage5_audit/cursors' / f'{rid}.json'
                    saved = json.loads(cursor.read_text()) if cursor.exists() else {}
                    lease.update(update=saved.get('completed_updates', 0), status='auditing', detail=dict(pid=child.pid, cursor=str(cursor)))
                    time.sleep(5)
                assert child.returncode == 0, f'audit failed: {job}/output.log'
            except BaseException:
                if child.poll() is None: child.terminate(); child.wait()
                j = json.loads((pipeline.JOB / 'job.json').read_text()); pid = j['pid']
                proc = Path(f'/proc/{pid}/cmdline')
                stopped = False
                if runtime.process_is_alive(pid) and proc.exists() and str(ROOT / 'scripts/precision_stage5_capacity_pipeline.py').encode() in proc.read_bytes().split(b'\0'):
                    os.kill(pid, signal.SIGTERM); stopped = True
                atomic_write_json(job / 'failure.json', dict(at_utc=utc_now(), status='failed',
                    pipeline_stop_requested=stopped, pipeline_pid=pid, launch=launch,
                    recovery='resolve the failed original audit before resuming precision_wdro_replay'))
                raise
        r = json.loads(report.read_text())
        assert r['status'] == 'pass' and all(r['checks'].values())
        assert r['binding']['result_sha256'] == file_sha256(rp)
        atomic_write_json(job / 'result.json', dict(status='pass', at_utc=utc_now(),
            report=str(report), report_resolved=str(report.resolve()), report_sha256=file_sha256(report),
            checks=len(r['checks']), training_nets=r['training_nets_checked'], stage6_complete=False))
        progress(f'R5完成项提前审计：{rid} 的2000 updates通过原完整审计器（{len(r["checks"])}项）；其余消融继续', str(job / 'result.json'), True)
        lease.finish(update=2000, status='complete')
        print(json.dumps(dict(status='pass', checks=len(r['checks']), result=str(job / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()
