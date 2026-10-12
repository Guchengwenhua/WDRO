"""Foreground recovery of two committed updates, retaining the 2000-update run."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / 'experiments'
sys.path.insert(0, str(ROOT))
import torch
import precision_queue_runtime as runtime
import precision_stage5_capacity_pipeline as pipeline
from src.io_utils import atomic_write_json
from src.job_control import JobLease, utc_now
from src.rest import file_sha256
from src.stage2_audit_checks import optimizer_checks
from src.trainer import load_trainer_config
from audit_stage3 import effective_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    pipeline.gate()
    task = next(t for t in json.loads((pipeline.JOB / 'all_tasks.json').read_text())['tasks'] if t['run_id'] == args.run_id)
    rid = args.run_id
    checkpoint = EXP / 'checkpoints/stage5' / rid / 'latest.pt'
    initial = torch.load(checkpoint, weights_only=False, map_location='cpu')
    start = initial['update']; stop = start + 2
    assert stop < 2000
    job = EXP / 'logs/stage6_remaining_20260908/resume_smoke' / rid
    job.mkdir(parents=True, exist_ok=True)
    events = EXP / 'logs/stage5' / rid / 'events.jsonl'
    old_replays = {}
    with events.open() as source:
        for line in source:
            e = json.loads(line)
            if e.get('event') == 'train' and start < e['update'] <= stop:
                old_replays[e['update']] = e
    before = dict(at_utc=utc_now(), checkpoint_sha256=file_sha256(checkpoint),
                  events_sha256=file_sha256(events), update=start, run_identity=initial['run_identity'])
    atomic_write_json(job / 'before.json', before, mode=0o444)
    summary = pipeline.preflight('stage5_foreground_resume_' + rid)
    assert summary, 'no current capacity; keep checkpoint and rerun after a fresh preflight'
    host = json.loads(Path(summary['manifest']).read_text())
    gpu = runtime.choose_gpu(host['gpu_decisions'], {}, [], {}, 1630 + 1024 + 2048, time.time())
    assert gpu is not None
    runtime.cleanup_dead_lock(EXP / 'logs/jobs' / rid)
    cmd = ['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(ROOT / 'scripts/train_stage5.py'),
           '--degree', '30', '--seed', '101', '--scope', 'ablation', '--wdro-config', task['config'],
           '--rho-mode', task['rho_mode'], '--run-id', rid, '--physical-gpu-index', str(gpu),
           '--host-runtime-manifest', summary['manifest'], '--geosteiner-workers', '8', '--resume', '--stop-after', str(stop)]
    exact = shlex.join([sys.executable, str(Path(__file__)), *sys.argv[1:]])
    with JobLease(job, run_id='foreground_resume_' + rid, exact_command=exact,
                  resume_command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {ROOT}/scripts/precision_stage5_capacity_pipeline.py',
                  monitor_command=f'cat {job}/heartbeat.json', expected_updates=2,
                  physical_gpu_index=gpu, host_runtime_manifest=summary['manifest']) as lease:
        with (job / 'output.log').open('a') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_write_json(job / 'launch.json', dict(at_utc=utc_now(), pid=child.pid, gpu=gpu,
            command=cmd, exact_command=shlex.join(cmd), preflight=summary,
            log=str(job / 'output.log'), log_resolved=str((job / 'output.log').resolve()),
            monitor_command=f'cat {job}/heartbeat.json', recovery_command=lease._job['resume_command']))
        samples = []
        try:
            while child.poll() is None:
                snapshot = pipeline.observe(); samples.append(snapshot)
                lease.update(update=0, status='foreground_resume', detail=dict(pid=child.pid, gpu=gpu, start_update=start, stop_update=stop))
                assert next(g for g in snapshot['gpus'] if g['index'] == gpu)['memory_free_mib'] > 2048
                assert snapshot['disk'][1]['free_gib'] >= 100
                time.sleep(5)
            assert child.returncode == 0, f'foreground smoke failed: {job}/output.log'
        finally:
            if child.poll() is None:
                runtime.stop_children({rid: dict(pid=child.pid, proc=child)}, ROOT / 'scripts/train_stage5.py', job)
            atomic_write_json(job / 'resources.json', dict(samples=samples))
        result_path = EXP / 'logs/stage5' / rid / 'result.json'
        result = json.loads(result_path.read_text())
        latest = torch.load(checkpoint, weights_only=False, map_location='cpu')
        checks = optimizer_checks(latest, load_trainer_config())
        checks.update(expected_budget_unchanged=latest['run_identity']['expected_updates'] == 2000,
                      same_identity=latest['run_identity'] == initial['run_identity'],
                      resumed_exact_checkpoint=result['started_this_segment_at_update'] == start + 1,
                      two_updates_committed=latest['update'] == result['completed_updates'] == stop,
                      paused_with_checkpoint=result['status'] == 'paused_for_resume',
                      checkpoint_hash=file_sha256(checkpoint) == result['latest_checkpoint_sha256'],
                      no_partial=not list(checkpoint.parent.glob('*.partial')),
                      no_lock=not (EXP / 'logs/jobs' / rid / 'run.lock').exists(),
                      no_test_or_ispd=not result['test_or_ispd_used'])
        train, _, _ = effective_events(events)
        checks['effective_log_contiguous'] = sorted(train) == list(range(1, stop + 1))
        for update, old in old_replays.items():
            new = train[update]
            keys = ['batch_bundle_sha256', 'training_stream_hash', 'actor_loss', 'critic_loss', 'mean_regret']
            detail_keys = ['support_sha256', 'selected_bundle_sha256', 'selected_indices', 'exact_relative_regrets',
                           'exact_transport_costs', 'lambda_before', 'lambda_after', 'rho']
            checks[f'replayed_update_{update}_exact_decisions'] = all(old[k] == new[k] for k in keys) and all(
                old['method_detail'][k] == new['method_detail'][k] for k in detail_keys)
        assert all(checks.values()), checks
        output = dict(status='pass', at_utc=utc_now(), run_id=rid, start_update=start, stop_update=stop,
                      checks=checks, preflight=summary, peak_reserved_mib=result['peak_memory_reserved_bytes'] / 2**20,
                      result_sha256=file_sha256(result_path), checkpoint_sha256=file_sha256(checkpoint),
                      script_sha256=file_sha256(Path(__file__)), logical_path=str(job), resolved_path=str(job.resolve()),
                      formal_run_complete=False)
        atomic_write_json(job / 'acceptance.json', output, mode=0o444)
        lease.finish(update=2, status='complete')
        print(json.dumps(dict(status='pass', run_id=rid, from_update=start, to_update=stop,
                              checks=len(checks), gpu=gpu, acceptance=str(job / 'acceptance.json'))), flush=True)


if __name__ == '__main__':
    main()
