"""Read-only recovery readiness review of live Stage 6 jobs; writes evidence only."""
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODEX = Path('/remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex')
ROOT = CODEX / 'wdro_rest_min'
EXP = CODEX / 'experiments'
ORIGINAL = Path('/remote-home/Guilin/DRO/Codex')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from src.stage2_audit_checks import optimizer_checks
from src.trainer import load_trainer_config, canonical_sha256
from stage6_pipeline import gate
from precision_replay_pipeline import cpu_preflight


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    return dict(logical_path=str(path), resolved_path=str(path.resolve()),
                sha256=file_sha256(path), size_bytes=path.stat().st_size)


def checkpoint_snapshot(path):
    # Hash and deserialize one inode even if the trainer atomically replaces its name.
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        value = torch.load(handle, map_location='cpu', weights_only=False)
        handle.seek(0)
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        after = os.fstat(handle.fileno())
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return value, dict(logical_path=str(path), resolved_path=str(path.resolve()),
                       sha256=digest, size_bytes=after.st_size, inode=after.st_ino,
                       saved_at_utc=value['saved_at_utc'], update=value['update'])


def main():
    preflight = cpu_preflight('stage6_network_resume_checkpoint_review')
    gate()
    trainer = load_trainer_config()
    queue = read(EXP / 'logs/stage6_pipeline/queue.json')
    rows = []
    for active in queue['active']:
        task = active['task']
        rid, pid = task['run_id'], active['pid']
        proc = Path('/proc') / str(pid)
        cmd = proc.joinpath('cmdline').read_bytes().split(b'\0')
        cmd = [arg.decode() for arg in cmd if arg]
        stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
        cwd = str(proc.joinpath('cwd').resolve())
        env = dict(line.split('=', 1) for line in proc.joinpath('environ').read_text().split('\0') if '=' in line)
        controlled_env = {key: env.get(key) for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'PYTHONPYCACHEPREFIX')}
        job_dir = EXP / 'logs/jobs' / rid
        job, heartbeat, lock = [read(job_dir / name) for name in ('job.json', 'heartbeat.json', 'run.lock')]
        now = datetime.now(timezone.utc)
        age = (now - datetime.fromisoformat(heartbeat['last_update_utc'].replace('Z', '+00:00'))).total_seconds()
        latest, latest_ref = checkpoint_snapshot(EXP / 'checkpoints/stage6' / rid / 'latest.pt')
        best, best_ref = checkpoint_snapshot(EXP / 'checkpoints/stage6' / rid / 'best.pt')
        identity = latest['run_identity']
        checks = dict(process_identity=stat[0] not in ('Z', 'X') and str(ROOT/'scripts/train_stage6.py') in cmd and rid in cmd,
                      cwd_inside_codex=cwd.startswith(str(ORIGINAL) + '/'),
                      runtime_paths_inside_codex=all(not value or value.startswith(str(ORIGINAL) + '/') for value in controlled_env.values()),
                      gpu_mapping=cmd[cmd.index('--physical-gpu-index')+1] == str(active['gpu']) == job['cuda_visible_devices'] and job['physical_gpu_index'] == active['gpu'],
                      lease_owned=job['pid'] == heartbeat['pid'] == lock['pid'] == pid,
                      heartbeat_recent=age < 600,
                      positive_recovery_checkpoint=0 < latest['update'] <= 2000,
                      checkpoint_identity=identity == best['run_identity'],
                      requested_identity=identity['degree'] == task['degree'] and identity['training_seed'] == task['seed'] and identity['stage'] == 6,
                      frozen_semantics=identity['expected_updates'] == 2000 and identity['batch_size'] == 256 and identity['rho_mode'] == 'q75',
                      frozen_configuration=file_sha256(Path(task['config'])) == task['config_sha256'] and identity['wdro_config_sha256'] == canonical_sha256(read(task['config'])),
                      corrected_reference=identity['reference_solver']['reference_id'] == 'geosteiner_exact_binary_scale_v2',
                      complete_wdro_state=set(latest['method_state']) >= {'lambda','rho','cost_history','budget','support_stream_hash','selection_stream_hash'},
                      validation_selected_best=best['update'] == best['best']['update'] and best['best']['validation_manifest_sha256'] is not None,
                      best_not_after_latest=best['update'] <= latest['update'])
        checks.update({'latest_' + key: value for key, value in optimizer_checks(latest, trainer).items()})
        if best['update']:
            checks.update({'best_' + key: value for key, value in optimizer_checks(best, trainer).items()})
        row = dict(run_id=rid, degree=task['degree'], seed=task['seed'], gpu=active['gpu'], pid=pid,
                   process_start_ticks=int(stat[19]), exact_command=shlex.join(cmd), cwd=cwd,
                   environment_paths=controlled_env, heartbeat_age_seconds=age,
                   heartbeat_update=heartbeat['completed_updates'], latest=latest_ref, best=best_ref,
                   job=job, checks=checks, status='pass' if all(checks.values()) else 'fail')
        rows.append(row)
        print(json.dumps({key:row[key] for key in ('run_id','gpu','pid','heartbeat_update','status')}, ensure_ascii=False), flush=True)
        del latest, best
    completion = ROOT / 'manifests/stage6_completion_v1.json'
    report = dict(status='pass' if all(row['status']=='pass' for row in rows) else 'fail', at_utc=utc_now(),
                  kind='live_stage6_recovery_readiness', stage6_status='DOING',
                  scope='read-only checkpoint and process review, not completed-training or model-test acceptance',
                  source=binding(Path(__file__)), preflight=binding(preflight), source_gates_passed=True,
                  frozen_operations=binding(ROOT/'manifests/stage6_operations_contract_v1.json'),
                  queue_snapshot=queue, active_run_reviews=rows,
                  formal_new_completed=len(queue['complete']), formal_new_total=len(queue['tasks']),
                  active_updates=sum(row['heartbeat_update'] for row in rows),
                  controller_heartbeat=read(EXP/'logs/precision_stage6_completion/heartbeat.json'),
                  pipeline_heartbeat=read(EXP/'logs/stage6_pipeline/heartbeat.json'),
                  root_launch=read(ORIGINAL/'experiments/logs/precision_v2/jobs/stage6_completion/launch.json'),
                  historical_capacity_recoveries=[json.loads(line) for line in (EXP/'logs/stage6_pipeline/recoveries.jsonl').read_text().splitlines()],
                  stage6_completion_exists=completion.exists(), stage7_started=False,
                  action='retained healthy tmux controller and workers without duplicate launch or scheduler change')
    report['review_implementation_correction'] = binding(Path(__file__).parent / 'network_review_environment_diagnosis.json')
    report['gpu_mapping_evidence'] = 'queue + exact live CLI + JobLease runtime CUDA_VISIBLE_DEVICES; /proc/PID/environ is only the exec-time environment'
    target = Path(__file__).parent / f'network_resume_review_{time.time_ns()}.json'
    atomic_write_json(target, report, mode=0o444)
    print(json.dumps(dict(status=report['status'], report=str(target), active=len(rows), active_updates=report['active_updates']), ensure_ascii=False), flush=True)
    assert report['status']=='pass'


if __name__ == '__main__':
    main()
