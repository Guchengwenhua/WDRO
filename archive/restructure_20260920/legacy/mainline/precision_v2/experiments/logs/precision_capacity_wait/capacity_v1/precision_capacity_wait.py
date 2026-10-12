"""Retry only a verified host-capacity denial, preserving every preflight."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.io_utils import append_jsonl, atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256

OPERATIONS = ROOT / 'manifests/precision_capacity_wait_operations_v1.json'


def check_operations():
    for item in json.loads(OPERATIONS.read_text())['files']:
        assert file_sha256(Path(item['logical_path'])) == item['sha256'], item['logical_path']


def classify_capture(result):
    """An exit code alone is never evidence that a failure is resource-only."""
    if result.returncode not in (0, 2):
        result.check_returncode()
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    path = Path(summary['manifest'])
    assert path.is_absolute() and path.is_relative_to(CODEX), 'preflight outside correction workspace'
    assert file_sha256(path) == summary['sha256'], 'preflight hash mismatch'
    manifest = json.loads(path.read_text())
    assert manifest['kind'] == 'host_runtime_preflight' and manifest['research_protocol_override'] is False
    assert manifest['sampling']['count'] >= 2
    assert len(manifest['sampling']['samples']) == manifest['sampling']['count']
    eligible = bool(manifest['eligible_gpu_indices'])
    disk_ok = manifest['checks']['experiments_disk_meets_floor']
    ready = eligible and disk_ok
    assert summary['preflight_passed'] == manifest['preflight_passed'] == ready
    assert (result.returncode == 0) == ready, 'preflight exit/status mismatch'
    return ready, summary


def wait_capture(command, *, context, delay_seconds=15, log_root=None):
    assert len(command) > 1 and Path(command[1]) == ROOT / 'scripts/capture_host_runtime.py'
    root = Path(log_root) if log_root else CODEX / 'experiments/logs/precision_capacity_wait'
    assert root.is_absolute() and root.is_relative_to(CODEX)
    while True:
        check_operations()
        result = subprocess.run(command, capture_output=True, text=True)
        record = dict(at_utc=utc_now(), pid=os.getpid(), context=context, command=command,
                      returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                      research_protocol_override=False)
        append_jsonl(root / 'attempts.jsonl', record)
        ready, summary = classify_capture(result)
        atomic_write_json(root / f'heartbeat_{os.getpid()}.json', dict(
            at_utc=utc_now(), pid=os.getpid(), context=context,
            status='ready' if ready else 'waiting_resources', preflight=summary,
            next_action='run the unchanged smoke command' if ready else 'capture fresh GPU/process/disk state',
            delay_seconds=delay_seconds))
        if ready:
            return result
        time.sleep(delay_seconds)
