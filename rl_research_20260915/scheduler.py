"""Audited, bounded queue for this isolated research branch.

The scheduler never signals, adopts, or restarts an already-live worker.  PID
identity requires Linux boot ID and /proc start ticks; file existence is not
completion evidence.  Only this branch's queue and worker artifacts are written.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OPS = ROOT / 'artifacts/operations'
_HASH_CACHE = {}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial.{os.getpid()}.{time.time_ns()}')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sha256(path):
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def ident(pid, proc_root=Path('/proc')):
    """Return kernel identity, treating zombies/exited processes as absent."""
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        fields = (Path(proc_root) / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            return None
        return dict(pid=pid, start_ticks=fields[19],
                    boot_id=(Path(proc_root) / 'sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_identity(current, saved):
    return bool(current and saved and all(current.get(k) == saved.get(k)
                for k in ('pid', 'start_ticks', 'boot_id')))


def lock_held(path):
    """Probe the kernel flock; a leftover filename alone is harmless."""
    path = Path(path)
    if not path.exists():
        return False
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def acquire_scheduler_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        prior_text = stream.read()
        try:
            prior = json.loads(prior_text) if prior_text.strip() else None
        except json.JSONDecodeError:
            prior = {'unparseable_stale_metadata': prior_text[:1024]}
        if prior and same_identity(ident(prior.get('pid', -1)), prior):
            raise RuntimeError('Scheduler owner is still live even though lock was released')
        record = dict(**ident(os.getpid()), acquired_at=time.time(), recovered_stale_metadata=prior)
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(record, indent=2))
        stream.flush()
        os.fsync(stream.fileno())
        return stream
    except BaseException:
        stream.close()
        raise


def discover_live_workers(root=ROOT, proc_root=Path('/proc')):
    """Cover the crash window between Popen and writing the queue receipt."""
    found = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            # Scheduler always launches an absolute runner path.  Smoke jobs
            # are outside this queue and cannot own production run locks.
            if str(Path(root) / 'runner.py') in argv and '--smoke' not in argv:
                current = ident(int(entry.name), proc_root)
                if current:
                    found.append(dict(identity=current, command=argv))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def ensure_no_prior_workers(ts, root=ROOT, identity_fn=ident, discover_fn=discover_live_workers):
    root = Path(root)
    ops = root / 'artifacts/operations'
    queue = ops / 'queue.json'
    old = json.loads(queue.read_text()) if queue.exists() else {}
    records = list(old.get('running', {}).values())
    for path in ops.glob('*_launch.json'):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and 'identity' in record:
            records.append(record)
    for task in ts:
        out = root / 'artifacts/runs' / task['id']
        if lock_held(out / 'run.lock'):
            raise RuntimeError(f"Prior worker holds run lock: {task['id']}")
        heartbeat = out / 'heartbeat.json'
        if heartbeat.exists():
            h = json.loads(heartbeat.read_text())
            records.append({'identity': h, 'pid': h.get('pid')})
    for record in records:
        saved = record.get('identity') or {}
        pid = record.get('pid', saved.get('pid'))
        if pid is None:
            continue
        current = identity_fn(pid)
        # Records from an older scheduler may lack boot_id.  If start ticks
        # match a live PID, refuse rather than infer that it is safe to replay.
        if same_identity(current, saved) or (current and not saved.get('boot_id')
                and current.get('start_ticks') == saved.get('start_ticks')):
            raise RuntimeError(f'Prior live worker; refusing duplicate queue launch: {pid}')
    found = discover_fn(root)
    if found:
        raise RuntimeError(f'Prior unrecorded live worker; refusing duplicate queue launch: {found}')
    return old


def snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpus = []
    for line in raw.splitlines():
        i, uuid, free, util = [x.strip() for x in line.split(',')]
        try:
            gpus.append(dict(index=int(i), uuid=uuid, free_mib=int(free), util=int(util)))
        except ValueError:
            continue  # Unknown utilization/free memory cannot authorize admission.
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                             '--format=csv'], text=True, timeout=15)
    except (subprocess.SubprocessError, OSError) as error:
        processes = f'process query unavailable: {error}'
    return dict(time=time.time(), gpus=gpus, load=os.getloadavg(), free_bytes=shutil.disk_usage(ROOT).free,
                processes=processes, logical_path=str(ROOT), resolved_path=str(ROOT.resolve()))


def tasks(protocol):
    ts = []
    for seed in protocol['seeds']:
        warm = f'warmup_s{seed}'
        ts.append(dict(id=warm, kind='warmup', method='erm', seed=seed, deps=[]))
        for method in protocol['learned_methods']:
            ts.append(dict(id=f'fixed_{method}_s{seed}', kind='fixed', method=method, seed=seed, deps=[warm]))
        for method in protocol['coev_methods']:
            deps = [warm]
            if method not in ('erm', 'random', 'es'):
                parent = 'ppo' if method == 'population_ppo' else method
                deps.append(f'fixed_{parent}_s{seed}')
            ts.append(dict(id=f'coev_{method}_s{seed}', kind='coev', method=method, seed=seed, deps=deps))
    if len({t['id'] for t in ts}) != len(ts):
        raise ValueError('Duplicate queue task IDs')
    known = {t['id'] for t in ts}
    if any(not set(t['deps']) <= known for t in ts):
        raise ValueError('Queue has unknown parent tasks')
    return sorted(ts, key=lambda t: ({'warmup': 0, 'fixed': 1, 'coev': 2}[t['kind']], t['seed'], t['method']))


def expected_binding(root=ROOT):
    root = Path(root)
    model_dir = Path('/remote-home/Guilin/DRO/Codex/shared/legacy_dro/WDRO-REST/models')
    dependencies = {str(p): sha256(p) for p in sorted(model_dir.glob('*.py'))}
    if not dependencies:
        raise RuntimeError('REST model dependency manifest is empty')
    return dict(protocol_sha256=sha256(root / 'protocol.json'),
                code_sha256={str(root / name): sha256(root / name) for name in ('runner.py', 'policies.py', 'env_geometry.py')},
                dependencies=dependencies)


def verify_source_freeze(root=ROOT):
    """Optional source manifest: {sha256: {absolute-or-root-relative-path: digest}}."""
    root = Path(root)
    path = root / 'artifacts/operations/source_freeze.json'
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    hashes = record.get('sha256', record.get('files'))
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError('Source freeze must contain a nonempty sha256 mapping')
    for name, digest in hashes.items():
        file = Path(name)
        file = file if file.is_absolute() else root / file
        if not file.is_file() or sha256(file) != digest:
            raise RuntimeError(f'Source freeze mismatch: {name}')
    return True


def validate_complete(task, protocol, root=ROOT, expected=None, identity_fn=ident):
    """Return (accepted, reason); never trust complete.json existence alone."""
    root = Path(root)
    out = root / 'artifacts/runs' / task['id']
    marker = out / 'complete.json'
    if not marker.exists():
        return False, 'missing_complete'
    try:
        complete = json.loads(marker.read_text())
        for key, value in dict(status='complete', run_id=task['id'], kind=task['kind'],
                               method=task['method'], seed=task['seed'],
                               step=protocol[task['kind'] + '_updates'], smoke=False).items():
            if key not in complete or type(complete[key]) is not type(value) or complete[key] != value:
                raise ValueError(f'metadata {key} mismatch')
        identity = complete['identity']
        if complete['device'] != 'cuda:0' or identity['device'] != 'cuda:0':
            raise ValueError('GPU device evidence missing')
        gpu = identity['physical_gpu_index']
        if type(gpu) is not int or gpu < 0 or str(gpu) != identity['CUDA_VISIBLE_DEVICES']:
            raise ValueError('Physical GPU mapping mismatch')
        if (type(identity['pid']) is not int or identity['pid'] <= 0
                or not str(identity['start_ticks']).isdigit() or not identity['boot_id']):
            raise ValueError('Incomplete process identity')
        if same_identity(identity_fn(identity['pid']), identity):
            return False, 'worker_still_live'
        if lock_held(out / 'run.lock'):
            return False, 'run_lock_held'
        if complete['peak_allocated_bytes'] <= 0:
            raise ValueError('No CUDA allocation evidence')
        if complete['calls'].get('oracle_calls') != 0:
            raise ValueError('Invalid oracle call count')
        expected = expected_binding(root) if expected is None else expected
        binding = complete['binding']
        for name in ('protocol_sha256', 'code_sha256', 'dependencies'):
            if binding.get(name) != expected[name]:
                raise ValueError(f'{name} mismatch')
        parents = {str(root / 'artifacts/runs' / parent / 'latest.pt'):
                   sha256(root / 'artifacts/runs' / parent / 'latest.pt') for parent in task['deps']}
        if binding.get('parents') != parents:
            raise ValueError('Parent checkpoint binding mismatch')
        names = {'latest.pt', 'run.json', 'progress.jsonl', 'actual_calls.jsonl'}
        if task['kind'] != 'warmup':
            names.add('results.npz')
        hashes = complete['artifacts_sha256']
        if set(hashes) != names:
            raise ValueError('Artifact manifest mismatch')
        for name, digest in hashes.items():
            path = out / name
            if not path.is_file() or not path.stat().st_size or sha256(path) != digest:
                raise ValueError(f'Artifact hash mismatch: {name}')
        run = json.loads((out / 'run.json').read_text())
        if (run['run_id'] != task['id'] or run['smoke'] is not False
                or run['binding'] != binding or run['identity'] != identity):
            raise ValueError('Run metadata mismatch')
        return True, 'validated'
    except (KeyError, TypeError, ValueError, OSError) as error:
        return False, str(error)


def validated_done(ts, protocol, root=ROOT, expected=None, running=()):
    done, invalid = set(), {}
    # tasks() is topological; a child can never be accepted if its parent failed.
    for task in ts:
        if task['id'] in running:
            continue
        ok, reason = validate_complete(task, protocol, root, expected)
        if ok:
            if set(task['deps']) <= done:
                done.add(task['id'])
            else:
                invalid[task['id']] = 'Unvalidated dependency'
        elif reason != 'missing_complete':
            invalid[task['id']] = reason
    return done, invalid


def ready_tasks(ts, done, running, attempts, max_attempts=3):
    return [t for t in ts if t['id'] not in done and t['id'] not in running
            and set(t['deps']) <= done and attempts.get(t['id'], 0) < max_attempts]


def admission(snapshot_now, history, running, resources):
    """Return capacity, eligible cards and reasons; no device/process effects."""
    max_workers = min(4, int(resources['max_workers']))
    capacity = max(0, max_workers - len(running))
    reserve = float(resources['reserve_disk_gib']) * 2**30
    worker_reserve = max(.25, float(resources.get('worker_disk_gib', 1))) * 2**30
    disk_slots = max(0, math.floor((snapshot_now['free_bytes'] - reserve) / worker_reserve) - len(running))
    capacity = min(capacity, disk_slots)
    load_limit = float(resources.get('max_load', 60))
    if snapshot_now['load'][0] > load_limit:
        capacity = 0
    used = {v['gpu'] for v in running.values()}
    available, reasons = [], {}
    for gpu in snapshot_now['gpus']:
        index = gpu['index']
        if index in used:
            reasons[index] = 'own_worker_already_on_card'
            continue
        if gpu['free_mib'] < max(6144, int(resources['min_free_mib'])):
            reasons[index] = 'insufficient_free_memory'
            continue
        samples = []
        for old in history[-3:]:
            match = next((g for g in old['gpus'] if g['uuid'] == gpu['uuid']), None)
            if match is not None:
                samples.append(match['util'])
        if not samples:
            samples = [gpu['util']]
        rolling = sum(samples) / len(samples)
        if rolling <= min(85, float(resources['max_admission_util'])):
            duty = 1.0
        elif len(samples) >= 3 and resources.get('saturated_duty_fraction', 0) > 0:
            duty = min(.25, float(resources['saturated_duty_fraction']))
        else:
            reasons[index] = 'utilization_over_limit_or_insufficient_history'
            continue
        available.append(dict(gpu, rolling_util=rolling, duty_fraction=duty))
    available.sort(key=lambda g: (-g['duty_fraction'], g['rolling_util'], -g['free_mib']))
    return capacity, available, reasons


def serial_running(running):
    return {key: {k: v for k, v in value.items() if k not in ('process', 'stream')}
            for key, value in running.items()}


def main():
    OPS.mkdir(parents=True, exist_ok=True)
    lock = acquire_scheduler_lock(OPS / 'scheduler.lock')
    running, attempts, done = {}, {}, set()
    try:
        protocol = json.loads((ROOT / 'protocol.json').read_text())
        ts = tasks(protocol)
        resources = protocol['resources']
        old = ensure_no_prior_workers(ts)
        attempts = {key: int(value) for key, value in old.get('attempts', {}).items()}
        frozen = verify_source_freeze()
        expected = expected_binding()
        history = []
        atomic(OPS / 'launch.json', dict(identity=ident(os.getpid()), time=time.time(), command=sys.argv,
                                       tasks=ts, source_freeze_verified=frozen, binding=expected))
        while True:
            verify_source_freeze()
            if expected_binding() != expected:
                raise RuntimeError('Source/protocol changed while queue was active')
            for key in list(running):
                item = running[key]
                code = item['process'].poll()
                if code is not None:
                    item['stream'].close()
                    atomic(OPS / (key + f'_exit_a{attempts[key]}.json'),
                           dict(returncode=code, identity=item['identity'], time=time.time()))
                    del running[key]
            done, invalid = validated_done(ts, protocol, expected=expected, running=running)
            if invalid:
                atomic(OPS / 'queue.json', dict(status='invalid_completion', invalid=invalid,
                       complete=sorted(done), total=len(ts), running=serial_running(running), attempts=attempts, time=time.time()))
                return 1
            if len(done) == len(ts) and not running:
                atomic(OPS / 'queue.json', dict(status='auditing', complete=sorted(done), total=len(ts), running={}, attempts=attempts))
                result = subprocess.run(['bash', str(ROOT / 'run_env.sh'), str(ROOT / 'report.py')], cwd=ROOT)
                atomic(OPS / 'queue.json', dict(status='complete' if result.returncode == 0 else 'audit_failed',
                       complete=sorted(done), total=len(ts), running={}, attempts=attempts, report_exit=result.returncode, time=time.time()))
                return result.returncode
            failed = [t['id'] for t in ts if attempts.get(t['id'], 0) >= 3
                      and t['id'] not in done and t['id'] not in running]
            if failed and not running:
                atomic(OPS / 'queue.json', dict(status='failed', stopped=True,
                       stop_reason='retry_limit_exhausted', failed=failed, complete=sorted(done),
                       total=len(ts), running={}, attempts=attempts, time=time.time()))
                return 1
            try:
                snap = snapshot()
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                atomic(OPS / 'queue.json', dict(status='resource_probe_failed', error=str(error),
                       complete=sorted(done), total=len(ts), running=serial_running(running), attempts=attempts, time=time.time()))
                time.sleep(15)
                continue
            history.append(snap)
            history = history[-3:]
            atomic(OPS / 'resources_latest.json', snap)
            with (OPS / 'resource_history.jsonl').open('a') as stream:
                stream.write(json.dumps(snap) + '\n')
            ready = ready_tasks(ts, done, running, attempts)
            # Re-read resources immediately before every launch.  Decisions
            # reserve one card and disk budget for each worker already admitted.
            for task in ready:
                try:
                    fresh = snapshot()
                except (subprocess.SubprocessError, OSError, ValueError) as error:
                    atomic(OPS / 'resource_probe_error.json', dict(error=str(error), time=time.time()))
                    break
                fresh_history = history[:-1] + [fresh]
                capacity, available, reasons = admission(fresh, fresh_history, running, resources)
                if capacity <= 0 or not available:
                    break
                gpu = available[0]
                key = task['id']
                attempts[key] = attempts.get(key, 0) + 1
                tag = f'{key}_a{attempts[key]}'
                atomic(OPS / (tag + '_preflight.json'), dict(snapshot=fresh, rolling_history=fresh_history,
                       admission=gpu, resources=resources, exclusion_reasons=reasons))
                out = ROOT / 'artifacts/runs' / key
                out.mkdir(parents=True, exist_ok=True)
                if lock_held(out / 'run.lock'):
                    raise RuntimeError(f'Worker lock became occupied before launch: {key}')
                cmd = ['nice', '-n', '10', 'bash', str(ROOT / 'run_env.sh'), str(ROOT / 'runner.py'),
                       '--kind', task['kind'], '--method', task['method'], '--seed', str(task['seed']),
                       '--physical-gpu-index', str(gpu['index']), '--duty-fraction', str(gpu['duty_fraction'])]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu['index']))
                stream = (out / f'console_a{attempts[key]}.log').open('a')
                try:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    raise
                running[key] = dict(process=proc, stream=stream, pid=proc.pid, identity=ident(proc.pid),
                                    gpu=gpu['index'], gpu_uuid=gpu['uuid'], duty_fraction=gpu['duty_fraction'], command=cmd, launch=time.time())
                atomic(OPS / (tag + '_launch.json'), serial_running(running)[key])
                # Persist every admission, not only the last worker in a batch.
                atomic(OPS / 'queue.json', dict(status='running', complete=sorted(done), total=len(ts),
                       running=serial_running(running), attempts=attempts, time=time.time()))
            atomic(OPS / 'queue.json', dict(status='running' if running else 'waiting_resources', complete=sorted(done),
                   total=len(ts), running=serial_running(running), attempts=attempts, time=time.time()))
            time.sleep(15)
    except BaseException as error:
        atomic(OPS / 'scheduler_error.json', dict(error=repr(error), identity=ident(os.getpid()),
               running=serial_running(running), attempts=attempts, complete=sorted(done), time=time.time()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
