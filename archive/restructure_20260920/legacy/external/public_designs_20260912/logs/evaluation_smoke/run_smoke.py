"""Reproducible real-input GPU resume, cross-device and idempotence audit."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/remote-home/Guilin/DRO/Codex/external_benchmarks/20260912_public')
OUT = BASE / 'logs/evaluation_smoke'
spec = importlib.util.spec_from_file_location('external_evaluate', BASE / 'evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
np = ev.np
sys.path.insert(0, str(ev.PACKAGE))
from src import geosteiner as gst


def capture(label):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True)
    info = dict(at=ev.now(), label=label, gpus=gpu, compute_apps=processes,
        disk=dict(zip(('total','used','free'), __import__('shutil').disk_usage(OUT))))
    ev.append_event(OUT / 'resources.jsonl', info)
    return info


def run(label, directory, gpu, pause=False):
    preflight = capture('before_' + label)
    selected = next(row for row in preflight['gpus'].splitlines() if row.split(',')[0].strip() == str(gpu))
    used = int(selected.split(',')[-2].strip())
    assert used < 4096, f'smoke GPU unexpectedly occupied: {selected}'
    command = ['bash', str(ev.PACKAGE / 'scripts/stage3_env.sh'), sys.executable, str(BASE / 'evaluate.py'),
        '--input', str(OUT / 'input.npz'), '--reference', str(OUT / 'reference.npz'),
        '--method', 'erm', '--degree', '50', '--seed', '101', '--gpu', str(gpu), '--output', str(directory)]
    if pause:
        command += ['--max-new-batches', '1']
    started = time.monotonic()
    result = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    (OUT / f'{label}.stdout.log').write_text(result.stdout)
    (OUT / f'{label}.stderr.log').write_text(result.stderr)
    receipt = dict(at=ev.now(), label=label, command=command, exit_code=result.returncode,
        wall_seconds=elapsed, stdout=ev.artifact(OUT / f'{label}.stdout.log'), stderr=ev.artifact(OUT / f'{label}.stderr.log'))
    ev.atomic_json(OUT / f'{label}.json', receipt)
    assert result.returncode == 0, result.stderr
    return receipt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ev.Lease(OUT / 'driver', -1) as lease:
        script_before = ev.artifact(BASE / 'evaluate.py')
        capture('preparation')
        parts, source_bindings = [], []
        for source in sorted((BASE / 'cache/designs').glob('ispd2008_gr_*/degree_50.npz')):
            source_bindings.append(ev.artifact(source))
            parts.append(ev.load_npz(source))
            if sum(len(p['pins']) for p in parts) >= 512:
                break
        assert sum(len(p['pins']) for p in parts) >= 512
        data = {key: np.concatenate([p[key] for p in parts])[:512] for key in parts[0]}
        ev.validate_inputs(data, 50)
        ev.atomic_npz(OUT / 'input.npz', data)
        evaluator = gst.GeoSteinerEvaluator()
        started = time.monotonic()
        primary = np.array([evaluator.solve(p, require_integral=True) for p in data['pins']])
        try:
            gst.SCALE = 2**28
            secondary = np.array([evaluator.solve(p, require_integral=True) for p in data['pins']])
        finally:
            gst.SCALE = 2**27
        np.testing.assert_array_equal(primary, secondary)
        reference_seconds = time.monotonic() - started
        ev.atomic_npz(OUT / 'reference.npz', dict(net_ids=data['net_ids'], lstar=primary, verification_lstar=secondary))
        ev.atomic_json(OUT / 'reference.json', dict(status='complete', input=ev.artifact(OUT / 'input.npz'),
            reference=ev.artifact(OUT / 'reference.npz'), reference_identity=gst.reference_identity(),
            scales=[2**27,2**28], completed_nets=512, calls=1024, seconds=reference_seconds,
            source_inputs=source_bindings, source_order='sorted design name, original net order, first 512'))
        lease.update(512, 'references_complete')
        # Two low-memory inference processes on the preflight-checked empty GPU2.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            full_future = pool.submit(run, 'continuous', OUT / 'continuous', 2)
            pause_future = pool.submit(run, 'interrupted', OUT / 'resumed', 2, True)
            while not (full_future.done() and pause_future.done()):
                capture('concurrent_inference')
                time.sleep(0.15)
            continuous, interrupted = full_future.result(), pause_future.result()
        state = ev.read_json(OUT / 'resumed/cursor.json')
        assert state['complete_nets'] == 256 and len(state['shards']) == 1
        first_shard = state['shards'][0]
        assert ev.read_json(OUT / 'resumed/heartbeat.json')['status'] == 'paused'
        assert ev.read_json(OUT / 'resumed/job.json')['lock_released']
        # The second batch is explicitly resumed on a different physical GPU.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            resume_future = pool.submit(run, 'cross_gpu_resume', OUT / 'resumed', 3)
            while not resume_future.done():
                capture('resume_inference')
                time.sleep(0.15)
            resumed = resume_future.result()
        before_predictions = ev.artifact(OUT / 'resumed/predictions.npz')
        before_result = ev.artifact(OUT / 'resumed/result.json')
        before_calls = ev.artifact(OUT / 'resumed/calls.jsonl')
        idempotent = run('idempotent', OUT / 'resumed', 3)
        assert before_predictions == ev.artifact(OUT / 'resumed/predictions.npz')
        assert before_result == ev.artifact(OUT / 'resumed/result.json')
        assert before_calls == ev.artifact(OUT / 'resumed/calls.jsonl')
        a, b = [ev.load_npz(OUT / name / 'predictions.npz') for name in ('continuous', 'resumed')]
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        ra, rb = [ev.read_json(OUT / name / 'result.json') for name in ('continuous', 'resumed')]
        assert ra['summary'] == rb['summary']
        final_cursor = ev.read_json(OUT / 'resumed/cursor.json')
        assert final_cursor['shards'][0] == first_shard
        assert [r['physical_gpu'] for r in final_cursor['shards']] == [2,3]
        assert script_before == ev.artifact(BASE / 'evaluate.py')
        capture('completed')
        audit = dict(status='pass', at=ev.now(), degree=50, method='erm', seed=101, nets=512,
            evaluator=script_before, input=ev.artifact(OUT / 'input.npz'), reference=ev.artifact(OUT / 'reference.json'),
            results=[ev.artifact(OUT / x / 'result.json') for x in ('continuous','resumed')],
            receipts=[continuous, interrupted, resumed, idempotent],
            reference_seconds=reference_seconds, all_prediction_fields=list(a),
            peak_reserved_mib=max(ra['peak_reserved_mib'],rb['peak_reserved_mib']),
            peak_allocated_mib=max(ra['peak_allocated_mib'],rb['peak_allocated_mib']),
            checks=dict(real_public_design_inputs=True, all_1024_reference_calls_equal=True,
                paused_after_256=True, resumed_on_physical_gpu_3=True, completed_512=True,
                all_arrays_exactly_equal=True, topology_actions_exactly_equal=True, summaries_exactly_equal=True,
                first_shard_reused_unchanged=True, idempotent_predictions_unchanged=True,
                idempotent_result_unchanged=True, idempotent_no_new_model_calls=True, evaluator_source_unchanged=True))
        ev.atomic_json(OUT / 'audit.json', audit)
        lease.update(512, 'complete')
        print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
