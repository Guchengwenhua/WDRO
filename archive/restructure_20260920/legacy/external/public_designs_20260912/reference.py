"""Recoverable multi-CPU, full dual-scale GeoSteiner reference generation."""
import argparse
import concurrent.futures
import multiprocessing
import os
import sys
import time

from common import *
sys.path.insert(0, str(PACKAGE))
from src import geosteiner as gst

_INPUT_CACHE = {}

def solve_task(task):
    degree, start, end, _, _ = task
    with lease(f'reference_shards/d{degree}_{start}_{end}'):
        return solve_locked_task(task)

def solve_locked_task(task):
    degree, start, end, input_path, input_sha = task
    out = BASE / 'cache/references/shards' / f'd{degree}_{start:08d}_{end:08d}.npz'
    meta = out.with_suffix('.json')
    binding = dict(input_sha256=input_sha, start=start, end=end, reference_identity=gst.reference_identity(),
                   wrapper_sha256=sha(__file__), scales=[2**27, 2**28])
    if meta.exists():
        m = read(meta)
        assert m['binding'] == binding and sha(out) == m['output']['sha256']
        return m
    if input_path not in _INPUT_CACHE:
        _INPUT_CACHE[input_path] = load(input_path)
    data = _INPUT_CACHE[input_path]
    evaluator = gst.GeoSteinerEvaluator()
    t = time.monotonic()
    write(meta.with_suffix('.started.json'), dict(binding=binding, at=now(), pid=os.getpid(), planned_calls=2*(end-start)))
    primary = np.array([evaluator.solve(p, require_integral=True) for p in data['pins'][start:end]])
    try:
        gst.SCALE = 2**28
        secondary = np.array([evaluator.solve(p, require_integral=True) for p in data['pins'][start:end]])
    finally:
        gst.SCALE = 2**27
    np.testing.assert_array_equal(primary, secondary)
    npz(out, dict(net_ids=data['net_ids'][start:end], lstar=primary, verification_lstar=secondary))
    m = dict(status='pass', binding=binding, output=artifact(out), seconds=time.monotonic()-t, calls=2*(end-start), at=now())
    write(meta, m)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=24)
    args = p.parse_args()
    with lease('references'):
        pf, _ = preflight('dual-scale reference cache, CPU only')
        plan = read(BASE / 'cache/input_manifest.json')
        tasks = []
        for group in plan['groups']:
            a = group['input']
            assert sha(a['logical_path']) == a['sha256']
            for start in range(0, group['count'], 256):
                tasks.append((group['degree'], start, min(start+256, group['count']), a['logical_path'], a['sha256']))
        records = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(solve_task, t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                records.append(f.result())
                write(BASE / 'logs/references/heartbeat.json', dict(status='running', at=now(), completed_shards=len(records), total_shards=len(tasks), committed_calls=sum(x['calls'] for x in records)))
        groups = []
        for g in plan['groups']:
            parts = [load(BASE / 'cache/references/shards' / f'd{g["degree"]}_{start:08d}_{min(start+256,g["count"]):08d}.npz') for start in range(0,g['count'],256)]
            arrays = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            data = load(g['input']['logical_path'])
            np.testing.assert_array_equal(arrays['net_ids'], data['net_ids'])
            np.testing.assert_array_equal(arrays['lstar'], arrays['verification_lstar'])
            assert np.all(arrays['lstar'] > 0) and np.isfinite(arrays['lstar']).all()
            out = BASE / 'cache/references' / f'degree_{g["degree"]}.npz'
            npz(out, arrays)
            write(out.with_suffix('.json'), dict(status='complete', input=g['input'], reference=artifact(out),
                  reference_identity=gst.reference_identity(), scales=[2**27,2**28], completed_nets=g['count']))
            groups.append(dict(degree=g['degree'], count=g['count'], reference=artifact(out)))
        result = dict(status='pass', at=now(), input_manifest=artifact(BASE/'cache/input_manifest.json'), preflight=artifact(pf), workers=args.workers,
                      groups=groups, total_nets=sum(g['count'] for g in groups), calls=sum(x['calls'] for x in records),
                      checks=dict(all_optimal=True, full_two_scales_equal=True, connected_integer_embeddings=True),
                      summed_worker_seconds=sum(x['seconds'] for x in records), implementation=artifact(__file__), reference_identity=gst.reference_identity())
        write(BASE/'cache/reference_manifest.json', result)
        write(BASE/'logs/references/heartbeat.json', dict(status='complete', at=now(), completed_shards=len(records), total_shards=len(tasks)))
        print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
