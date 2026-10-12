"""Read-only real GPU greedy-tree batch-shape regression diagnostic."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adv_v3.common import atomic_json, atomic_npz, load_parent, RestTreeEvaluator, sha256

OLD = Path('/remote-home/Guilin/DRO/Codex/adversarial_v3_20260912')
OUT = ROOT / 'experiments/diagnostics/actor_batchshape_probe'


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(123)
    actor, _, _, meta = load_parent('erm', 10, 101, torch.device('cuda:0'))
    actor.eval()
    inputs = sorted((OLD/'experiments/fixed/erm_d10_s101/inputs').glob('*.npz'))
    points = np.concatenate([np.load(p)['pins'] for p in inputs])
    assert len(points) == 1024
    rest = RestTreeEvaluator()
    arrays = {'pins': points}

    def predict(pins, request_batch, padded):
        outputs = []
        with torch.no_grad():
            for request_start in range(0, len(pins), request_batch):
                request = pins[request_start:request_start+request_batch]
                step = 256 if padded else request_batch
                for start in range(0, len(request), step):
                    chunk = request[start:start+step]
                    count = len(chunk)
                    if padded and count < 256:
                        chunk = np.concatenate((chunk, np.repeat(chunk[-1:], 256-count, axis=0)))
                    result, _ = actor(chunk, True)
                    outputs.append(result[:count].cpu().numpy())
        return np.concatenate(outputs)

    reference_tree = predict(points, 256, False)
    reference_length = rest.lengths(points, reference_tree)
    arrays.update(reference_tree=reference_tree, reference_length=reference_length)
    comparisons = []
    def compare(name, tree):
        length = rest.lengths(points, tree)
        tree_differences = np.flatnonzero(np.any(tree != reference_tree, axis=1))
        length_differences = np.flatnonzero(length != reference_length)
        record = dict(name=name, tree_exact=not len(tree_differences),
                      rest_length_exact=not len(length_differences),
                      tree_difference_count=len(tree_differences),
                      tree_difference_indices=tree_differences.tolist(),
                      length_difference_count=len(length_differences),
                      length_difference_indices=length_differences.tolist(),
                      max_abs_length_difference=float(np.max(np.abs(length-reference_length))))
        comparisons.append(record)
        arrays[name+'_tree'] = tree
        arrays[name+'_length'] = length
        print(json.dumps(record), flush=True)
        return record

    for batch in [64, 128, 256, 512, 63, 129]:
        compare('native_'+str(batch), predict(points, batch, False))
    rng = np.random.default_rng(730912)
    orders = {'identity': np.arange(len(points)), 'reverse': np.arange(len(points))[::-1],
              'random': rng.permutation(len(points))}
    failures = []
    for order_name, order in orders.items():
        inverse = np.argsort(order)
        for batch in [64, 128, 256, 512, 63, 129]:
            name = f'padded256_request{batch}_{order_name}'
            result = compare(name, predict(points[order], batch, True)[inverse])
            if not result['tree_exact'] or not result['rest_length_exact']:
                failures.append(name)
    atomic_npz(OUT/'outputs.npz', **arrays)
    result = dict(status='passed' if not failures else 'failed', physical_gpu_index=2,
                  pid=os.getpid(), started_at=started, finished_at=time.time(),
                  count=len(points), degree=10, parent='erm', seed=101,
                  parent_checkpoint=meta['best'],
                  input_artifacts=[dict(path=str(p), sha256=sha256(p)) for p in inputs],
                  script_sha256=sha256(__file__),
                  output_sha256=sha256(OUT/'outputs.npz'), comparisons=comparisons,
                  failed_padded_cases=failures,
                  inference_mode='greedy eval no_grad deterministic algorithms',
                  real_rest_library_sha256=sha256(rest.library_path),
                  maximum_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  maximum_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    atomic_json(OUT/'result.json', result)
    print(json.dumps({'status': result['status'], 'seconds': time.time()-started,
                      'failed_padded_cases': failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
