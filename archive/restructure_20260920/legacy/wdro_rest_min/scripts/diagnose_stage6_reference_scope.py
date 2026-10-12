"""Bounded coordinate-only audit of existing validation and training references."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.geosteiner import GeoSteinerEvaluator
from src.datasets import load_dataset
from src.training_data import materialize_training_batch
from src.protocol import load_protocol
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from stage6.evaluation import artifact

OUT = ROOT.parent / 'experiments/logs/stage6_negative_regret'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    assert preflight['preflight_passed']
    protocol = load_protocol()
    evaluator = GeoSteinerEvaluator()
    path = OUT / 'reference_scope.json'
    binding = dict(preflight=artifact(args.preflight), validation_prefix_per_cell=64,
                   training_prefix_per_degree=128, training_update=1, training_seed=101,
                   degrees=[10,20,30,40,50], scales=[1,2**27,2**28],
                   scope='coordinates and reference lengths only; no actor or model selection')
    state = json.loads(path.read_text()) if path.exists() else dict(status='running', binding=binding, groups=[])
    done = {row['name'] for row in state['groups']}
    calls = 0
    for degree in binding['degrees']:
        groups = []
        for distribution in protocol['data']['distributions']:
            source = ROOT.parent / f'experiments/cache/datasets/protocol_v1/validation/{distribution}_d{degree}_s202_n2000.npz'
            data, meta = load_dataset(source, verify=True)
            reference = ROOT.parent / f"experiments/cache/lstar/protocol_v1/{meta['array_bundle_sha256']}.npz"
            ref, _ = load_dataset(reference, verify=True)
            np.testing.assert_array_equal(data['net_ids'], ref['net_ids'])
            groups.append((f'validation:{distribution}:d{degree}', data['pins'][:64], data['net_ids'][:64],
                           ref['lstar'][:64], dict(dataset=artifact(source), reference=artifact(reference))))
        training = materialize_training_batch(protocol=protocol, degree=degree, training_seed=101, update=1, batch_size=256)
        groups.append((f'training:uniform:d{degree}:s101:u1', training.pins[:128], training.net_ids[:128], None,
                       dict(training_bundle=training.bundle_sha256)))
        for name, pins, ids, cache, source in groups:
            if name in done:
                continue
            points = pins.astype(np.float64)
            scaled = points * (2**27)
            assert np.array_equal(points, scaled / (2**27)) and np.array_equal(scaled, np.rint(scaled))
            old = evaluator.lengths(points); calls += len(points)
            new = evaluator.lengths(scaled) / (2**27); calls += len(points)
            check = evaluator.lengths(points * (2**28)) / (2**28); calls += len(points)
            if cache is not None:
                np.testing.assert_array_equal(old, cache)
            delta = old - new
            changed = np.flatnonzero(np.abs(delta) > 1e-10)
            row = dict(name=name, n=len(points), sources=source, default_matches_cache=cache is None or bool(np.array_equal(old, cache)),
                       scaling_round_trip_exact=True, two_integer_scales_identical=bool(np.array_equal(new, check)),
                       changed_count=len(changed), default_greater_count=int(np.sum(delta > 1e-10)),
                       default_lower_count=int(np.sum(delta < -1e-10)),
                       maximum_absolute_delta=float(np.max(np.abs(delta))),
                       changed=[dict(net_id=ids[i].decode(), index=int(i), default=float(old[i]), scaled_27=float(new[i]),
                                     scaled_28=float(check[i]), delta=float(delta[i])) for i in changed],
                       geosteiner_calls=3*len(points))
            state['groups'].append(row)
            state.update(updated_at_utc=utc_now(), pid=os.getpid(), invocation_calls=calls)
            atomic_write_json(path, state)
            print(json.dumps({k:v for k,v in row.items() if k not in ('sources','changed')}), flush=True)
    state.update(status='complete', completed_at_utc=utc_now(), total_nets=sum(row['n'] for row in state['groups']),
                 geosteiner_calls=sum(row['geosteiner_calls'] for row in state['groups']))
    atomic_write_json(path, state)

if __name__ == '__main__':
    main()
