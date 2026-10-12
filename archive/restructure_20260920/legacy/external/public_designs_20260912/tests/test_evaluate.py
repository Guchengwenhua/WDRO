"""CPU-only geometry, provenance and crash/recovery integrity tests."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('external_evaluate', ROOT / 'evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class AdapterIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix='evaluate_test_', dir=ROOT))
        self.pins = np.array([[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]], dtype=np.float32)
        self.data = dict(pins=self.pins, net_ids=np.array(['public:chip:n1:d3']),
            design_ids=np.array(['chip']), dataset_ids=np.array(['public']), family_ids=np.array(['real']),
            source_index=np.array([42], dtype=np.int64), offsets=np.array([[500., 1000.]]), scales=np.array([0.008]))
        self.topology = np.array([[0, 1, 1, 2]], dtype=np.int64)
        length = ev.python_tree_lengths(self.pins, self.topology)
        self.reference = dict(net_ids=self.data['net_ids'].copy(), lstar=length.copy(), verification_lstar=length.copy())
        self.arrays = dict(self.data, **{k: v for k, v in self.reference.items() if k != 'net_ids'},
            topology=self.topology, actions=self.topology.copy(), actor_log_probability=np.array([-0.1]),
            rest_lengths=length, regret=np.zeros(1), rest_lengths_inverse=length/self.data['scales'],
            lstar_inverse=length/self.data['scales'])

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_legal_geometry_metadata_and_reference(self):
        ev.validate_inputs(self.data, 3)
        ev.validate_reference(self.reference, self.data)
        ev.audit_predictions(self.arrays, self.data, self.reference)
        expected = 2 * (float(np.float32(0.9)) - float(np.float32(0.1)))
        self.assertEqual(float(self.arrays['rest_lengths'][0]), expected)

    def test_invalid_spacing_degree_and_reference_order(self):
        with self.assertRaisesRegex(ValueError, 'shape/degree'):
            ev.validate_inputs(self.data, 10)
        bad = {**self.data, 'pins': self.pins.copy()}
        bad['pins'][0, 1] = bad['pins'][0, 0]
        with self.assertRaisesRegex(ValueError, 'spacing'):
            ev.validate_inputs(bad, 3)
        with self.assertRaisesRegex(ValueError, 'identity/order'):
            ev.validate_reference({**self.reference, 'net_ids': np.array(['wrong'])}, self.data)
        with self.assertRaisesRegex(ValueError, 'dual scales'):
            ev.validate_reference({**self.reference, 'verification_lstar': self.reference['lstar'] + 0.01}, self.data)

    def test_disconnected_cycle_and_bad_index_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cycle/self-loop'):
            ev.python_tree_lengths(self.pins, np.array([[0, 1, 1, 0]]))
        with self.assertRaisesRegex(ValueError, 'out of range'):
            ev.python_tree_lengths(self.pins, np.array([[0, 1, 1, 3]]))

    def test_prediction_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            ev.audit_predictions({**self.arrays, 'rest_lengths': self.arrays['rest_lengths'] + 1}, self.data, self.reference)
        with self.assertRaisesRegex(ValueError, 'identity changed: family_ids'):
            ev.audit_predictions({**self.arrays, 'family_ids': np.array(['wrong'])}, self.data, self.reference)

    def test_atomic_roundtrip_shard_cursor_and_hash_corruption(self):
        path = self.directory / 'shard.npz'
        ev.atomic_npz(path, self.arrays)
        binding = ev.artifact(path)
        state = dict(complete_nets=1, shards=[dict(start=0, end=1, artifact=binding)])
        rebuilt = ev.read_shards(state, self.data, self.reference)
        self.assertTrue(np.array_equal(rebuilt['topology'], self.topology))
        with self.assertRaisesRegex(ValueError, 'cursor'):
            ev.read_shards({**state, 'complete_nets': 0}, self.data, self.reference)
        with self.assertRaisesRegex(ValueError, 'noncontiguous'):
            ev.read_shards({**state, 'shards': [dict(start=1, end=2, artifact=binding)]}, self.data, self.reference)
        with path.open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            ev.read_shards(state, self.data, self.reference)
        self.assertFalse(list(self.directory.glob('*.partial')))

    def test_duplicate_job_and_release_on_error(self):
        with self.assertRaisesRegex(RuntimeError, 'intentional interruption'):
            with ev.Lease(self.directory, 0):
                with self.assertRaisesRegex(RuntimeError, 'duplicate'):
                    with ev.Lease(self.directory, 0):
                        self.fail('duplicate lease acquired')
                raise RuntimeError('intentional interruption')
        job = json.loads((self.directory / 'job.json').read_text())
        self.assertEqual(job['status'], 'failed')
        self.assertTrue(job['lock_released'])
        with ev.Lease(self.directory, 0) as lease:
            lease.update(1, 'complete')
        self.assertEqual(json.loads((self.directory / 'job.json').read_text())['outcome'], 'complete')

    def test_canonical_model_binding_rejects_changed_checkpoint(self):
        model = ev.load_model(ev.FROZEN, 'erm', 10, 101)
        modified = json.loads(json.dumps(model))
        modified['best']['sha256'] = '0' * 64
        index = self.directory / 'models.json'
        ev.atomic_json(index, {'models': [modified]})
        with self.assertRaisesRegex(ValueError, 'differs from accepted'):
            ev.load_model(index, 'erm', 10, 101)

    def test_python_audit_against_existing_precision_v2_predictions_all_degrees(self):
        # These independently generated historical predictions are a real-data
        # regression fixture. Never write to any frozen input or output.
        base = ev.REVISION / 'experiments'
        total = 0
        for degree in (10, 20, 30, 40, 50):
            data = ev.load_npz(base / f'cache/stage7/inputs/degree_{degree}.npz')
            arrays = ev.load_npz(base / f'logs/stage7/evaluation/erm_d{degree}_s101/predictions.npz')
            ev.validate_inputs(data, degree)
            np.testing.assert_array_equal(ev.python_tree_lengths(data['pins'], arrays['topology']), arrays['rest_lengths'])
            total += len(data['pins'])
        self.assertEqual(total, 22498)


if __name__ == '__main__':
    unittest.main(verbosity=2)
