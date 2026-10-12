import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.protocol import load_protocol
from src.training_data import materialize_training_batch
from src.baseline_data import balanced_training_batch, update_group_weights
from src.candidates import generate_support, fixed_surrogate, shortlist_support, select_hard_mining, ReferenceCache
from src.geometry import pin_set_sha256, has_spacing_violation

ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.config = json.loads((ROOT / "configs/stage3_baselines_v1.json").read_text())

    def batch(self, update=1, count=4):
        return materialize_training_batch(protocol=self.protocol, degree=10, training_seed=101, update=update, batch_size=count)

    def test_balanced_counts_ids_and_shared_uniform_slice(self):
        mixed = balanced_training_batch(protocol=self.protocol, degree=10, training_seed=101, update=1, batch_size=256)
        nominal = self.batch(count=256)
        self.assertTrue(np.array_equal(mixed.pins[:64], nominal.pins[:64]))
        self.assertTrue(np.array_equal(mixed.net_ids[:64], nominal.net_ids[:64]))
        for group, name in enumerate(("uniform", "mix_gauss", "edge_bias", "grid_cluster")):
            ids = mixed.net_ids[group*64:(group+1)*64]
            self.assertTrue(all(i.decode().startswith(f"train_u000001:{name}:") for i in ids))
        again = balanced_training_batch(protocol=self.protocol, degree=10, training_seed=101, update=1, batch_size=256)
        self.assertEqual(mixed.bundle_sha256, again.bundle_sha256)

    def test_group_weights_increase_hardest_and_resume_stably(self):
        weights = np.full(4, -np.log(4))
        losses = np.asarray([0.0, 0.1, 0.2, 0.3])
        for _ in range(500):
            weights, q = update_group_weights(weights, losses, 0.01)
        self.assertTrue(np.all(np.diff(q) > 0))
        self.assertAlmostEqual(q.sum(), 1.0)
        resumed, _ = update_group_weights(json.loads(json.dumps(weights.tolist())), losses, 0.01)
        uninterrupted, _ = update_group_weights(weights, losses, 0.01)
        np.testing.assert_array_equal(resumed, uninterrupted)
        equal, _ = update_group_weights(np.full(4, -np.log(4)), np.ones(4)*1e6, 0.01)
        np.testing.assert_allclose(equal, np.full(4, -np.log(4)), atol=1e-10)
        with self.assertRaises(ValueError):
            update_group_weights(weights, [0, 1, np.nan, 2], 0.01)

    def test_candidate_identity_legality_and_permutation(self):
        batch = self.batch()
        support = generate_support(batch, training_seed=101, update=1, search=self.config["search"])
        again = generate_support(batch, training_seed=101, update=1, search=self.config["search"])
        self.assertEqual(support.sha256, again.sha256)
        self.assertEqual(support.pins.shape, (4, 33, 10, 2))
        np.testing.assert_array_equal(support.pins[:, 0], batch.pins)
        for candidates in support.pins:
            self.assertEqual(len({pin_set_sha256(net) for net in candidates}), 33)
            self.assertFalse(any(has_spacing_violation(net) for net in candidates))
            self.assertTrue(np.isfinite(candidates).all())
            self.assertGreaterEqual(float(candidates.min()), 0.1)
            self.assertLessEqual(float(candidates.max()), 0.9)
        from dataclasses import replace
        permuted = replace(batch, pins=batch.pins[:, ::-1].copy())
        other = generate_support(permuted, training_seed=101, update=1, search=self.config["search"])
        for original, perm in zip(support.pins.reshape(-1, 10, 2), other.pins.reshape(-1, 10, 2)):
            self.assertEqual(pin_set_sha256(original), pin_set_sha256(perm))
        changed = generate_support(batch, training_seed=101, update=11, search=self.config["search"])
        self.assertNotEqual(support.sha256, changed.sha256)

    def test_fixed_surrogate_known_mst_and_nominal_top_four(self):
        square = np.asarray([[[0.1,0.1],[0.1,0.9],[0.9,0.1],[0.9,0.9]]])
        self.assertAlmostEqual(float(fixed_surrogate(square)[0]), 0.5)
        support = generate_support(self.batch(), training_seed=101, update=1, search=self.config["search"])
        shortlist, scores = shortlist_support(support)
        self.assertEqual(shortlist.shape, (4,4))
        self.assertTrue(np.all(shortlist[:, 0] == 0))
        for ids, values in zip(shortlist, scores):
            self.assertEqual(len(set(ids)), 4)
            np.testing.assert_array_equal(ids[1:], np.argsort(-values[1:], kind="stable")[:3]+1)

    def test_hard_mining_per_center_uses_true_regret_without_cost(self):
        ids = np.asarray([[0,8,2,9],[0,3,5,6]])
        scores = np.asarray([[0.0,0.5,0.3,0.4],[0.9,0.1,0.5,0.4]])
        selected, columns = select_hard_mining(ids,scores)
        np.testing.assert_array_equal(selected, [8,0])
        np.testing.assert_array_equal(columns, [1,0])

    def test_reference_cache_content_key_and_restart(self):
        class Exact:
            def lengths(self, pins):
                return np.ptp(pins, axis=1).sum(axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"lstar.sqlite"
            pins = self.batch().pins
            cache = ReferenceCache(path)
            first, misses = cache.lengths(pins, Exact())
            self.assertEqual(misses, 4)
            cache.close()
            cache = ReferenceCache(path)
            second, misses = cache.lengths(pins[:,::-1], Exact())
            self.assertEqual(misses, 0)
            np.testing.assert_array_equal(first, second)
            cache.close()


if __name__ == "__main__":
    unittest.main()
