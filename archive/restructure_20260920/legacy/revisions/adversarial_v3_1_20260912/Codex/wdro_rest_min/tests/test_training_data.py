import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol import load_protocol
from src.training_data import EMPTY_STREAM_SHA256, extend_stream_hash, materialize_training_batch


class TrainingPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def test_stateless_pool_is_fixed_and_update_disjoint(self) -> None:
        first = materialize_training_batch(
            protocol=self.protocol, degree=30, training_seed=101, update=1, batch_size=8
        )
        repeated = materialize_training_batch(
            protocol=self.protocol, degree=30, training_seed=101, update=1, batch_size=8
        )
        second = materialize_training_batch(
            protocol=self.protocol, degree=30, training_seed=101, update=2, batch_size=8
        )
        self.assertEqual(first.bundle_sha256, repeated.bundle_sha256)
        np.testing.assert_array_equal(first.pins, repeated.pins)
        self.assertNotEqual(first.bundle_sha256, second.bundle_sha256)
        self.assertTrue(set(first.net_ids).isdisjoint(set(second.net_ids)))
        self.assertTrue(all(value.decode().startswith("train_u000001:") for value in first.net_ids))
        self.assertFalse(any(value.decode().startswith("validation:") for value in first.net_ids))
        self.assertFalse(any(value.decode().startswith("test:") for value in first.net_ids))
        self.assertGreaterEqual(float(first.pins.min()), 0.1)
        self.assertLessEqual(float(first.pins.max()), 0.9)

    def test_stream_hash_is_order_sensitive(self) -> None:
        left = "1" * 64
        right = "2" * 64
        self.assertEqual(
            extend_stream_hash(EMPTY_STREAM_SHA256, left),
            extend_stream_hash(EMPTY_STREAM_SHA256, left),
        )
        self.assertNotEqual(
            extend_stream_hash(extend_stream_hash(EMPTY_STREAM_SHA256, left), right),
            extend_stream_hash(extend_stream_hash(EMPTY_STREAM_SHA256, right), left),
        )


if __name__ == "__main__":
    unittest.main()
