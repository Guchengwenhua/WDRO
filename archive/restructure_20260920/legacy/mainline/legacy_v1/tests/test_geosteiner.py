from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geosteiner import GeoSteinerEvaluator, ParallelGeoSteinerEvaluator
from src.geometry import inverse_net, normalize_net


class GeoSteinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = GeoSteinerEvaluator()

    def test_scale_equivalence_under_inverse_transform(self) -> None:
        pins = np.asarray([[10.0, 5.0], [14.0, 5.0], [12.0, 8.0], [11.0, 7.0]])
        normalized, transform = normalize_net(pins)
        restored = inverse_net(normalized, transform)
        original_length = self.evaluator.length(pins)
        restored_length = self.evaluator.length(restored)
        normalized_length = self.evaluator.length(normalized)
        self.assertAlmostEqual(original_length, restored_length, places=10)
        self.assertAlmostEqual(original_length, normalized_length / transform.scale, places=9)

    def test_parallel_evaluator_is_exact_and_order_preserving(self) -> None:
        rng = np.random.default_rng(91)
        nets = rng.random((6, 10, 2))
        expected = self.evaluator.lengths(nets)
        with ParallelGeoSteinerEvaluator(2) as evaluator:
            actual = evaluator.lengths(nets)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
