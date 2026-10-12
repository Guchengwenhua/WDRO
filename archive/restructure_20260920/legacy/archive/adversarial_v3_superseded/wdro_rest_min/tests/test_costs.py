from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.costs import hungarian_l1_mean, symmetric_chamfer_l1


COSTS = (hungarian_l1_mean, symmetric_chamfer_l1)


class CostContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.asarray([[0.1, 0.1], [0.4, 0.2], [0.7, 0.8], [0.9, 0.4]])

    def test_1_identity_is_zero(self) -> None:
        for cost in COSTS:
            self.assertAlmostEqual(cost(self.x, self.x), 0.0)

    def test_2_permutation_invariance(self) -> None:
        permutation = [2, 0, 3, 1]
        for cost in COSTS:
            self.assertAlmostEqual(cost(self.x, self.x[permutation]), 0.0)

    def test_3_small_translation_has_expected_l1_cost(self) -> None:
        shifted = self.x + np.asarray([0.01, 0.02])
        for cost in COSTS:
            self.assertAlmostEqual(cost(self.x, shifted), 0.03, places=12)

    def test_4_perturbation_amplitude_is_monotone(self) -> None:
        for cost in COSTS:
            small = cost(self.x, self.x + np.asarray([0.005, 0.0]))
            large = cost(self.x, self.x + np.asarray([0.02, 0.0]))
            self.assertGreater(large, small)

    def test_5_distinct_structure_does_not_collapse(self) -> None:
        clustered = np.asarray([[0.45, 0.45], [0.46, 0.46], [0.47, 0.47], [0.48, 0.48]])
        for cost in COSTS:
            self.assertGreater(cost(self.x, clustered), 0.05)

    def test_6_mean_scaling_is_comparable_across_degree(self) -> None:
        x2 = np.asarray([[0.2, 0.2], [0.7, 0.7]])
        z2 = x2 + np.asarray([0.01, 0.02])
        x6 = np.repeat(x2, 3, axis=0)
        z6 = x6 + np.asarray([0.01, 0.02])
        for cost in COSTS:
            self.assertAlmostEqual(cost(x2, z2), cost(x6, z6), places=12)

    def test_7_duplicates_are_finite_but_nan_and_degree_mismatch_are_rejected(self) -> None:
        duplicate = np.asarray([[0.2, 0.2], [0.2, 0.2], [0.8, 0.8]])
        shifted = duplicate + np.asarray([0.01, 0.0])
        for cost in COSTS:
            self.assertTrue(np.isfinite(cost(duplicate, shifted)))
            with self.assertRaisesRegex(ValueError, "finite"):
                cost(duplicate, np.asarray([[np.nan, 0.2], [0.2, 0.2], [0.8, 0.8]]))
            with self.assertRaisesRegex(ValueError, "same degree"):
                cost(duplicate, shifted[:2])


if __name__ == "__main__":
    unittest.main()

