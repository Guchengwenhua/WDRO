from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import paired_bootstrap_mean_difference, relative_regret, summarize_groups


class MetricTests(unittest.TestCase):
    def test_relative_regret_is_per_net_not_ratio_of_means(self) -> None:
        rest = np.asarray([2.0, 6.0])
        optimal = np.asarray([1.0, 4.0])
        np.testing.assert_allclose(relative_regret(rest, optimal), [1.0, 0.5])

    def test_worst_group_uses_group_mean(self) -> None:
        summary = summarize_groups(np.asarray([0.1, 0.2, 0.8]), ["a", "a", "b"])
        self.assertEqual(summary["worst_group"], "b")
        self.assertAlmostEqual(summary["worst_group_mean"], 0.8)

    def test_paired_bootstrap_is_reproducible(self) -> None:
        candidate = np.asarray([0.1, 0.2, 0.3, 0.4])
        baseline = candidate + 0.05
        first = paired_bootstrap_mean_difference(candidate, baseline, samples=500, seed=7)
        second = paired_bootstrap_mean_difference(candidate, baseline, samples=500, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], -0.05)

    def test_nonpositive_optimal_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            relative_regret(np.asarray([1.0]), np.asarray([0.0]))


if __name__ == "__main__":
    unittest.main()

