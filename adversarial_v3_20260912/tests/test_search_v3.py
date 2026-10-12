"""CPU contracts for v3 true-score search; no real models or solvers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wdro_rest_min"))

from adv_v3.common import legalize
from adv_v3.search import BUDGETS, antithetic_gradient, attack
from src.geometry import has_spacing_violation, normalize_net


class FakeOracle:
    def __init__(self, constant=False):
        self.requested = 0
        self.batches = []
        self.constant = constant
        self.device = "cpu"
        self.ledger = {}

    def evaluate(self, pins):
        self.requested += len(pins)
        self.batches.append(pins.copy())
        # Symmetric across pin order, with geometry-sensitive nonconvex values.
        regret = np.full(len(pins), 0.5) if self.constant else (
            0.5 + 0.08 * np.mean(np.sin(17 * pins[..., 0]) + np.cos(11 * pins[..., 1]), axis=1)
        )
        lstar = np.ones(len(pins))
        return {"regret": regret, "lstar": lstar, "length": lstar * (1 + regret)}


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        cls.centers = np.asarray([
            normalize_net(rng.uniform(size=(10, 2)), expected_degree=10)[0]
            for _ in range(2)
        ], dtype=np.float32)

    @staticmethod
    def proposer(centers, price, count, seed):
        # Deterministic, permutation equivariant proposals without invoking a
        # learned module. The callback contract concerns legal Z coordinates.
        amplitudes = np.linspace(0.002, 0.02, count)
        delta = amplitudes[None, :, None, None] * np.sin(centers[:, None] * 13)
        return legalize(centers, delta)

    def run_attack(self, method, centers=None, *, constant=False):
        oracle = FakeOracle(constant=constant)
        centers = self.centers if centers is None else centers
        result = attack(method, centers, oracle, 0.2, 73, proposer=self.proposer)
        return result, oracle

    def test_every_method_true_budget_legality_and_best_history(self):
        for method, budget in BUDGETS.items():
            with self.subTest(method=method):
                result, oracle = self.run_attack(method)
                self.assertEqual(oracle.requested, len(self.centers) * budget)
                self.assertEqual(result["calls_per_center"], budget)
                self.assertEqual(result["evaluated_pins"].shape, (2, budget, 10, 2))
                np.testing.assert_array_equal(result["evaluated_pins"][:, 0], self.centers)
                self.assertTrue(np.all(result["score"] >= result["original_score"]))
                self.assertTrue(all(not has_spacing_violation(x) for x in result["evaluated_pins"].reshape(-1, 10, 2)))
                self.assertEqual(result["pins"].dtype, np.float32)
                np.testing.assert_allclose(result["score"], result["regret"] - 0.2 * result["cost"])
                np.testing.assert_array_equal(result["pins"], result["evaluated_pins"][np.arange(2), result["selected_index"]])

    def test_nominal_wins_constant_regret_with_positive_price(self):
        for method in BUDGETS:
            with self.subTest(method=method):
                result, _ = self.run_attack(method, constant=True)
                np.testing.assert_array_equal(result["pins"], self.centers)
                np.testing.assert_array_equal(result["selected_index"], 0)

    def test_determinism_pin_equivariance_and_batch_independence(self):
        permutation = np.random.default_rng(9).permutation(10)
        for method in BUDGETS:
            with self.subTest(method=method):
                result, _ = self.run_attack(method)
                again, _ = self.run_attack(method)
                np.testing.assert_array_equal(result["evaluated_pins"], again["evaluated_pins"])
                permuted, _ = self.run_attack(method, self.centers[:, permutation])
                np.testing.assert_allclose(result["evaluated_pins"][:, :, permutation], permuted["evaluated_pins"], rtol=0, atol=2e-7)
                single, _ = self.run_attack(method, self.centers[:1])
                np.testing.assert_array_equal(result["evaluated_pins"][:1], single["evaluated_pins"])

    def test_es_exact_pair_and_center_schedule(self):
        result, oracle = self.run_attack("es64")
        self.assertEqual([len(x) // 2 for x in oracle.batches], [1] + [8, 1] * 7)
        self.assertEqual(result["diagnostics"]["rounds"], 7)
        self.assertFalse(result["diagnostics"]["history_dependent_duplicate_repair"])

    def test_es_does_not_modify_repeated_function_values(self):
        def collapse(centers, delta):
            if delta.ndim == 4:
                return np.repeat(centers[:, None], delta.shape[1], axis=1)
            return centers.copy()
        with patch("adv_v3.search.legalize", side_effect=collapse):
            result, oracle = self.run_attack("es64")
        self.assertEqual(oracle.requested, 128)
        np.testing.assert_array_equal(result["evaluated_pins"], np.repeat(self.centers[:, None], 64, axis=1))
        self.assertEqual(result["diagnostics"]["nominal_repeat_count"], 126)
        np.testing.assert_array_equal(result["diagnostics"]["mean_gradient_rms_by_round"], 0)

    def test_antithetic_estimator_linear_direction_and_offset_cancellation(self):
        # Coordinate quadrature with sqrt(d)*e_j has mean uu^T=I, making
        # the estimator exact for this linear objective, without MC tolerance.
        n = 3
        d = 2 * n
        directions = (np.sqrt(d) * np.eye(d)).reshape(1, d, n, 2)
        target = np.arange(1, d + 1, dtype=np.float64).reshape(1, n, 2)
        mu = 0.02
        values = np.einsum("bmnc,bnc->bm", directions, target)
        gradient = antithetic_gradient(13 + mu * values, 13 - mu * values, directions, mu)
        np.testing.assert_allclose(gradient, target, rtol=0, atol=1e-12)
        self.assertGreater(float(np.sum(gradient * target)), 0)

    def test_bad_inputs_fail_before_oracle(self):
        oracle = FakeOracle()
        for method, price, proposer in [("unknown", 0, None), ("es64", -1, None), ("rl4", 0, None)]:
            with self.assertRaises(ValueError):
                attack(method, self.centers, oracle, price, 73, proposer=proposer)
        self.assertEqual(oracle.requested, 0)
        bad = lambda x, p, c, s: np.zeros((len(x), c, x.shape[1], 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "normalized bounds"):
            attack("rl4", self.centers, oracle, 0, 73, proposer=bad)
        self.assertEqual(oracle.requested, 0)


if __name__ == "__main__":
    unittest.main()
