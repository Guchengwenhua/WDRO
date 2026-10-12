"""Fixed-shape inference accounting and cross-method baseline rejection."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from adv_v3 import common, report, worker


class ShapeSensitiveActor(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.batches = []
        self.fail = fail

    def forward(self, pins, deterministic):
        self.batches.append(np.array(pins, copy=True))
        if self.fail:
            raise RuntimeError("injected forward failure")
        # Deliberately batch-size-sensitive: without fixed padding, partitioning
        # the same rows changes their exact scores.
        code = torch.from_numpy(pins[:, 0, 0].copy()) + len(pins) / 1024
        return code[:, None], None


class FakeRest:
    def __init__(self):
        self.rows = 0

    def lengths(self, pins, outputs):
        self.rows += len(pins)
        return 2 + outputs[:, 0].astype(np.float64)


class FakeReference:
    def __init__(self, workers):
        self.closed = False

    def close(self):
        self.closed = True


class FakeCache:
    def __init__(self, path):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE calls (requested INT, misses_started INT, misses_completed INT)")
        self.rows = 0

    def lengths(self, pins, reference):
        self.rows += len(pins)
        self.db.execute("INSERT INTO calls VALUES (?,?,?)", (len(pins), len(pins), len(pins)))
        return np.full(len(pins), 2.), None

    def close(self):
        self.db.close()


class PaddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(common, "RestTreeEvaluator", FakeRest),
                        patch.object(common, "ParallelGeoSteinerEvaluator", FakeReference),
                        patch.object(common, "ReferenceCache", FakeCache)]
        for context in self.patches:
            context.start()

    def tearDown(self):
        for context in reversed(self.patches):
            context.stop()
        self.temp.cleanup()

    def evaluate_parts(self, points, sizes, name):
        actor = ShapeSensitiveActor()
        actor.train()
        oracle = common.Oracle(actor, "cpu", self.root / name)
        results, offset = [], 0
        for size in sizes:
            results.append(oracle.evaluate(points[offset:offset + size]))
            offset += size
        self.assertEqual(offset, len(points))
        self.assertTrue(actor.training)
        self.assertTrue(all(len(batch) == 256 for batch in actor.batches))
        self.assertEqual(oracle.rest.rows, len(points))
        self.assertEqual(oracle.cache.rows, len(points))
        expected_forward = sum(((size + 255) // 256) * 256 for size in sizes)
        for suffix in ("requested", "completed"):
            self.assertEqual(oracle.ledger["actor_" + suffix], len(points))
            self.assertEqual(oracle.ledger["actor_forward_rows_" + suffix], expected_forward)
            self.assertEqual(oracle.ledger["actor_padding_rows_" + suffix], expected_forward - len(points))
            self.assertEqual(oracle.ledger["rest_" + suffix], len(points))
        self.assertEqual(oracle.ledger["reference_requests"], len(points))
        report.ledger_details(oracle.ledger)
        recorded = json.loads(oracle.ledger_path.read_text())
        self.assertEqual(recorded, oracle.ledger)
        oracle.close()
        self.assertTrue(oracle.geosteiner.closed)
        return {key: np.concatenate([result[key] for result in results]) for key in results[0]}

    def test_partition_invariance_and_real_vs_padding_accounting(self):
        points = np.random.default_rng(9).uniform(.1, .9, (519, 3, 2)).astype(np.float32)
        full = self.evaluate_parts(points, [519], "full")
        partitioned = self.evaluate_parts(points, [1, 255, 263], "partitioned")
        for key in full:
            np.testing.assert_array_equal(full[key], partitioned[key])

    def test_exact_multiple_does_not_add_padding_or_references(self):
        points = np.full((512, 3, 2), .5, dtype=np.float32)
        self.evaluate_parts(points, [256, 256], "exact")

    def test_failed_forward_keeps_requested_rows_and_no_false_completion(self):
        oracle = common.Oracle(ShapeSensitiveActor(fail=True), "cpu", self.root / "failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            oracle.evaluate(np.full((3, 3, 2), .5, dtype=np.float32))
        ledger = json.loads(oracle.ledger_path.read_text())
        self.assertEqual(ledger["actor_requested"], 3)
        self.assertEqual(ledger["actor_forward_rows_requested"], 256)
        self.assertEqual(ledger["actor_padding_rows_requested"], 253)
        self.assertEqual(ledger.get("actor_completed", 0), 0)
        self.assertEqual(ledger["reference_requests"], 0)
        self.assertEqual(oracle.rest.rows, 0)
        self.assertEqual(report.ledger_details(ledger)["actor_forward_rows_uncertain_requests"], 256)
        oracle.close()

    def test_report_preserves_counters_and_checks_optional_padding_bounds(self):
        legacy = dict(actor_requested=3, actor_completed=3, rest_requested=3, rest_completed=3)
        report.ledger_details(legacy)
        ledger = dict(legacy, actor_forward_rows_requested=256, actor_forward_rows_completed=256,
                      actor_padding_rows_requested=253, actor_padding_rows_completed=253,
                      custom_counter=17)
        result = report.ledger_details(ledger)
        self.assertEqual(result["custom_counter"], 17)
        for key, value in ledger.items():
            self.assertEqual(result[key], value)
        with self.assertRaisesRegex(report.AuditError, "bounds"):
            report.ledger_details(dict(ledger, actor_padding_rows_completed=254))
        with self.assertRaisesRegex(report.AuditError, "accounting"):
            report.ledger_details(dict(ledger, actor_padding_rows_requested=252,
                                       actor_padding_rows_completed=252))


class NominalGuardTests(unittest.TestCase):
    def test_mismatch_writes_diagnostic_without_oracle_calls_or_overwriting_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, diagnostic = root / "baseline.npz", root / "nominal_mismatch.json"
            payload = dict(centers=np.zeros((2, 3, 2)), original_regret=np.array([.1, .2]))
            common.atomic_npz(baseline, **payload)
            original_bytes = baseline.read_bytes()
            with patch.object(common.Oracle, "evaluate", side_effect=AssertionError("unexpected scoring")) as evaluate:
                worker.check_nominal(payload, baseline, diagnostic, "matching")
                self.assertFalse(diagnostic.exists())
                broken = dict(payload, original_regret=np.array([.1, .21]))
                with self.assertRaisesRegex(RuntimeError, "Cross-method nominal greedy mismatch"):
                    worker.check_nominal(broken, baseline, diagnostic, "rl4, uniform, 0")
                evaluate.assert_not_called()
            evidence = json.loads(diagnostic.read_text())
            self.assertEqual(evidence["context"], "rl4, uniform, 0")
            self.assertEqual(evidence["indices"], [1])
            self.assertEqual(evidence["expected"], [.2])
            self.assertEqual(evidence["actual"], [.21])
            self.assertEqual(baseline.read_bytes(), original_bytes)

    def test_existing_npz_guard_and_center_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, result, diagnostic = root / "baseline.npz", root / "resumed.npz", root / "mismatch.json"
            payload = dict(centers=np.zeros((2, 3, 2)), original_regret=np.array([.1, .2]))
            common.atomic_npz(baseline, **payload)
            common.atomic_npz(result, **dict(payload, original_regret=np.array([.2, .2])))
            with np.load(result, allow_pickle=False) as existing:
                with self.assertRaisesRegex(RuntimeError, "Cross-method nominal"):
                    worker.check_nominal(existing, baseline, diagnostic, "resumed rl4")
            with self.assertRaises(AssertionError):
                worker.check_nominal(dict(payload, centers=np.ones((2, 3, 2))), baseline,
                                     diagnostic, "wrong centers")


if __name__ == "__main__":
    unittest.main()
