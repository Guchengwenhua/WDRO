"""Statistics/audit fixtures never consult final experiment performance."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report as r


class ReportTests(unittest.TestCase):
    def test_shared_center_draws_preserve_cross_seed_cancellation(self):
        x = np.arange(32).reshape(4, 8)/100
        values = np.stack((x, np.zeros_like(x), -x))
        result = r.paired_bootstrap(values)
        self.assertAlmostEqual(result["estimate"], 0, places=15)
        self.assertEqual(result["ci95_low"], 0)
        self.assertEqual(result["simultaneous_high"], 0)

    def test_bonferroni_intervals_contain_pointwise_intervals(self):
        values = np.random.default_rng(7).normal(size=(3, 4, 64))
        result = r.paired_bootstrap(values)
        self.assertLessEqual(result["simultaneous_low"], result["ci95_low"])
        self.assertGreaterEqual(result["simultaneous_high"], result["ci95_high"])
        self.assertEqual(result, r.paired_bootstrap(values))

    def test_primary_requires_all_seeds_and_correct_lower_better_sign(self):
        rows, distributions, seeds = [], [], []
        r.add_contrast(rows, distributions, seeds, "D7_x", "a", "b", np.full((3,4,8), -.01), "lower", True)
        self.assertEqual(rows[-1]["conclusion"], "simultaneous_supported_improvement")
        mixed = np.full((3, 4, 8), -.01); mixed[0] = .001
        r.add_contrast(rows, distributions, seeds, "D7_x", "a", "b", mixed, "lower", True)
        self.assertEqual(rows[-1]["conclusion"], "inconclusive")

    def test_formal_source_hash_remains_strict(self):
        with tempfile.TemporaryDirectory(dir=r.EXP/"audit") as directory:
            path = Path(directory)/"source.py"; path.write_text("x=1\n")
            binding = dict(path=str(path), sha256="0"*64)
            audit = r.Audit()
            with self.assertRaisesRegex(AssertionError, "hash mismatch"):
                audit.source_binding(binding, smoke=False)
            audit.source_binding(binding, smoke=True)
            self.assertEqual(len(audit.smoke_source_drift), 1)

    def test_default_report_rejects_incomplete_formal_tasks(self):
        with tempfile.TemporaryDirectory(dir=r.EXP/"audit") as directory, patch.object(r, "EXP", Path(directory)):
            with self.assertRaisesRegex(RuntimeError, "requires all 16"):
                r.main([])
            self.assertFalse((Path(directory)/"summary.json").exists())

    def test_transport_known_shift_and_pin_permutation(self):
        x = np.stack((np.linspace(.1,.9,30), np.linspace(.1,.8,30)), axis=-1)[None]
        candidates = np.stack((x[0, ::-1], x[0]+.01))[None]
        np.testing.assert_allclose(r.transport(x,candidates), [[0,.02]], rtol=0, atol=1e-14)

    def test_csv_figures_render_all_three_png_and_pdf_pairs(self):
        with tempfile.TemporaryDirectory(dir=r.EXP/"audit") as directory:
            path = Path(directory); tables = path/"tables"
            rows, distributions, seeds = [], [], []
            for label in ("D1", "D2", "D3", "D4", "D5", "D6", "no_move", "random_direction"):
                r.add_contrast(rows, distributions, seeds, label, "a", "b", np.full((3,4,8), .001), "higher", label.startswith("D"))
            for label in ("D7_refresh8_raw", "D7_refresh32_ema", "D7_refresh8_ema"):
                r.add_contrast(rows, distributions, seeds, label, "a", "b", np.full((3,4,8), -.001), "lower", True)
            r.csv_write(tables/"contrasts.csv", rows)
            dynamics = [dict(seed=s, variant=v, update=t, lambda_after=.01*t,
                             mean_cost=.015+.001*t, rho=.02) for s in r.SEEDS for v in r.VARIANTS for t in range(1,5)]
            metrics = [dict(seed=s, variant=v, checkpoint=c, distribution=d, mean_regret=.01,
                            p95_regret=.02) for s in r.SEEDS for v in r.VARIANTS
                       for c in ("initial", "best", "latest") for d in r.DISTS]
            r.csv_write(tables/"training_dynamics.csv", dynamics)
            r.csv_write(tables/"training_metrics.csv", metrics)
            outputs = r.figures(tables, path/"figures")
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(p.stat().st_size > 1000 for p in outputs))


if __name__ == "__main__":
    unittest.main()
