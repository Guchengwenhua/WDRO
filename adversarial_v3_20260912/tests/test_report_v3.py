import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adv_v3 import report
from adv_v3.common import costs


class ReportTests(unittest.TestCase):
    def fixture(self):
        x = np.asarray([[[.1, .1], [.9, .3], [.4, .8]],
                        [[.1, .1], [.9, .8], [.2, .3]]], dtype=np.float32)
        z = x.copy()
        z[:, 2, 0] += .02
        c = costs(x, z[:, None])[:, 0]
        regret = np.array([.04, .06])
        data = dict(centers=x, pins=z, regret=regret, lstar=np.ones(2), length=1 + regret,
                    price=np.full(2, .2), cost=c, score=regret-.2*c,
                    original_regret=np.array([.01, .02]), original_score=np.array([.01, .02]),
                    calls=np.full(2, 4))
        return x, data

    def test_selected_scores_costs_and_budget_are_independently_checked(self):
        x, data = self.fixture()
        report.audit_arrays(data, x, degree=3, budget=4, price=.2)
        for key, changed, message in (("cost", data["cost"]+.01, "Hungarian"),
                                      ("score", data["score"]+.01, "score"),
                                      ("calls", np.array([4, 64]), "budget")):
            broken = copy.deepcopy(data)
            broken[key] = changed
            with self.assertRaisesRegex(report.AuditError, message):
                report.audit_arrays(broken, x, degree=3, budget=4, price=.2)

    def test_original_lower_bound_and_input_identity(self):
        x, data = self.fixture()
        broken = copy.deepcopy(data)
        broken["original_regret"] = broken["original_score"] = np.ones(2)
        with self.assertRaisesRegex(report.AuditError, "below original"):
            report.audit_arrays(broken, x, degree=3, budget=4, price=.2)
        with self.assertRaisesRegex(report.AuditError, "identity"):
            report.audit_arrays(data, x[::-1], degree=3, budget=4, price=.2)

    def test_paired_seed_resampling_and_effect_sign(self):
        base = np.arange(30, dtype=float).reshape(3, 10)
        result = report.paired_bootstrap(base + 2, base, seed=44, repeats=2000)
        self.assertEqual(result["difference"], 2)
        self.assertAlmostEqual(result["ci_low"], 2)
        self.assertAlmostEqual(result["ci_high"], 2)
        self.assertTrue(result["all_seeds_positive"])
        self.assertEqual(result["centers"], 10)
        self.assertEqual(result["seed_count"], 3)
        repeated = report.paired_bootstrap(base + 2, base, seed=44, repeats=2000, metric="p95")
        self.assertAlmostEqual(repeated["difference"], 2)

    def test_incomplete_status_cannot_be_a_pass(self):
        protocol = {"parents": ["erm", "wdro_rest"], "degrees": [10,20,30,40,50], "seeds": [101,102,103]}
        with tempfile.TemporaryDirectory() as temporary:
            status = report.phase_status("fixed", protocol, Path(temporary))
            self.assertEqual(status["status"], "incomplete")
            self.assertEqual(status["expected_tasks"], 30)
            self.assertEqual(len(status["missing"]), 30)

    def test_source_split_overlap_fails_and_hash_is_permutation_invariant(self):
        x, _ = self.fixture()
        self.assertEqual(report.array_keys(x), report.array_keys(x[:, ::-1]))
        with self.assertRaisesRegex(report.AuditError, "overlap"):
            report.finish_split_audit({3: {"train": {"a"}, "validation": {"a"}, "evaluation": {"b"}}})


if __name__ == "__main__":
    unittest.main()
