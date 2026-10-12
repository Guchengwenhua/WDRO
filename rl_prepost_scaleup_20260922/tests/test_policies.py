"""CPU tests of budget accounting, policy gradients, invariance and recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rl_research_policies_under_test", ROOT / "policies.py")
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
torch.set_num_threads(1)


class Scorer:
    def __init__(self):
        self.count = 0
        self.batches = []

    def __call__(self, x):
        self.count += len(x)
        self.batches.append(x.copy())
        # Smooth permutation-invariant objective with useful nonzero gradients.
        return (2*x[..., 0] + .5*x[..., 1]**2).sum(1)


def parameters(module):
    return torch.cat([v.detach().cpu().reshape(-1) for v in module.parameters()])


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


class TestPolicies(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(112)
        self.centers = np.random.default_rng(18).uniform(.15, .85, (5, 6, 2)).astype(np.float32)
        self.config = dict(width=12, epochs=2, replay_batch=12, replay_capacity=64)

    def make(self, method):
        return p.LearnedAttacker(method, 6, "cpu", self.config)

    def test_all_methods_train_finite_and_exact_budget(self):
        for method in sorted(p.LearnedAttacker.METHODS):
            with self.subTest(method=method):
                attack = self.make(method)
                before = parameters(attack.actor)
                scorer = Scorer()
                result = attack.search(self.centers, .2, scorer, np.random.default_rng(13), train=True)
                self.assertEqual(scorer.count, 40)
                self.assertEqual(result["diagnostics"]["scored_coordinates"], 40)
                expected_batches = 8 if method in {"mappo", "sequential_ppo"} else 2
                self.assertEqual(len(scorer.batches), expected_batches)
                self.assertTrue(np.array_equal(result["candidates"][:, 0], self.centers))
                self.assertTrue(np.isfinite(result["objective"]).all())
                self.assertTrue(np.isfinite(list(v for v in result["diagnostics"].values() if isinstance(v, (float, int)))).all())
                self.assertFalse(torch.equal(before, parameters(attack.actor)), "actor did not learn")
                z = result["candidates"]
                self.assertLessEqual(float(np.abs(z-self.centers[:, None]).max()), .0800001)
                self.assertGreaterEqual(float(z.min()), .1)
                self.assertLessEqual(float(z.max()), .9)
                np.testing.assert_allclose(result["objective"], result["lengths"]-.2*p.costs(self.centers, z), rtol=0, atol=1e-12)
                np.testing.assert_array_equal(result["index"], result["objective"].argmax(1))
                self.assertTrue(np.all(result["objective"].max(1) >= result["objective"][:, 0]))
                if method in {"mappo", "sequential_ppo"}:
                    self.assertLess(result["diagnostics"]["telescoping_error"], 1e-12)

    def test_eval_does_not_mutate_and_is_rng_reproducible(self):
        for method in sorted(p.LearnedAttacker.METHODS):
            with self.subTest(method=method):
                attack = self.make(method)
                state = attack.state_dict()
                a = attack.search(self.centers, .2, Scorer(), np.random.default_rng(201))
                b = attack.search(self.centers, .2, Scorer(), np.random.default_rng(201))
                np.testing.assert_array_equal(a["candidates"], b["candidates"])
                self.assertTrue(equal_tree(state, attack.state_dict()))

    def test_replacement_baseline_exact_with_center_and_ties(self):
        objective = np.array([[3., 1., 5., 4.], [8., 2., 4., 1.], [2., 5., 5., 0.]])
        np.testing.assert_array_equal(p.replacement_advantages(objective), [[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
        np.testing.assert_array_equal(p.replacement_advantages(np.array([[3., 5.], [7., 2.]])), [[2.], [0.]])

    def test_permutation_equivariance_and_lam_conditioning(self):
        attack = self.make("ppo")
        x = torch.from_numpy(self.centers)
        d = torch.randn_like(x)*.01
        lam = torch.linspace(.1, 3., len(x))
        context = torch.rand(len(x), 3)
        perm = torch.tensor([4, 1, 5, 0, 3, 2])
        with torch.no_grad():
            original = attack.actor(x, d, lam, context)
            permuted = attack.actor(x[:, perm], d[:, perm], lam, context)
            torch.testing.assert_close(permuted, original[:, perm], rtol=1e-5, atol=1e-7)
            # Value starts at zero; randomize its final layer before the test.
            torch.nn.init.normal_(attack.value.head[-1].weight)
            torch.testing.assert_close(attack.value(x, d, lam, context),
                                       attack.value(x[:, perm], d[:, perm], lam, context), rtol=1e-5, atol=1e-6)
            self.assertFalse(torch.equal(original, attack.actor(x, d, lam+2, context)))

    def test_resume_replay_optimizer_and_next_step_exact(self):
        for method in sorted(p.LearnedAttacker.METHODS):
            with self.subTest(method=method):
                first = self.make(method)
                rng = np.random.default_rng(100)
                first.search(self.centers, .3, Scorer(), rng, train=True)
                state = first.state_dict()
                rng_state = deepcopy(rng.bit_generator.state)
                second = self.make(method)
                second.load_state_dict(state)
                resumed_rng = np.random.default_rng()
                resumed_rng.bit_generator.state = rng_state
                a = first.search(self.centers, .3, Scorer(), rng, train=True)
                b = second.search(self.centers, .3, Scorer(), resumed_rng, train=True)
                np.testing.assert_array_equal(a["candidates"], b["candidates"])
                self.assertTrue(equal_tree(first.state_dict(), second.state_dict()))
                # Exported snapshots must not alias live tensors/replay buffers.
                self.assertFalse(equal_tree(state, first.state_dict()))
                if method == "maddpg_bandit":
                    self.assertEqual(len(first.replay["reward"]), 64)
                    self.assertFalse(torch.equal(parameters(first.target_actor), parameters(first.actor)))

    def test_geometry_matches_existing_geometry_and_duplicate_fallback(self):
        geometry_path = ROOT.parent / "rl_only_20260915" / "geometry.py"
        spec = importlib.util.spec_from_file_location("rl_only_geometry_for_policy_test", geometry_path)
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        rng = np.random.default_rng(71)
        delta = np.clip(rng.normal(0, .1, (5, 7, 6, 2)), -.08, .08)
        np.testing.assert_array_equal(p.legalize(self.centers, delta), g.legalize(self.centers, delta))
        np.testing.assert_array_equal(p.costs(self.centers, p.legalize(self.centers, delta)),
                                      g.costs(self.centers, g.legalize(self.centers, delta)))
        centers = np.array([[[.4, .4], [.41, .41]]], dtype=np.float32)
        deltas = np.array([[[.01, .01], [0., 0.]]], dtype=np.float32)
        np.testing.assert_array_equal(p.legalize(centers, deltas), centers)

    def test_single_proposal_and_per_center_lambda(self):
        for method in sorted(p.LearnedAttacker.METHODS):
            with self.subTest(method=method):
                attack = self.make(method)
                score = Scorer()
                result = attack.search(self.centers, np.arange(5), score, np.random.default_rng(4), train=True, k=2)
                self.assertEqual(score.count, 10)
                self.assertEqual(result["candidates"].shape, (5, 2, 6, 2))

    def test_learning_improves_known_analytic_objective(self):
        # This checks reward/gradient direction against a held-out random draw,
        # rather than merely checking that optimizer parameters changed.
        centers = np.random.default_rng(9).uniform(.25, .75, (32, 6, 2)).astype(np.float32)
        for method in sorted(p.LearnedAttacker.METHODS):
            with self.subTest(method=method):
                torch.manual_seed(221)
                attack = p.LearnedAttacker(method, 6, "cpu", dict(width=12, epochs=2, replay_batch=64))
                before = attack.search(centers, .2, Scorer(), np.random.default_rng(14))["diagnostics"]["best_gain_mean"]
                training_rng = np.random.default_rng(15)
                for _ in range(40):
                    attack.search(centers, .2, Scorer(), training_rng, train=True)
                after = attack.search(centers, .2, Scorer(), np.random.default_rng(14))["diagnostics"]["best_gain_mean"]
                self.assertGreater(after, before + .02)

    def test_invalid_scorer_and_inputs_rejected(self):
        attack = self.make("ppo")
        with self.assertRaises(ValueError):
            attack.search(self.centers, -1, Scorer(), np.random.default_rng(2))
        with self.assertRaises(ValueError):
            attack.search(self.centers, 1, lambda x: np.nan*np.ones(len(x)), np.random.default_rng(2))
        with self.assertRaises(ValueError):
            attack.search(self.centers, 1, Scorer(), np.random.default_rng(2), k=1)
        duplicate = self.centers.copy()
        duplicate[:, 0] = duplicate[:, 1]
        with self.assertRaises(ValueError):
            attack.search(duplicate, 1, Scorer(), np.random.default_rng(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
