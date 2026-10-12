"""Small CPU tests for candidate 7 dual dynamics and transactional recovery."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import training as T


class TrainingMechanismTests(unittest.TestCase):
    def test_dual_cost_ema_and_projection(self):
        raw, _, signal = T.update_price(0.4, 0.08, 0.03, 0.01, smoothing=False)
        smooth, ema, _ = T.update_price(0.4, 0.08, 0.03, 0.01, smoothing=True)
        self.assertAlmostEqual(raw, 0.45)
        self.assertAlmostEqual(signal, 0.08)
        self.assertAlmostEqual(ema, 0.017)
        self.assertAlmostEqual(smooth, 0.387)
        for smoothing in (False, True):
            price, ema, signal = T.update_price(0., 0.01, 0.03, None, smoothing=smoothing)
            self.assertEqual(price, 0.)
            self.assertEqual(ema, 0.01)
            self.assertEqual(signal, 0.01)
        with self.assertRaises(ValueError):
            T.update_price(0., np.nan, 0.03, None, smoothing=True)

    def test_factorial_only_changes_assigned_factors(self):
        settings = {name: T.settings_for(name) for name in T.VARIANTS}
        shared = [{k: v for k, v in row.items() if k not in ("critic_interval", "lambda_cost_smoothing")}
                  for row in settings.values()]
        for row in shared[1:]:
            self.assertEqual(row, shared[0])
        self.assertEqual({(s["critic_interval"], s["lambda_cost_smoothing"]) for s in settings.values()},
                         {(8, True), (8, False), (32, True), (32, False)})

    def test_validation_constraint_and_initial_tie(self):
        initial = dict(nominal_mean=.05, ood_worst_group_mean=.10)
        self.assertEqual(T.J._is_better(initial, initial, initial), (True, False))
        excessive = dict(nominal_mean=.0521, ood_worst_group_mean=.09)
        self.assertEqual(T.J._is_better(excessive, initial, initial), (False, False))
        improved = dict(nominal_mean=.0519, ood_worst_group_mean=.09)
        self.assertEqual(T.J._is_better(improved, initial, initial), (True, True))

    def test_checkpoint_restores_rng_optimizer_and_ema(self):
        torch.manual_seed(34)
        actor = torch.nn.Linear(2, 1)
        critic = torch.nn.Linear(2, 1)
        surrogate = torch.nn.Linear(2, 1)
        optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=.01)
        sur_optim = torch.optim.Adam(surrogate.parameters(), lr=.01)
        objects = (actor, critic, optim, None, None, surrogate, sur_optim)
        x = torch.tensor([[.3, .2], [.6, .7]])
        def train_once():
            optim.zero_grad(set_to_none=True)
            loss = (actor(x) + critic(x) - torch.randn(2, 1)).square().mean()
            loss.backward()
            optim.step()
            return copy.deepcopy(actor.state_dict())
        train_once()
        oracle = type("Oracle", (), {"ledger": {"actor_completed": 64}})()
        state = T._pack(7, .8, .03, objects, oracle)
        expected = train_once()
        with tempfile.TemporaryDirectory(dir=T.ROOT / "artifacts") as td:
            path = Path(td) / "checkpoint.pt"
            T.C.atomic_torch(path, state)
            state = torch.load(path, weights_only=False)
        T.J._restore(state, *objects)
        actual = train_once()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(state["ema_cost"], .03)
        self.assertEqual(state["update"], 7)
        self.assertEqual(state["oracle_ledger_snapshot"]["actor_completed"], 64)

    def test_training_and_evaluation_namespaces_are_disjoint(self):
        sets = []
        for family in ("tmp2-candidate07-train-u000001", "tmp2-candidate07-validation", "tmp2-candidate07-test"):
            pins = T.C.make_data(30, 8, "uniform", family, seed=101)
            sets.append({T.C.pin_set_sha256(x) for x in pins})
        for i in range(3):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])


if __name__ == "__main__":
    (T.ROOT / "artifacts").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
