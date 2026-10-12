"""Independently enumerate a finite policy to audit the best-of-K gradient.

This tests an expectation identity, not just examples of winner selection.
No reference routing solver, model checkpoint, or GPU is used.
"""
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("objective_identity_policies", ROOT / "policies.py")
POLICIES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICIES
SPEC.loader.exec_module(POLICIES)


class BestOfGradientIdentity(unittest.TestCase):
    def test_exact_categorical_enumeration(self):
        # Independent actions share trainable categorical logits.  Repeated
        # reward values exercise ties; centers can dominate all proposals.
        for count in (1, 2, 3, 5):
            for center, support in ((.2, (-1., .4, 1.7)),
                                    (3., (-1., .4, 1.7)),
                                    (-2., (-1., .4, 1.7)),
                                    (.2, (-1., 1.7, 1.7))):
                with self.subTest(count=count, center=center, support=support):
                    logits = torch.tensor([-.6, .1, .8], dtype=torch.float64, requires_grad=True)
                    probability = logits.softmax(dim=0)
                    actions = torch.tensor(list(itertools.product(range(3), repeat=count)))
                    weights = probability[actions].prod(dim=1)
                    rewards = np.array(support)[actions.numpy()]
                    objectives = np.column_stack((np.full(len(rewards), center), rewards))
                    maximum = torch.as_tensor(objectives.max(axis=1), dtype=torch.float64)
                    exact_return = (weights * maximum).sum()
                    exact_gradient, = torch.autograd.grad(exact_return, logits)

                    # grad(log p(action)) = one_hot(action) - p.  Other
                    # actions are held fixed in each conditional baseline.
                    scores = torch.nn.functional.one_hot(actions, num_classes=3).to(torch.float64)
                    scores = scores - probability.detach()[None, None, :]
                    advantage = torch.as_tensor(POLICIES.replacement_advantages(objectives))
                    estimator = (advantage[:, :, None] * scores).sum(dim=1)
                    expected_estimator = (weights.detach()[:, None] * estimator).sum(dim=0)
                    torch.testing.assert_close(expected_estimator, exact_gradient, rtol=1e-12, atol=1e-12)
                    self.assertAlmostEqual(float(weights.detach().sum()), 1., places=12)

    def test_bernoulli_closed_form(self):
        # With rewards {0,1} and center=0, J=1-(1-p)^K and
        # dJ/dlogit=K*p*(1-p)^K.  This provides an additional analytic check.
        for count in (1, 2, 4):
            probability = .37
            actions = np.asarray(list(itertools.product((0., 1.), repeat=count)))
            weights = np.prod(np.where(actions == 1., probability, 1-probability), axis=1)
            objectives = np.column_stack((np.zeros(len(actions)), actions))
            advantages = POLICIES.replacement_advantages(objectives)
            actual = np.sum(weights * np.sum(advantages * (actions-probability), axis=1))
            expected = count * probability * (1-probability)**count
            self.assertAlmostEqual(actual, expected, places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
