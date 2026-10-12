"""Small CPU checks for score-function gradients, accounting, and exact resume."""
import copy
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from adv_v3 import rl


class FakeOracle:
    device = torch.device("cpu")

    def __init__(self):
        self.ledger = {"real_score_requests": 0}

    def evaluate(self, pins):
        self.ledger["real_score_requests"] += len(pins)
        regret = np.square(np.asarray(pins, dtype=np.float64)[..., 0]).mean(axis=1)
        return {"regret": regret, "lstar": np.ones(len(pins)), "length": regret + 1}


def fake_common():
    def legalize(x, delta):
        centers = x[:, None] if delta.ndim == 4 else x
        return np.clip(centers + delta, 0.1, 0.9).astype(np.float32)

    def make_data(degree, count, distribution, namespace, seed=0):
        raw = hashlib.sha256(f"{namespace}:{seed}".encode()).digest()
        return np.random.default_rng(int.from_bytes(raw[:8], "little")).uniform(0.2, 0.8, (count, degree, 2)).astype(np.float32)

    def atomic_torch(path, value):
        temporary = Path(str(path) + ".partial")
        torch.save(value, temporary)
        temporary.replace(path)

    def atomic_json(path, value):
        Path(path).write_text(json.dumps(value))

    return types.SimpleNamespace(legalize=legalize, make_data=make_data,
                                 costs=lambda x, z: np.abs(x[:, None] - z).sum(-1).mean(-1),
                                 atomic_torch=atomic_torch, atomic_json=atomic_json)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.common = fake_common()
        self.patch = patch.object(rl, "_common", return_value=self.common)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        torch.manual_seed(9)

    def test_equivariant_forward_and_original_action_log_probability(self):
        policy = rl.Policy(4, hidden_size=8)
        torch.nn.init.normal_(policy.decoder[-1].weight, std=0.1)
        x = self.common.make_data(4, 3, "uniform", "equivariance")
        order = [2, 0, 3, 1]
        mean, std = policy(x, 0.2)
        perm_mean, perm_std = policy(x[:, order], 0.2)
        torch.testing.assert_close(perm_mean, mean[:, order], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(perm_std, std[:, order])
        # A constant positive advantage with one sampled action must produce a
        # nonzero mean-head gradient. A non-detached rsample would cancel it.
        policy = rl.Policy(4, hidden_size=8)
        _, log_probability = policy.sample(x[:1], 0.2, generator=torch.Generator().manual_seed(17))
        (-log_probability.mean()).backward()
        self.assertGreater(float(policy.decoder[-1].bias.grad.abs().sum()), 1e-5)

    def test_propose_deterministic_no_global_rng_consumption(self):
        policy = rl.Policy(4, hidden_size=8)
        x = self.common.make_data(4, 3, "uniform", "propose")
        rng = torch.get_rng_state().clone()
        one = rl.propose(policy, x, 0.2, 3, 42)
        two = rl.propose(policy, x, 0.2, 3, 42)
        np.testing.assert_array_equal(one, two)
        self.assertEqual(one.shape, (3, 3, 4, 2))
        self.assertEqual(one.dtype, np.float32)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertTrue(np.all(np.abs(one - x[:, None]) <= 0.080001))

    def test_fixed_seed_proposal_is_equivariant_in_pin_order(self):
        policy = rl.Policy(4, hidden_size=8)
        torch.nn.init.normal_(policy.decoder[-1].weight, std=0.02)
        x = self.common.make_data(4, 3, "uniform", "proposal-equivariance")
        order = [2, 0, 3, 1]
        original = rl.propose(policy, x, .2, 3, 17)
        permuted = rl.propose(policy, x[:, order], .2, 3, 17)
        np.testing.assert_array_equal(permuted, original[:, :, order])

    def test_proposal_equivariance_with_real_normalization(self):
        from adv_v3 import common
        policy = rl.Policy(10, hidden_size=8)
        torch.nn.init.normal_(policy.decoder[-1].weight, std=0.02)
        x = common.make_data(10, 3, "uniform", "unit-test-real-rl-equivariance", seed=5)
        order = np.random.default_rng(27).permutation(10)
        with patch.object(rl, "_common", return_value=common):
            original = rl.propose(policy, x, .2, 3, 17)
            permuted = rl.propose(policy, x[:, order], .2, 3, 17)
        np.testing.assert_array_equal(permuted, original[:, :, order])
        for net in original.reshape(-1, 10, 2):
            self.assertFalse(common.has_spacing_violation(net))
            self.assertTrue(np.allclose(net.min(0), .1))
            self.assertAlmostEqual(float(np.ptp(net, axis=0).max()), .8, places=6)

    def test_online_update_counts_baseline_and_changes_policy(self):
        policy = rl.Policy(4, hidden_size=8)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        before = copy.deepcopy(policy.state_dict())
        oracle = FakeOracle()
        x = self.common.make_data(4, 8, "uniform", "online")
        result = rl.online_update(policy, optimizer, oracle, x, 0.2)
        self.assertEqual(result["baseline_requests"], 8)
        self.assertEqual(result["candidate_requests"], 8)
        self.assertEqual(oracle.ledger["real_score_requests"], 16)
        self.assertGreater(result["gradient_norm"], 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in policy.state_dict().items()))

    def test_resume_matches_uninterrupted_and_completed_reuse_is_read_only(self):
        config = {"updates": 4, "batch_size": 4, "validation_count": 3,
                  "validation_interval": 2, "hidden_size": 8}
        with tempfile.TemporaryDirectory() as temporary:
            full_dir, split_dir = Path(temporary) / "full", Path(temporary) / "split"
            full_oracle, split_oracle = FakeOracle(), FakeOracle()
            full_policy, full = rl.train_policy(full_oracle, 4, 19, 0.2, full_dir, config)
            _, partial = rl.train_policy(split_oracle, 4, 19, 0.2, split_dir, {**config, "stop_after": 2})
            self.assertEqual(partial["status"], "partial")
            # Simulate log publication interrupted after a checkpoint commit.
            (split_dir / "training.jsonl").write_text('{"interrupted":')
            resumed_policy, resumed = rl.train_policy(split_oracle, 4, 19, 0.2, split_dir, config)
            records = [json.loads(line) for line in (split_dir / "training.jsonl").read_text().splitlines()]
            self.assertEqual([record["update"] for record in records], [1, 2, 3, 4])
            self.assertEqual(resumed["counts"], full["counts"])
            self.assertEqual(resumed["best_update"], full["best_update"])
            self.assertEqual(resumed["best_f"], full["best_f"])
            self.assertEqual(split_oracle.ledger, full_oracle.ledger)
            self.assertEqual(resumed["counts"]["training_requests"], 32)
            self.assertEqual(resumed["counts"]["validation_requests"], 12)
            for key, value in full_policy.state_dict().items():
                torch.testing.assert_close(resumed_policy.state_dict()[key], value, rtol=0, atol=0)
            full_latest = torch.load(full_dir / "latest.pt", weights_only=False)
            split_latest = torch.load(split_dir / "latest.pt", weights_only=False)
            for key, value in full_latest["policy_state_dict"].items():
                torch.testing.assert_close(split_latest["policy_state_dict"][key], value, rtol=0, atol=0)
            before = dict(split_oracle.ledger)
            _, reused = rl.train_policy(split_oracle, 4, 19, 0.2, split_dir, config)
            self.assertEqual(split_oracle.ledger, before)
            self.assertEqual(reused["status"], "complete")
            with self.assertRaisesRegex(ValueError, "identity"):
                rl.train_policy(split_oracle, 4, 19, 0.3, split_dir, config)


if __name__ == "__main__":
    unittest.main()
