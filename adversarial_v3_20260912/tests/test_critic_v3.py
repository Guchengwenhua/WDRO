"""CPU checks for independent Critic adaptation, proposals and resumability."""
import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adv_v3.critic import (InputCritic, calibrate_online, gradient_diagnostics,
                           matched_l1, propose, train_surrogate)


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        return self.bn(self.linear(x).transpose(1, 2)).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask):
        return x + self.drop(self.linear(x))


class Glimpse(nn.Module):
    def forward(self, x):
        return x.mean(1)


class ParentCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit_embedder = Embed()
        self.crit_encoder = Encoder()
        self.glimpse = Glimpse()
        self.critic_l1 = nn.Linear(8, 8)
        self.critic_l2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        with torch.no_grad():
            self.critic_l1.bias.fill_(1)
            self.critic_l2.weight.fill_(0.05)
            self.critic_l2.bias.fill_(-2)

    def forward(self, points):
        z = torch.tensor(points, dtype=torch.float32)
        encoded = self.crit_encoder(self.crit_embedder(z), None)
        return self.relu(self.critic_l2(self.relu(self.critic_l1(self.glimpse(encoded))))).squeeze(-1)


class FakeOracle:
    device = torch.device("cpu")

    def __init__(self):
        self.ledger = {"nets": 0}

    def evaluate(self, pins):
        self.ledger["nets"] += len(pins)
        return {"regret": np.square(np.asarray(pins, dtype=np.float64)).mean(axis=(1, 2))}


def fake_common():
    def make_data(degree, count, distribution, namespace, seed=0):
        number = int.from_bytes(hashlib.sha256(f"{namespace}:{seed}".encode()).digest()[:8], "little")
        return np.random.default_rng(number).uniform(0.15, 0.85, (count, degree, 2)).astype(np.float32)

    def legalize(centers, delta):
        return np.clip(np.asarray(centers) + delta, 0.1, 0.9).astype(np.float32)

    def atomic_torch(path, value):
        temporary = Path(str(path) + ".partial")
        torch.save(value, temporary)
        temporary.replace(path)

    def atomic_json(path, value):
        temporary = Path(str(path) + ".partial")
        temporary.write_text(json.dumps(value, allow_nan=False))
        temporary.replace(path)

    return types.SimpleNamespace(make_data=make_data, legalize=legalize,
                                 atomic_torch=atomic_torch, atomic_json=atomic_json)


class CriticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        self.parent = ParentCritic().eval()
        self.points = np.random.default_rng(62).uniform(0.1, 0.9, (4, 5, 2)).astype(np.float32)

    def test_raw_forward_preserved_but_input_connection_restored(self):
        z = torch.tensor(self.points, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = self.parent(z)
        self.assertIsNone(torch.autograd.grad(original.sum(), z, allow_unused=True)[0])
        adapted = InputCritic(self.parent)
        output = adapted(z)
        self.assertTrue(torch.equal(original, output))
        gradient = torch.autograd.grad(output.sum(), z)[0]
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
        self.assertEqual(gradient_diagnostics(adapted, self.points)["zero_gradient_fraction"], 1.0)
        linear = InputCritic(self.parent, calibrated=True)
        linear_grad = torch.autograd.grad(linear(z).sum(), z)[0]
        self.assertTrue(torch.isfinite(linear_grad).all())
        self.assertGreater(float(linear_grad.abs().sum()), 0)
        self.assertTrue(all(a.data_ptr() != b.data_ptr()
                            for a, b in zip(self.parent.parameters(), adapted.parameters())))

    def test_hungarian_branch_keeps_permutation_and_coordinate_gradients(self):
        x = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        z = torch.tensor([[[0.7, 0.6], [0.3, 0.4]]], requires_grad=True)
        cost = matched_l1(x, z)
        gradient = torch.autograd.grad(cost.sum(), z)[0]
        torch.testing.assert_close(cost, torch.tensor([0.3]))
        torch.testing.assert_close(gradient, torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]]))

    def test_proposals_are_deterministic_and_do_not_mutate_model_or_bn(self):
        model = InputCritic(self.parent, calibrated=True)
        before = copy.deepcopy(model.state_dict())
        with patch("adv_v3.critic._common", return_value=fake_common()):
            first = propose(model, self.points, 0.2, 3, 18)
            second = propose(model, self.points, 0.2, 3, 18)
        self.assertEqual(first.shape, (4, 3, 5, 2))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue((first >= 0.1).all() and (first <= 0.9).all())
        self.assertTrue(np.max(np.abs(first - self.points[:, None])) <= 0.080001)
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_real_critic_proposer_is_fixed_seed_permutation_equivariant(self):
        from adv_v3.common import build_critic, make_data
        model = InputCritic(build_critic(10, torch.device("cpu")), calibrated=True)
        centers = make_data(10, 2, "uniform", "test_real_critic_permutation", seed=19)
        permutation = np.asarray([3, 1, 8, 9, 0, 5, 7, 2, 6, 4])
        inverse = np.argsort(permutation)
        direct = propose(model, centers, 0.2, 2, 18)
        reordered = propose(model, centers[:, permutation], 0.2, 2, 18)
        np.testing.assert_allclose(direct, reordered[:, :, inverse], rtol=0, atol=0)

    def test_training_resume_reuses_labels_and_best_and_rejects_wrong_identity(self):
        oracle = FakeOracle()
        config = {"train_count": 16, "validation_count": 8, "epochs": 2,
                  "batch_size": 8, "label_microbatch": 5}
        before = copy.deepcopy(self.parent.state_dict())
        with tempfile.TemporaryDirectory() as directory, patch("adv_v3.critic._common", return_value=fake_common()):
            model, summary = train_surrogate(oracle, self.parent, 5, 101, directory, config)
            self.assertEqual(oracle.ledger["nets"], 24)
            resumed, repeated = train_surrogate(oracle, self.parent, 5, 101, directory, config)
            self.assertEqual(oracle.ledger["nets"], 24)
            self.assertEqual(summary["best_epoch"], repeated["best_epoch"])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
            state = torch.load(Path(directory) / "latest.pt", weights_only=False)
            self.assertIn("optimizer_state_dict", state)
            self.assertIn("rng_state", state)
            self.assertEqual(state["epoch"], 2)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                train_surrogate(oracle, self.parent, 5, 102, directory, config)
        for name, value in before.items():
            torch.testing.assert_close(value, self.parent.state_dict()[name], rtol=0, atol=0)

    def test_interrupted_labels_resume_without_repeating_completed_chunks(self):
        oracle = FakeOracle()
        config = {"train_count": 8, "validation_count": 4, "epochs": 1,
                  "batch_size": 4, "label_microbatch": 4}
        def interrupt():
            raise InterruptedError("simulated after atomic label chunk")
        with tempfile.TemporaryDirectory() as directory, patch("adv_v3.critic._common", return_value=fake_common()):
            with self.assertRaises(InterruptedError):
                train_surrogate(oracle, self.parent, 5, 101, directory, config, heartbeat=interrupt)
            self.assertEqual(oracle.ledger["nets"], 4)
            _, summary = train_surrogate(oracle, self.parent, 5, 101, directory, config)
            self.assertEqual(oracle.ledger["nets"], 12)
            self.assertEqual(summary["label_requests"], 12)

    def test_online_calibration_updates_only_independent_surrogate(self):
        model = InputCritic(self.parent, calibrated=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = copy.deepcopy(model.state_dict())
        summary = calibrate_online(model, optimizer, self.points, np.ones(4) * 0.3)
        self.assertEqual(summary["real_oracle_calls"], 0)
        self.assertLess(summary["mse_after"], summary["mse_before"])
        torch.testing.assert_close(before["critic.crit_embedder.bn.running_mean"],
                                   model.state_dict()["critic.crit_embedder.bn.running_mean"])

    def test_epoch_checkpoint_resume_matches_continuous_optimizer_and_rng(self):
        config = {"train_count": 8, "validation_count": 4, "epochs": 2,
                  "batch_size": 4, "label_microbatch": 4}
        common = fake_common()
        original_save = common.atomic_torch

        def interrupt_after_first_epoch(path, value):
            original_save(path, value)
            if Path(path).name == "latest.pt" and value["epoch"] == 1:
                raise InterruptedError("simulated after complete epoch publication")

        with tempfile.TemporaryDirectory() as directory:
            continuous_dir, resumed_dir = Path(directory) / "continuous", Path(directory) / "resumed"
            with patch("adv_v3.critic._common", return_value=common):
                continuous, continuous_summary = train_surrogate(
                    FakeOracle(), self.parent, 5, 101, continuous_dir, config)
                oracle = FakeOracle()
                common.atomic_torch = interrupt_after_first_epoch
                with self.assertRaises(InterruptedError):
                    train_surrogate(oracle, self.parent, 5, 101, resumed_dir, config)
                common.atomic_torch = original_save
                resumed, resumed_summary = train_surrogate(
                    oracle, self.parent, 5, 101, resumed_dir, config)
            self.assertEqual(oracle.ledger["nets"], 12)
            self.assertEqual(continuous_summary["history"], resumed_summary["history"])
            for name, value in continuous.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
            first = torch.load(continuous_dir / "latest.pt", weights_only=False)
            second = torch.load(resumed_dir / "latest.pt", weights_only=False)
            torch.testing.assert_close(first["rng_state"]["torch"], second["rng_state"]["torch"])
            for index, item in first["optimizer_state_dict"]["state"].items():
                for key, value in item.items():
                    torch.testing.assert_close(value, second["optimizer_state_dict"]["state"][index][key])


if __name__ == "__main__":
    unittest.main()
