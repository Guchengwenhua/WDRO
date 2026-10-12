"""Meaningful CPU checks for scale, local pairing, labels, and resumable fitting."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import surrogate as s


class TinySurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.dropout = nn.Dropout(.1)
        self.calibrated = True

    def forward(self, pins):
        parameter = next(self.parameters())
        x = torch.as_tensor(pins, device=parameter.device, dtype=parameter.dtype)
        return self.linear(self.dropout(x).mean(1)).squeeze(-1)


class FakeOracle:
    def __init__(self):
        self.ledger = {"actor_requested": 0}

    def evaluate(self, pins):
        self.ledger["actor_requested"] += len(pins)
        regret = np.square(pins.astype(np.float64)).mean((1, 2)) / 10
        return {"regret": regret, "length": 1 + regret, "lstar": np.ones(len(pins))}


def fake_legacy():
    common = types.SimpleNamespace(GEOMETRY_LEDGER={"proposals": 0, "repairs": 0, "fallbacks": 0})

    def derived_seed(*parts):
        return int.from_bytes(hashlib.sha256(str(parts).encode()).digest()[:4], "little")

    def make_data(degree, count, distribution, namespace, seed):
        return np.random.default_rng(derived_seed(namespace, seed)).uniform(
            .15, .85, (count, degree, 2)).astype(np.float32)

    def legalize(centers, delta):
        common.GEOMETRY_LEDGER["proposals"] += len(centers)
        return np.clip(centers + delta, .1, .9).astype(np.float32)

    def atomic_torch(path, obj):
        torch.save(obj, str(path) + ".partial")
        Path(str(path) + ".partial").replace(path)

    def atomic_json(path, obj):
        Path(str(path) + ".partial").write_text(json.dumps(obj, allow_nan=False))
        Path(str(path) + ".partial").replace(path)

    def atomic_npz(path, **arrays):
        with open(str(path) + ".partial", "wb") as stream:
            np.savez_compressed(stream, **arrays)
        Path(str(path) + ".partial").replace(path)

    def append_json(path, obj):
        with Path(path).open("a") as stream:
            stream.write(json.dumps(obj) + "\n")

    common.derived_seed, common.make_data, common.legalize = derived_seed, make_data, legalize
    common.atomic_torch, common.atomic_json = atomic_torch, atomic_json
    common.atomic_npz, common.append_json = atomic_npz, append_json
    common.sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    critic = types.SimpleNamespace(matched_l1=lambda a, b: (a-b).abs().sum(-1).mean(-1))
    return common, critic


class SurrogateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.model = TinySurrogate().eval()

    def test_local_scale_is_anchored_by_absolute_regression(self):
        target = torch.tensor([.01, .03, .08, .12], dtype=torch.float64)
        groups = np.array([0, 0, 1, 1])
        perfect = s.regression_loss(target, target, groups, "rankdiff")
        biased = s.regression_loss(target + .2, target, groups, "rankdiff")
        self.assertAlmostEqual(float(biased-perfect), .2**2, places=12)
        scaled = s.regression_loss(target*10, target, groups, "rankdiff")
        self.assertGreater(float(scaled), float(perfect))

    def test_local_loss_pairs_never_cross_centers_and_ignore_true_ties(self):
        target = torch.tensor([.01, .04, .01, .04], dtype=torch.float64)
        groups = np.array([0, 0, 1, 1])
        base = s.regression_loss(target, target, groups, "rankdiff")
        shifted = target + torch.tensor([.5, .5, -.5, -.5])
        changed = s.regression_loss(shifted, target, groups, "rankdiff")
        self.assertAlmostEqual(float(changed-base), .25, places=12)
        tied = torch.ones(4, dtype=torch.float64)*.02
        self.assertEqual(float(s.regression_loss(tied, tied, groups, "rankdiff")), 0.)

    def test_huber_caps_outlier_gradient_without_rescaling_prediction(self):
        prediction = torch.tensor([1.], requires_grad=True)
        loss = s.regression_loss(prediction, torch.zeros(1), [0], "huber")
        loss.backward()
        self.assertAlmostEqual(float(prediction.grad), .01, places=8)
        self.assertAlmostEqual(float(loss), .00995, places=7)

    def test_group_batches_cover_every_row_and_keep_all_local_pairs(self):
        groups = np.repeat(np.arange(7), 4)
        batches = list(s._group_batches(groups, 8, 123))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(28))
        for batch in batches:
            for group in np.unique(groups[batch]):
                self.assertEqual(np.sum(groups[batch] == group), 4)

    def test_real_degree30_trajectory_is_legal_equivariant_and_keeps_initial_model(self):
        common, critic = s._legacy()
        search = __import__("adv_v3.search", fromlist=["_check_pins"])
        model = critic.InputCritic(common.build_critic(30, torch.device("cpu")), calibrated=True)
        centers = common.make_data(30, 2, "uniform", "unit_surrogate_trajectory", 17)
        before = s._hash_model(model)
        direct, diagnostics = s._trajectory_points(model, centers, .1, 13)
        permutation = np.random.default_rng(5).permutation(30)
        reordered, _ = s._trajectory_points(model, centers[:, permutation], .1, 13)
        np.testing.assert_array_equal(direct, reordered[:, :, np.argsort(permutation)])
        np.testing.assert_array_equal(direct[:, 0], centers)
        search._check_pins(direct, 30)
        self.assertEqual(s._hash_model(model), before)
        self.assertEqual(diagnostics["surrogate_input_backward_rows"], 16)

    def test_shared_validation_label_reuse_and_hash_rejection(self):
        oracle = FakeOracle()
        with tempfile.TemporaryDirectory(dir=s.ROOT) as directory, patch.object(
                s, "_legacy", return_value=fake_legacy()):
            first, first_summary = s.build_refresh_data(
                self.model, oracle, 5, 101, Path(directory)/"uniform", "uniform", "unit",
                .1, train_count=16, validation_centers=4)
            second, _ = s.build_refresh_data(
                self.model, oracle, 5, 101, Path(directory)/"region", "search_region", "unit",
                .1, train_count=16, validation_centers=4)
            self.assertEqual(oracle.ledger["actor_requested"], 48)
            np.testing.assert_array_equal(first["validation_pins"], second["validation_pins"])
            self.assertEqual(first_summary["exact_train_validation_overlap"], 0)
            s.build_refresh_data(self.model, oracle, 5, 101, Path(directory)/"uniform", "uniform",
                                 "unit", .1, train_count=16, validation_centers=4)
            self.assertEqual(oracle.ledger["actor_requested"], 48)
            path = Path(directory)/"uniform"/"labels.npz"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "identity/hash mismatch"):
                s.build_refresh_data(self.model, oracle, 5, 101, Path(directory)/"uniform", "uniform",
                                     "unit", .1, train_count=16, validation_centers=4)

    def test_interrupted_labeling_preserves_paid_completed_chunk(self):
        oracle = FakeOracle()
        def stop():
            raise InterruptedError("after a committed label chunk")
        with tempfile.TemporaryDirectory(dir=s.ROOT) as directory, patch.object(
                s, "_legacy", return_value=fake_legacy()):
            with self.assertRaises(InterruptedError):
                s._label_pool(oracle, self.model, 5, 101, directory, "uniform", "unit", 256, .1, stop)
            self.assertEqual(oracle.ledger["actor_requested"], 128)
            _, summary = s._label_pool(oracle, self.model, 5, 101, directory,
                                       "uniform", "unit", 256, .1)
            self.assertEqual(oracle.ledger["actor_requested"], 256)
            self.assertEqual(summary["actual_requested"], 256)
            self.assertEqual(summary["request_overhead"], 0)

    def test_epoch_resume_and_init_immutability(self):
        common, critic = fake_legacy()
        oracle = FakeOracle()
        before = s._hash_model(self.model)
        with tempfile.TemporaryDirectory(dir=s.ROOT) as directory, patch.object(
                s, "_legacy", return_value=(common, critic)):
            data, _ = s.build_refresh_data(self.model, oracle, 5, 101, Path(directory)/"data",
                                           "search_region", "unit", .1,
                                           train_count=16, validation_centers=4)
            full, summary = s.train_refresh(self.model, data, 101, Path(directory)/"full",
                                            "rankdiff", epochs=3, batch_size=8, lr=.001)
            original_save = common.atomic_torch
            def interrupted_save(path, obj):
                original_save(path, obj)
                if Path(path).name == "latest.pt" and obj.get("epoch") == 1:
                    raise InterruptedError("after epoch checkpoint publication")
            common.atomic_torch = interrupted_save
            with self.assertRaises(InterruptedError):
                s.train_refresh(self.model, data, 101, Path(directory)/"resumed",
                                "rankdiff", epochs=3, batch_size=8, lr=.001)
            common.atomic_torch = original_save
            resumed, recovered = s.train_refresh(self.model, data, 101, Path(directory)/"resumed",
                                                 "rankdiff", epochs=3, batch_size=8, lr=.001)
            self.assertEqual(s._hash_model(full), s._hash_model(resumed))
            self.assertEqual(summary["best_epoch"], recovered["best_epoch"])
            self.assertEqual(s._hash_model(self.model), before)
            reused, _ = s.train_refresh(self.model, data, 101, Path(directory)/"resumed",
                                        "rankdiff", epochs=3, batch_size=8, lr=.001)
            self.assertEqual(s._hash_model(reused), s._hash_model(resumed))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                s.train_refresh(self.model, data, 102, Path(directory)/"resumed",
                                "rankdiff", epochs=3, batch_size=8, lr=.001)


if __name__ == "__main__":
    unittest.main()
