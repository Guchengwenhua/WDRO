"""CPU-only joint-training contracts, including exact interrupted resume."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wdro_rest_min"))

from adv_v3 import joint
from adv_v3.common import atomic_json


class TinyActor(torch.nn.Module):
    def __init__(self, degree=30):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.degree = degree

    def forward(self, pins, deterministic=False):
        distribution = torch.distributions.Bernoulli(logits=self.weight.expand(len(pins)))
        action = (self.weight.detach() > 0).float().expand(len(pins)) if deterministic else distribution.sample()
        topology = action.long()[:, None].expand(-1, 2 * (self.degree - 1))
        return topology, distribution.log_prob(action)


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, pins):
        return self.weight.expand(len(pins))


class TinyRest:
    def lengths(self, pins, topology):
        return 1.0 + 0.2 * topology[:, 0]


class FakeOracle:
    def __init__(self, actor, device, outdir, microbatch=256):
        self.actor, self.device = actor, torch.device(device)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.path = self.outdir / "oracle_ledger.json"
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.rest = TinyRest()

    def add(self, key, value):
        self.ledger[key] = self.ledger.get(key, 0) + int(value)

    def persist(self):
        atomic_json(self.path, self.ledger)

    def evaluate(self, pins):
        for key in ("actor_requested", "actor_completed", "rest_requested", "rest_completed", "reference_requests"):
            self.add(key, len(pins))
        self.persist()
        # Strictly positive and permutation invariant. This fake makes reduced
        # probability of the expensive sampled action improve greedy scores.
        regret = np.full(len(pins), float(self.actor.weight.detach().sigmoid()) * 0.2)
        return {"regret": regret, "lstar": np.ones(len(pins)), "length": 1.0 + regret}

    def close(self):
        self.persist()


def fake_parent(method, degree, seed, device):
    return TinyActor(degree).to(device), TinyCritic().to(device), {}, {"best": {"sha256": "fake"}}


def fake_identity(args, settings):
    return {"experiment": "cpu_fake_only", "method": args.method, "seed": args.seed,
            "degree": args.degree, "smoke": args.smoke, "config": settings}


class JointTests(unittest.TestCase):
    def arguments(self, method="random_top4"):
        return SimpleNamespace(degree=30, seed=101, method=method, physical_gpu_index=-1, smoke=True)

    def invoke(self, directory, *, stop_after=None, method="random_top4"):
        with patch.object(joint, "EXP", Path(directory)), \
             patch.object(joint, "_build_identity", side_effect=fake_identity), \
             patch.object(joint, "load_parent", side_effect=fake_parent), \
             patch.object(joint, "parent_price", return_value=(0.2, 0.03)), \
             patch.object(joint, "Oracle", FakeOracle), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.is_initialized", return_value=False):
            return joint.run(self.arguments(method), stop_after=stop_after)

    def test_initial_selection_candidate_and_nominal_gate(self):
        initial = {"nominal_mean": 0.1, "ood_worst_group_mean": 0.3}
        cases = [
            ({"nominal_mean": 0.1, "ood_worst_group_mean": 0.31}, (True, False)),
            ({"nominal_mean": 0.1, "ood_worst_group_mean": 0.3}, (True, False)),
            ({"nominal_mean": 0.101, "ood_worst_group_mean": 0.29}, (True, True)),
            ({"nominal_mean": 0.103, "ood_worst_group_mean": 0.1}, (False, False)),
        ]
        for candidate, expected in cases:
            self.assertEqual(joint._is_better(candidate, initial, initial), expected)

    def test_sampled_update_reuses_reference_and_has_separate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            actor, critic = TinyActor(), TinyCritic()
            optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-5)
            oracle = FakeOracle(actor, "cpu", directory)
            pins = np.zeros((8, 30, 2), dtype=np.float32)
            original_critic = critic.weight.detach().clone()
            torch.manual_seed(19)
            result = joint.sampled_update(actor, critic, optimizer, oracle, pins, np.ones(8))
            for key in ("sampled_actor_requested", "sampled_actor_completed", "sampled_rest_requested",
                        "sampled_rest_completed", "sampled_reference_reuses"):
                self.assertEqual(oracle.ledger[key], 8)
            self.assertNotIn("reference_requests", oracle.ledger)
            self.assertNotIn("actor_requested", oracle.ledger)
            self.assertNotEqual(float(critic.weight.detach()), float(original_critic))
            self.assertTrue(np.isfinite(result["actor_loss"]))

    def test_checkpoint_pack_does_not_alias_live_models_and_restores_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            actor, critic = TinyActor(), TinyCritic()
            optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-5)
            oracle = FakeOracle(actor, "cpu", directory)
            torch.manual_seed(33)
            saved = joint._pack(4, 0.2, actor, critic, optimizer, None, None, None, None, oracle)
            expected_random = torch.rand(3)
            with torch.no_grad():
                actor.weight.add_(1)
            self.assertNotEqual(float(actor.weight.detach()), float(saved["actor_state_dict"]["weight"]))
            joint._restore(saved, actor, critic, optimizer, None, None, None, None)
            np.testing.assert_array_equal(torch.rand(3).numpy(), expected_random.numpy())
            self.assertEqual(float(actor.weight.detach()), float(saved["actor_state_dict"]["weight"]))

    def test_smoke_end_to_end_budget_artifacts_and_completed_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = self.invoke(directory)
            self.assertEqual(complete["status"], "pass")
            self.assertEqual(complete["updates"], 4)
            self.assertIn(complete["best_update"], [0, 2, 4])
            self.assertEqual(complete["ledger"]["sampled_actor_completed"], 16)
            # Training 4*4*4; validation 3*4*4; plain evaluation 2*4*8;
            # independent attack 4 distributions * 2 centers * 64 candidates.
            expected = 64 + 48 + 64 + 512
            self.assertEqual(complete["ledger"]["actor_completed"], expected)
            self.assertEqual(complete["ledger"]["reference_requests"], expected)
            self.assertEqual(len(complete["history"]), 4)
            self.assertEqual(complete["data_audit"]["train"], 16)
            for artifact in complete["artifacts"]:
                joint._checked_artifact(artifact)
            again = self.invoke(directory)
            self.assertEqual(again["ledger"], complete["ledger"])
            self.assertEqual(again["artifacts"], complete["artifacts"])

    def test_interrupted_resume_matches_uninterrupted_model_inputs_and_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            uninterrupted = Path(directory) / "uninterrupted"
            resumed = Path(directory) / "resumed"
            final_a = self.invoke(uninterrupted)
            paused = self.invoke(resumed, stop_after=2)
            self.assertEqual(paused["status"], "paused")
            final_b = self.invoke(resumed)
            self.assertEqual(final_a["ledger"], final_b["ledger"])
            self.assertEqual(final_a["best_update"], final_b["best_update"])
            self.assertEqual(final_a["final_price"], final_b["final_price"])
            for record_a, record_b in zip(final_a["history"], final_b["history"]):
                for key in ("input_sha256", "selected_sha256", "lambda_after", "search_ledger_delta", "sampled"):
                    self.assertEqual(record_a[key], record_b[key])
            suffix = "joint_smoke/random_top4_d30_s101/latest.pt"
            saved_a = torch.load(uninterrupted / suffix, map_location="cpu", weights_only=False)
            saved_b = torch.load(resumed / suffix, map_location="cpu", weights_only=False)
            for key in ("actor_state_dict", "critic_state_dict"):
                for name in saved_a["current"][key]:
                    torch.testing.assert_close(saved_a["current"][key][name], saved_b["current"][key][name], rtol=0, atol=0)
            torch.testing.assert_close(saved_a["current"]["rng_state"]["torch"], saved_b["current"]["rng_state"]["torch"], rtol=0, atol=0)

    def test_corrupted_existing_evaluation_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = self.invoke(directory)
            artifact = next(a for a in complete["artifacts"] if "/evaluation/" in a["path"])
            with Path(artifact["path"]).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "bound artifact changed"):
                self.invoke(directory)


if __name__ == "__main__":
    unittest.main()
