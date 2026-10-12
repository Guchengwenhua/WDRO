import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stage2_audit_checks import all_finite, rebuild_stream, validation_checks
from src.protocol import load_protocol, protocol_sha256
from src.trainer import load_trainer_config
from src.training_data import EMPTY_STREAM_SHA256, extend_stream_hash, materialize_training_batch
from src.validation import validation_manifest_hash


class FormalAuditTests(unittest.TestCase):
    def test_nested_nan_and_inf_are_detected(self):
        self.assertTrue(all_finite({"value": torch.ones(2), "rng": np.array([1, 2])}))
        self.assertFalse(all_finite({"state": [torch.tensor(float("nan"))]}))
        self.assertFalse(all_finite({"metric": float("inf")}))

    def test_stream_cursor_resume_and_identity_rejection(self):
        protocol = load_protocol()
        stream = EMPTY_STREAM_SHA256
        for update in (1, 2):
            batch = materialize_training_batch(protocol=protocol, degree=10, training_seed=101,
                                               update=update, batch_size=2)
            stream = extend_stream_hash(stream, batch.bundle_sha256)
        result = {"run_id": "unit", "protocol_canonical_sha256": protocol_sha256(protocol),
                  "training_stream_hash": stream, "expected_updates": 2, "batch_size": 2}
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(protocol=protocol, degree=10, seed=101, result=result, root=Path(directory),
                          dataset_root=Path(directory), heartbeat=lambda *_: None)
            heldout = ({"pins": np.zeros((1, 10, 2))}, {"npz_sha256": "a" * 64})
            with patch("src.stage2_audit_checks.load_dataset", return_value=heldout):
                initial = rebuild_stream(**kwargs)
                self.assertEqual(initial["status"], "pass")
                with patch("src.stage2_audit_checks.materialize_training_batch", side_effect=AssertionError("replayed")):
                    resumed = rebuild_stream(**kwargs)
                self.assertEqual(initial["stream_hash"], resumed["stream_hash"])
                result["training_stream_hash"] = "b" * 64
                with self.assertRaises(ValueError):
                    rebuild_stream(**kwargs)

    def test_validation_argmin_and_missing_schedule_detected(self):
        protocol, trainer = load_protocol(), load_trainer_config()
        groups = {name: {"summary": {"count": 2000, "mean": 0.02},
                         "dataset_array_bundle_sha256": "a" * 64}
                  for name in protocol["data"]["distributions"]}
        events = []
        for update in range(0, 2001, 100):
            result = {"split": "validation", "seed": 202, "degree": 10, "groups": copy.deepcopy(groups),
                      "inference": {"mode": "deterministic_greedy", "symmetry_transforms": 1, "batch_size": 256},
                      "selection": {"scope": protocol["training"]["selection"]["worst_group_scope"],
                                    "worst_group_mean": 0.02, "nominal_mean": 0.02, "test_or_ispd_used": False}}
            events.append({"update": update, "result": result})
        best = {"update": 0, "best": {"selection_objective": 0.02, "nominal_mean": 0.02,
                 "validation_manifest_sha256": validation_manifest_hash(events[0]["result"])}}
        with patch("src.stage2_audit_checks.load_validation_group", return_value=(None, None, None, {"array_bundle_sha256": "a" * 64})):
            self.assertTrue(all(validation_checks(events, best, protocol, trainer, 10).values()))
            best["update"] = 100
            self.assertFalse(validation_checks(events, best, protocol, trainer, 10)["selected_best_is_earliest_global_validation_minimum"])
            self.assertFalse(validation_checks(events[:-1], best, protocol, trainer, 10)["all_21_validation_updates_present"])


if __name__ == "__main__":
    unittest.main()
