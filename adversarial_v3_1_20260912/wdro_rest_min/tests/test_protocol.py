from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol import DEFAULT_PROTOCOL_PATH, ProtocolError, load_protocol, protocol_sha256, validate_protocol


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()

    def test_frozen_protocol_validates(self) -> None:
        validate_protocol(self.protocol)

    def test_hash_is_stable_across_key_order(self) -> None:
        reversed_items = dict(reversed(list(self.protocol.items())))
        self.assertEqual(protocol_sha256(self.protocol), protocol_sha256(reversed_items))

    def test_test_seed_cannot_overlap_training(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["data"]["seeds"]["test"] = 101
        with self.assertRaisesRegex(ProtocolError, "test seed changed|disjoint"):
            validate_protocol(changed)

    def test_test_leakage_is_rejected(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["training"]["selection"]["test_data_used_for_selection"] = True
        with self.assertRaisesRegex(ProtocolError, "test leakage"):
            validate_protocol(changed)

    def test_distribution_parameter_drift_is_rejected(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["data"]["distributions"]["mix_gauss"]["sigma"] = 0.05
        with self.assertRaisesRegex(ProtocolError, "distribution parameters"):
            validate_protocol(changed)

    def test_relative_regret_drift_is_rejected(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["loss_and_metrics"]["formula"] = "L_REST - L_star"
        with self.assertRaisesRegex(ProtocolError, "relative regret formula"):
            validate_protocol(changed)

    def test_batch_disk_floor_cannot_be_relaxed(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["runtime"]["minimum_free_disk_gib_for_batch"] = 99
        with self.assertRaisesRegex(ProtocolError, "100 GiB"):
            validate_protocol(changed)

    def test_config_has_no_write_bits(self) -> None:
        self.assertEqual(DEFAULT_PROTOCOL_PATH.stat().st_mode & 0o222, 0)


if __name__ == "__main__":
    unittest.main()
