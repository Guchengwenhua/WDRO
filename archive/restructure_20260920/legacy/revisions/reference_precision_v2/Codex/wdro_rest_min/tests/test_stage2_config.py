import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol import load_protocol
from src.trainer import load_trainer_config


class Stage2ConfigTests(unittest.TestCase):
    def test_shared_settings_do_not_override_protocol(self) -> None:
        protocol = load_protocol()
        config = load_trainer_config()
        self.assertEqual(config["status"], "frozen")
        self.assertFalse(config["research_protocol_override"])
        self.assertEqual(config["loss"], protocol["loss_and_metrics"]["training_loss"])
        self.assertEqual(config["stage2_smoke_updates"], 200)
        self.assertEqual(
            config["training_pool"]["materialized_nets_per_formal_run"],
            protocol["training"]["policy_updates"] * protocol["training"]["batch_size"],
        )
        self.assertEqual(config["checkpoint_schema"]["retention"], ["latest", "best"])
        self.assertFalse(config["selection"]["test_or_ispd_allowed"])


if __name__ == "__main__":
    unittest.main()
