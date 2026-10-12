from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.accounting import CallLedger


class CallLedgerTests(unittest.TestCase):
    def test_add_and_merge_count_nets(self) -> None:
        first = CallLedger()
        first.add("geosteiner_nets", 10)
        second = CallLedger(actor_forward_nets=5, geosteiner_nets=2)
        first.merge(second)
        self.assertEqual(first.to_dict()["actor_forward_nets"], 5)
        self.assertEqual(first.to_dict()["geosteiner_nets"], 12)

    def test_invalid_field_and_negative_count_are_rejected(self) -> None:
        ledger = CallLedger()
        with self.assertRaises(KeyError):
            ledger.add("unknown", 1)
        with self.assertRaises(ValueError):
            ledger.add("geosteiner_nets", -1)


if __name__ == "__main__":
    unittest.main()

