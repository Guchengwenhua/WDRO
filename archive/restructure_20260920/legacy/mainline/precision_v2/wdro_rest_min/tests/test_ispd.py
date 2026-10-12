from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ispd import discover_base_designs, normalize_ispd_degree_file, parse_degree_file


class IspdTests(unittest.TestCase):
    def test_exactly_twenty_base_designs_and_no_metal5(self) -> None:
        root = PROJECT_ROOT.parents[2] / "DRO" / "benchmark"
        designs = discover_base_designs(root)
        self.assertEqual(len(designs), 20)
        self.assertTrue(all("metal5" not in design.name for design in designs))

    def test_parser_preserves_net_name_and_counts_invalid_duplicate(self) -> None:
        content = """net n1 10 3
0 0
1 0
0 1
END
net n2 20 3
0 0
0 0
1 1
END
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "degree_3.txt"
            path.write_text(content, encoding="utf-8")
            parsed = list(parse_degree_file(path))
            self.assertEqual([net.net_name for net in parsed], ["n1", "n2"])
            arrays, metadata = normalize_ispd_degree_file(path, "design", 3)
            self.assertEqual(metadata["parsed_count"], 2)
            self.assertEqual(metadata["valid_count"], 1)
            self.assertEqual(metadata["invalid_reasons"], {"degree_changed_by_deduplication": 1})
            self.assertEqual(arrays["net_ids"][0].decode(), "ispd:design:n1:d3")


if __name__ == "__main__":
    unittest.main()

