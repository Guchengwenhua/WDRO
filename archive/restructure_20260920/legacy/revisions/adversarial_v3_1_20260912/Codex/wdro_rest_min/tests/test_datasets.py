from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (
    build_synthetic_dataset,
    generate_terminals,
    load_dataset,
    namespaced_seed,
    save_dataset,
)
from src.geometry import (
    MINIMUM_L1_SPACING,
    NORMALIZED_LOWER,
    NORMALIZED_UPPER,
    ZERO_SPAN_EPSILON,
)


class DatasetTests(unittest.TestCase):
    def test_data_contract_matches_implementation_constants(self) -> None:
        import json

        contract_path = PROJECT_ROOT / "configs" / "data_contract_v1.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        coordinates = contract["coordinates"]
        self.assertEqual(coordinates["normalized_lower"], NORMALIZED_LOWER)
        self.assertEqual(coordinates["normalized_upper"], NORMALIZED_UPPER)
        self.assertEqual(coordinates["zero_bbox_span_epsilon"], ZERO_SPAN_EPSILON)
        self.assertEqual(coordinates["minimum_normalized_l1_spacing"], MINIMUM_L1_SPACING)
        self.assertEqual(contract_path.stat().st_mode & 0o222, 0)

    def test_generator_is_reproducible(self) -> None:
        parameters = {"nonuniform": 0.8, "clusters": 6, "sigma": 0.04}
        first = generate_terminals(5, 10, 123, "mix_gauss", parameters)
        second = generate_terminals(5, 10, 123, "mix_gauss", parameters)
        np.testing.assert_array_equal(first, second)

    def test_seed_namespaces_are_split_and_distribution_specific(self) -> None:
        seeds = {
            namespaced_seed("p", 202, split, distribution, 10)
            for split in ("validation", "test")
            for distribution in ("uniform", "mix_gauss")
        }
        self.assertEqual(len(seeds), 4)

    def test_dataset_round_trip_and_hash(self) -> None:
        arrays, metadata = build_synthetic_dataset(
            protocol_id="test-protocol",
            split="validation",
            distribution="uniform",
            degree=10,
            base_seed=202,
            count=8,
            parameters={"nonuniform": 0.0},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.npz"
            saved = save_dataset(path, arrays, metadata)
            loaded, loaded_metadata = load_dataset(path)
            self.assertEqual(saved["array_bundle_sha256"], loaded_metadata["array_bundle_sha256"])
            for name in arrays:
                np.testing.assert_array_equal(arrays[name], loaded[name])
            self.assertEqual(path.stat().st_mode & 0o222, 0)


if __name__ == "__main__":
    unittest.main()
