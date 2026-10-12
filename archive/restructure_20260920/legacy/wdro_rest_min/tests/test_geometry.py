from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import (
    InvalidNetError,
    inverse_net,
    minimum_pair_l1_distance,
    normalize_net,
    pin_set_sha256,
    project_normalized_net,
    validate_normalized_net,
)


class GeometryTests(unittest.TestCase):
    def test_normalize_and_inverse_round_trip(self) -> None:
        pins = np.asarray([[10.0, -3.0], [14.0, -2.0], [12.0, 1.0]])
        normalized, transform = normalize_net(pins, expected_degree=3)
        np.testing.assert_allclose(inverse_net(normalized, transform), pins, atol=1e-12)
        self.assertAlmostEqual(float(normalized.min(axis=0)[0]), 0.1)
        self.assertAlmostEqual(float(normalized.min(axis=0)[1]), 0.1)
        self.assertAlmostEqual(float((normalized.max(axis=0) - normalized.min(axis=0)).max()), 0.8)

    def test_aspect_ratio_is_preserved(self) -> None:
        pins = np.asarray([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0]])
        normalized, _ = normalize_net(pins)
        spans = normalized.max(axis=0) - normalized.min(axis=0)
        self.assertAlmostEqual(spans[0] / spans[1], 2.0)

    def test_one_axis_zero_span_is_valid(self) -> None:
        pins = np.asarray([[2.0, 1.0], [2.0, 4.0], [2.0, 9.0]])
        normalized, transform = normalize_net(pins)
        self.assertTrue(np.allclose(normalized[:, 0], 0.1))
        np.testing.assert_allclose(inverse_net(normalized, transform), pins)

    def test_both_axes_zero_span_is_invalid(self) -> None:
        with self.assertRaisesRegex(InvalidNetError, "degree_below_two"):
            normalize_net(np.asarray([[2.0, 2.0], [2.0, 2.0]]))

    def test_duplicate_changes_fixed_degree_and_is_invalid(self) -> None:
        pins = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        with self.assertRaisesRegex(InvalidNetError, "degree_changed_by_deduplication"):
            normalize_net(pins, expected_degree=3)

    def test_nonfinite_coordinate_is_invalid(self) -> None:
        with self.assertRaisesRegex(InvalidNetError, "nonfinite_coordinate"):
            normalize_net(np.asarray([[0.0, 0.0], [np.nan, 1.0]]))

    def test_minimum_spacing_violation_is_invalid(self) -> None:
        pins = np.asarray([[0.0, 0.0], [1.0, 1.0], [1e-8, 0.0]])
        with self.assertRaisesRegex(InvalidNetError, "minimum_spacing_violation"):
            normalize_net(pins, expected_degree=3)

    def test_projection_clips_and_resolves_duplicates(self) -> None:
        pins = np.asarray([[-1.0, 0.1], [-1.0, 0.1], [2.0, 2.0]])
        projected = project_normalized_net(pins)
        validate_normalized_net(projected, expected_degree=3)
        self.assertGreaterEqual(minimum_pair_l1_distance(projected), 1e-6)

    def test_pin_set_hash_is_permutation_invariant(self) -> None:
        pins = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.assertEqual(pin_set_sha256(pins), pin_set_sha256(pins[[2, 0, 1]]))


if __name__ == "__main__":
    unittest.main()

