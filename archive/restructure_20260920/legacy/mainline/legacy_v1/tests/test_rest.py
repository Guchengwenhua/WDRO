import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rest import LEGACY_ROOT, RestTreeEvaluator, build_actor, load_trusted_legacy_actor


class RestBindingTests(unittest.TestCase):
    def test_trusted_checkpoint_and_absolute_evaluator(self) -> None:
        device = torch.device("cpu")
        actor = build_actor(10, device)
        info = load_trusted_legacy_actor(
            actor, LEGACY_ROOT / "save" / "DAC21" / "rsmt10b.pt", map_location=device
        )
        self.assertEqual(info["weights_only"], False)
        self.assertEqual(info["loaded_components"], ["actor_state_dict"])
        pins = np.random.default_rng(7).random((2, 10, 2)).astype(np.float32)
        actor.eval()
        with torch.no_grad():
            outputs, _ = actor(pins, True)
        evaluator = RestTreeEvaluator()
        lengths = evaluator.lengths(pins, outputs.numpy())
        self.assertEqual(lengths.shape, (2,))
        self.assertTrue(np.isfinite(lengths).all())
        translated_scaled = pins.astype(np.float64) * 3.25 + np.asarray([9.0, -4.0])
        changed = evaluator.lengths(translated_scaled, outputs.numpy())
        np.testing.assert_allclose(changed, lengths * 3.25, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
