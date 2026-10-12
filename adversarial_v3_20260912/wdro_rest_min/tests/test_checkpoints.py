import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoints import atomic_torch_save, capture_rng_state, restore_rng_state


class CheckpointTests(unittest.TestCase):
    def test_rng_round_trip_and_atomic_save(self) -> None:
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(4))
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "checkpoint.pt"
            atomic_torch_save({"tensor": expected[2]}, destination)
            self.assertTrue(destination.is_file())
            self.assertEqual(list(destination.parent.glob("*.partial")), [])
            loaded = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(loaded["tensor"], expected[2]))

    def test_restore_accepts_rng_tensors_after_device_normalization(self) -> None:
        torch.manual_seed(17)
        state = capture_rng_state()
        # A clone exercises the same normalization path used after checkpoint IO.
        state["torch_cpu"] = state["torch_cpu"].clone()
        state["torch_cuda"] = [item.clone() for item in state["torch_cuda"]]
        restore_rng_state(state)
        self.assertEqual(state["torch_cpu"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
