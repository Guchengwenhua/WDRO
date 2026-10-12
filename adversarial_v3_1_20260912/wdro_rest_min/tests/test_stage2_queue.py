import importlib.util
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_queue_module():
    path = PROJECT_ROOT / "scripts" / "run_stage2_erm_queue.py"
    specification = importlib.util.spec_from_file_location("stage2_queue", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class Stage2QueueTests(unittest.TestCase):
    def test_formal_task_matrix_and_memory_profile(self) -> None:
        module = load_queue_module()
        tasks = module.build_tasks()
        self.assertEqual(len(tasks), 15)
        self.assertEqual(
            {(task["degree"], task["seed"]) for task in tasks},
            {(degree, seed) for degree in [10, 20, 30, 40, 50] for seed in [101, 102, 103]},
        )
        self.assertTrue(all("u2000" in task["run_id"] for task in tasks))
        profile = json.loads(module.MEMORY_PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertFalse(profile["research_protocol_override"])
        self.assertEqual(set(profile["profiles"]), {"10", "20", "30", "40", "50"})
        self.assertTrue(
            all(
                value["capacity_peak_mib"] >= value["measured_torch_reserved_mib"]
                for value in profile["profiles"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
