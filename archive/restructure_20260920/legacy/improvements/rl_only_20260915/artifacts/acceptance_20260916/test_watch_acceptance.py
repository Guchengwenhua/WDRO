"""Isolated stdlib tests; no watcher, training or acceptance subprocess is started."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().with_name("watch_acceptance.py")
SPEC = importlib.util.spec_from_file_location("acceptance_watch", SCRIPT)
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "artifacts/operations").mkdir(parents=True)
        (self.root / "artifacts/report").mkdir()
        self.out = self.root / "artifacts/acceptance_20260916"
        self.out.mkdir()
        for name in ("watch_acceptance.py", "audit_acceptance.py", "statistics_review.py"):
            (self.out / name).write_text("# isolated test fixture\n")
        (self.root / "run_env.sh").write_text("#!/bin/bash\n")
        self.identity = {"pid": 123, "start_ticks": "456", "boot_id": "test"}
        self.put("protocol.json", {"frozen": True})
        self.put("artifacts/operations/source_hashes.json",
                 {"hashes": {"protocol.json": watcher.sha256(self.root / "protocol.json")}})
        self.put("queue.json", {"tasks": [{"id": "one"}]})
        self.put("artifacts/operations/launch.json", {"launch_id": "L", "identity": self.identity})
        self.queue = {"launch_id": "L", "scheduler_identity": self.identity,
                      "status": "running", "at": time.time(), "completed": {},
                      "pending": [], "running": [{"task_id": "one"}], "failures": {}}
        self.put("artifacts/operations/queue.json", self.queue)

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, path, value):
        (self.root / path).write_text(json.dumps(value))

    def assess(self, alive):
        with patch.object(watcher, "is_alive", return_value=alive), \
             patch.object(watcher, "lock_released", return_value=True):
            return watcher.assess_main(self.root, "L", self.identity)

    def complete(self):
        self.queue.update(status="tasks_complete", completed={"one": {}}, running=[])
        self.put("artifacts/operations/queue.json", self.queue)
        self.put("artifacts/operations/exit.json", {"launch_id": "L", "status": "complete", "running": [], "pending": []})
        self.put("artifacts/report/complete.json", {"status": "complete", "acceptance": "accepted_gpu",
                                                   "protocol_sha256": watcher.sha256(self.root / "protocol.json")})

    def test_running_waits(self):
        self.assertEqual(self.assess(True)["state"], "waiting")

    def test_dead_scheduler_fails_without_markers(self):
        with self.assertRaisesRegex(watcher.GateError, "exited without"):
            self.assess(False)

    def test_wrong_report_fails(self):
        self.put("artifacts/report/complete.json", {"status": "complete", "acceptance": "diagnostic_only"})
        with self.assertRaisesRegex(watcher.GateError, "accepted_gpu"):
            self.assess(True)

    def test_failed_postprocess_fails(self):
        self.put("artifacts/operations/exit.json", {"launch_id": "L", "status": "postprocess_failed"})
        with self.assertRaisesRegex(watcher.GateError, "postprocess_failed"):
            self.assess(False)

    def test_source_drift_fails(self):
        self.queue["source_error"] = "changed"
        self.put("artifacts/operations/queue.json", self.queue)
        with self.assertRaisesRegex(watcher.GateError, "source_error"):
            self.assess(True)

    def test_finalization_waits_for_scheduler_exit(self):
        self.complete()
        self.assertEqual(self.assess(True)["state"], "waiting_for_scheduler_exit")
        self.assertEqual(self.assess(False)["state"], "ready")
        watcher.require_final_pass({"status": "PASS"})
        for status in ("complete", "accepted", "FAIL", None):
            with self.subTest(final_status=status), self.assertRaisesRegex(watcher.GateError, "does not indicate PASS"):
                watcher.require_final_pass({"status": status})

    def test_source_binding_detects_edit(self):
        with patch.object(watcher, "SCRIPT", self.out / "watch_acceptance.py"):
            binding = watcher.build_binding(self.root, self.out)
        statistics_key = "artifacts/acceptance_20260916/statistics_review.py"
        self.assertIn(statistics_key, binding["files"])
        watcher.verify_binding(self.root, binding)
        (self.out / "statistics_review.py").write_text("# changed independent statistics implementation\n")
        with self.assertRaisesRegex(watcher.GateError, "statistics_review.py"):
            watcher.verify_binding(self.root, binding)
        (self.out / "statistics_review.py").unlink()
        with patch.object(watcher, "SCRIPT", self.out / "watch_acceptance.py"), \
             self.assertRaisesRegex(watcher.GateError, "binding input missing"):
            watcher.build_binding(self.root, self.out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
