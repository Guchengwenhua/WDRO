"""CPU-only tests for scheduling safety and artifact gates; never start GPU work."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import scheduler


class SchedulerTests(unittest.TestCase):
    def test_multi_sample_compute_and_memory_gate(self):
        args = argparse.Namespace(admission_samples=3, max_gpu_util=50, min_free_mib=8192,
                                  memory_margin_mib=2048)
        task = {"required_free_mib": 6144}
        idle = {"utilization": 20, "free_mib": 9000}
        self.assertFalse(scheduler.admissible_gpu([idle, idle], task, args))
        self.assertTrue(scheduler.admissible_gpu([idle] * 3, task, args))
        self.assertFalse(scheduler.admissible_gpu([idle, {**idle, "utilization": 100}, idle], task, args))
        self.assertFalse(scheduler.admissible_gpu([idle, {**idle, "free_mib": 8191}, idle], task, args))

    def test_identity_rejects_pid_reuse(self):
        actual = scheduler.identity(os.getpid())
        self.assertTrue(scheduler.is_alive(actual))
        self.assertFalse(scheduler.is_alive({**actual, "start_ticks": "invalid"}))
        self.assertFalse(scheduler.is_alive({**actual, "boot_id": "previous_boot"}))

    def test_dependency_cycle_rejected(self):
        tasks = [{"id": "a", "args": ["train.py"], "complete": "a.json", "depends_on": ["b"]},
                 {"id": "b", "args": ["train.py"], "complete": "b.json", "depends_on": ["a"]}]
        with self.assertRaisesRegex(ValueError, "cycle"):
            scheduler.validate_manifest({"tasks": tasks})

    def test_completion_requires_cuda_acceptance_and_valid_json(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(scheduler, "ROOT", Path(temp)):
            path = Path(temp) / "complete.json"
            task = {"complete": "complete.json", "completion_checks": {"acceptance": "accepted_gpu"}}
            path.write_text('{"status":"complete","acceptance":"cpu_diagnostic"}')
            self.assertIsNone(scheduler.completion(task))
            path.write_text('{"status":"complete","acceptance":"accepted_gpu"}')
            self.assertIsNotNone(scheduler.completion(task))
            path.write_text('{"status":')
            self.assertIsNone(scheduler.completion(task))
            path.write_text('{"status":"complete","artifacts":null}')
            self.assertIsNone(scheduler.completion({"complete": "complete.json", "verify_artifacts": True}))

    def test_saturated_gpus_dry_run_never_launches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"tasks": [{"id": "a", "args": ["train.py"], "complete": "a.json"}]}
            (root / "queue.json").write_text(json.dumps(manifest))
            snapshot = {"gpus": [{"index": 0, "free_mib": 40000, "utilization": 100}],
                        "disk_free_bytes": 100 * 2**30, "at": 0}
            with mock.patch.object(scheduler, "ROOT", root), mock.patch.object(scheduler, "OPS", root / "ops"), \
                 mock.patch.object(scheduler, "resource_snapshot", return_value=snapshot), \
                 mock.patch.object(scheduler.subprocess, "Popen") as popen:
                self.assertEqual(scheduler.main(["--dry-run", "--admission-samples", "1"]), 0)
                popen.assert_not_called()
                result = json.loads((root / "ops/dry_run.json").read_text())
                self.assertEqual(result["eligible_resources"], [])
                self.assertEqual(result["status"], "waiting_for_resources")
                self.assertFalse((root / "ops/source_hashes.json").exists())

    def test_corrupted_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(scheduler, "ROOT", Path(temp)):
            root = Path(temp)
            for filename in ("latest.pt", "best.pt", "results.npz", "run.json", "protocol.json", "train.py", "dependency.py"):
                (root / filename).write_bytes(b"test")
            files = {name: scheduler.sha256(root / name) for name in ("latest.pt", "best.pt", "results.npz", "run.json")}
            record = {"status": "complete", "acceptance": "accepted_gpu", "identity": {"device": "cuda:0"},
                      "artifacts": files, "binding": {"protocol_sha256": scheduler.sha256(root / "protocol.json"),
                      "code_sha256": {str(root / "train.py"): scheduler.sha256(root / "train.py")},
                      "dependencies": {str(root / "dependency.py"): scheduler.sha256(root / "dependency.py")}}}
            scheduler.atomic(root / "complete.json", record)
            task = {"complete": "complete.json", "verify_artifacts": True}
            self.assertIsNotNone(scheduler.completion(task))
            (root / "results.npz").write_bytes(b"corrupted")
            self.assertIsNone(scheduler.completion(task))

    def test_missing_cpu_preflight_blocks_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"tasks": [{"id": "a", "args": ["train.py"], "complete": "a.json"}],
                        "cpu_diagnostic_gate": {"report": "missing.json"}}
            (root / "queue.json").write_text(json.dumps(manifest))
            snapshot = {"gpus": [{"index": 0, "free_mib": 40000, "utilization": 0}],
                        "disk_free_bytes": 100 * 2**30, "at": 0}
            with mock.patch.object(scheduler, "ROOT", root), mock.patch.object(scheduler, "OPS", root / "ops"), \
                 mock.patch.object(scheduler, "resource_snapshot", return_value=snapshot), \
                 mock.patch.object(scheduler.subprocess, "Popen") as popen:
                self.assertEqual(scheduler.main(["--dry-run", "--admission-samples", "1"]), 0)
                popen.assert_not_called()
                self.assertEqual(json.loads((root / "ops/dry_run.json").read_text())["status"], "blocked_cpu_preflight")
                scheduler.atomic(root / "missing.json", {"status": "complete", "acceptance": "diagnostic_only",
                    "task_count": 6, "all_oracle_calls_zero": True, "all_run_locks_released": True,
                    "all_reference_solver_calls_zero": False})
                self.assertIsNotNone(scheduler.preflight_gate(manifest))

    def test_valid_completed_artifact_clears_old_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ops").mkdir()
            (root / "queue.json").write_text(json.dumps({"tasks": [{"id": "a", "args": ["train.py"], "complete": "a.json"}]}))
            (root / "a.json").write_text('{"status":"complete"}')
            (root / "ops/queue.json").write_text(json.dumps({"failures": {"a": {"retry_exhausted": True}}}))
            snapshot = {"gpus": [], "disk_free_bytes": 100 * 2**30, "at": 0}
            with mock.patch.object(scheduler, "ROOT", root), mock.patch.object(scheduler, "OPS", root / "ops"), \
                 mock.patch.object(scheduler, "resource_snapshot", return_value=snapshot):
                self.assertEqual(scheduler.main(["--dry-run"]), 0)
                state = json.loads((root / "ops/dry_run.json").read_text())
                self.assertEqual(state["status"], "tasks_complete")
                self.assertEqual(state["failures"], {})


if __name__ == "__main__":
    unittest.main()
