import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stage3_pipeline
import audit_stage3_completed


def cursor_then_wait(task):
    from src.io_utils import atomic_write_json
    atomic_write_json(task["cursor"], {"completed_updates": 25, "status": "running"})
    atomic_write_json(task["queue"], {"status": "complete"})
    time.sleep(60)
    raise AssertionError("The parent should hand off before this worker finishes")


def failed_audit(task):
    raise RuntimeError("held-out content overlap")


def passing_audit(task):
    time.sleep(0.2)
    return {"run_id": task["run_id"], "status": "pass"}


class CompletedAuditGateTests(unittest.TestCase):
    def test_explicit_foreground_audit_runs_after_training_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "queue.json"
            queue.write_text(json.dumps({"status": "complete"}))
            with patch.object(audit_stage3_completed, "QUEUE", queue), patch.object(audit_stage3_completed, "audit_run", passing_audit):
                result = audit_stage3_completed.run_audit_task({"run_id": "fake"}, Mock(), 0,
                    poll_seconds=0.1, handoff_when_training_complete=False)
            self.assertEqual(result, {"run_id": "fake", "status": "pass"})

    def test_active_cpu_worker_yields_with_cursor_and_no_false_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.json"
            queue.write_text(json.dumps({"status": "running"}))
            cursor = root / "logs/stage3_audit/cursors/fake.json"
            job = root / "job"
            task = {"run_id": "fake", "cursor": str(cursor), "queue": str(queue)}
            with patch.object(audit_stage3_completed, "EXP", root), patch.object(audit_stage3_completed, "JOB", job), patch.object(audit_stage3_completed, "QUEUE", queue), patch.object(audit_stage3_completed, "audit_run", cursor_then_wait):
                result = audit_stage3_completed.run_audit_task(task, Mock(), 0, poll_seconds=0.1)
            self.assertIsNone(result)
            handoff = json.loads((job / "handoff.json").read_text())
            self.assertEqual(handoff["cursor_completed_updates"], 25)
            self.assertTrue(handoff["full_audit_still_required"])
            self.assertFalse(handoff["training_processes_signaled"])
            self.assertFalse((job / "result.json").exists())

    def test_real_worker_exception_is_not_hidden_as_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "queue.json"
            queue.write_text(json.dumps({"status": "running"}))
            with patch.object(audit_stage3_completed, "QUEUE", queue), patch.object(audit_stage3_completed, "audit_run", failed_audit):
                with self.assertRaisesRegex(RuntimeError, "held-out content overlap"):
                    audit_stage3_completed.run_audit_task({"run_id": "fake"}, Mock(), 0, poll_seconds=0.1)

    def test_audit_failure_blocks_pipeline_before_any_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = root / "logs/stage3_completed_audit/failure.json"
            failure.parent.mkdir(parents=True)
            failure.write_text(json.dumps({"reason": "held-out overlap"}))
            with patch.object(stage3_pipeline, "EXP", root):
                with self.assertRaisesRegex(RuntimeError, "unresolved completed-run audit failure"):
                    stage3_pipeline.verify_sources()


if __name__ == "__main__":
    unittest.main()
