"""CPU regressions for logical/physical artifact aliases in the delivery report."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adv_v3 import report


OPERATION = Path(__file__).resolve().parents[1] / "operations/report_path_binding_20260912"


class ReportPathBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test_", dir=OPERATION)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.physical = self.root / "storage"
        self.physical.mkdir()
        self.logical = self.root / "experiments"
        self.logical.symlink_to(self.physical, target_is_directory=True)
        self.task = self.logical / "task"
        self.task.mkdir()
        self.artifact = self.task / "input.npz"
        self.artifact.write_bytes(b"bound artifact bytes")
        (self.root / "protocol.json").write_text("{}")
        (self.task / "exit.json").write_text(json.dumps({"complete": True}))
        (self.task / "launch.json").write_text(json.dumps({"pid": -1}))
        (self.task / "task.lock").touch()
        self.done = {
            "status": "pass", "degree": 30, "seed": 101,
            "method": "random_top4", "smoke": False,
            "identity": {"source": {}, "protocol": {
                "sha256": report.sha256(self.root / "protocol.json")}},
            "artifacts": [self.binding(self.artifact.resolve())],
        }
        self.patch = mock.patch.object(report, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def binding(self, path):
        return {"path": str(path), "sha256": report.sha256(path)}

    def verify(self):
        (self.task / "complete.json").write_text(json.dumps(self.done))
        return report.verify_task(self.task, ("random_top4", 30, 101), "joint")

    def test_resolved_producer_binding_accepts_logical_required_path(self):
        _, listed = self.verify()
        report.require_bound(self.artifact, listed)
        self.assertEqual(set(listed), {str(self.artifact.resolve())})

    def test_logical_producer_binding_accepts_physical_required_path(self):
        self.done["artifacts"] = [self.binding(self.artifact)]
        _, listed = self.verify()
        report.require_bound(self.artifact.resolve(), listed)
        self.assertEqual(set(listed), {str(self.artifact.resolve())})

    def test_symlink_alias_duplicate_binding_is_rejected(self):
        self.done["artifacts"].append(self.binding(self.artifact))
        with self.assertRaisesRegex(report.AuditError, "duplicate artifact binding"):
            self.verify()

    def test_changed_bytes_still_fail_hash_gate(self):
        self.artifact.write_bytes(b"tampered bytes")
        with self.assertRaisesRegex(report.AuditError, "artifact hash mismatch"):
            self.verify()

    def test_distinct_unbound_file_with_identical_bytes_is_rejected(self):
        other = self.task / "other.npz"
        other.write_bytes(self.artifact.read_bytes())
        _, listed = self.verify()
        with self.assertRaisesRegex(report.AuditError, "unbound required artifact"):
            report.require_bound(other, listed)

    def test_symlink_retarget_cannot_reuse_previous_binding(self):
        _, listed = self.verify()
        other_storage = self.root / "other_storage"
        other_task = other_storage / "task"
        other_task.mkdir(parents=True)
        (other_task / "input.npz").write_bytes(self.artifact.read_bytes())
        self.logical.unlink()
        self.logical.symlink_to(other_storage, target_is_directory=True)
        with self.assertRaisesRegex(report.AuditError, "unbound required artifact"):
            report.require_bound(self.artifact, listed)

    def test_missing_bound_artifact_is_rejected(self):
        self.artifact.unlink()
        with self.assertRaisesRegex(report.AuditError, "artifact hash mismatch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
