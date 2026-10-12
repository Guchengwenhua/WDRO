"""Operational hard-resume reconciliation and session-independent child lifetime."""
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from adv_v3 import queue
from adv_v3.common import atomic_json, sha256


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.task = Path(self.temp.name) / "task"
        self.oracle = self.task / "oracle"
        self.oracle.mkdir(parents=True)
        self.ledger_path = self.oracle / "oracle_ledger.json"
        self.database = self.oracle / "reference.sqlite"
        self.before = dict(actor_requested=20, actor_completed=18,
                           actor_forward_rows_requested=256, actor_forward_rows_completed=256,
                           actor_padding_rows_requested=236, actor_padding_rows_completed=238,
                           rest_requested=18, rest_completed=16, reference_requests=7,
                           geosteiner_misses_started=5, geosteiner_misses_completed=3,
                           custom_uncertainty=99)
        atomic_json(self.ledger_path, self.before)
        db = sqlite3.connect(self.database)
        db.execute("CREATE TABLE calls (id INTEGER PRIMARY KEY, requested INTEGER, misses_started INTEGER, misses_completed INTEGER)")
        db.executemany("INSERT INTO calls (requested,misses_started,misses_completed) VALUES (?,?,?)",
                       [(7, 5, 3), (5, 4, 3)])
        db.commit()
        db.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_only_reference_counters_change_with_bound_before_after_receipts(self):
        database_hash = sha256(self.database)
        result = queue.reconcile_reference_ledger(self.task)
        self.assertEqual(result["status"], "reconciled")
        after = json.loads(self.ledger_path.read_text())
        expected = dict(self.before, reference_requests=12,
                        geosteiner_misses_started=9, geosteiner_misses_completed=6)
        self.assertEqual(after, expected)
        self.assertEqual(sha256(self.database), database_hash)
        receipt_path = Path(result["receipt"])
        applied = json.loads(receipt_path.read_text())
        prepared_path = Path(applied["prepared_receipt"])
        prepared = json.loads(prepared_path.read_text())
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(applied["status"], "applied")
        self.assertEqual(applied["ledger_before"], self.before)
        self.assertEqual(applied["ledger_after"], expected)
        self.assertEqual(applied["prepared_receipt_sha256"], sha256(prepared_path))
        self.assertEqual(applied["ledger_after_sha256"], sha256(self.ledger_path))
        self.assertEqual(applied["sqlite_call_rows"], 2)
        self.assertEqual(applied["sqlite_last_call_id"], 2)
        self.assertEqual(queue.reconcile_reference_ledger(self.task)["status"], "unchanged")
        self.assertEqual(len(list((self.oracle / "reconciliations").glob("*.json"))), 2)
        # The prelaunch helper releases ownership before Popen/worker acquisition.
        with (self.task / "task.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_running_pid_and_manual_lock_each_prevent_mutation(self):
        ticks = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
        atomic_json(self.task / "launch.json", dict(pid=os.getpid(), process_start_ticks=ticks))
        original = self.ledger_path.read_bytes()
        self.assertEqual(queue.reconcile_reference_ledger(self.task)["status"], "active")
        self.assertEqual(self.ledger_path.read_bytes(), original)
        (self.task / "launch.json").unlink()
        with (self.task / "task.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(queue.reconcile_reference_ledger(self.task)["status"], "busy")
            self.assertEqual(self.ledger_path.read_bytes(), original)
        self.assertFalse((self.oracle / "reconciliations").exists())

    def test_missing_ledger_and_completed_task_are_skipped(self):
        self.ledger_path.unlink()
        self.assertEqual(queue.reconcile_reference_ledger(self.task)["status"], "missing_ledger")
        self.assertFalse(self.ledger_path.exists())
        atomic_json(self.ledger_path, self.before)
        atomic_json(self.task / "complete.json", dict(status="pass"))
        self.assertEqual(queue.reconcile_reference_ledger(self.task)["status"], "complete")
        self.assertEqual(json.loads(self.ledger_path.read_text()), self.before)

    def test_stale_pid_identity_is_reconciled_without_signalling_it(self):
        atomic_json(self.task / "launch.json", dict(pid=os.getpid(), process_start_ticks="-1"))
        self.assertEqual(queue.reconcile_reference_ledger(self.task)["status"], "reconciled")


class SessionIsolationTests(unittest.TestCase):
    def wait_path(self, path, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(.02)
        self.fail(f"test process did not create {path.name}")

    def test_new_session_child_survives_own_controller_group_sighup(self):
        child_code = '''
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
ticks = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
ready = root / "child_ready.json"
partial = root / "child_ready.partial"
partial.write_text(json.dumps(dict(pid=os.getpid(), sid=os.getsid(0), pgid=os.getpgrp(), ticks=ticks)))
partial.rename(ready)
deadline = time.monotonic() + 15
while not (root / "release").exists() and time.monotonic() < deadline:
    time.sleep(.02)
if (root / "release").exists():
    (root / "child_done").write_text("survived controller SIGHUP")
'''
        controller_code = '''
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c", sys.argv[2], sys.argv[1]],
                 start_new_session=True, stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(20)
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = subprocess.Popen([sys.executable, "-c", controller_code, temporary, child_code],
                                          start_new_session=True, stdin=subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child = None
            try:
                self.wait_path(root / "child_ready.json")
                child = json.loads((root / "child_ready.json").read_text())
                self.assertEqual(child["sid"], child["pid"])
                self.assertEqual(child["pgid"], child["pid"])
                self.assertNotEqual(child["pgid"], controller.pid)
                # Only this test's newly-created controller process group is signalled.
                os.killpg(controller.pid, signal.SIGHUP)
                self.assertEqual(controller.wait(timeout=5), -signal.SIGHUP)
                self.assertTrue(queue.process_active(child["pid"], child["ticks"]))
                (root / "release").touch()
                self.wait_path(root / "child_done")
                self.assertEqual((root / "child_done").read_text(), "survived controller SIGHUP")
                print(json.dumps(dict(proof="isolated_child_survived_controller_group_sighup",
                                      controller_pid=controller.pid, controller_exit=controller.returncode,
                                      child=child, completion="survived controller SIGHUP")), flush=True)
            finally:
                if controller.poll() is None:
                    controller.terminate()
                    controller.wait(timeout=5)
                if child and queue.process_active(child["pid"], child["ticks"]):
                    os.kill(child["pid"], signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
