"""CPU-only checks of resource, corruption, replay and process-lease boundaries."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil
import signal
import time
from unittest.mock import patch

BRANCH = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("papers_scheduler", BRANCH / "scheduler.py")
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BRANCH / "artifacts/operations")
        self.root = Path(self.temp.name)
        self.patch_root = patch.object(s, "ROOT", self.root)
        self.patch_ops = patch.object(s, "OPS", self.root / "ops")
        self.patch_root.start()
        self.patch_ops.start()

    def tearDown(self):
        self.patch_ops.stop()
        self.patch_root.stop()
        self.temp.cleanup()

    def test_real_pid_identity_and_reuse_guard(self):
        record = s.identity(os.getpid())
        self.assertTrue(s.alive(record))
        self.assertFalse(s.alive(dict(record, start_ticks="0")))
        self.assertFalse(s.alive(dict(record, boot_id="wrong")))
        self.assertIsNone(s.identity(-1))

    def test_multisample_admission_rejects_saturated_compute(self):
        args = argparse.Namespace(admission_samples=3, max_mean_util=80, required_low_util=70,
                                  max_sample_util=95, memory_margin_mib=2048)
        task = {"required_free_mib": 4096}
        sample = lambda values: [{"utilization": n, "free_mib": 10000} for n in values]
        self.assertTrue(s.admissible(sample([70, 80, 90]), task, args))
        self.assertFalse(s.admissible(sample([89, 89, 89]), task, args))
        self.assertFalse(s.admissible(sample([71, 71, 71]), task, args))
        self.assertFalse(s.admissible(sample([60, 60, 96]), task, args))
        self.assertFalse(s.admissible(sample([10, 10]), task, args))
        rows = sample([30, 40, 50])
        rows[0]["free_mib"] = 6143
        self.assertFalse(s.admissible(rows, task, args))
        rows[0]["free_mib"] = 6144
        self.assertTrue(s.admissible(rows, task, args))

    def test_formal_memory_requirement_uses_verified_smoke_peak_without_mutation(self):
        out = self.root / "artifacts/runs/gpu_smoke_buffer"
        out.mkdir(parents=True)
        metrics = out / "metrics.json"
        metrics.write_text(json.dumps({"cuda_max_reserved_mib": 6100.5}))
        complete = out / "complete.json"
        complete.write_text(json.dumps({"status": "complete", "artifacts": {"metrics.json": s.digest(metrics)}}))
        smoke = dict(id="gpu_smoke_buffer", complete=str(complete))
        task = dict(id="buffer_wdro_s101", required_free_mib=4096, depends_on=["gpu_smoke_buffer"])
        task_map = {smoke["id"]: smoke, task["id"]: task}
        accepted = {smoke["id"]: s.completion(smoke)}
        effective = s.effective_requirement(task, task_map, accepted)
        self.assertEqual(effective["required_free_mib"], 7125)
        self.assertEqual(task["required_free_mib"], 4096)
        self.assertEqual(effective["resource_requirement_evidence"][0]["cuda_max_reserved_mib"], 6100.5)
        args = argparse.Namespace(admission_samples=3, max_mean_util=80, required_low_util=70,
                                  max_sample_util=95, memory_margin_mib=2048)
        rows = [{"utilization": 50, "free_mib": 9172}] * 3
        self.assertFalse(s.admissible(rows, effective, args))
        rows = [{"utilization": 50, "free_mib": 9173}] * 3
        self.assertTrue(s.admissible(rows, effective, args))
        metrics.write_text(json.dumps({"cuda_max_reserved_mib": 1}))
        with self.assertRaises(ValueError):
            s.effective_requirement(task, task_map, accepted)

    def test_completion_rejects_corruption_and_wrong_binding(self):
        (self.root / "result.bin").write_bytes(b"verified artifact")
        task = dict(complete="complete.json", binding_checks={"protocol_sha256": "abc"})
        s.atomic(self.root / "complete.json", dict(status="complete", binding={"protocol_sha256": "abc"},
                                                 artifacts={"result.bin": s.digest(self.root / "result.bin")}))
        self.assertTrue(s.completion(task))
        self.assertIsNone(s.completion(dict(task, binding_checks={"protocol_sha256": "wrong"})))
        (self.root / "result.bin").write_bytes(b"corrupt")
        self.assertIsNone(s.completion(task))
        s.atomic(self.root / "complete.json", dict(status="complete", artifacts={}))
        self.assertIsNone(s.completion(task))

    def test_paths_and_dependency_cycles(self):
        with self.assertRaises(ValueError):
            s.local("../outside")
        a = dict(id="a", command=["python", "runner.py"], complete="a/complete.json", depends_on=["b"])
        b = dict(id="b", command=["python", "runner.py"], complete="b/complete.json", depends_on=["a"])
        with self.assertRaises(ValueError):
            s.validate_manifest({"tasks": [a, b]})
        b["depends_on"] = []
        self.assertEqual(len(s.validate_manifest({"tasks": [a, b]})), 2)

    def test_checkpoint_controls_resume_and_literal_substitution(self):
        task = dict(command=["python", "{root}/runner.py", "--physical-gpu-index", "{gpu}", "--threads", "{cpu_threads}"],
                    checkpoint="latest.pt", resume_args=["--resume"])
        first = s.make_command(task, 7, [3, 4])
        self.assertEqual(first[-3:], ["7", "--threads", "2"])
        self.assertNotIn("--resume", first)
        (self.root / "latest.pt").write_bytes(b"checkpoint")
        self.assertEqual(s.make_command(task, 7, [3, 4])[-1], "--resume")

    def test_child_retains_kernel_lease_after_wrapper_descriptor_close(self):
        path = self.root / "gpu.lock"
        lease = s.acquire(path)
        self.assertIsNotNone(lease)
        child = subprocess.Popen([sys.executable, "-c", "import sys;sys.stdin.buffer.read(1)"],
                                 stdin=subprocess.PIPE, pass_fds=(lease.fileno(),))
        try:
            lease.close()
            self.assertTrue(s.held(path))
            self.assertTrue(s.alive(s.identity(child.pid)))
            child.communicate(input=b"x", timeout=10)
            self.assertFalse(s.held(path))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_atomic_write_and_terminal_failure(self):
        s.atomic(self.root / "new/nested.json", {"value": 12})
        self.assertEqual(s.read(self.root / "new/nested.json"), {"value": 12})
        self.assertEqual(s.status_for([], {}, {}, {"a"}, set()), "failed")
        self.assertEqual(s.status_for(["b"], {}, {}, {"a"}, {"a", "b"}), "failed_dependencies")
        self.assertEqual(s.status_for([], {}, {"a": {}}, set(), set()), "tasks_complete")

    def test_mock_telemetry_queue_retries_capacity_from_checkpoint_and_closes(self):
        """No CUDA calls: a stub nvidia-smi drives the actual detached queue."""
        shutil.copy(BRANCH / "scheduler.py", self.root / "scheduler.py")
        bin_path = self.root / "bin"
        bin_path.mkdir()
        fake_smi = bin_path / "nvidia-smi"
        fake_smi.write_text("#!/usr/bin/env python3\nimport sys\n"
                            "print('0, GPU-FAKE-CPU-TEST, fake, 48000, 30000, 10' if '--query-gpu=' in ' '.join(sys.argv) else '')\n")
        fake_smi.chmod(0o755)
        child = self.root / "child.py"
        child.write_text("import hashlib,json,pathlib,sys,time\n"
                         "p=pathlib.Path('result');p.mkdir(exist_ok=True)\n"
                         "if '--resume' not in sys.argv:\n"
                         " (p/'latest.pt').write_bytes(b'resume-state');sys.exit(75)\n"
                         "(p/'answer.txt').write_text('resumed successfully')\n"
                         "(p/'complete.json').write_text(json.dumps({'status':'complete','artifacts':{'answer.txt':hashlib.sha256((p/'answer.txt').read_bytes()).hexdigest()}}))\n")
        task = dict(id="cpu_standin", command=[sys.executable, str(child)], complete="result/complete.json",
                    checkpoint="result/latest.pt", resume_args=["--resume"], worker_poll_seconds=1, cpu_threads=1)
        (self.root / "queue.json").write_text(json.dumps({"tasks": [task], "source_paths": ["child.py"]}))
        env = dict(os.environ, PATH=str(bin_path) + os.pathsep + os.environ["PATH"])
        result = subprocess.run([sys.executable, str(self.root / "scheduler.py"), "--poll-seconds", "1", "--retry-seconds", "0"],
                                cwd=self.root, env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = json.loads((self.root / "artifacts/operations/queue_state.json").read_text())
        self.assertEqual(state["status"], "tasks_complete")
        self.assertEqual(state["attempts"]["cpu_standin"], 2)
        self.assertEqual(state["failures"], {})
        self.assertIn("--resume", json.loads(sorted((self.root / "artifacts/operations").glob("attempt_*.json"))[-1].read_text())["command"])
        self.assertEqual(json.loads((self.root / "artifacts/operations/exit.json").read_text())["status"], "complete")
        self.assertFalse(s.held(self.root / "artifacts/operations/task_cpu_standin.lock"))
        self.assertFalse(s.held(self.root / "artifacts/operations/gpu_0.lock"))

    def test_restart_preserves_live_worker_and_does_not_duplicate(self):
        shutil.copy(BRANCH / "scheduler.py", self.root / "scheduler.py")
        bin_path = self.root / "bin"
        bin_path.mkdir()
        fake_smi = bin_path / "nvidia-smi"
        fake_smi.write_text("#!/usr/bin/env python3\nimport sys\nprint('0, GPU-FAKE-RESTART, fake, 48000, 30000, 10' if '--query-gpu=' in ' '.join(sys.argv) else '')\n")
        fake_smi.chmod(0o755)
        child = self.root / "child.py"
        child.write_text("import hashlib,json,pathlib,time\n"
                         "p=pathlib.Path('result');p.mkdir(exist_ok=True)\n"
                         "(p/'started').write_text('started')\n"
                         "while not (p/'release').exists():time.sleep(0.1)\n"
                         "(p/'complete.json').write_text(json.dumps({'status':'complete','artifacts':{'started':hashlib.sha256((p/'started').read_bytes()).hexdigest()}}))\n")
        task = dict(id="restart", command=[sys.executable, str(child)], complete="result/complete.json",
                    worker_poll_seconds=1, cpu_threads=1)
        (self.root / "queue.json").write_text(json.dumps({"tasks": [task], "source_paths": ["child.py"]}))
        env = dict(os.environ, PATH=str(bin_path) + os.pathsep + os.environ["PATH"])
        command = [sys.executable, str(self.root / "scheduler.py"), "--poll-seconds", "1", "--retry-seconds", "0"]
        first = subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        second = None
        try:
            limit = time.monotonic() + 20
            while not (self.root / "result/started").exists() and time.monotonic() < limit:
                time.sleep(0.1)
            self.assertTrue((self.root / "result/started").exists())
            first.terminate()
            output, _ = first.communicate(timeout=10)
            self.assertEqual(first.returncode, 0, output)
            spec = json.loads(next((self.root / "artifacts/operations").glob("attempt_*.json")).read_text())
            self.assertTrue(s.alive(spec["child_identity"]))
            second = subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            time.sleep(1)
            (self.root / "result/release").write_text("release")
            output, _ = second.communicate(timeout=20)
            self.assertEqual(second.returncode, 0, output)
            state = json.loads((self.root / "artifacts/operations/queue_state.json").read_text())
            self.assertEqual(state["attempts"]["restart"], 1)
            self.assertEqual(state["status"], "tasks_complete")
            self.assertEqual(state["failures"], {})
        finally:
            for process in (first, second):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
            for path in (self.root / "artifacts/operations").glob("attempt_*.json"):
                spec = json.loads(path.read_text())
                for field in ("child_identity", "worker_identity"):
                    record = spec.get(field)
                    if s.alive(record):
                        os.killpg(record["pid"], signal.SIGKILL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
