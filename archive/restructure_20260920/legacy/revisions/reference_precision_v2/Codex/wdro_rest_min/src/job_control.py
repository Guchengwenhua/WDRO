"""Duplicate-run lock, PID, task cursor, and heartbeat records."""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobLease:
    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        exact_command: str,
        resume_command: str,
        monitor_command: str,
        expected_updates: int,
        physical_gpu_index: int | None,
        host_runtime_manifest: str | None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.lock_path = self.root / "run.lock"
        self.job_path = self.root / "job.json"
        self.pid_path = self.root / "pid.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.task_path = self.root / "task.json"
        self._started = utc_now()
        self._job = {
            "schema_version": 1,
            "run_id": run_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at_utc": self._started,
            "exact_command": exact_command,
            "resume_command": resume_command,
            "monitor_command": monitor_command,
            "physical_gpu_index": physical_gpu_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tmux_session": os.environ.get("WDRO_TMUX_SESSION"),
            "host_runtime_manifest": host_runtime_manifest,
            "status": "running",
        }
        self._expected_updates = expected_updates

    def __enter__(self) -> "JobLease":
        try:
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            try:
                existing = json.loads(self.lock_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            pid = int(existing.get("pid", -1))
            if process_is_alive(pid):
                raise RuntimeError(f"run is already locked by live PID {pid}: {self.lock_path}") from error
            stale = self.root / f"run.lock.stale.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            os.replace(self.lock_path, stale)
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "run_id": self.run_id, "created_at_utc": self._started}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_write_json(self.job_path, self._job)
        atomic_write_json(self.pid_path, {"pid": os.getpid(), "run_id": self.run_id})
        self.update(update=0, status="running")
        return self

    def update(self, *, update: int, status: str, detail: dict[str, Any] | None = None) -> None:
        timestamp = utc_now()
        heartbeat = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "last_update_utc": timestamp,
            "completed_updates": update,
            "expected_updates": self._expected_updates,
            "status": status,
        }
        if detail:
            heartbeat["detail"] = detail
        atomic_write_json(self.heartbeat_path, heartbeat)
        atomic_write_json(
            self.task_path,
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "task": "erm_training_and_validation",
                "completed_updates": update,
                "expected_updates": self._expected_updates,
                "status": status,
                "last_update_utc": timestamp,
            },
        )

    def finish(self, *, update: int, status: str, detail: dict[str, Any] | None = None) -> None:
        self.update(update=update, status=status, detail=detail)
        self._job.update({"status": status, "finished_at_utc": utc_now(), "completed_updates": update})
        atomic_write_json(self.job_path, self._job)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc is not None:
            try:
                current = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
                update = int(current.get("completed_updates", 0))
            except Exception:
                update = 0
            self.finish(update=update, status="failed", detail={"error": f"{type(exc).__name__}: {exc}"})
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
