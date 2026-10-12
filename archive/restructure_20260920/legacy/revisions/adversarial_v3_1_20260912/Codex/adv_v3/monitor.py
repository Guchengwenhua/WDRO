"""Read-only experiment observer; writes only its own logs and ROOT/STATUS.md.

This operational module deliberately imports no science modules, NumPy, Torch,
or model checkpoints. It never signals, restarts, or changes experiment workers.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FOLDER = EXP / "monitor"
COUNTERS = ("actor_requested", "actor_completed", "rest_completed", "reference_requests",
            "geosteiner_misses_completed", "sampled_actor_completed", "sampled_rest_completed")
_CACHE = {}


def _utc(timestamp):
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _read(path, errors, fields=None):
    path = Path(path)
    key = (str(path), fields)
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        old = _CACHE.get(key)
        if old and old[0] == stamp:
            return old[1]
        value = json.loads(path.read_text())
        if fields and isinstance(value, dict):
            value = {name: value[name] for name in fields if name in value}
        _CACHE[key] = stamp, value
        return value
    except FileNotFoundError:
        _CACHE.pop(key, None)
        return None
    except (OSError, ValueError) as error:
        errors.append(f"无法读取 {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {error}")
        return None


def _last_jsonl(path):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262144))
            lines = handle.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
    except OSError:
        pass
    return None


def _process(pid, expected_start=None):
    if pid is None:
        return {"alive": False, "pid": None, "state": "missing"}
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        matches = expected_start is None or str(expected_start) == fields[19]
        return {"pid": int(pid), "alive": matches and fields[0] not in ("Z", "X"),
                "state": fields[0], "start_ticks": fields[19], "identity_matches": matches}
    except (OSError, ValueError, IndexError):
        return {"alive": False, "pid": pid, "state": "exited"}


def _tasks(phase):
    if phase == "fixed":
        return [f"{method}_d{degree}_s{seed}" for degree in (10, 20, 30, 40, 50)
                for seed in (101, 102, 103) for method in ("erm", "wdro_rest")]
    return [f"{method}_d30_s{seed}" for seed in (101, 102, 103)
            for method in ("random_top4", "critic_calibrated4", "rl_frozen4", "rl_joint4", "random64", "es64")]


def _controller(folder, errors):
    launch = _read(folder / "launch.json", errors) or {}
    heartbeat = _read(folder / "heartbeat.json", errors) or {}
    return {"launch": launch, "heartbeat": heartbeat,
            "process": _process(launch.get("pid", heartbeat.get("pid")), launch.get("process_start_ticks"))}


def _gpu(errors):
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=10)
        rows = []
        for line in result.stdout.splitlines():
            index, uuid, total, used, free, utilization = [value.strip() for value in line.split(",")]
            rows.append({"index": int(index), "uuid": uuid, "total_mib": int(total), "used_mib": int(used),
                         "free_mib": int(free), "utilization_percent": int(utilization), "our_active_count": 0})
        return rows
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        errors.append(f"nvidia-smi 采样失败: {error}")
        return []


def _rates(now, previous, totals, tasks):
    if not previous:
        return {"seconds": None, "counters_per_second": {}}
    seconds = now - float(previous["at_unix"])
    if seconds <= 0:
        return {"seconds": seconds, "counters_per_second": {}, "clock_nonmonotone": True}
    delta = {key: totals.get(key, 0) - previous.get("totals", {}).get(key, 0) for key in COUNTERS}
    prior_tasks = {row["key"]: row for row in previous.get("tasks", [])}
    for row in tasks:
        old = prior_tasks.get(row["key"])
        if old:
            count = row["ledger"].get("actor_completed", 0) - old.get("ledger", {}).get("actor_completed", 0)
            row["real_scores_per_second"] = count / seconds if count >= 0 else None
    return {"seconds": seconds, "counter_delta": delta,
            "counters_per_second": {key: value / seconds if value >= 0 else None for key, value in delta.items()},
            "decreased_counters": [key for key, value in delta.items() if value < 0]}


def sample(previous=None):
    now, errors = time.time(), []
    pipeline = _controller(EXP / "pipeline", errors)
    pipeline["state"] = _read(EXP / "pipeline" / "state.json", errors) or {}
    pipeline["complete"] = _read(EXP / "pipeline" / "complete.json", errors, ("status", "at"))
    queues = {phase: _controller(EXP / "queue" / phase, errors) for phase in ("fixed", "joint")}
    totals = {key: 0 for key in COUNTERS}
    tasks, counts, historical_failures = [], {}, []
    for phase in ("fixed", "joint"):
        ids = _tasks(phase)
        counts[phase] = {"total": len(ids), "complete": 0, "active": 0, "failed": 0, "pending": 0}
        queue_active = {row["task"]: row for row in queues[phase]["heartbeat"].get("active", [])}
        blocked = _read(EXP / "queue" / phase / "blocked.json", errors)
        if blocked and not queues[phase]["process"]["alive"]:
            errors.append(f"{phase} 队列阻塞: {blocked.get('status', 'unknown')}")
        for ident in ids:
            out = EXP / phase / ident
            launch = _read(out / "launch.json", errors) or {}
            heartbeat = _read(out / "heartbeat.json", errors) or {}
            complete = _read(out / "complete.json", errors, ("status", "best_update", "updates"))
            failure = _read(out / "failure.json", errors) or {}
            ledger = _read(out / "oracle" / "oracle_ledger.json", errors) or {}
            queued = queue_active.get(ident, {})
            process = _process(launch.get("pid", queued.get("pid")), launch.get("process_start_ticks"))
            finished = bool(complete and complete.get("status") == "pass" and not process["alive"])
            failed = bool(failure and not process["alive"] and not finished)
            status = "complete" if finished else ("active" if process["alive"] else ("failed" if failed else "pending"))
            counts[phase][status] += 1
            for key in COUNTERS:
                totals[key] += int(ledger.get(key, 0))
            if not launch and not queued and not ledger and not complete:
                continue
            gpu = launch.get("physical_gpu_index", queued.get("gpu"))
            current_phase = heartbeat.get("phase", "starting" if process["alive"] else status)
            heartbeat_age = now - heartbeat["at"] if heartbeat.get("at") else None
            evaluation_chunks = len(list((out / "evaluation").rglob("*.npz")))
            attack_chunks = len(list((out / "attacks").rglob("*.npz"))) if phase == "joint" else 0
            update_files = list((out / "updates").glob("u*.json")) if phase == "joint" else []
            committed_update = max((int(path.stem[1:]) for path in update_files), default=0)
            rl_last = _last_jsonl(out / "rl" / "training.jsonl") if phase == "fixed" else None
            critic_summary = _read(out / "critic" / "summary.json", errors, ("status", "best_epoch")) if phase == "fixed" else None
            row = {"key": f"{phase}/{ident}", "task": ident, "experiment_phase": phase, "status": status,
                   "gpu": gpu, "pid": process["pid"], "process": process, "phase": current_phase,
                   "heartbeat_age_seconds": heartbeat_age,
                   "elapsed_process_seconds": max(0, now - float(launch["at"])) if launch.get("at") else None,
                   "distribution": heartbeat.get("distribution"), "searcher": heartbeat.get("searcher"),
                   "offset": heartbeat.get("offset"), "heartbeat_update": heartbeat.get("update"),
                   "committed_update": committed_update, "evaluation_chunks": evaluation_chunks,
                   "independent_attack_chunks": attack_chunks,
                   "rl_update": (rl_last or {}).get("update"), "critic_training_complete": bool(critic_summary),
                   "ledger": ledger, "reference_workers": launch.get("reference_workers"),
                   "real_scores_per_second": None, "completion_stamp": bool(complete)}
            if failure:
                row["last_error"] = failure.get("error", "unknown")
                row["last_error_at"] = failure.get("at")
                if failed:
                    errors.append(f"{row['key']}: {row['last_error']}")
                else:
                    historical_failures.append({"task": row["key"], "error": row["last_error"], "at": row["last_error_at"]})
            if process["alive"] and heartbeat_age is not None and heartbeat_age > 90:
                errors.append(f"{row['key']} 进程存活但心跳已 {int(heartbeat_age)} 秒未更新")
            tasks.append(row)
    gpus = _gpu(errors)
    for gpu in gpus:
        gpu["our_active_count"] = sum(row["status"] == "active" and row["gpu"] == gpu["index"] for row in tasks)
    throughput = _rates(now, previous, totals, tasks)
    if throughput.get("decreased_counters"):
        errors.append("累计计数下降，该区间不提供吞吐估计: " + ", ".join(throughput["decreased_counters"]))
    state = pipeline["state"]
    if state.get("status") == "failed":
        errors.append(f"pipeline 阶段 {state.get('phase')} 失败，退出码 {state.get('code')}")
    elif state.get("status") == "running" and not pipeline["process"]["alive"]:
        errors.append("pipeline 状态仍为running但控制进程已退出；任务进程状态另列")
    terminal = ("pipeline_complete" if pipeline["complete"] and pipeline["complete"].get("status") == "pass"
                else ("pipeline_failed_and_exited" if state.get("status") == "failed" and not pipeline["process"]["alive"] else None))
    return {"schema_version": 1, "at_unix": now, "at_utc": _utc(now), "pipeline": pipeline,
            "queues": queues, "counts": counts, "gpus": gpus, "tasks": tasks, "totals": totals,
            "throughput": throughput, "errors": errors, "historical_failures": historical_failures,
            "disk_free_gib": shutil.disk_usage(EXP).free / 2**30, "terminal_reason": terminal,
            "scope": "formal fixed/joint only; workers read-only; completion stamps not artifact re-audits"}


def _duration(seconds):
    if seconds is None:
        return "—"
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def _number(value):
    return "—" if value is None else f"{value:.1f}"


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(snapshot):
    state = snapshot["pipeline"]["state"]
    phase_names = {"fixed": "A 固定模型验证", "fixed_audit": "A 完整性审计", "joint": "B 交替训练",
                   "final_audit": "最终审计与报告", "all": "全部阶段"}
    phase = phase_names.get(state.get("phase"), state.get("phase", "等待pipeline状态"))
    a, b = snapshot["counts"]["fixed"], snapshot["counts"]["joint"]
    active = [task for task in snapshot["tasks"] if task["status"] == "active"]
    rate = snapshot["throughput"].get("counters_per_second", {})
    lines = ["# 对抗生成框架 v3：运行状态", "", f"更新时间：{snapshot['at_utc']}。",
             f"当前：**{phase} / {state.get('status', 'unknown')}**；A 完成 **{a['complete']}/30**，B 完成 **{b['complete']}/18**；本版本活动任务 {len(active)}。",
             f"A：活动 {a['active']}，待运行 {a['pending']}，失败待处理 {a['failed']}；B：活动 {b['active']}，待运行 {b['pending']}，失败待处理 {b['failed']}。",
             "", f"累计真实greedy评分完成 {snapshot['totals']['actor_completed']:,}，GeoSteiner新求解完成 {snapshot['totals']['geosteiner_misses_completed']:,}；sampled主更新评分 {snapshot['totals']['sampled_actor_completed']:,}。",
             f"最近 {_number(snapshot['throughput'].get('seconds'))} 秒：真实greedy评分 {_number(rate.get('actor_completed'))} net/s，GeoSteiner新求解 {_number(rate.get('geosteiner_misses_completed'))} net/s。磁盘可用 {snapshot['disk_free_gib']:.1f} GiB。",
             "", "| GPU | 可用显存 GiB | 利用率 % | 本版本活动任务 |", "|---|---:|---:|---:|"]
    for gpu in snapshot["gpus"]:
        lines.append(f"| {gpu['index']} | {gpu['free_mib'] / 1024:.2f} | {gpu['utilization_percent']} | {gpu['our_active_count']} |")
    lines.extend(["", "| 活动任务 | GPU | 当前环节 / 已提交进度 | 本进程时长 | 真实评分完成 | net/s | 心跳年龄 |",
                  "|---|---:|---|---:|---:|---:|---:|"])
    for task in sorted(active, key=lambda row: (row["gpu"] if row["gpu"] is not None else 99, row["task"])):
        if task["experiment_phase"] == "fixed":
            progress = f"RL u{task['rl_update']}/256" if task["rl_update"] is not None and "rl" in task["phase"] else f"评价块 {task['evaluation_chunks']}/144"
            if "critic" in task["phase"]:
                progress = "Critic校准完成" if task["critic_training_complete"] else "生成标签/校准中"
        else:
            progress = f"u{task['committed_update']}/512；评价块 {task['evaluation_chunks']}/64；攻击块 {task['independent_attack_chunks']}/4"
        detail = " / ".join(str(value) for value in (task["phase"], task.get("distribution"), task.get("searcher")) if value)
        lines.append(f"| {_escape(task['task'])} | {task['gpu']} | {_escape(detail)}；{progress} | {_duration(task['elapsed_process_seconds'])} | {task['ledger'].get('actor_completed', 0):,} | {_number(task['real_scores_per_second'])} | {_number(task['heartbeat_age_seconds'])}s |")
    if not active:
        lines.append("| 暂无活动任务 | — | — | — | — | — | — |")
    lines.extend(["", "当前异常："])
    if snapshot["errors"]:
        lines.extend(f"- {_escape(error)}" for error in snapshot["errors"])
    else:
        lines.append("- 未发现当前异常。")
    if snapshot["historical_failures"]:
        lines.append(f"- 另有 {len(snapshot['historical_failures'])} 个已重试或已完成任务保留历史failure记录，详见原始采样。")
    lines.extend(["", "[冻结协议](protocol.json) · [实验计划](EXPERIMENT_PLAN.md) · [操作手册](RUNBOOK.md) · [pipeline状态](experiments/pipeline/state.json) · [原始采样](experiments/monitor/samples.jsonl) · [监控代码](adv_v3/monitor.py)",
                  "", "本页只读采样任务状态、PID、心跳和账目；每15秒保存采样，每60秒刷新正文。完成数要求任务complete.json为pass且对应进程退出；最终交付仍以report审计及pipeline完成文件为准。GPU任务数来自活跃任务的明确physical_gpu_index，不从进程环境推断。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="sample and publish once, then exit; never starts workers")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--status-every", type=float, default=60)
    args = parser.parse_args()
    if args.interval < 1 or args.status_every < 1:
        parser.error("intervals must be positive")
    FOLDER.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    with (FOLDER / "monitor.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, "已有monitor持有独占锁；不重复启动。\n")
        process = _process(os.getpid())
        _atomic_json(FOLDER / "launch.json", {"pid": os.getpid(), "process_start_ticks": process["start_ticks"],
                     "at": time.time(), "command": sys.argv, "once": args.once, "interval": args.interval,
                     "status_every": args.status_every, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        previous = _last_jsonl(FOLDER / "samples.jsonl")
        written, status_at, reason = 0, 0.0, "signal"
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stop.set())
        try:
            while not stop.is_set():
                began = time.monotonic()
                current = sample(previous)
                with (FOLDER / "samples.jsonl").open("a") as log:
                    log.write(json.dumps(current, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                written += 1
                terminal = current["terminal_reason"]
                if args.once or terminal or began - status_at >= args.status_every:
                    _atomic_text(ROOT / "STATUS.md", render(current))
                    status_at = began
                    print(json.dumps({"at_utc": current["at_utc"], "phase": current["pipeline"]["state"],
                                      "counts": current["counts"], "throughput": current["throughput"],
                                      "errors": current["errors"]}, ensure_ascii=False), flush=True)
                _atomic_json(FOLDER / "heartbeat.json", {"pid": os.getpid(), "at": time.time(),
                             "samples_this_process": written, "last_sample_at": current["at_unix"]})
                previous = current
                if args.once or terminal:
                    reason = "once" if args.once else terminal
                    break
                stop.wait(max(0.1, args.interval - (time.monotonic() - began)))
        finally:
            _atomic_json(FOLDER / "exit.json", {"pid": os.getpid(), "at": time.time(), "reason": reason,
                         "samples_this_process": written, "workers_modified": False})
            fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
