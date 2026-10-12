"""Read-only progress/ETA view, calibrated by this branch's CUDA measurements.

Timing ranges are engineering scenarios, not statistical confidence intervals.
Unobserved queue waiting is always unknown. CPU smoke never calibrates GPU ETA.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def jsonl(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            pass  # A live writer may be halfway through its last line.
    return rows


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def category_kind(category):
    if category.startswith("validation_"):
        return "validation"
    if category.startswith("test_"):
        return "test"
    if category.startswith("attack_"):
        return "attack"
    return "training"


def call_durations(rows):
    """Pair events by attempt/update/category; retain actual scored padded nets."""
    pending = {}
    durations = []
    for row in rows:
        key = (row.get("attempt"), row.get("update"), row.get("category"))
        if row.get("event") == "started":
            pending[key] = row
        elif row.get("event") == "completed" and key in pending:
            start = pending.pop(key)
            elapsed = row.get("time", 0) - start.get("time", 0)
            if elapsed > 0 and row.get("nets", 0) > 0:
                durations.append(dict(attempt=key[0], update=key[1], category=key[2],
                                      seconds=elapsed, nets=row["nets"], time=row["time"]))
    return durations


def run_records(root):
    records = []
    for path in sorted((root / "artifacts/runs").glob("*/run.json")):
        run = read(path, {})
        identity = run.get("identity", {})
        heartbeat = read(path.parent / "heartbeat.json", {})
        records.append(dict(path=path.parent, identity=identity, heartbeat=heartbeat,
                            progress=jsonl(path.parent / "progress.jsonl"),
                            validation=jsonl(path.parent / "validation.jsonl"),
                            calls=jsonl(path.parent / "actual_calls.jsonl"),
                            metrics=read(path.parent / "metrics.json", {}),
                            cuda=str(heartbeat.get("device", "")).startswith("cuda")))
    return records


def calibration(records, family, method, protocol):
    same = [r for r in records if r["cuda"] and r["identity"].get("family") == family
            and r["identity"].get("method") == method]
    formal = [r for r in same if r["identity"].get("scope") != "smoke"]
    measured_formal = [r for r in formal if r["progress"]]
    selected = measured_formal if sum(len(r["progress"]) for r in measured_formal) >= 4 else same
    source = "formal_cuda" if selected and selected == measured_formal else "cuda_smoke_extrapolation"
    # Prefer formal update samples whenever any exist; fewer than four are
    # explicitly provisional, rather than mixed with smaller-batch smoke rows.
    if measured_formal:
        selected = measured_formal
        source = "formal_cuda" if sum(len(r["progress"]) for r in selected) >= 4 else "formal_cuda_provisional"
    train = []
    for record in selected:
        train.extend(float(row["seconds"]) for row in record["progress"][-64:]
                     if row.get("seconds", 0) > 0)
    if not train:
        return None
    sampled_sizes = [call["nets"] for record in selected for call in call_durations(record["calls"])
                     if call["category"] == "train_sampled"]
    observed_batch = statistics.median(sampled_sizes) if sampled_sizes else 8
    batch_scale = protocol["batch_size"] / observed_batch if source == "cuda_smoke_extrapolation" else 1
    if source == "cuda_smoke_extrapolation" and observed_batch == protocol["batch_size"]:
        source = "cuda_smoke_same_batch"
    # Formal CUDA windows include CPU adapter/ref costs, optimizer and copies.
    # Smoke B8 -> formal B64 cannot be identified from a single batch size:
    # bound throughput scenarios by constant-per-batch vs linear scaling.
    train_interval = [quantile(train, .2) * .8, quantile(train, .9) * 1.4 * max(1, batch_scale)]
    greedy_records = [r for r in formal if call_durations(r["calls"])] or same
    greedy = []
    greedy_batches = []
    for record in greedy_records:
        for call in call_durations(record["calls"]):
            if category_kind(call["category"]) != "training":
                greedy.append(call["seconds"] / call["nets"])
                greedy_batches.append(call["nets"])
    if greedy:
        typical_batch = statistics.median(greedy_batches)
        batch_advantage = min(1, typical_batch / protocol["greedy_microbatch"])
        greedy_interval = [quantile(greedy, .2) * .8 * batch_advantage, quantile(greedy, .9) * 1.5]
    else:
        greedy_interval = None
    # Validation JSON lacks its own wall-time field. Measure validation wall
    # spans from actual calls to the next training call where available. This
    # includes best-checkpoint writes and validation's Python work.
    validation_spans = []
    for record in greedy_records:
        began = None
        for event in record["calls"]:
            if event.get("event") != "started":
                continue
            kind = category_kind(event.get("category", ""))
            if kind == "validation" and began is None:
                began = event
            elif kind != "validation" and began is not None:
                if event.get("attempt") == began.get("attempt"):
                    duration = event.get("time", 0) - began.get("time", 0)
                    if duration > 0:
                        nets = sum(call["nets"] for call in call_durations(record["calls"])
                                   if call["attempt"] == began.get("attempt") and call["update"] == began.get("update")
                                   and category_kind(call["category"]) == "validation")
                        if nets:
                            validation_spans.append(dict(seconds=duration, nets=nets))
                began = None
    # The per-net measured span provides an additional upper bound for overhead.
    if greedy_interval and validation_spans:
        span_rates = [v["seconds"] / v["nets"] for v in validation_spans]
        greedy_interval[1] = max(greedy_interval[1], quantile(span_rates, .8) * 1.3)
    return dict(source=source, training_samples=len(train), seconds_per_update=train_interval,
                greedy_seconds_per_padded_net=greedy_interval,
                source_runs=[str(r["path"].name) for r in selected],
                validation_span_samples=len(validation_spans),
                measured_completed_run_seconds=[r["metrics"].get("seconds") for r in same if r["metrics"].get("seconds")],
                batch_scale_upper=batch_scale,
                observed_training_batch=observed_batch,
                limitation="Ranges reflect sampling and batch-size scenarios, not confidence intervals; contention and buffering DP CPU costs may change.")


def mul(interval, amount):
    return None if interval is None else [float(v * amount) for v in interval]


def add(intervals):
    return None if any(value is None for value in intervals) else [sum(value[i] for value in intervals) for i in (0, 1)]


def scored_nets(count, microbatch):
    return math.ceil(count / microbatch) * microbatch


def task_dimensions(task):
    command = task.get("command", [])
    values = {}
    for key in ("family", "method", "seed", "run-id", "smoke-batch-size"):
        flag = "--" + key
        if flag in command and command.index(flag) + 1 < len(command):
            values[key.replace("-", "_")] = command[command.index(flag) + 1]
        elif key in task:
            values[key] = task[key]
    return values


def cpu_references(records):
    groups = {}
    for record in records:
        if record["cuda"] or not record["progress"]:
            continue
        key = (record["identity"].get("family"), record["identity"].get("method"))
        row = groups.setdefault(key, dict(family=key[0], method=key[1], seconds=[], batches=set(), source_runs=[]))
        row["seconds"].extend(float(p["seconds"]) for p in record["progress"][-64:] if p.get("seconds", 0) > 0)
        row["batches"].update(c["nets"] for c in call_durations(record["calls"]) if c["category"] == "train_sampled")
        row["source_runs"].append(record["path"].name)
    return [dict(family=row["family"], method=row["method"], observed_batches=sorted(row["batches"]),
                 seconds_per_update_observed_range=[quantile(row["seconds"], .2), quantile(row["seconds"], .9)],
                 source_runs=row["source_runs"], note="CPU measurements only; may include earlier implementation versions; not a GPU ETA")
            for row in groups.values() if row["seconds"]]


def historical_rest_reference(root):
    path = Path('/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/logs/stage6/formal_wdro_d10_s101_u2000_v1/result.json')
    value = read(path, {})
    if value.get("completed_updates") != 2000 or value.get("started_this_segment_at_update") != 1:
        return None
    return dict(source=str(path), batch_size=value.get("run_identity", {}).get("batch_size"), updates=2000,
                elapsed_seconds=value.get("segment_wall_seconds"),
                seconds_per_update=value["segment_wall_seconds"] / 2000,
                note="Historical basic REST WDRO with different batch/runtime; not current four-family GPU timing")


def build_report(root=ROOT, now=None):
    now = time.time() if now is None else now
    root = Path(root)
    protocol = read(root / "protocol.json", {})
    queue = read(root / "queue.json", {})
    state = read(root / "artifacts/operations/queue_state.json", {})
    records = run_records(root)
    by_path = {str(r["path"].resolve()): r for r in records}
    measured = {(f, m): calibration(records, f, m, protocol)
                for f in protocol.get("families", []) for m in protocol.get("methods", [])}
    completed = state.get("completed", {})
    running = state.get("running", {})
    if isinstance(running, list):
        running = {row["task_id"]: row for row in running}
    tasks = queue.get("tasks", [])
    if not tasks:
        tasks = [dict(id=f"{f}_{m}_s{seed}", family=f, method=m, seed=seed,
                      complete=f"artifacts/runs/{f}_{m}_s{seed}/complete.json")
                 for f in protocol.get("families", []) for m in protocol.get("methods", []) for seed in protocol.get("seeds", [])]
    reports = []
    n_dist = len(protocol["distributions"])
    for task in tasks:
        dims = task_dimensions(task)
        task_id = task["id"]
        path = Path(task["complete"])
        path = path if path.is_absolute() else root / path
        record = by_path.get(str(path.parent.resolve()), {})
        identity = record.get("identity", {})
        family = identity.get("family", dims.get("family"))
        method = identity.get("method", dims.get("method"))
        # Postprocessing-only manifest entries cannot be extrapolated as training.
        if family is None or method is None:
            reports.append(dict(task_id=task_id, status="nontraining_task", total_seconds=None))
            continue
        config = dict(protocol)
        smoke = "--smoke" in task.get("command", []) or identity.get("scope") == "smoke"
        if smoke:
            size = int(dims.get("smoke_batch_size", 8))
            config.update(updates=6, batch_size=size, validation_every=2, validation_count=size,
                          test_count=size, attack_count=min(32, size) if "smoke_batch_size" in dims else 4,
                          greedy_microbatch=size)
        total_updates = config["updates"]
        step = config["greedy_microbatch"]
        heartbeat = record.get("heartbeat", {})
        current = min(total_updates, int(heartbeat.get("update", 0)))
        is_running = task_id in running
        finished = task_id in completed
        status = "complete" if finished else "running" if is_running else "failed" if task_id in state.get("exhausted", []) else "pending"
        if finished:
            current = total_updates
        remaining_updates = total_updates - current
        validation_done = {int(row["update"]) for row in record.get("validation", []) if "update" in row}
        scheduled = [0, *range(config["validation_every"], total_updates + 1, config["validation_every"])]
        validation_rounds = [u for u in scheduled if u >= current and u not in validation_done]
        validation_nets = len(validation_rounds) * n_dist * scored_nets(config["validation_count"], step)
        test_nets = 2 * n_dist * scored_nets(config["test_count"], step)
        attack_nets = n_dist * scored_nets(config["attack_count"] * (config["search"]["candidate_count"] + 1), step)
        # Only this live attempt can finish a partly evaluated phase. A restart
        # at update U unconditionally reruns test/attack in runner.py.
        if is_running and record:
            attempt = heartbeat.get("attempt")
            calls = [row for row in call_durations(record["calls"]) if row["attempt"] == attempt]
            validation_nets -= sum(row["nets"] for row in calls if category_kind(row["category"]) == "validation"
                                   and row["update"] in validation_rounds)
            test_nets -= sum(row["nets"] for row in calls if category_kind(row["category"]) == "test")
            attack_nets -= sum(row["nets"] for row in calls if category_kind(row["category"]) == "attack")
        validation_nets, test_nets, attack_nets = [max(0, v) for v in (validation_nets, test_nets, attack_nets)]
        c = measured.get((family, method))
        if finished:
            training = validation = testing = attack = lp = [0., 0.]
        elif c:
            training = mul(c["seconds_per_update"], remaining_updates)
            validation = mul(c["greedy_seconds_per_padded_net"], validation_nets)
            testing = mul(c["greedy_seconds_per_padded_net"], test_nets)
            attack = mul(c["greedy_seconds_per_padded_net"], attack_nets)
            # LP has no dedicated timer in current logs. Explicit conservative
            # engineering allowance, separate from measured neural forward time.
            remaining_lp = n_dist if attack_nets else (1 if heartbeat.get("phase", "").startswith("attack_") else 0)
            lp = [0., remaining_lp * (2 + .002 * config["attack_count"] * (config["search"]["candidate_count"] + 1))]
        else:
            training = validation = testing = attack = lp = None
        reports.append(dict(task_id=task_id, family=family, method=method, scope="smoke" if smoke else "formal", status=status, update=current,
                            total_updates=total_updates, phase=heartbeat.get("phase"),
                            heartbeat_age_seconds=now - heartbeat.get("time", now) if heartbeat else None,
                            remaining_updates=remaining_updates, validation_rounds_remaining=len(validation_rounds),
                            remaining_scored_nets=dict(validation=validation_nets, test=test_nets, attack=attack_nets),
                            training_seconds=training, validation_seconds=validation, test_seconds=testing,
                            attack_forward_seconds=attack, attack_lp_allowance_seconds=lp,
                            total_seconds=add([training, validation, testing, attack, lp]),
                            calibration=c))
    pending_reports = [r for r in reports if r["status"] != "complete"]
    unknown = [r["task_id"] for r in pending_reports if r["total_seconds"] is None]
    known = [r for r in pending_reports if r["total_seconds"] is not None]
    total = add([r["total_seconds"] for r in known]) if known else (None if unknown else [0., 0.])
    weights = {r["task_id"]: r["total_seconds"] for r in known}
    deps = {task["id"]: task.get("depends_on", []) for task in tasks}
    def span(key, endpoint, visiting=frozenset()):
        if key in visiting:
            return 0  # Scheduler validates DAG; malformed read-only snapshots do not hang ETA.
        parent = max((span(dep, endpoint, visiting | {key}) for dep in deps.get(key, [])), default=0.)
        return weights.get(key, [0., 0.])[endpoint] + parent
    longest = [max((span(key, i) for key in deps), default=0.) for i in (0, 1)]
    conditional = {}
    for concurrency in sorted({1, max(1, len(running)), min(8, max(1, len(pending_reports)))}):
        conditional[str(concurrency)] = dict(continuous_gpu_workers=concurrency,
                                            seconds=None if total is None else [max(total[0] / concurrency, longest[0]),
                                                     total[1] / concurrency + (1 - 1 / concurrency) * longest[1]],
                                            assumption="continuous workers and listed task dependencies; excludes capacity waiting and postprocess")
    return dict(at=now, queue_status=state.get("status", "not_started"),
                queue_state_age_seconds=now - state.get("at", now) if state else None,
                total_tasks=len(reports), completed_tasks=len(reports) - len(pending_reports), running_tasks=len(running),
                unknown_calibration_tasks=unknown, known_remaining_work_seconds=total,
                conditional_wall_scenarios=conditional, queue_wait_seconds=None,
                queue_wait_explanation="Unknown: GPUs and CPU throughput depend on external load; free memory alone does not establish compute capacity.",
                fully_estimated=not unknown, tasks=reports,
                cpu_reference_timings=cpu_references(records), historical_rest_reference=historical_rest_reference(root),
                notes=["Intervals are engineering scenarios, not confidence intervals or completion promises.",
                       "Training, validation, test, 33-candidate attack forwards and LP allowance are separated; postprocess is excluded.",
                       "CUDA smoke uses batch-size extrapolation; formal CUDA samples replace it as soon as available.",
                       "Only scheduler-verified completion counts as complete; run files alone do not establish acceptance."])


def duration(value):
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.2f}h"


def show_range(value):
    return "待 CUDA 实测" if value is None else f"{duration(value[0])}–{duration(value[1])}"


def render(report):
    lines = [f"队列: {report['queue_status']} | 已验收 {report['completed_tasks']}/{report['total_tasks']} | 运行 {report['running_tasks']}",
             "GPU 资源排队等待: 未知；下列为持续获得算力时的估算区间。"]
    if report["queue_state_age_seconds"] is not None:
        lines.append(f"状态更新时间距今: {duration(max(0, report['queue_state_age_seconds']))}")
    if report["unknown_calibration_tasks"]:
        lines.append(f"尚缺 CUDA 标定: {len(report['unknown_calibration_tasks'])} 个任务；以下总量仅覆盖已标定任务。")
        if report.get("cpu_reference_timings"):
            lines.append("现有 CPU 每更新耗时参考（不能换算为本轮 GPU ETA）：")
            for value in report["cpu_reference_timings"]:
                interval = value["seconds_per_update_observed_range"]
                lines.append(f"  {value['family']}/{value['method']} B={value['observed_batches']}: {interval[0]:.3f}–{interval[1]:.3f}s/update")
        history = report.get("historical_rest_reference")
        if history:
            lines.append(f"历史基本 REST WDRO d10、B{history['batch_size']}: {history['seconds_per_update']:.2f}s/update；不同运行条件，仅作量级参考。")
    lines.append(f"剩余计算工作量: {show_range(report['known_remaining_work_seconds'])} GPU-worker 时间")
    for value in report["conditional_wall_scenarios"].values():
        lines.append(f"  持续 {value['continuous_gpu_workers']} 个 GPU worker: {show_range(value['seconds'])}，另加排队和后处理")
    lines.append("任务 | 状态/更新 | 训练 | 验证 | 两次 test | 攻评+LP")
    for task in report["tasks"]:
        if task["status"] == "complete":
            continue
        if task["status"] == "nontraining_task":
            lines.append(f"{task['task_id']} | 非训练任务；未估时")
            continue
        attack = add([task["attack_forward_seconds"], task["attack_lp_allowance_seconds"]])
        label = task.get("calibration", {}).get("source") if task.get("calibration") else "uncalibrated"
        lines.append(f"{task['task_id']} | {task['status']} {task['update']}/{task['total_updates']} | "
                     f"{show_range(task['training_seconds'])} | {show_range(task['validation_seconds'])} | "
                     f"{show_range(task['test_seconds'])} | {show_range(attack)} [{label}]")
    lines.append("区间为工程估算；共享负载、CPU 动态规划和 smoke 的 batch 差异会改变速度。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", help="Optionally save JSON under this branch; otherwise read-only")
    args = parser.parse_args()
    report = build_report()
    if args.output:
        path = Path(args.output)
        path = path if path.is_absolute() else ROOT / path
        path.resolve().relative_to(ROOT.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))


if __name__ == "__main__":
    main()
