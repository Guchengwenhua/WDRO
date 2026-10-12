"""Read-only experiment auditing and reproducible v3 result/statistics export.

This module never trains models or generates new oracle scores.  A full report
requires the complete frozen grid; --status is an explicitly incomplete progress
view. Statistical intervals condition on the recorded data and trained seeds.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .common import (ROOT, EXP, DISTRIBUTIONS, atomic_json, derived_seed, make_data,
                     sha256, costs, pin_set_sha256, has_spacing_violation)


FIXED_METHODS = ("random_top4", "critic_raw4", "critic_calibrated4", "rl4",
                 "random64", "cem64", "es64", "critic_calibrated64", "rl64")
FIXED_PAIRS = (("es64", "random64"), ("es64", "cem64"),
               ("critic_calibrated4", "random_top4"), ("rl4", "random_top4"),
               ("critic_calibrated4", "critic_raw4"),
               ("critic_calibrated64", "random64"), ("rl64", "random64"))
JOINT_METHODS = ("random_top4", "critic_calibrated4", "rl_frozen4", "rl_joint4", "random64", "es64")
JOINT_PAIRS = (("rl_joint4", "rl_frozen4"), ("rl_frozen4", "random_top4"),
               ("critic_calibrated4", "random_top4"), ("es64", "random64"))


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    return {"path": str(path), "resolved_path": str(path.resolve()), "sha256": sha256(path)}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(rows), f"refusing empty table {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def array_keys(pins):
    return {pin_set_sha256(net) for net in np.asarray(pins)}


def audit_geometry(pins, degree, context):
    points = np.asarray(pins)
    require(points.ndim == 3 and points.shape[1:] == (degree, 2), f"{context}: bad geometry shape")
    require(np.isfinite(points).all(), f"{context}: nonfinite geometry")
    require(points.min() >= 0.1 - 2e-7 and points.max() <= 0.9 + 2e-7, f"{context}: coordinate bounds")
    require(np.allclose(points.min(axis=1), 0.1, rtol=0, atol=2e-7), f"{context}: bbox origin")
    require(np.allclose(np.ptp(points, axis=1).max(axis=1), 0.8, rtol=0, atol=2e-7),
            f"{context}: longest-span normalization")
    require(not any(has_spacing_violation(net) for net in points), f"{context}: minimum spacing")


def audit_arrays(data, centers, *, degree, budget=None, price=None, context="array"):
    """Recompute every selected cost and algebraic score; no oracle evaluation."""
    n = len(centers)
    for key in ("pins", "centers", "regret", "lstar", "length"):
        require(key in data, f"{context}: missing {key}")
    require(np.array_equal(data["centers"], centers), f"{context}: center identity mismatch")
    audit_geometry(centers, degree, context + " centers")
    audit_geometry(data["pins"], degree, context)
    for key in ("regret", "lstar", "length"):
        value = np.asarray(data[key])
        require(value.shape == (n,) and np.isfinite(value).all(), f"{context}: invalid {key}")
    require(np.all(data["lstar"] > 0) and np.all(data["length"] > 0), f"{context}: nonpositive length")
    require(np.allclose(data["regret"], data["length"] / data["lstar"] - 1, rtol=1e-8, atol=2e-9),
            f"{context}: regret identity")
    require(np.all(data["regret"] >= -1e-7), f"{context}: REST below reference")
    if budget is None:
        require(np.array_equal(data["pins"], centers), f"{context}: plain evaluation pins differ")
        return
    for key in ("cost", "score", "original_regret", "original_score", "calls", "price"):
        value = np.asarray(data[key])
        require(value.shape in ((n,), ()) and np.isfinite(value).all(), f"{context}: invalid {key}")
    require(np.all(data["calls"] == budget), f"{context}: real-score budget mismatch")
    if price is not None:
        require(np.all(np.asarray(data["price"]) == price), f"{context}: price mismatch")
    actual_cost = costs(np.asarray(centers), np.asarray(data["pins"])[:, None])[:, 0]
    require(np.allclose(actual_cost, data["cost"], rtol=1e-9, atol=1e-10), f"{context}: Hungarian cost mismatch")
    expected_score = data["regret"] - np.asarray(data["price"]) * actual_cost
    require(np.allclose(data["score"], expected_score, rtol=1e-8, atol=2e-9), f"{context}: penalized score mismatch")
    require(np.allclose(data["original_score"], data["original_regret"], rtol=0, atol=1e-10),
            f"{context}: original zero-cost score mismatch")
    require(np.all(data["score"] >= data["original_score"] - 1e-9), f"{context}: selected score below original")
    if "ledger_delta_json" in data:
        delta = json.loads(str(np.asarray(data["ledger_delta_json"]).item()))
        for key in ("actor_requested", "actor_completed", "rest_requested", "rest_completed", "reference_requests"):
            require(delta.get(key) == n * budget, f"{context}: {key} call reconciliation")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def paired_bootstrap(left, right, *, seed, metric="mean", repeats=2000):
    """Paired resampling of centers, shared across already-trained seed rows.

    Arrays are [N] or [trained_seed,N].  The latter retains all seed predictions
    for a sampled center and then macro-averages seed statistics; seeds are not
    treated as independent copies of the same evaluation center.
    """
    a, b = np.atleast_2d(np.asarray(left, dtype=np.float64)), np.atleast_2d(np.asarray(right, dtype=np.float64))
    require(a.shape == b.shape and a.shape[1] > 0, "paired bootstrap shape mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "paired bootstrap nonfinite input")
    require(metric in {"mean", "p95"}, "unsupported paired metric")
    statistic = (lambda v: v.mean(axis=-1)) if metric == "mean" else (lambda v: np.quantile(v, 0.95, axis=-1))
    per_seed = statistic(a) - statistic(b)
    rng, boot = np.random.default_rng(seed), np.empty(repeats)
    for start in range(0, repeats, 100):
        stop = min(start + 100, repeats)
        indices = rng.integers(0, a.shape[1], size=(stop - start, a.shape[1]))
        boot[start:stop] = (statistic(a[:, indices]) - statistic(b[:, indices])).mean(axis=0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"difference": float(per_seed.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "per_seed_differences": per_seed.tolist(), "all_seeds_positive": bool(np.all(per_seed > 0)),
            "all_seeds_negative": bool(np.all(per_seed < 0)), "seed_count": a.shape[0],
            "centers": a.shape[1], "bootstrap_replicates": repeats,
            "ci_positive": bool(lo > 0), "ci_negative": bool(hi < 0),
            "independent_unit": "center_shared_across_trained_seeds"}


def expected_tasks(phase, protocol):
    if phase == "fixed":
        return {f"{p}_d{d}_s{s}": (p, d, s) for p in protocol["parents"]
                for d in protocol["degrees"] for s in protocol["seeds"]}
    return {f"{m}_d{protocol['joint']['degree']}_s{s}": (m, protocol["joint"]["degree"], s)
            for m in protocol["joint"]["methods"] for s in protocol["seeds"]}


def phase_status(phase, protocol, exp=EXP):
    expected = expected_tasks(phase, protocol)
    directory = Path(exp) / phase
    complete, failed, active, invalid = [], [], [], []
    for name in expected:
        task = directory / name
        if (task / "complete.json").exists():
            try:
                done = read_json(task / "complete.json")
                (complete if done.get("status") == "pass" and not done.get("smoke", False) else invalid).append(name)
            except (ValueError, OSError):
                invalid.append(name)
        elif (task / "failure.json").exists():
            failed.append(name)
        elif task.exists():
            active.append(name)
    extra = sorted(p.parent.name for p in directory.glob("*/complete.json") if p.parent.name not in expected)
    missing = sorted(set(expected) - set(complete))
    return {"phase": phase, "status": "ready_for_audit" if not missing and not extra else "incomplete",
            "expected_tasks": len(expected), "completed_tasks": len(complete), "completed": sorted(complete),
            "missing": missing, "failed": failed, "active_or_partial": active, "invalid": invalid, "unexpected": extra}


def verify_task(path, expected, phase):
    done = read_json(path / "complete.json")
    require(done.get("status") == "pass" and not done.get("smoke", False), f"{path}: incomplete/smoke task")
    label, degree, seed = expected
    require(done.get("degree") == degree and done.get("seed") == seed, f"{path}: task coordinates mismatch")
    require(done.get("parent" if phase == "fixed" else "method") == label, f"{path}: task method mismatch")
    identity = done["identity"]
    source_hashes = identity["source_hashes"] if phase == "fixed" else identity["source"]
    for name, expected_hash in source_hashes.items():
        require(sha256(ROOT / "adv_v3" / name) == expected_hash, f"{path}: task source drift: {name}")
    task_protocol_hash = identity["protocol_sha256"] if phase == "fixed" else identity["protocol"]["sha256"]
    require(task_protocol_hash == sha256(ROOT / "protocol.json"), f"{path}: task protocol drift")
    listed = {}
    for record in done.get("artifacts", []):
        artifact = Path(record["path"])
        require(artifact.exists() and sha256(artifact) == record["sha256"], f"{path}: artifact hash mismatch {artifact}")
        require(str(artifact) not in listed, f"{path}: duplicate artifact binding")
        listed[str(artifact)] = record["sha256"]
    require(bool(listed), f"{path}: no artifact bindings")
    require((path / "exit.json").exists(), f"{path}: no exit evidence")
    exit_record = read_json(path / "exit.json")
    require(exit_record.get("complete") is True, f"{path}: incomplete exit record")
    require((path / "task.lock").exists(), f"{path}: missing task lock provenance")
    with (path / "task.lock").open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AuditError(f"{path}: task lock still owned") from exc
        fcntl.flock(handle, fcntl.LOCK_UN)
    launch = read_json(path / "launch.json")
    process_file = Path(f"/proc/{launch['pid']}/stat")
    if process_file.exists():
        fields = process_file.read_text().rsplit(") ", 1)[1].split()
        same_process = fields[19] == str(launch.get("process_start_ticks", ""))
        require(not same_process or fields[0] == "Z", f"{path}: owning task process remains alive")
    return done, listed


def require_bound(path, listed):
    require(str(path) in listed, f"unbound required artifact: {path}")


def ledger_details(ledger):
    result = dict(ledger)
    for prefix in ("actor", "rest"):
        low, high = ledger.get(prefix + "_completed", 0), ledger.get(prefix + "_requested", 0)
        require(0 <= low <= high, f"invalid {prefix} ledger bounds")
        result[prefix + "_uncertain_requests"] = high - low
    low, high = ledger.get("geosteiner_misses_completed", 0), ledger.get("geosteiner_misses_started", 0)
    require(0 <= low <= high, "invalid reference ledger bounds")
    result["reference_uncertain_misses"] = high - low
    return result


def input_audit(task, done, evaluation_keys, split_sets, listed):
    """Audit source centers/label datasets; generated RL trajectory Zs not retained."""
    degree = done["degree"]
    group = split_sets.setdefault(degree, {"train": set(), "validation": set(), "evaluation": set()})
    group["evaluation"].update(evaluation_keys)
    require_bound(task / "critic" / "labels.pt", listed)
    require_bound(task / "rl" / "latest.pt", listed)
    labels = torch.load(task / "critic" / "labels.pt", map_location="cpu", weights_only=False)
    require(labels["completed"] == 8192 + 1024 and len(labels["train_pins"]) == 8192 and len(labels["validation_pins"]) == 1024,
            f"{task}: incomplete/nonformal Critic labels")
    for split in ("train", "validation"):
        points = np.asarray(labels[f"{split}_pins"])
        audit_geometry(points, degree, f"{task}: critic {split}")
        group[split].update(array_keys(points))
    state = torch.load(task / "rl" / "latest.pt", map_location="cpu", weights_only=False)
    require(state["update"] == 256, f"{task}: incomplete RL updates")
    cfg, seed, namespace = state["identity"]["config"], state["identity"]["seed"], state["data_namespace"]
    require(cfg["updates"] == 256 and cfg["batch_size"] == 64 and cfg["validation_count"] == 128,
            f"{task}: nonformal RL config")
    val = make_data(degree, cfg["validation_count"], "uniform", namespace + "_validation", seed=seed)
    from .rl import _array_digest
    require(_array_digest(val) == state["validation_centers_sha256"], f"{task}: RL validation reconstruction mismatch")
    group["validation"].update(array_keys(val))
    records = state["training_records"]
    require([row["update"] for row in records] == list(range(1, 257)), f"{task}: RL training cursor coverage")
    log_records = [json.loads(line) for line in (task / "rl" / "training.jsonl").read_text().splitlines()]
    require(log_records == records, f"{task}: RL JSONL/checkpoint records mismatch")
    for row in records:
        x = make_data(degree, cfg["batch_size"], "uniform", namespace + f"_train_update{row['update']:06d}", seed=seed)
        require(_array_digest(x) == row["centers_sha256"], f"{task}: RL train reconstruction mismatch")
        group["train"].update(array_keys(x))
    return {"task": task.name, "critic_train": len(labels["train_pins"]),
            "critic_validation": len(labels["validation_pins"]), "rl_train": 256 * 64,
            "rl_validation": 128, "rl_training_center_reconstruction": "all_batch_hashes_match",
            "scope": "training_source_centers_and_critic_labels_not_unretained_rl_action_trajectories"}


def finish_split_audit(split_sets):
    rows = []
    for degree, sets in sorted(split_sets.items()):
        row = {"degree": degree, **{k + "_unique": len(v) for k, v in sets.items()}}
        for left, right in (("train", "validation"), ("train", "evaluation"), ("validation", "evaluation")):
            count = len(sets[left] & sets[right])
            require(count == 0, f"degree {degree}: {left}/{right} source-data overlap: {count}")
            row[left + "_" + right + "_overlap"] = count
        rows.append(row)
    return rows


def summarize(data):
    regret = data["regret"]
    row = {"nets": len(regret), "mean_regret": float(np.mean(regret)), "p95_regret": float(np.quantile(regret, .95))}
    if "score" in data:
        row.update(mean_score=float(data["score"].mean()), mean_cost=float(data["cost"].mean()),
                   p95_cost=float(np.quantile(data["cost"], .95)), maximum_cost=float(data["cost"].max()),
                   mean_original_regret=float(data["original_regret"].mean()),
                   mean_delta_score=float((data["score"] - data["original_score"]).mean()),
                   improved_fraction=float(np.mean(data["score"] > data["original_score"] + 1e-9)))
    return row


def fixed_report(protocol, exp=EXP):
    status = phase_status("fixed", protocol, exp)
    require(status["status"] == "ready_for_audit", f"fixed grid incomplete: {status}")
    require(tuple(protocol["methods"]) == FIXED_METHODS, "fixed method protocol mismatch")
    cells, efficiency, groups, artifacts, splits, input_rows, diagnostics = [], [], {}, [], {}, [], []
    common_inputs = {}
    for name, (parent, degree, seed) in expected_tasks("fixed", protocol).items():
        task = Path(exp) / "fixed" / name
        done, listed = verify_task(task, (parent, degree, seed), "fixed")
        require(done["methods"] == list(FIXED_METHODS) and done["evaluation_centers"] == 1024,
                f"{name}: fixed method/count coverage")
        eval_keys = set()
        evaluation_score_requests = 0
        for distribution in DISTRIBUTIONS:
            input_path = task / "inputs" / f"{distribution}.npz"
            require_bound(input_path, listed)
            centers = load_npz(input_path)["pins"]
            require(len(centers) == 256, f"{name}: evaluation size")
            audit_geometry(centers, degree, str(input_path))
            key = (degree, distribution)
            if key in common_inputs:
                require(np.array_equal(common_inputs[key], centers), f"{name}: shared evaluation center mismatch")
            else:
                common_inputs[key] = centers
            eval_keys.update(array_keys(centers))
            original_regret = None
            for method in FIXED_METHODS:
                chunks, seconds, delta_ledger, geometry = [], 0.0, {}, {}
                budget = 4 if method in FIXED_METHODS[:4] else 64
                for start in range(0, 256, 64):
                    path = task / "evaluation" / f"{distribution}_{method}_{start:05d}.npz"
                    require_bound(path, listed)
                    data = load_npz(path)
                    audit_arrays(data, centers[start:start + 64], degree=degree, budget=budget,
                                 price=done["price"], context=str(path))
                    require(np.all(np.asarray(data["rho"]) == done["rho"]), f"{path}: rho mismatch")
                    seconds += float(data["elapsed_seconds"])
                    if "ledger_delta_json" in data:
                        for k, v in json.loads(str(data["ledger_delta_json"].item())).items():
                            delta_ledger[k] = delta_ledger.get(k, 0) + v
                    if "geometry_delta_json" in data:
                        for k, v in json.loads(str(data["geometry_delta_json"].item())).items():
                            geometry[k] = geometry.get(k, 0) + v
                    chunks.append(data)
                joined = {k: np.concatenate([c[k] for c in chunks]) for k in
                          ("regret", "score", "cost", "original_regret", "original_score")}
                if original_regret is None:
                    original_regret = joined["original_regret"]
                else:
                    require(np.allclose(original_regret, joined["original_regret"], rtol=0, atol=2e-9),
                            f"{name}: original greedy score changed between methods")
                evaluation_score_requests += 256 * budget
                cells.append({"parent": parent, "degree": degree, "seed": seed, "distribution": distribution,
                              "searcher": method, "budget": budget, "price": done["price"], "rho": done["rho"],
                              **summarize(joined), "evaluation_seconds": seconds,
                              "real_score_requests": 256 * budget, "ledger_delta": delta_ledger,
                              "geometry": geometry, "selected_legal_fraction": 1.0})
                groups[(parent, degree, distribution, method, seed)] = joined
        expected_evaluation_files = {str(task / "evaluation" / f"{d}_{m}_{start:05d}.npz")
                                     for d in DISTRIBUTIONS for m in FIXED_METHODS for start in range(0, 256, 64)}
        require({str(p) for p in (task / "evaluation").glob("*.npz")} == expected_evaluation_files,
                f"{name}: unexpected/missing evaluation files")
        rl_counts = done["rl_summary"]["counts"]
        cs, rs = done["critic_summary"], done["rl_summary"]
        diagnostics.append({"task": name, "parent": parent, "degree": degree, "seed": seed,
                            "raw_validation_mse": cs["raw_validation"]["mse"],
                            "calibrated_validation_mse": cs["validation"]["mse"],
                            "raw_validation_spearman": cs["raw_validation"]["spearman"],
                            "calibrated_validation_spearman": cs["validation"]["spearman"],
                            "raw_zero_gradient_fraction": cs["raw_gradient"]["zero_gradient_fraction"],
                            "calibrated_zero_gradient_fraction": cs["calibrated_gradient"]["zero_gradient_fraction"],
                            "raw_zero_prediction_fraction": cs["raw_gradient"]["zero_prediction_fraction"],
                            "calibrated_zero_prediction_fraction": cs["calibrated_gradient"]["zero_prediction_fraction"],
                            "rl_best_update": rs["best_update"], "rl_best_validation_f": rs["best_f"],
                            "critic_best_epoch": cs["best_epoch"]})
        train_requests = int(done["critic_summary"]["label_requests"]) + rl_counts["training_requests"] + rl_counts["validation_requests"]
        ledger = ledger_details(done["ledger"])
        require(ledger["actor_completed"] >= evaluation_score_requests + train_requests,
                f"{name}: actual oracle count smaller than committed training/evaluation")
        efficiency.append({"task": name, "parent": parent, "degree": degree, "seed": seed,
                           "critic_label_requests": done["critic_summary"]["label_requests"],
                           "rl_training_requests": rl_counts["training_requests"],
                           "rl_validation_requests": rl_counts["validation_requests"],
                           "rl_baseline_requests": rl_counts["baseline_requests"],
                           "search_requests": evaluation_score_requests,
                           "extra_completed_requests": ledger["actor_completed"] - evaluation_score_requests - train_requests,
                           "critic_seconds_current_invocation": done["critic_seconds"],
                           "rl_seconds_current_invocation": done["rl_seconds"],
                           "rl_checkpoint_training_seconds": done["rl_summary"]["training_wall_seconds"],
                           "task_seconds_current_invocation": done["task_seconds"],
                           "peak_allocated_mib": done.get("peak_allocated_mib"),
                           "peak_reserved_mib": done.get("peak_reserved_mib"), "actual_ledger": ledger,
                           "walltime_scope": "shared_server_current_invocation_except_explicit_checkpoint_total"})
        input_rows.append(input_audit(task, done, eval_keys, splits, listed))
        artifacts.extend({"path": p, "sha256": h} for p, h in listed.items())
        artifacts.extend([binding(task / "complete.json"), binding(task / "exit.json"),
                          binding(task / "launch.json"), binding(task / "rl" / "training.jsonl")])
        parent_checkpoint = done["parent_checkpoint"]
        require(sha256(parent_checkpoint["logical_path"]) == parent_checkpoint["sha256"], f"{name}: parent model hash mismatch")
        artifacts.append(binding(parent_checkpoint["logical_path"]))
        print(json.dumps({"audit": "fixed", "task": name, "completed_tasks": len(efficiency)}), flush=True)
    paired = []
    for parent in protocol["parents"]:
        for degree in protocol["degrees"]:
            for distribution in DISTRIBUTIONS:
                for left, right in FIXED_PAIRS:
                    for field, metric in (("score", "mean"), ("regret", "mean"), ("regret", "p95"), ("cost", "mean")):
                        a = np.stack([groups[(parent, degree, distribution, left, s)][field] for s in protocol["seeds"]])
                        b = np.stack([groups[(parent, degree, distribution, right, s)][field] for s in protocol["seeds"]])
                        result = paired_bootstrap(a, b, seed=derived_seed("fixed-bootstrap", parent, degree, distribution, left, right, field, metric), metric=metric)
                        paired.append({"parent": parent, "degree": degree, "distribution": distribution,
                                       "left": left, "right": right, "metric": field + "_" + metric,
                                       "interpretation": "higher_score_is_stronger_attack; cost_has_no_standalone_win_direction",
                                       **result, "stable_positive": result["ci_positive"] and result["all_seeds_positive"],
                                       "stable_negative": result["ci_negative"] and result["all_seeds_negative"]})
    require(len(cells) == 1080 and len(efficiency) == 30, "fixed final table coverage")
    return {"phase": "fixed", "cells": cells, "paired": paired, "efficiency": efficiency,
            "diagnostics": diagnostics,
            "input_audit": input_rows, "split_audit": finish_split_audit(splits), "artifacts": artifacts,
            "selected_nets_audited": 30 * 4 * 9 * 256, "tasks": 30,
            "limitations": ["No new REST/GeoSteiner calls in report; frozen exact lengths checked by artifact identity and arithmetic.",
                            "Input split hashes cover source centers and Critic labels, not unretained RL action trajectories.",
                            "CI conditions on trained seeds and fixed data; no multiple-comparison correction or controlled speedup claim."]}


def _joint_chunks(task, listed, distribution, stage, count=512):
    result = []
    for start in range(0, count, 64):
        path = task / "evaluation" / distribution / f"{stage}_{start:05d}.npz"
        require_bound(path, listed)
        data = load_npz(path)
        audit_arrays(data, data["centers"], degree=30, budget=1, price=0, context=str(path))
        require(np.array_equal(data["pins"], data["centers"]), f"{path}: plain evaluation moved pins")
        result.append(data)
    return {k: np.concatenate([p[k] for p in result]) for k in ("centers", "pins", "regret", "lstar", "length")}


def audit_joint_training(task, done, listed):
    history = done["history"]
    require([row["update"] for row in history] == list(range(1, 513)), f"{task}: update cursor coverage")
    log = [json.loads(line) for line in (task / "training.jsonl").read_text().splitlines()]
    require(log == history, f"{task}: history JSONL mismatch")
    cfg = done["identity"]["config"]
    require(cfg["updates"] == 512 and cfg["batch_size"] == 64 and cfg["validation_interval"] == 64,
            f"{task}: nonformal B settings")
    families = {role: set() for role in ("train", "validation", "evaluation", "independent_attack", "rl_online")}
    from .joint import _array_hash
    for path in sorted((task / "inputs").glob("*.npz")):
        require_bound(path, listed)
        arrays = load_npz(path)
        role = path.name.split("__", 1)[0]
        require(role in families, f"{path}: unknown input role")
        audit_geometry(arrays["pins"], 30, str(path))
        spec = json.loads(str(arrays["spec_json"].item()))
        require(spec["identity_hash"] == done["identity_hash"] and spec["degree"] == 30 and spec["count"] == len(arrays["pins"]),
                f"{path}: input specification mismatch")
        keys = array_keys(arrays["pins"])
        require(len(keys) == len(arrays["pins"]), f"{path}: duplicate source net")
        families[role].update(keys)
        if role == "train":
            update = int(path.stem.split("__u")[1])
            require(_array_hash(arrays["pins"]) == history[update-1]["input_sha256"], f"{path}: update input hash mismatch")
    expected_counts = {"train": 512 * 64, "validation": 4 * 128, "evaluation": 4 * 512,
                       "independent_attack": 4 * 64, "rl_online": 128 * 64 if done["method"] == "rl_joint4" else 0}
    require({role: len(keys) for role, keys in families.items()} == expected_counts, f"{task}: B source count coverage")
    names = list(families)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            require(not families[left] & families[right], f"{task}: {left}/{right} source overlap")
    price, search_requests, online_requests = done["initial_price"], 0, 0
    budget = 64 if done["method"] in ("random64", "es64") else 4
    for row in history:
        update = row["update"]
        require_bound(task / "updates" / f"u{update:06d}.json", listed)
        require(read_json(task / "updates" / f"u{update:06d}.json") == row, f"{task}: update record mismatch")
        require(row["search_calls_per_center"] == budget and row["rho"] == done["rho"], f"{task}: B budget/rho mismatch")
        expected_price = max(0, price + cfg["lambda_learning_rate"] * (row["mean_cost"] - done["rho"]))
        require(np.isclose(row["lambda_before"], price, rtol=0, atol=1e-12) and np.isclose(row["lambda_after"], expected_price, rtol=0, atol=1e-12),
                f"{task}: lambda update direction/value mismatch")
        price = row["lambda_after"]
        for key in ("actor_requested", "actor_completed", "rest_requested", "rest_completed", "reference_requests"):
            require(row["search_ledger_delta"].get(key) == 64 * budget, f"{task}: training search budget {key}")
        search_requests += 64 * budget
        for key in ("sampled_actor_requested", "sampled_actor_completed", "sampled_rest_requested", "sampled_rest_completed", "sampled_reference_reuses"):
            require(row["sampled"]["ledger_delta"].get(key) == 64, f"{task}: sampled training budget {key}")
        online = row["online"]
        if done["method"] == "rl_joint4" and update % 4 == 0:
            require(online is not None and online["kind"] == "rl" and online["real_score_requests"] == 128 and online["baseline_requests"] == 64,
                    f"{task}: RL online update schedule/accounting mismatch")
            online_requests += 128
        elif done["method"] == "critic_calibrated4" and update % 32 == 0:
            require(online is not None and online["kind"] == "critic" and online["real_oracle_calls"] == 0,
                    f"{task}: Critic online update schedule/accounting mismatch")
        else:
            require(online is None, f"{task}: unplanned online update")
    require(np.isclose(price, done["final_price"], rtol=0, atol=1e-12), f"{task}: final price mismatch")
    best_update, best_worst = 0, None
    initial_nominal = None
    for update in range(0, 513, 64):
        means = {}
        for distribution in DISTRIBUTIONS:
            parts = []
            source_path = task / "inputs" / f"validation__{distribution}.npz"
            centers = load_npz(source_path)["pins"]
            for start in (0, 64):
                path = task / "validation" / f"u{update:06d}" / f"{distribution}_{start:05d}.npz"
                require_bound(path, listed)
                arrays = load_npz(path)
                audit_arrays(arrays, centers[start:start+64], degree=30, budget=1, price=0, context=str(path))
                parts.append(arrays["regret"])
            means[distribution] = float(np.concatenate(parts).mean())
        worst = max(value for key, value in means.items() if key != "uniform")
        if update == 0:
            initial_nominal, best_worst = means["uniform"], worst
        elif means["uniform"] <= initial_nominal + .002 and worst < best_worst:
            best_update, best_worst = update, worst
    require(done["best_update"] == best_update, f"{task}: validation-only best selection mismatch")
    require(done["best_validation"]["nominal_mean"] <= initial_nominal + .002 + 1e-12, f"{task}: nominal gate failed")
    expected_validation = {str(task / "validation" / f"u{u:06d}" / f"{d}_{start:05d}.npz")
                           for u in range(0, 513, 64) for d in DISTRIBUTIONS for start in (0, 64)}
    require({str(p) for p in (task / "validation").glob("*/*.npz")} == expected_validation,
            f"{task}: validation file coverage mismatch")
    return {"task": task.name, **{role + "_nets": count for role, count in expected_counts.items()},
            "source_overlap_pairs": 0, "best_selection_recomputed": True,
            "training_search_requests": search_requests, "online_attacker_requests": online_requests,
            "sampled_main_actor_nets": 512 * 64, "sampled_main_reference_reuses": 512 * 64,
            "committed_training_wall_seconds": sum(row["wall_seconds"] for row in history)}


def joint_report(protocol, exp=EXP):
    status = phase_status("joint", protocol, exp)
    require(status["status"] == "ready_for_audit", f"joint grid incomplete: {status}")
    require(tuple(protocol["joint"]["methods"]) == JOINT_METHODS, "joint method protocol mismatch")
    rows, efficiency, artifacts, groups, shared, common_centers, input_rows = [], [], [], {}, {}, {}, []
    for name, (method, degree, seed) in expected_tasks("joint", protocol).items():
        task = Path(exp) / "joint" / name
        done, listed = verify_task(task, (method, degree, seed), "joint")
        require(done["updates"] == 512, f"{task}: incomplete joint updates")
        history = done["history"]
        require(len(history) == 512, f"{task}: history cursor coverage")
        training_audit = audit_joint_training(task, done, listed)
        input_rows.append(training_audit)
        for distribution in DISTRIBUTIONS:
            initial = _joint_chunks(task, listed, distribution, "initial")
            final = _joint_chunks(task, listed, distribution, "final")
            require(len(initial["regret"]) == 512 and np.array_equal(initial["centers"], final["centers"]),
                    f"{task}: before/after evaluation pairing")
            key = (distribution, seed)
            if distribution in common_centers:
                require(np.array_equal(common_centers[distribution], initial["centers"]), f"{task}: cross-seed evaluation centers differ")
            else:
                common_centers[distribution] = initial["centers"]
            if key in shared:
                require(np.array_equal(shared[key]["centers"], initial["centers"]), f"{task}: method center pairing")
                require(np.array_equal(shared[key]["regret"], initial["regret"]), f"{task}: initial model predictions differ")
            else:
                shared[key] = initial
            attack_path = task / "attacks" / distribution / "cem64_00000.npz"
            require_bound(attack_path, listed)
            attacked = load_npz(attack_path)
            require(len(attacked["regret"]) == 64, f"{task}: independent attack count")
            audit_arrays(attacked, attacked["centers"], degree=30, budget=64, price=0, context=str(attack_path))
            for stage, data in (("initial", initial), ("final", final), ("attacked_final", attacked)):
                rows.append({"method": method, "degree": 30, "seed": seed, "distribution": distribution,
                             "stage": stage, "best_update": done["best_update"], **summarize(data)})
                groups[(method, seed, distribution, stage)] = data
        expected_evaluation = {str(task / "evaluation" / d / f"{stage}_{start:05d}.npz")
                               for d in DISTRIBUTIONS for stage in ("initial", "final") for start in range(0, 512, 64)}
        expected_attack = {str(task / "attacks" / d / "cem64_00000.npz") for d in DISTRIBUTIONS}
        require({str(p) for p in (task / "evaluation").glob("*/*.npz")} == expected_evaluation,
                f"{task}: evaluation file coverage mismatch")
        require({str(p) for p in (task / "attacks").glob("*/*.npz")} == expected_attack,
                f"{task}: independent attack file coverage mismatch")
        efficiency.append({"task": name, "method": method, "degree": degree, "seed": seed,
                           "best_update": done["best_update"], "initial_price": done.get("initial_price"),
                           "final_price": done.get("final_price"), "rho": done["rho"],
                           "actual_ledger": ledger_details(done["ledger"]),
                           "training_search_requests": training_audit["training_search_requests"],
                           "online_attacker_requests": training_audit["online_attacker_requests"],
                           "sampled_main_actor_nets": training_audit["sampled_main_actor_nets"],
                           "committed_training_wall_seconds": training_audit["committed_training_wall_seconds"],
                           "task_seconds_this_attempt": done["task_seconds_this_attempt"],
                           "history_updates": len(history), "initial_validation": done["initial_validation"],
                           "best_validation": done["best_validation"]})
        artifacts.extend({"path": p, "sha256": h} for p, h in listed.items())
        artifacts.extend([binding(task / "complete.json"), binding(task / "exit.json")])
    paired = []
    for distribution in DISTRIBUTIONS:
        for metric in ("mean", "p95"):
            for method in JOINT_METHODS:
                a = np.stack([groups[(method, s, distribution, "final")]["regret"] for s in protocol["seeds"]])
                b = np.stack([groups[(method, s, distribution, "initial")]["regret"] for s in protocol["seeds"]])
                result = paired_bootstrap(a, b, seed=derived_seed("joint-before-after", method, distribution, metric), metric=metric)
                paired.append({"comparison": "after_minus_before", "left": method, "right": "initial_erm",
                               "distribution": distribution, "metric": "regret_" + metric,
                               **result, "stable_improvement": result["ci_negative"] and result["all_seeds_negative"]})
            for stage in ("final", "attacked_final"):
                for left, right in JOINT_PAIRS:
                    for seed in protocol["seeds"]:
                        require(np.array_equal(groups[(left, seed, distribution, stage)]["centers"], groups[(right, seed, distribution, stage)]["centers"]),
                                "joint cross-method paired centers differ")
                    a = np.stack([groups[(left, s, distribution, stage)]["regret"] for s in protocol["seeds"]])
                    b = np.stack([groups[(right, s, distribution, stage)]["regret"] for s in protocol["seeds"]])
                    result = paired_bootstrap(a, b, seed=derived_seed("joint-pair", left, right, distribution, stage, metric), metric=metric)
                    paired.append({"comparison": stage, "left": left, "right": right,
                                   "distribution": distribution, "metric": "regret_" + metric,
                                   **result, "stable_improvement": result["ci_negative"] and result["all_seeds_negative"]})
    worst = []
    for method in JOINT_METHODS:
        for seed in protocol["seeds"]:
            for stage in ("initial", "final", "attacked_final"):
                means = {d: float(groups[(method, seed, d, stage)]["regret"].mean()) for d in DISTRIBUTIONS if d != "uniform"}
                worst.append({"method": method, "seed": seed, "stage": stage, "worst_group": max(means, key=means.get),
                              "worst_group_mean": max(means.values()), "group_means": means})
    require(len(rows) == 216 and len(efficiency) == 18, "joint final table coverage")
    return {"phase": "joint", "cells": rows, "paired": paired, "efficiency": efficiency,
            "input_audit": input_rows,
            "worst_group": worst, "artifacts": artifacts, "tasks": 18,
            "limitations": ["Minimal alternating one-step policies; full multi-step MARL was not implemented.",
                            "Attacked-final results use independent CEM64; after-minus-before uses paired plain evaluation.",
                            "CI conditions on recorded seeds and center strata; no multiple-comparison correction."]}


def _snapshot_records(snapshot):
    entries = snapshot.get("files", snapshot.get("source_files"))
    require(isinstance(entries, (list, dict)), "source snapshot lacks files/source_files")
    if isinstance(entries, dict):
        return [{"path": k, "sha256": v} for k, v in entries.items()]
    return entries


def audit_provenance(root=ROOT):
    """All provenance gates are mandatory for a completion manifest."""
    root = Path(root)
    required = [root / "protocol.json", root / "EXPERIMENT_PLAN.md",
                root / "manifests" / "source_snapshot.json", root / "manifests" / "parent_snapshot.json",
                root / "manifests" / "tests.json", root / "manifests" / "preflight.json"]
    for path in required:
        require(path.exists(), f"missing completion evidence: {path}")
    protocol = read_json(required[0])
    require(protocol["plan_sha256"] == sha256(root / "EXPERIMENT_PLAN.md"), "frozen plan hash mismatch")
    checked = []
    source = read_json(root / "manifests" / "source_snapshot.json")
    for record in _snapshot_records(source):
        path = Path(record.get("path", record.get("source", "")))
        if not path.is_absolute():
            path = root / path
        require(path.exists() and sha256(path) == record["sha256"], f"frozen v3 source hash mismatch: {path}")
        checked.append(binding(path))
    parent = read_json(root / "manifests" / "parent_snapshot.json")
    require(parent.get("originals_preserved") is True, "parent snapshot did not preserve originals")
    for record in parent["source_files"]:
        source_path = Path(record["source"])
        require(source_path.exists() and sha256(source_path) == record["sha256"], f"parent source mutated: {source_path}")
    tests = read_json(root / "manifests" / "tests.json")
    require(tests.get("status") in ("pass", "passed") and tests.get("exit_code", 0) == 0, "tests evidence not passing")
    require("log" in tests and "sha256" in tests, "tests output binding missing")
    tests_log = Path(tests["log"])
    require(tests_log.exists() and sha256(tests_log) == tests["sha256"], "tests log hash mismatch")
    required.append(tests_log)
    preflight = read_json(root / "manifests" / "preflight.json")
    require(preflight.get("status") not in ("fail", "failed", "blocked"), "preflight failed")
    supplementary = []

    def bind_nested_artifacts(evidence):
        for row in evidence.get("artifacts", []):
            artifact = Path(row["path"])
            if not artifact.is_absolute():
                artifact = root / artifact
            require(artifact.exists() and sha256(artifact) == row["sha256"], f"GPU integration artifact mismatch: {artifact}")
            if artifact in required:
                continue
            required.append(artifact)
            if artifact.suffix == ".json":
                bind_nested_artifacts(read_json(artifact))

    for path in (root / "manifests" / "gpu_smoke.json",
                 root / "manifests" / "final_integration.json",
                 root / "experiments" / "diagnostics" / "gpu_recovery.json",
                 root / "experiments" / "diagnostics" / "gpu_memory_profile.json",
                 root / "experiments" / "diagnostics" / "joint_gpu_recovery" / "comparison.json"):
        if path.exists():
            evidence = read_json(path)
            require(evidence.get("status") in ("pass", "passed"), f"GPU evidence not passing: {path}")
            required.append(path)
            supplementary.append({"path": str(path), "kind": evidence.get("kind", path.stem),
                                  "scope": evidence.get("scope", "GPU smoke and resource evidence")})
            bind_nested_artifacts(evidence)
            for row in evidence.get("results", []):
                if "log" in row and "log_sha256" in row:
                    log = Path(row["log"])
                    require(log.exists() and sha256(log) == row["log_sha256"], f"GPU smoke log mismatch: {log}")
                    required.append(log)
    postflight = root / "experiments" / "diagnostics" / "joint_gpu_recovery" / "postflight.json"
    if postflight.exists():
        evidence = read_json(postflight)
        require(evidence["all_child_pids_exited"] is True and
                all(value == "released_and_reacquirable" for value in evidence["locks"].values()),
                "joint GPU recovery processes or locks not released")
        required.append(postflight)
    # These operational programs are allowed to evolve independently of the
    # frozen experiment implementations; bind the versions that performed this
    # audit so the resulting statistics remain reproducible.
    operational = [root / "adv_v3" / name for name in ("report.py", "queue.py", "pipeline.py")]
    required.extend(path for path in operational if path.exists())
    return {"status": "pass", "parent_sources_checked": len(parent["source_files"]),
            "v3_sources_checked": len(checked), "bindings": [binding(p) for p in required],
            "source_bindings": checked, "tests": tests, "preflight": preflight,
            "supplementary_gpu_evidence": supplementary}


def write_results(results, protocol, provenance, exp=EXP, root=ROOT):
    tables = Path(exp) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = []
    for phase, result in results.items():
        for key in ("cells", "paired", "efficiency", "input_audit", "split_audit", "worst_group", "diagnostics"):
            if result.get(key):
                path = tables / f"{phase}_{key}.csv"
                write_csv(path, result[key])
                outputs.append(binding(path))
        compact = {k: v for k, v in result.items() if k not in {"artifacts", "cells", "paired", "efficiency", "input_audit", "worst_group"}}
        compact["row_counts"] = {key: len(result[key]) for key in ("cells", "paired", "efficiency")}
        path = tables / f"{phase}_audit.json"
        atomic_json(path, dict(status="pass", **compact))
        outputs.append(binding(path))
    lines = ["# 对抗生成框架 v3 实验结果", "", "本文件由 adv_v3.report 从完整已绑定产物生成；没有新增训练或 oracle 评分。", ""]
    if "fixed" in results:
        fixed = results["fixed"]
        lines += [f"A：30个父模型任务、1080个评价格完整，逐网复核 {fixed['selected_nets_audited']} 个选中net的合法性、Hungarian成本、真实分数恒等式及原X下界。", "",
                  "| 比较（左减右） | score均值稳定为正的格数 | 稳定为负的格数 | 其余格数 |", "|---|---:|---:|---:|"]
        for left, right in FIXED_PAIRS:
            selected = [r for r in fixed["paired"] if r["left"] == left and r["right"] == right and r["metric"] == "score_mean"]
            pos, neg = sum(r["stable_positive"] for r in selected), sum(r["stable_negative"] for r in selected)
            lines.append(f"| {left} − {right} | {pos} | {neg} | {len(selected)-pos-neg} |")
        lines += ["", "每格为固定父方法×degree×分布、三个已训练seed的配对比较；正值代表攻击分数更高。稳定方向要求95%配对CI不跨零且三个seed同向。cost变化单列，不能把走得更远直接称为更好。", ""]
    if "joint" in results:
        joint = results["joint"]
        count = sum(r["best_update"] == 0 for r in joint["efficiency"])
        lines += [f"B：18个交替训练任务、216行训练前后及独立攻击统计完整；{count}个任务选中起始模型（best_update=0）。", "",
                  "训练前后mean/P95、逐seed最差OOD组和独立CEM64比较见 joint_cells.csv、joint_worst_group.csv、joint_paired.csv。RL联合版新增奖励调用单独计账，其总训练预算不同于冻结攻击器版。", ""]
    else:
        lines += ["B尚未纳入本次报告，不能把A报告称为全部三路线交付完成。", ""]
    lines += ["全部CI采用2000次按中心配对bootstrap，三个seed共享同一中心重采样；条件于本次固定数据和已训练模型，未作多重比较校正。未证明连续空间最坏分布、Wasserstein风险证书或新方法的普遍优势。", "",
              "调用账目保留请求/完成上下界及恢复额外调用。共享服务器和跨恢复分段耗时不支持受控加速比；逐项效率范围见efficiency表。输入分离审计覆盖训练来源中心和Critic标签集，未保留的RL动作轨迹不在逐内容交集审计内。", "",
              "原precision_v2实验与代码均保留，父源文件hash再次核验通过；阶段9论文工作没有启动。", ""]
    phase = "all" if set(results) == {"fixed", "joint"} else next(iter(results))
    report_path = Path(root) / ("RESULTS.md" if phase == "all" else f"RESULTS_{phase.upper()}.md")
    temporary = report_path.with_name(report_path.name + ".partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    outputs.append(binding(report_path))
    if phase != "all":
        # The stable phase report is bound by its manifest; the combined preview
        # can later change without invalidating the earlier phase's evidence.
        preview = Path(root) / "RESULTS.md"
        temporary = preview.with_name(preview.name + ".partial")
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(preview)
    all_bindings = list(provenance["bindings"]) + outputs
    for result in results.values():
        all_bindings.extend(result["artifacts"])
    unique = {}
    for record in all_bindings:
        path = record["path"]
        require(path not in unique or unique[path]["sha256"] == record["sha256"], f"conflicting artifact bindings {path}")
        unique[path] = record
    completion = {"status": "pass", "phase": phase, "all_experiments_complete": phase == "all",
                  "protocol_sha256": sha256(Path(root) / "protocol.json"), "provenance": provenance,
                  "phases": {p: {"tasks": r["tasks"], "cells": len(r["cells"])} for p, r in results.items()},
                  "artifacts": list(unique.values()), "artifact_bindings": len(unique),
                  "created_at_unix": time.time()}
    atomic_json(Path(root) / "manifests" / f"{phase}_completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fixed", "joint", "all"), default="all")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    protocol = read_json(ROOT / "protocol.json")
    phases = ("fixed", "joint") if args.phase == "all" else (args.phase,)
    if args.status:
        print(json.dumps({p: phase_status(p, protocol) for p in phases}, indent=2))
        return
    provenance = audit_provenance()
    results = {p: (fixed_report(protocol) if p == "fixed" else joint_report(protocol)) for p in phases}
    done = write_results(results, protocol, provenance)
    print(json.dumps({"status": done["status"], "phase": done["phase"], "phases": done["phases"],
                      "artifact_bindings": done["artifact_bindings"]}))


if __name__ == "__main__":
    main()
