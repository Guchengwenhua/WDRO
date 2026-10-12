#!/usr/bin/env python3
"""Independent artifact audit and stratified paired statistics for RL research.

No training code is imported. Missing or invalid formal tasks make the report
INCOMPLETE and cause a nonzero exit. Bootstrap resamples training seeds and
paired centers crossed across seeds; seed-net pairs are never treated as
independent training replicates.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def expected_tasks(protocol):
    tasks = protocol.get("tasks")
    if isinstance(tasks, list) and tasks:
        result = []
        for item in tasks:
            if isinstance(item, str):
                result.append(item)
            else:
                result.append(item.get("run_id", item.get("id", item.get("task_id"))))
        if not all(result):
            raise ValueError("Every protocol task needs run_id/id/task_id")
        return result
    seeds = protocol.get("seeds", [201, 202, 203])
    methods = protocol.get("methods", [])
    result = [f"warmup_s{s}" for s in seeds]
    for kind in ("fixed", "coev"):
        section = protocol.get(kind, {})
        choices = protocol.get(kind + "_methods", protocol.get("learned_methods", []) if kind == "fixed" else section.get("methods", methods))
        if not choices:
            raise ValueError(f"Protocol must declare {kind}_methods or tasks")
        result += [f"{kind}_{m}_s{s}" for s in seeds for m in choices]
    return result


def own_lengths(pins, topology):
    """Independent REST terminal-span evaluator, including topology checks."""
    x = np.asarray(pins, dtype=np.float64)
    t = np.asarray(topology)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError("pins must have shape [B,N,2]")
    b, n, _ = x.shape
    if t.shape != (b, 2 * (n - 1)) or not np.issubdtype(t.dtype, np.integer):
        raise ValueError("bad topology shape or dtype")
    if np.any(t < 0) or np.any(t >= n):
        raise ValueError("topology index outside net")
    xl = x[:, :, 0].copy()
    xr = xl.copy()
    yl = x[:, :, 1].copy()
    yh = yl.copy()
    for i, edges in enumerate(t.reshape(b, n - 1, 2)):
        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a
        for u, v in edges:
            u, v = int(u), int(v)
            ru, rv = find(u), find(v)
            if ru == rv:
                raise ValueError("topology is disconnected/cyclic")
            parent[ru] = rv
            yl[i, u] = min(yl[i, u], x[i, v, 1])
            yh[i, u] = max(yh[i, u], x[i, v, 1])
            xl[i, v] = min(xl[i, v], x[i, u, 0])
            xr[i, v] = max(xr[i, v], x[i, u, 0])
    return (xr - xl + yh - yl).sum(-1)


def check_points(array, name, errors, degree, epsilon=None, centers=None):
    a = np.asarray(array)
    if a.ndim != 3 or a.shape[1:] != (degree, 2):
        errors.append(f"{name}: invalid coordinate shape {a.shape}")
        return
    if not np.isfinite(a).all() or np.any(a < .1 - 1e-6) or np.any(a > .9 + 1e-6):
        errors.append(f"{name}: nonfinite/out-of-window coordinates")
    for net in a:
        d = np.abs(net[:, None].astype(np.float64) - net[None].astype(np.float64)).sum(-1)
        np.fill_diagonal(d, np.inf)
        if np.any(d < 1e-6):
            errors.append(f"{name}: coincident/too-close pins")
            break
    if centers is not None:
        if a.shape != centers.shape:
            errors.append(f"{name}: center shape mismatch")
        elif epsilon is not None and np.max(np.abs(a - centers)) > epsilon + 1e-6:
            errors.append(f"{name}: displacement cap violated")


def assert_close(actual, expected, label, errors, atol=2e-6):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, rtol=1e-6, atol=atol):
        errors.append(f"{label}: independent recomputation mismatch")


def values_for(arrays, *keys):
    for key in keys:
        if key in arrays:
            return arrays[key]
    return None


def frozen_number(protocol, kind, key, default):
    section = protocol.get(kind, {})
    if not isinstance(section, dict):
        section = {}
    return section.get(key, protocol.get(kind + "_" + key, protocol.get(key, default)))


def verify_budget(manifest, protocol, errors):
    kind = manifest["kind"]
    updates = int(frozen_number(protocol, kind, "updates", 256))
    batch = int(frozen_number(protocol, kind, "batch_size", 32))
    candidates = int(frozen_number(protocol, kind, "candidates", 8))
    groups = len(protocol.get("distributions", ["uniform", "mix_gauss", "edge_bias", "grid_cluster"]))
    calls = manifest.get("calls", {})
    expected = {"search_greedy_scores": 0, "evaluation_greedy_scores": 0,
                "sampled_actor_scores": 0, "oracle_calls": 0}
    if kind in ("fixed", "coev") and manifest.get("method") != "erm":
        expected["search_greedy_scores"] = updates * batch * candidates
    if kind == "fixed":
        count = int(frozen_number(protocol, kind, "test_per_distribution", 128))
        prices = protocol.get("test_lambdas", protocol.get("lambdas", protocol.get("fixed", {}).get("lambdas", [0., 1., 5.])))
        expected["evaluation_greedy_scores"] = groups * count * candidates * len(prices) * 3
    elif kind == "coev":
        count = int(frozen_number(protocol, kind, "test_per_distribution", 1024))
        attack_count = int(frozen_number(protocol, kind, "attack_per_distribution", 128))
        attack_k = int(frozen_number(protocol, kind, "attack_candidates", 32))
        expected.update(sampled_actor_scores=updates * batch,
                        evaluation_greedy_scores=groups * count + groups * attack_count * attack_k * 2)
    elif kind == "warmup":
        expected["sampled_actor_scores"] = updates * batch
    else:
        errors.append(f"Unknown task kind: {kind}")
    observed = {}
    for metric, amount in expected.items():
        found = calls.get(metric)
        observed[metric] = found
        if found is None:
            errors.append(f"budget: missing {metric} counter; available={list(calls)}")
        elif int(found) != amount:
            errors.append(f"budget: {metric}={found}, expected {amount}")
    if int(manifest.get("step", -1)) != updates:
        errors.append(f"training step={manifest.get('step')}, expected {updates}")
    return {"expected": expected, "observed": observed}


def sample_id(pins):
    pins = np.asarray(pins, dtype="<f4")
    order = np.lexsort((pins[:, 1], pins[:, 0]))
    return hashlib.sha256(pins[order].tobytes()).digest()


def regenerate_training_ids(protocol):
    """Independent uniform input regeneration from frozen seed namespaces."""
    ids = set()
    count, degree = int(protocol.get("batch_size", 32)), int(protocol.get("degree", 30))
    for seed in protocol.get("seeds", [201, 202, 203]):
        for kind in ("warmup", "fixed", "coev"):
            for step in range(1, int(protocol[kind + "_updates"]) + 1):
                accepted = []
                for attempt in range(100):
                    parts = (f"rl-research-v1/{kind}/train/{step}", seed, "uniform", degree, attempt)
                    digest_seed = hashlib.sha256(("rl-only-v1:" + ":".join(map(str, parts))).encode()).digest()
                    rng = np.random.default_rng(int.from_bytes(digest_seed[:4], "little"))
                    points = (.1 + .8 * rng.random((count, degree, 2))).astype(np.float32)
                    for p in points:
                        distance = np.abs(p[:, None].astype(float) - p[None].astype(float)).sum(-1)
                        np.fill_diagonal(distance, np.inf)
                        if np.all(distance >= 1e-6):
                            accepted.append(p)
                    if len(accepted) >= count:
                        break
                if len(accepted) < count:
                    raise ValueError("Independent uniform regeneration failed")
                ids.update(sample_id(p) for p in accepted[:count])
    return ids


def verify_call_log(path, manifest, errors):
    totals = dict(search_greedy_scores=0, evaluation_greedy_scores=0, sampled_actor_scores=0,
                  search_padded_forward_rows=0, evaluation_padded_forward_rows=0)
    pending, completed = {}, set()
    try:
        with (path / "actual_calls.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                event = row["event"]
                if event == "sampled_complete":
                    totals["sampled_actor_scores"] += int(row["count"])
                elif event == "start":
                    if row["id"] in pending or row["id"] in completed:
                        errors.append("actual call log: duplicate call id")
                    pending[row["id"]] = row
                elif event == "complete":
                    start = pending.pop(row["id"], None)
                    if start is None or any(start.get(k) != row.get(k) for k in ("phase", "count", "padded")):
                        errors.append("actual call log: missing/mismatched start event")
                    completed.add(row["id"])
                    totals[row["phase"] + "_greedy_scores"] += int(row["count"])
                    totals[row["phase"] + "_padded_forward_rows"] += int(row["padded"])
                else:
                    errors.append(f"actual call log: unknown event {event}")
        for key, actual in totals.items():
            if actual < manifest.get("calls", {}).get(key, 0):
                errors.append(f"actual call log: {key}={actual}, manifest={manifest.get('calls', {}).get(key)}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"actual call log invalid: {exc}")
    return {"completed_totals": totals,
            "retry_or_uncommitted_completed_overhead": {key: value - manifest.get("calls", {}).get(key, 0) for key, value in totals.items()},
            "unpaired_inflight_starts": len(pending),
            "inflight_requested_rows": sum(int(row.get("count", 0)) for row in pending.values()),
            "inflight_requested_padded_rows": sum(int(row.get("padded", 0)) for row in pending.values()),
            "inflight_interpretation": "Requested rows of interrupted calls, not claimed completed GPU evaluations."}


def audit_run(path, protocol, protocol_hash, training_ids=None):
    errors = []
    m = json.loads((path / "complete.json").read_text())
    if m.get("status") != "complete":
        errors.append("manifest status is not complete")
    if m.get("run_id") != path.name:
        errors.append("run_id/path mismatch")
    if m.get("smoke") is not False:
        errors.append("formal manifest must explicitly set smoke=false")
    if not str(m.get("device", "")).startswith("cuda"):
        errors.append("formal task was not run on CUDA")
    if not np.isfinite(float(m.get("elapsed_seconds", float("nan")))):
        errors.append("missing/nonfinite elapsed_seconds")
    binding = m.get("binding", {})
    if binding.get("protocol_sha256") != protocol_hash:
        errors.append("protocol hash mismatch")
    verified_hashes = 0
    for field in ("code_sha256", "dependencies", "parents"):
        mapping = binding.get(field)
        if not isinstance(mapping, dict) or (not mapping and (field != "parents" or m.get("kind") != "warmup")):
            errors.append(f"missing {field} bindings")
            continue
        for name, expected in mapping.items():
            target = Path(name)
            if not target.is_absolute():
                target = ROOT / target
            if not target.is_file() or digest(target) != expected:
                errors.append(f"{field} mismatch: {name}")
            else:
                verified_hashes += 1
    artifacts = m.get("artifacts_sha256", {})
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("missing artifact hash bindings")
    for name, expected in artifacts.items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()):
            errors.append(f"artifact escaped run directory: {name}")
        elif not target.is_file() or digest(target) != expected:
            errors.append(f"artifact hash mismatch: {name}")
        else:
            verified_hashes += 1
    budget = verify_budget(m, protocol, errors)
    actual_calls = verify_call_log(path, m, errors)
    lock_path = path / "run.lock"
    if not lock_path.is_file():
        errors.append("run lock missing")
    else:
        try:
            with lock_path.open("r") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
        except BlockingIOError:
            errors.append("run lock is still held")
    arrays = {}
    if m.get("kind") != "warmup":
        if "results.npz" not in artifacts:
            errors.append("results.npz is not hash-bound")
        if not (path / "results.npz").exists():
            errors.append("results.npz missing")
        else:
            with np.load(path / "results.npz", allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
    degree = int(protocol.get("degree", 30))
    epsilon = float(protocol.get("epsilon", .08))
    distributions = protocol.get("distributions", [])
    required_arrays = {}
    if m.get("kind") == "fixed":
        n = int(protocol.get("fixed_test_per_distribution", 128))
        for dist in distributions:
            required_arrays[dist + "__pins"] = (n, degree, 2)
            for price in protocol.get("test_lambdas", [1., 5., 10.]):
                for method in (m["method"], "random", "es"):
                    prefix = f"{dist}__lambda{price:g}__{method}__"
                    for key in ("gain", "length", "cost", "index", "center_length"):
                        required_arrays[prefix + key] = (n,)
                    required_arrays[prefix + "selected"] = (n, degree, 2)
                    k = int(protocol.get("candidates", 8))
                    required_arrays[prefix + "all_candidates"] = (n, k, degree, 2)
                    required_arrays[prefix + "all_lengths"] = (n, k)
                    required_arrays[prefix + "all_objective"] = (n, k)
    elif m.get("kind") == "coev":
        n, na = int(protocol.get("test_per_distribution", 1024)), int(protocol.get("attack_per_distribution", 128))
        for dist in distributions:
            required_arrays.update({dist + "__nominal_pins": (n, degree, 2), dist + "__nominal_length": (n,),
                                    dist + "__nominal_topology": (n, 2 * (degree - 1)), dist + "__attack_pins": (na, degree, 2)})
            for attack in ("random32", "es32"):
                required_arrays.update({dist + "__" + attack + "_length": (na,), dist + "__" + attack + "_cost": (na,),
                                        dist + "__" + attack + "_selected": (na, degree, 2)})
                k = int(protocol.get("attack_candidates", 32))
                required_arrays.update({dist + "__" + attack + "_all_candidates": (na, k, degree, 2),
                                        dist + "__" + attack + "_all_lengths": (na, k),
                                        dist + "__" + attack + "_index": (na,)})
    for key, shape in required_arrays.items():
        if key not in arrays or arrays[key].shape != shape:
            errors.append(f"{key}: missing/wrong shape; expected {shape}")
    heldout_ids = {}
    if training_ids is not None:
        for key, array in arrays.items():
            if key.endswith("_pins") and array.ndim == 3:
                ids = {sample_id(p) for p in array}
                heldout_ids[key] = ids
                if len(ids) != len(array):
                    errors.append(f"{key}: duplicate held-out inputs")
                if ids.intersection(training_ids):
                    errors.append(f"{key}: held-out input overlaps regenerated training data")
        for left, right in itertools.combinations(heldout_ids, 2):
            if heldout_ids[left].intersection(heldout_ids[right]):
                errors.append(f"held-out datasets overlap: {left}, {right}")
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            errors.append(f"{key}: nonnumeric/nonfinite array")
        if key.endswith("length") and (value.ndim != 1 or np.any(value <= 0)):
            errors.append(f"{key}: invalid length vector")
        if key.endswith("_pins") or key.endswith("_selected"):
            check_points(value, key, errors, degree)
    for key in arrays:
        if not key.endswith("_cost"):
            continue
        suffix = "__" if key.endswith("__cost") else "_"
        prefix = key[:-(len(suffix) + 4)]
        dist = key.split("__")[0]
        selected = arrays.get(prefix + suffix + "selected")
        centers = values_for(arrays, prefix + "__pins", prefix + "__centers",
                             dist + "__attack_pins", dist + "__pins", dist + "__centers")
        if selected is None or centers is None:
            errors.append(f"{prefix}: missing selected/center coordinates for cost audit")
            continue
        check_points(selected, prefix + "__selected", errors, degree, epsilon, centers)
        if selected.shape == centers.shape:
            recomputed = np.abs(selected.astype(np.float64) - centers.astype(np.float64)).sum(-1).mean(-1)
            assert_close(arrays[key], recomputed, key, errors)
    for key in arrays:
        if key.endswith("all_candidates"):
            prefix = key[:-len("all_candidates")]
            dist = key.split("__")[0]
            centers = values_for(arrays, dist + "__attack_pins", dist + "__pins")
            try:
                candidates = arrays[key]
                if centers is None or candidates.ndim != 4 or candidates.shape[0] != len(centers):
                    raise ValueError("candidate/center shape mismatch")
                b, k, n, _ = candidates.shape
                center_copies = np.broadcast_to(centers[:, None], candidates.shape).reshape(-1, n, 2)
                check_points(candidates.reshape(-1, n, 2), key, errors, degree, epsilon, center_copies)
                assert_close(candidates[:, 0], centers, prefix + "center candidate", errors, atol=0)
                all_lengths = arrays[prefix + "all_lengths"]
                if all_lengths.shape != (b, k) or np.any(all_lengths <= 0):
                    raise ValueError("nonpositive/wrong-shape candidate lengths")
                cost = np.abs(candidates.astype(float) - centers[:, None].astype(float)).sum(-1).mean(-1)
                price = float(key.split("__")[1].removeprefix("lambda")) if m["kind"] == "fixed" else 0.
                objective = all_lengths - price * cost
                idx = arrays[prefix + "index"]
                if not np.array_equal(idx, objective.argmax(1)):
                    errors.append(f"{prefix}: selected index is not first true-objective argmax")
                if idx.shape != (b,) or not np.issubdtype(idx.dtype, np.integer) or np.any(idx < 0) or np.any(idx >= k):
                    raise ValueError("invalid selected index vector")
                rows = np.arange(b)
                assert_close(arrays[prefix + "selected"], candidates[rows, idx], prefix + "selected", errors, atol=0)
                assert_close(arrays[prefix + "length"], all_lengths[rows, idx], prefix + "length", errors)
                if m["kind"] == "fixed":
                    assert_close(arrays[prefix + "all_objective"], objective, prefix + "all_objective", errors)
                    assert_close(arrays[prefix + "center_length"], all_lengths[:, 0], prefix + "center_length", errors)
            except (KeyError, ValueError, IndexError) as exc:
                errors.append(f"{key}: cannot verify complete candidate selection: {exc}")
        if key.endswith("__nominal_length"):
            dist = key.split("__")[0]
            try:
                recomputed = own_lengths(arrays[dist + "__nominal_pins"], arrays[dist + "__nominal_topology"])
                assert_close(arrays[key], recomputed, key, errors)
            except (KeyError, ValueError) as exc:
                errors.append(f"{key}: cannot recompute length: {exc}")
        if key.endswith("__gain"):
            prefix = key[:-6]
            parts = prefix.split("__")
            try:
                price = float(parts[1].removeprefix("lambda"))
                reconstructed = arrays[prefix + "__length"] - price * arrays[prefix + "__cost"] - arrays[prefix + "__center_length"]
                assert_close(arrays[key], reconstructed, key, errors)
                if np.min(arrays[key]) < -2e-6:
                    errors.append(f"{key}: center inclusion violated")
                idx = arrays[prefix + "__index"]
                k = int(frozen_number(protocol, "fixed", "candidates", 8))
                if not np.issubdtype(idx.dtype, np.integer) or np.any(idx < 0) or np.any(idx >= k):
                    errors.append(f"{prefix}: invalid candidate indices")
            except (KeyError, ValueError) as exc:
                errors.append(f"{key}: cannot recompute objective gain: {exc}")
    return m, arrays, {"run_id": path.name, "passed": not errors, "errors": errors,
                       "verified_hashes": verified_hashes, "budget": budget,
                       "actual_call_log_totals": actual_calls,
                       "heldout_vs_training_overlap_checked": training_ids is not None,
                       "manifest_sha256": digest(path / "complete.json")}


def bootstrap(groups_by_seed, replicates=2000, seed=70019):
    """Crossed seed/common-center bootstrap, equal seed and distribution weight."""
    keys = sorted(groups_by_seed)
    distributions = sorted(groups_by_seed[keys[0]])
    if any(sorted(groups_by_seed[s]) != distributions for s in keys):
        raise ValueError("Unequal distributions across training seeds")
    means = {s: float(np.mean([np.mean(groups_by_seed[s][d]) for d in distributions])) for s in keys}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for start in range(0, replicates, 100):
        size = min(100, replicates - start)
        chosen = rng.integers(len(keys), size=(size, len(keys)))
        weights = np.stack([(chosen == i).mean(1) for i in range(len(keys))], 1)
        values = np.zeros((size, len(keys)))
        for d in distributions:
            counts = [len(groups_by_seed[s][d]) for s in keys]
            if len(set(counts)) != 1:
                raise ValueError("Common-center bootstrap requires equal matched center counts")
            indices = rng.integers(counts[0], size=(size, counts[0]))
            for i, s in enumerate(keys):
                values[:, i] += np.asarray(groups_by_seed[s][d])[indices].mean(1) / len(distributions)
        draws[start:start + size] = (values * weights).sum(1)
    low, high = np.quantile(draws, [.025, .975])
    return dict(mean_difference=float(np.mean(list(means.values()))), ci_low=float(low), ci_high=float(high),
                seed_count=len(keys), seed_differences=json.dumps(means, sort_keys=True),
                all_seed_signs_agree=bool(all(v > 0 for v in means.values()) or all(v < 0 for v in means.values())),
                bootstrap_replicates=replicates,
                bootstrap_unit="crossed_seed_and_shared_paired_center_within_distribution" if len(keys) > 1 else "paired_center_within_distribution_conditional_on_one_seed")


def calculate_metrics(runs):
    rows = []
    for m, a in runs.values():
        if m["kind"] == "fixed":
            for key, value in a.items():
                if not key.endswith("__gain"):
                    continue
                dist, price, attacker, _ = key.split("__")
                prefix = key[:-6]
                for metric in ("gain", "length", "cost"):
                    v = a[prefix + "__" + metric]
                    rows.append(dict(kind="fixed", method=m["method"], evaluated_attacker=attacker,
                                     seed=m["seed"], distribution=dist, price=price,
                                     metric=metric, mean=float(np.mean(v)), p95=float(np.quantile(v, .95)),
                                     n=len(v), better="larger" if metric == "gain" else "descriptive"))
        elif m["kind"] == "coev":
            by_suite = {}
            for key, value in a.items():
                if not key.endswith("_length"):
                    continue
                dist, metric = key.split("__")
                rows.append(dict(kind="coev", method=m["method"], seed=m["seed"], distribution=dist,
                                 metric=metric, mean=float(np.mean(value)), p95=float(np.quantile(value, .95)),
                                 n=len(value), better="smaller"))
                by_suite.setdefault(metric, []).append(value)
            for metric, groups in by_suite.items():
                rows.append(dict(kind="coev", method=m["method"], seed=m["seed"], distribution="equal_group_aggregate",
                                 metric=metric, mean=float(np.mean([v.mean() for v in groups])),
                                 p95=float(np.quantile(np.concatenate(groups), .95)),
                                 worst_group_mean=float(max(v.mean() for v in groups)),
                                 n=sum(map(len, groups)), better="smaller"))
    return rows


def comparisons(runs, replicates, errors):
    batches = {}
    def add(kind, suite, method, reference, seed, dist, difference):
        key = (kind, suite, method, reference)
        batches.setdefault(key, {}).setdefault(int(seed), {})[dist] = np.asarray(difference)
    fixed = {(m["method"], int(m["seed"])): (m, a) for m, a in runs.values() if m["kind"] == "fixed"}
    for (method, seed), (m, a) in fixed.items():
        for key in a:
            if not key.endswith(f"__{method}__gain"):
                continue
            dist, price, _, _ = key.split("__")
            for ref in ("random", "es", "reinforce"):
                if ref == method:
                    continue
                other_key = f"{dist}__{price}__{ref}__gain"
                other = a if other_key in a else fixed.get((ref, seed), ({}, {}))[1]
                if other_key not in other:
                    if ref != "reinforce":
                        errors.append(f"fixed {method}/s{seed}: missing comparison {other_key}")
                    continue
                centers_a = values_for(a, dist + "__pins", dist + "__centers")
                centers_b = values_for(other, dist + "__pins", dist + "__centers")
                if centers_a is None or centers_b is None or not np.array_equal(centers_a, centers_b):
                    errors.append(f"fixed {method} vs {ref}/s{seed}: center pairing missing or unequal")
                    continue
                own_center = f"{dist}__{price}__{method}__center_length"
                ref_center = f"{dist}__{price}__{ref}__center_length"
                assert_close(a[own_center], other[ref_center], f"fixed shared router centers {method}/{ref}/s{seed}", errors)
                add("fixed", price + "__gain", method, ref, seed, dist, a[key] - other[other_key])
    coev = {(m["method"], int(m["seed"])): (m, a) for m, a in runs.values() if m["kind"] == "coev"}
    methods = sorted({method for method, seed in coev})
    for left, right in itertools.combinations(methods, 2):
        for seed in sorted({seed for method, seed in coev}):
            if (left, seed) not in coev or (right, seed) not in coev:
                continue
            a, b = coev[left, seed][1], coev[right, seed][1]
            for key in a:
                if not key.endswith("_length") or key not in b:
                    continue
                dist, metric = key.split("__")
                pkey = dist + ("__nominal_pins" if metric == "nominal_length" else "__attack_pins")
                if pkey not in a or pkey not in b or not np.array_equal(a[pkey], b[pkey]):
                    errors.append(f"coev {left} vs {right}/s{seed}/{metric}: center pairing missing or unequal")
                    continue
                add("coev", metric, left, right, seed, dist, a[key] - b[key])
    rows = []
    for (kind, suite, method, reference), groups in sorted(batches.items()):
        common = dict(kind=kind, suite=suite, method=method, reference=reference,
                      difference="method-minus-reference", better="positive" if kind == "fixed" else "negative")
        rows.append(dict(common, seed="aggregate", distribution="equal_group_aggregate", **bootstrap(groups, replicates)))
        for seed, values in sorted(groups.items()):
            rows.append(dict(common, seed=seed, distribution="equal_group_aggregate", **bootstrap({seed: values}, replicates)))
            for dist, value in sorted(values.items()):
                rows.append(dict(common, seed=seed, distribution=dist, **bootstrap({seed: {dist: value}}, replicates)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/runs")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts/report")
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.replicates != 2000:
        parser.error("Formal bootstrap count is frozen to 2000")
    out = args.outdir.resolve()
    if not out.is_relative_to(ROOT):
        parser.error("Report outputs must stay in the research branch")
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    required = expected_tasks(protocol)
    training_ids = regenerate_training_ids(protocol)
    errors, audits, runs, missing = [], [], {}, []
    for run_id in required:
        path = args.runs / run_id
        if not (path / "complete.json").is_file():
            missing.append(run_id)
            continue
        try:
            m, a, audit = audit_run(path, protocol, digest(args.protocol), training_ids)
            audits.append(audit)
            if audit["passed"]:
                runs[run_id] = (m, a)
            else:
                errors += [run_id + ": " + e for e in audit["errors"]]
        except Exception as exc:
            errors.append(f"{run_id}: audit exception {type(exc).__name__}: {exc}")
    metrics = calculate_metrics(runs)
    shared_inputs = {}
    for run_id, (manifest, arrays) in runs.items():
        for key, value in arrays.items():
            if key.endswith("_pins"):
                group = (manifest["kind"], key)
                if group in shared_inputs and not np.array_equal(value, shared_inputs[group]):
                    errors.append(f"{run_id}/{key}: common input ordering differs across methods or seeds")
                else:
                    shared_inputs[group] = value
    stats = comparisons(runs, args.replicates, errors)
    # Aggregated inference requires every frozen training seed, even when the
    # means from a partial run would otherwise be easy to display.
    seeds = len(protocol.get("seeds", [201, 202, 203]))
    for row in stats:
        row["full_seed_set"] = row["seed_count"] == seeds if row["seed"] == "aggregate" else "conditional"
    status = "PASS" if not missing and not errors and len(runs) == len(required) else "INCOMPLETE"
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "comparisons.csv", stats)
    audit = {"status": status, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "protocol_sha256": digest(args.protocol), "report_code_sha256": digest(__file__),
             "expected_tasks": required, "missing_tasks": missing, "errors": errors,
             "passed_task_count": len(runs), "expected_task_count": len(required), "runs": audits,
             "regenerated_unique_training_inputs": len(training_ids),
             "bootstrap_replicates": args.replicates,
             "statistics": "paired within center; crossed seed and shared-center resampling within distribution; equal group and seed weighting",
             "limitations": ["Only three training seeds; seed-population inference is fragile.",
                             "95% intervals are exploratory and unadjusted for multiplicity.",
                             "Finite attacks do not prove continuous WDRO guarantees or optimality gaps."]}
    lines = ["# RL 优化方向：独立验收与统计", "", f"**状态：{status}**。已通过 {len(runs)}/{len(required)} 个协议任务。", "",
             "## 统计口径", "",
             "所有差值均为方法减对照。固定模型攻击的 gain 越大越好；共同训练后的自身动作线长越小越好。",
             "先在每个训练 seed 内对相同中心配对，再对训练 seed 和各分布内共同中心做交叉 bootstrap（2000 次）：每次抽取一组 seed，所有 seed 共用同一组抽样中心；各 seed、各分布等权。没有把 seed×net 当作独立训练重复。",
             "仅三个训练 seed，95% 区间不稳定；多方法、多分布比较未做多重比较校正，只作探索性证据。候选攻击不能证明连续 Wasserstein 鲁棒性或最优性差距。", "",
             "metrics.csv 给出逐 seed、逐分布的 mean/P95，以及共同训练的 worst-group mean。comparisons.csv 包含逐 seed 配对区间和 seed 汇总区间；区间针对平均差，P95 与 worst-group 是描述性统计。", "",
             "## 汇总比较", "", "| 阶段 | 指标 | 方法 − 对照 | 均值差 | 95% CI | seeds |", "|---|---|---|---:|---|---:|"]
    for row in stats:
        if row["seed"] == "aggregate":
            lines.append(f"| {row['kind']} | {row['suite']} | {row['method']} − {row['reference']} | {row['mean_difference']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] | {row['seed_count']} |")
    if missing:
        lines += ["", "## 缺失任务", ""] + ["- " + name for name in missing]
    if errors:
        lines += ["", "## 未通过检查", ""] + ["- " + error for error in errors]
    lines += ["", "## 独立检查范围", "", "协议、源码、依赖及产物 SHA256；CUDA 执行；正式更新及调用预算；有限正线长；坐标窗口、pin 间距、位移限制；运输成本和攻击 gain 独立重算；nominal topology 重新计算 REST 线长。", "",
              "actual_call_log_totals 同时保留正式预算、已完成重试额外调用、以及未配对的中断开始事件；未配对请求行数不能当成已完成评分。任一任务缺失、正式预算不符、配对失败或绑定失败均不能标记 DONE。此报告不会启动、恢复或修改任何训练任务。", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    audit["report_artifacts_sha256"] = {name: digest(out / name) for name in ("REPORT.md", "metrics.csv", "comparisons.csv")}
    atomic_json(out / "audit.json", audit)
    print(json.dumps({"status": status, "passed": len(runs), "expected": len(required), "missing": len(missing), "errors": len(errors), "outdir": str(out)}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
