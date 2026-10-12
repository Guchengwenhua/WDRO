"""Independent artifact audit and prespecified paired statistics; no model tuning."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "artifacts"
LEGACY = ROOT.parents[1] / "revisions/adversarial_v3_1_20260912/Codex"
sys.path.insert(0, str(LEGACY))
from adv_v3.common import make_data, atomic_json, sha256
from adv_v3.joint import _array_hash
from search import configurations, FAMILIES

DISTS = ("uniform", "mix_gauss", "edge_bias", "grid_cluster")
SEEDS = (101, 102, 103)
VARIANTS = ("refresh32_raw", "refresh8_raw", "refresh32_ema", "refresh8_ema")
FIXED_METHODS = ("baseline", "no_move", "random_direction", "direction3", "direction4",
                 "direction5", "direction6", "uniform_mse", "region_mse", "region_huber", "region_rankdiff")
GREEDY = ("actor_requested", "actor_completed", "rest_requested", "rest_completed", "reference_requests")
SAMPLED = ("sampled_actor_requested", "sampled_actor_completed", "sampled_rest_requested",
           "sampled_rest_completed", "sampled_reference_reuses")


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=1e-12):
    if not np.allclose(actual, expected, atol=atol, rtol=1e-12):
        raise AssertionError(message)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def state_hash(state, surrogate_order=False):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous().numpy()
        shape_type = (value.dtype.str, value.shape) if surrogate_order else (value.shape, value.dtype.str)
        h.update(key.encode()); h.update(str(shape_type).encode()); h.update(value.tobytes())
    return h.hexdigest()


def canonical_hash(points):
    p = np.asarray(points, dtype=np.float32)
    p = np.ascontiguousarray(p[np.lexsort((p[:, 1], p[:, 0]))])
    return hashlib.sha256(p.tobytes()).hexdigest()


def legal(points):
    points = np.asarray(points)
    require(points.ndim >= 3 and points.shape[-2:] == (30, 2), "degree30 coordinate shape")
    flat = points.reshape(-1, 30, 2)
    require(np.isfinite(flat).all(), "nonfinite geometry")
    require(np.all(flat >= np.float32(.1)) and np.all(flat <= np.float32(.9)), "geometry outside normalized box")
    close(flat.min(1), .1, "noncanonical bounding-box origin", atol=3e-7)
    close(np.ptp(flat, axis=1).max(1), .8, "noncanonical bounding-box span", atol=3e-7)
    # Independent vectorized minimum-L1-spacing check in bounded memory chunks.
    tri = np.triu_indices(30, 1)
    for start in range(0, len(flat), 512):
        p = flat[start:start+512].astype(np.float64)
        distance = np.abs(p[:, tri[0]] - p[:, tri[1]]).sum(-1)
        require(np.all(distance >= 1e-6), "minimum pin spacing violated")


def transport(centers, support):
    result = np.empty(support.shape[:2], dtype=np.float64)
    for i, center in enumerate(centers.astype(np.float64)):
        distances = np.abs(center[None, :, None] - support[i, :, None].astype(np.float64)).sum(-1)
        for j, distance in enumerate(distances):
            rows, cols = linear_sum_assignment(distance)
            result[i, j] = distance[rows, cols].mean()
    return result


def process_matches(launch):
    pid = launch.get("pid")
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    try:
        # comm may contain spaces; split only after its closing parenthesis.
        fields = stat.read_text().rsplit(")", 1)[1].split()
        return fields[19] == str(launch.get("start_ticks", launch.get("process_start_ticks")))
    except (FileNotFoundError, ProcessLookupError):
        return False


def lock_free(path):
    require(Path(path).is_file(), f"missing lock file {path}")
    with Path(path).open("r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def paired_bootstrap(deltas, *, replicates=5000, seed=20260915, family_size=9):
    """Input [models,distributions,shared_centers]; models are NOT resampled.

    Within each distribution, one center-index draw is shared by every model.
    Independent draws across distributions preserve equal distribution weights.
    """
    values = np.asarray(deltas, dtype=np.float64)
    require(values.ndim == 3 and np.isfinite(values).all(), "bootstrap shape/finite values")
    conditional = values.mean(0)
    rng = np.random.default_rng(seed)
    distribution_samples = np.empty((replicates, conditional.shape[0]))
    for d, row in enumerate(conditional):
        indices = rng.integers(len(row), size=(replicates, len(row)))
        distribution_samples[:, d] = row[indices].mean(1)
    macro = distribution_samples.mean(1)
    alpha = .05
    return dict(estimate=float(values.mean()),
                ci95_low=float(np.quantile(macro, alpha/2)),
                ci95_high=float(np.quantile(macro, 1-alpha/2)),
                simultaneous_low=float(np.quantile(macro, alpha/(2*family_size))),
                simultaneous_high=float(np.quantile(macro, 1-alpha/(2*family_size))),
                seed_deltas=values.mean((1, 2)).tolist(),
                distribution_estimate=values.mean((0, 2)).tolist(),
                distribution_ci95_low=np.quantile(distribution_samples, alpha/2, axis=0).tolist(),
                distribution_ci95_high=np.quantile(distribution_samples, 1-alpha/2, axis=0).tolist())


class Audit:
    def __init__(self, root=ROOT, exp=EXP):
        self.root, self.exp = Path(root), Path(exp)
        self.hash_cache, self.artifacts = {}, []
        self.roles = {"train": set(), "validation": set(), "test": set()}
        self.namespace_rows, self.task_rows = [], []
        self.fixed, self.training = {}, {}
        self.fixed_rows, self.training_rows, self.dynamics_rows, self.surrogate_rows = [], [], [], []
        self.validation_scores = {}
        self.smoke_source_drift = []
        self.shared_inputs = {}

    def source_binding(self, binding, smoke=False):
        path = Path(binding["path"])
        if smoke and sha256(path) != binding["sha256"]:
            self.smoke_source_drift.append(dict(path=str(path), frozen_sha256=binding["sha256"],
                                               current_sha256=sha256(path)))
            return
        self.binding(binding)

    def binding(self, binding):
        path = Path(binding.get("path", binding.get("logical_path", "")))
        require(path.is_file(), f"missing bound artifact {path}")
        key = str(path.resolve())
        if key not in self.hash_cache:
            self.hash_cache[key] = sha256(path)
        require(self.hash_cache[key] == binding["sha256"], f"bound artifact hash mismatch {path}")
        self.artifacts.append({"path": key, "sha256": self.hash_cache[key]})
        return path

    def geometry_role(self, role, points, namespace, path):
        flat = np.asarray(points).reshape(-1, 30, 2)
        legal(flat)
        hashes = {canonical_hash(p) for p in flat}
        self.roles[role].update(hashes)
        self.namespace_rows.append(dict(role=role, namespace=namespace, path=str(path),
                                       rows=len(flat), unique_nets=len(hashes), arrays_sha256=_array_hash(flat)))

    def completion(self, out, kind, smoke=False):
        complete = read_json(out / "complete.json")
        identity = read_json(out / "identity.json")
        require(complete["status"] == "pass", f"task did not pass {out}")
        require(complete["identity"] == identity, f"completion identity differs {out}")
        if kind == "training":
            require(bool(identity["smoke"]) == smoke, "smoke/formal identity mismatch")
            require(complete["identity_hash"] == digest(identity), "training identity digest mismatch")
            for binding in identity["dependencies"].values(): self.binding(binding)
            self.binding(identity["theory"]); self.binding(identity["protocol"])
            for name, expected in identity["source_sha256"].items():
                source = LEGACY/name if name.startswith("adv_v3/") else self.root/name
                self.source_binding(dict(path=str(source), sha256=expected), smoke)
        else:
            require(identity["phase"] == ("smoke" if smoke else out.name.split("_s")[0]), "phase identity mismatch")
            for binding in identity["bindings"].values(): self.binding(binding)
            self.binding(dict(path=str(self.root/"protocol.json"), sha256=identity["protocol_sha256"]))
            for name, expected in identity["source"].items():
                self.source_binding(dict(path=str(self.root/name), sha256=expected), smoke)
            if identity["phase"] == "test":
                self.binding(dict(path=str(self.exp/"selection.json"), sha256=identity["selection_sha256"]))
        bound = set()
        for item in complete["artifacts"]:
            bound.add(str(self.binding(item).resolve()))
        require(str((out/"summary.json").resolve()) in bound, "summary not completion-bound")
        if kind == "training":
            excluded = {"complete.json", "heartbeat.json", "launch.json", "exit.json", "failure.json", "paused.json"}
            for path in out.rglob("*"):
                if path.is_file() and path.suffix in (".npz", ".pt", ".json", ".jsonl") and path.name not in excluded:
                    require(str(path.resolve()) in bound, f"unbound scientific training artifact {path}")
        launch, exit_record = read_json(out/"launch.json"), read_json(out/"exit.json")
        require(exit_record.get("complete") is True, f"missing successful exit {out}")
        require(not process_matches(launch), f"task process still alive {out}")
        require(lock_free(out/"task.lock"), f"task writer lock remains held {out}")
        return complete, identity, bound

    def ledger(self, out, committed_greedy, committed_sampled=0):
        ledger = read_json(out/"oracle/oracle_ledger.json")
        for key in GREEDY:
            require(ledger.get(key, 0) >= committed_greedy, f"greedy actual < committed: {out}/{key}")
        for key in SAMPLED:
            require(ledger.get(key, 0) >= committed_sampled, f"sampled actual < committed: {out}/{key}")
        for family in ("actor", "rest", "actor_forward_rows", "actor_padding_rows", "sampled_actor", "sampled_rest"):
            require(ledger.get(family+"_requested", 0) >= ledger.get(family+"_completed", 0), "completed exceeds requested")
        for end in ("requested", "completed"):
            require(ledger.get("actor_forward_rows_"+end, 0) == ledger.get("actor_"+end, 0)
                    + ledger.get("actor_padding_rows_"+end, 0), "forward/padding accounting inconsistent")
        database = out/"oracle/reference.sqlite"
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
            request, started, finished = db.execute(
                "SELECT COALESCE(SUM(requested),0),COALESCE(SUM(misses_started),0),COALESCE(SUM(misses_completed),0) FROM calls").fetchone()
        require((request, started, finished) == tuple(ledger[key] for key in
                    ("reference_requests", "geosteiner_misses_started", "geosteiner_misses_completed")), "SQLite/ledger mismatch")
        return dict(actual_greedy_requested=ledger.get("actor_requested", 0),
                    actual_greedy_completed=ledger.get("actor_completed", 0),
                    committed_greedy=committed_greedy,
                    interrupted_or_replayed_greedy=ledger.get("actor_requested", 0)-committed_greedy,
                    actual_sampled_requested=ledger.get("sampled_actor_requested", 0),
                    committed_sampled=committed_sampled,
                    actor_forward_rows_requested=ledger.get("actor_forward_rows_requested", 0),
                    actor_padding_rows_requested=ledger.get("actor_padding_rows_requested", 0),
                    reference_requests=request, geosteiner_misses_started=started,
                    geosteiner_misses_completed=finished)

    def fixed_task(self, out, smoke=False):
        done, identity, bound = self.completion(out, "fixed", smoke)
        phase, seed, price = identity["phase"], identity["seed"], identity["price"]
        methods = list(configurations()) if phase == "validation" else list(FIXED_METHODS)
        if smoke: methods = ["baseline", "no_move", "random_direction", "differentiable_normalization", "trajectory21_8"]
        dists = ("uniform",) if smoke else DISTS
        count = 2 if smoke else (32 if phase == "validation" else 64)
        summary = read_json(out/"summary.json")
        require(len(summary) == len(methods)*len(dists), "fixed summary coverage")
        for dist in dists:
            inputs = read_npz(out/"inputs"/f"{dist}.npz")["pins"]
            expected = make_data(30, count, dist, "critic-search-20260915-"+phase, seed=0)
            require(np.array_equal(inputs, expected), "fixed namespace data differs")
            self.geometry_role("test" if phase == "test" else "validation", inputs,
                               "critic-search-20260915-"+phase, out/"inputs"/f"{dist}.npz")
            nominal = None
            for method in methods:
                chunks, diags, times = [], [], []
                for offset in range(0, count, 16):
                    path = out/"evaluation"/f"{dist}_{method}_{offset:04d}.npz"
                    require(str(path.resolve()) in bound, "unbound fixed result")
                    a = read_npz(path); b = len(a["centers"])
                    require(np.array_equal(a["centers"], inputs[offset:offset+16]), "fixed center mismatch")
                    require(a["support"].shape == (b, 64, 30, 2), "fixed support shape")
                    legal(a["support"])
                    require(np.array_equal(a["support"][:, 0], a["centers"]), "nominal omitted")
                    require(np.all(a["calls"] == 64), "fixed score budget !=64")
                    close(a["price"], price, "fixed lambda drift")
                    require(np.all(a["all_lstar"] > 0), "nonpositive reference")
                    close(a["all_regret"], a["all_length"]/a["all_lstar"]-1, "real regret mismatch")
                    cost = transport(a["centers"], a["support"])
                    close(a["all_cost"], cost, "Hungarian cost mismatch")
                    close(a["all_score"], a["all_regret"]-price*cost, "all_score mismatch")
                    selected = a["all_score"].argmax(1)
                    require(np.array_equal(selected, a["selected_index"]), "incorrect true-score argmax/tie")
                    for key in ("score", "cost", "regret"):
                        close(a[key], a["all_"+key][np.arange(b), selected], "selected "+key)
                    close(a["original_regret"], a["all_regret"][:, 0], "nominal regret mismatch")
                    close(cost[:, 0], 0, "nominal transport nonzero")
                    require(np.all(a["score"] >= a["original_regret"]-1e-12), "lost nominal best")
                    ledger = json.loads(str(a["ledger_json"]))
                    for key in GREEDY: require(ledger.get(key) == b*64, "fixed chunk ledger budget")
                    chunks.append(a); diags.append(json.loads(str(a["diagnostics_json"])))
                    times.append(float(a["elapsed_seconds"]))
                data = {key: np.concatenate([a[key] for a in chunks]) for key in
                        ("score", "cost", "regret", "original_regret", "selected_index", "centers", "all_score")}
                if nominal is None: nominal = data["original_regret"]
                require(np.array_equal(data["original_regret"], nominal), "nominal scores differ across methods")
                row = next(r for r in summary if r["distribution"] == dist and r["method"] == method)
                close(row["score"], data["score"].mean(), "fixed summary mean mismatch")
                close(row["seconds"], sum(times), "fixed summary time mismatch")
                if phase == "test": self.fixed[seed, dist, method] = data
                if phase == "validation": self.validation_scores[dist, method] = float(data["score"].mean())
                diagnostic = {k: sum(d.get(k, 0) for d in diags) for k in
                              ("forward_samples", "backward_samples", "clipped_coordinates", "total_coordinates", "proposals", "repairs", "fallbacks", "cosine_sum", "cosine_count")}
                self.fixed_rows.append(dict(phase=phase, seed=seed, distribution=dist, method=method,
                    centers=count, mean_score=float(data["score"].mean()), p95_score=float(np.quantile(data["score"], .95)),
                    mean_regret=float(data["regret"].mean()), p95_regret=float(np.quantile(data["regret"], .95)),
                    maximum_regret=float(data["regret"].max()), mean_hungarian_cost=float(data["cost"].mean()),
                    p95_hungarian_cost=float(np.quantile(data["cost"], .95)),
                    mean_score_gain=float((data["score"]-data["original_regret"]).mean()),
                    elapsed_seconds=sum(times), seconds_per_center=sum(times)/count,
                    gradient_forward_rows=diagnostic["forward_samples"], gradient_backward_rows=diagnostic["backward_samples"],
                    clipped_fraction=diagnostic["clipped_coordinates"]/max(1, diagnostic["total_coordinates"]),
                    movement_cosine=diagnostic["cosine_sum"]/max(1, diagnostic["cosine_count"]),
                    repairs=diagnostic["repairs"], fallbacks=diagnostic["fallbacks"],
                    duplicate_fraction=float(np.mean([d["duplicate_fraction"] for d in diags]))))
        labels = self.surrogates(out) if phase == "test" else 0
        budget = self.ledger(out, len(dists)*len(methods)*count*64+labels)
        self.task_rows.append(dict(task=str(out.relative_to(self.exp)), kind="fixed", seed=seed,
            phase=phase, elapsed_seconds=done["elapsed_seconds"], peak_reserved_mib=done["peak_reserved_mib"],
            **budget, exited=True, lock_released=True, bound_artifacts=len(bound)))

    def surrogates(self, out):
        import torch
        directory = out/"surrogates"
        summary = read_json(directory/"summary.json")
        fixed_identity = read_json(out/"identity.json")
        parent = torch.load(fixed_identity["bindings"]["actor"]["logical_path"], map_location="cpu", weights_only=False)
        actor_hash = state_hash(parent["actor_state_dict"], surrogate_order=True)
        del parent
        base = torch.load(fixed_identity["bindings"]["surrogate"]["path"], map_location="cpu", weights_only=False)
        base_hash = state_hash(base["model_state_dict"], surrogate_order=True)
        del base
        require(summary["initial_model_sha256"] == base_hash, "refresh base differs from frozen surrogate")
        total_labels = 0
        for name, role in (("uniform", "train"), ("search_region", "train"), ("shared_validation", "validation")):
            pool = directory/"data"/name; manifest = read_json(pool/"manifest.json")
            self.binding(dict(path=str(pool/"labels.npz"), sha256=manifest["labels_sha256"]))
            self.binding(dict(path=str(pool/"label_requests.jsonl"), sha256=manifest["request_ledger_sha256"]))
            arrays = read_npz(pool/"labels.npz")
            identity = manifest["identity"]
            require(identity["label_actor_sha256"] == actor_hash and identity["initial_model_sha256"] == base_hash,
                    "surrogate labels actor/base binding")
            close(identity["price"], fixed_identity["price"], "surrogate collection price drift")
            require(identity["seed"] == fixed_identity["seed"] and identity["degree"] == 30, "surrogate seed/degree")
            suffix = f"validation/local/d30/s{identity['seed']}" if role == "validation" else f"train/{name}/d30/s{identity['seed']}"
            require(identity["namespace"] == "critic_search_20260915/"+suffix, "surrogate namespace differs from plan")
            close(arrays["regret"], arrays["length"]/arrays["lstar"]-1, "surrogate real label mismatch")
            expected_count = 512 if role == "validation" else 1024
            require(len(arrays["pins"]) == expected_count, "surrogate label count mismatch")
            from surrogate import _hash_arrays
            require(_hash_arrays(arrays) == manifest["arrays_sha256"], "surrogate array hash mismatch")
            self.geometry_role(role, arrays["pins"], manifest["identity"]["namespace"], pool/"labels.npz")
            total_labels += expected_count
        for name, row in summary["variants"].items():
            item = read_json(directory/"models"/name/"summary.json")
            require(row == item and item["status"] == "complete", "surrogate summary identity")
            require(item["identity"]["initial_model_sha256"] == base_hash, "surrogate optimizer initialization binding")
            history = item["history"]
            require(len(history) == 6, "surrogate epochs !=6")
            best = min(history, key=lambda r: r["validation"]["mse"])
            require(best["epoch"] == item["best_epoch"], "surrogate selection not earliest minimum MSE")
            self.binding(dict(path=item["best_path"], sha256=item["best_sha256"]))
            self.binding(dict(path=item["latest_path"], sha256=item["latest_sha256"]))
            self.surrogate_rows.append(dict(seed=summary["seed"], variant=name, best_epoch=item["best_epoch"],
                train_labels=1024, shared_validation_labels=512, train_rows=6144,
                training_seconds=sum(r["training_seconds"] for r in history), **item["validation"]))
        require(total_labels == 2560, "surrogate total label budget")
        return total_labels

    def selection(self):
        selection = read_json(self.exp/"selection.json")
        configs = configurations()
        means = {name: float(np.mean([self.validation_scores[d, name] for d in DISTS])) for name in configs}
        for name, value in means.items(): close(value, selection["validation_macro_scores"][name], "selection score drift")
        expected = {family: max(names, key=lambda n: (means[n], -list(configs).index(n))) for family, names in FAMILIES.items()}
        require(expected == selection["selected"], "validation winner/tie-break mismatch")
        require(selection["source_identity"] == read_json(self.exp/"fixed/validation_s101/identity.json"), "selection source identity")
        for seed in SEEDS:
            for row in read_json(self.exp/"fixed"/f"test_s{seed}"/"summary.json"):
                if row["method"] in expected:
                    require(row["spec"].get("selected_name") == expected[row["method"]], "test uses wrong validation winner")
                    require(row["spec"]["config"] == configs[expected[row["method"]]], "selected config drift")
        self.binding(dict(path=str(self.exp/"selection.json"), sha256=sha256(self.exp/"selection.json")))
        return selection

    def evaluation(self, out, folder, prefix, count, batch, identity_hash, actor_hash, update, role=None):
        result = {}
        for dist in DISTS:
            values = []
            family = "validation" if folder == "validation" else "evaluation"
            inputs = read_npz(out/"inputs"/f"{family}__{dist}.npz")["pins"]
            for offset in range(0, count, batch):
                path = (out/"validation"/f"u{update:06d}"/f"{dist}_{offset:05d}.npz" if folder == "validation"
                        else out/"evaluation"/dist/f"{prefix}_{offset:05d}.npz")
                a = read_npz(path)
                require(str(a["identity_hash"]) == identity_hash and str(a["actor_sha256"]) == actor_hash, "evaluation actor/identity binding")
                require(int(a["update"]) == update, "evaluation checkpoint update")
                require(np.array_equal(a["centers"], inputs[offset:offset+batch]), "training evaluation centers")
                require(np.array_equal(a["pins"], a["centers"]), "nominal evaluation pins changed")
                require(np.all(a["calls"] == 1), "nominal evaluation calls !=1")
                close(a["cost"], 0, "nominal evaluation transport")
                close(a["price"], 0, "nominal evaluation price")
                require(np.all(a["lstar"] > 0), "reference length nonpositive")
                close(a["regret"], a["length"]/a["lstar"]-1, "evaluation real regret")
                for key in ("score", "original_regret", "original_score"): close(a[key], a["regret"], "nominal evaluation "+key)
                for key in GREEDY:
                    require(json.loads(str(a["ledger_delta_json"])).get(key) == len(a["centers"]), "evaluation ledger")
                values.append(a["regret"])
            result[dist] = np.concatenate(values)
        return result

    def training_task(self, out, smoke=False):
        import torch
        done, identity, bound = self.completion(out, "training", smoke)
        summary = read_json(out/"summary.json")
        settings, seed, variant = identity["settings"], identity["seed"], identity["variant"]
        updates, bs = settings["updates"], settings["batch_size"]
        require(updates == (4 if smoke else 128), "incorrect outer update budget")
        latest = torch.load(out/"latest.pt", map_location="cpu", weights_only=False)
        initial = torch.load(out/"initial.pt", map_location="cpu", weights_only=False)
        best = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
        parent = torch.load(identity["dependencies"]["erm"]["path"], map_location="cpu", weights_only=False)
        require(state_hash(initial["actor_state_dict"]) == state_hash(parent["actor_state_dict"]), "initial Actor differs from bound ERM parent")
        del parent
        history = summary["history"]
        require(latest["identity_hash"] == digest(identity), "latest identity hash")
        require(latest["history"] == history and len(history) == updates, "history/checkpoint mismatch")
        require(latest["current"]["update"] == updates, "latest incomplete")
        require(state_hash(best["actor_state_dict"]) == state_hash(latest["best"]["actor_state_dict"]), "best actor differs from committed best")
        require(best["update"] == summary["best_update"], "best update summary mismatch")
        jsonl = [json.loads(line) for line in (out/"training.jsonl").read_text().splitlines()]
        require(jsonl == history, "jsonl history mismatch")
        for path in sorted((out/"inputs").glob("*.npz")):
            a = read_npz(path); spec = json.loads(str(a["spec_json"]))
            require(spec["identity_hash"] == digest(identity), "training input identity")
            family, suffix = path.stem.split("__", 1)
            require(family in ("train", "validation", "evaluation"), "unexpected input family")
            if family == "train":
                step = int(suffix[1:]); require(1 <= step <= updates, "train input step outside budget")
                expected_spec = dict(degree=30, count=bs, distribution="uniform",
                    namespace=identity["data_namespaces"][family].format(step=step), seed=seed,
                    identity_hash=digest(identity))
            else:
                require(suffix in DISTS, "unexpected input distribution")
                expected_spec = dict(degree=30, count=settings["validation_count" if family == "validation" else "evaluation_count"],
                    distribution=suffix, namespace=identity["data_namespaces"][family], seed=0,
                    identity_hash=digest(identity))
            require(spec == expected_spec, "input namespace/seed/count differs from frozen identity")
            expected = make_data(spec["degree"], spec["count"], spec["distribution"], spec["namespace"], spec["seed"])
            require(np.array_equal(a["pins"], expected), "training namespace regeneration mismatch")
            self.shared_inputs[seed, variant, family, suffix] = _array_hash(a["pins"])
            role = "test" if family == "evaluation" else family
            self.geometry_role(role, a["pins"], spec["namespace"], path)
        price, ema, rho = summary["initial_price"], None, summary["rho"]
        best_step, best_score = 0, summary["initial_validation"]["ood_worst_group_mean"]
        selection_rows = [(0, initial["validation"], state_hash(initial["actor_state_dict"]))]
        for t, row in enumerate(history, 1):
            require(row["update"] == t, "nonconsecutive training update")
            require(read_json(out/"updates"/f"u{t:06d}.json") == row, "update log differs")
            a = read_npz(out/"search"/f"u{t:06d}.npz")
            require(str(a["identity_hash"]) == digest(identity), "search identity")
            require(str(a["actor_sha256"]) == row["label_actor_sha256"], "search label actor hash")
            require(_array_hash(a["centers"]) == row["input_sha256"], "training center identity")
            require(np.array_equal(a["centers"], read_npz(out/"inputs"/f"train__u{t:06d}.npz")["pins"]), "search center differs from train input")
            require(a["evaluated_pins"].shape == (bs, 4, 30, 2), "training search budget !=4")
            legal(a["evaluated_pins"])
            self.roles["train"].update(canonical_hash(p) for p in a["evaluated_pins"].reshape(-1, 30, 2))
            require(np.array_equal(a["evaluated_pins"][:, 0], a["centers"]), "training omitted nominal")
            cost = transport(a["centers"], a["evaluated_pins"])
            close(a["price"], price, "training price drift")
            score = a["evaluated_regrets"]-price*cost
            selected = score.argmax(1); rows = np.arange(bs)
            require(np.array_equal(selected, a["selected_index"]), "training true argmax")
            require(np.array_equal(a["selected_pins"], a["evaluated_pins"][rows, selected]), "selected training pins")
            close(a["selected_regret"], a["selected_length"]/a["selected_lstar"]-1, "selected training real regret")
            close(a["selected_regret"], a["evaluated_regrets"][rows, selected], "selected regret")
            close(a["selected_cost"], cost[rows, selected], "selected training Hungarian cost")
            close(a["score"], score[rows, selected], "training score")
            close(a["original_score"], score[:, 0], "training nominal score")
            mean_cost = float(cost[rows, selected].mean())
            close(row["mean_cost"], mean_cost, "training mean cost")
            close(row["rho"], rho, "rho changed")
            close(row["lambda_before"], price, "lambda before")
            ema = mean_cost if ema is None else settings["ema_beta"]*ema+(1-settings["ema_beta"])*mean_cost
            signal = ema if settings["lambda_cost_smoothing"] else mean_cost
            price = max(0., price+settings["lambda_learning_rate"]*(signal-rho))
            close(row["ema_cost"], ema, "EMA recurrence")
            close(row["dual_signal_cost"], signal, "dual cost signal")
            close(row["lambda_after"], price, "lambda projected update")
            for key in GREEDY: require(row["search_ledger_delta"].get(key) == bs*4, "search committed ledger")
            for key in SAMPLED: require(row["sampled"]["ledger_delta"].get(key) == bs, "sampled committed ledger")
            online = row["online"]
            require((online is not None) == (t % settings["critic_interval"] == 0), "refresh schedule")
            if online:
                require(online["label_actor_sha256"] == row["label_actor_sha256"] and online["label_actor_update"] == t-1, "stale online labels")
                require(online["before_actor_update"] is True, "online label timing")
                require(all(online["ledger_delta"].get(k, 0) == 0 for k in GREEDY), "unreported online labeling")
            if row["validation"] is not None:
                checkpoint = torch.load(out/"checkpoints"/f"u{t:06d}.pt", map_location="cpu", weights_only=False)
                selection_rows.append((t, row["validation"], state_hash(checkpoint["current"]["actor_state_dict"])))
                feasible = row["validation"]["nominal_mean"] <= summary["initial_validation"]["nominal_mean"]+.002
                improved = feasible and row["validation"]["ood_worst_group_mean"] < best_score
                require(row["validation_feasible"] == feasible and row["best_improved"] == improved, "validation selection decision")
                if improved: best_step, best_score = t, row["validation"]["ood_worst_group_mean"]
            self.dynamics_rows.append(dict(seed=seed, variant=variant, update=t, mean_cost=mean_cost, rho=rho,
                cost_minus_rho=mean_cost-rho, ema_cost=ema, dual_signal_cost=signal,
                lambda_before=row["lambda_before"], lambda_after=price, surrogate_mse=row["surrogate_error"]["mse"],
                calibration_performed=online is not None, search_seconds=row["search_seconds"],
                calibration_seconds=row["calibration_seconds"], policy_update_seconds=row["policy_update_seconds"]))
        require(best_step == best["update"], "best checkpoint selection mismatch")
        require(state_hash(best["actor_state_dict"]) == next(h for t, _, h in selection_rows if t == best_step), "best actor differs from selected validation checkpoint")
        close(summary["final_price"], price, "final lambda")
        close(latest["current"]["price"], price, "checkpoint final lambda")
        close(latest["current"]["ema_cost"], ema, "checkpoint final EMA cost")
        for step, target, actor_hash in selection_rows:
            values = self.evaluation(out, "validation", "validation", settings["validation_count"],
                settings["evaluation_batch_size"], digest(identity), actor_hash, step)
            self.check_eval_summary(values, target)
        for label, prefix, state, target in (("initial", "initial", initial, summary["initial_evaluation"]),
                ("best", "final", best, summary["final_evaluation"]),
                ("latest", "last", latest["current"], summary["latest_evaluation"])):
            values = self.evaluation(out, "evaluation", prefix, settings["evaluation_count"],
                settings["evaluation_batch_size"], digest(identity), state_hash(state["actor_state_dict"]), state["update"])
            self.check_eval_summary(values, target)
            for dist, regrets in values.items():
                self.training[seed, variant, label, dist] = regrets
                self.training_rows.append(dict(seed=seed, variant=variant, checkpoint=label, update=state["update"],
                    distribution=dist, count=len(regrets), mean_regret=float(regrets.mean()),
                    p95_regret=float(np.quantile(regrets, .95)), maximum_regret=float(regrets.max())))
        committed = updates*bs*4 + len(selection_rows)*len(DISTS)*settings["validation_count"] + 3*len(DISTS)*settings["evaluation_count"]
        budget = self.ledger(out, committed, updates*bs)
        self.task_rows.append(dict(task=str(out.relative_to(self.exp)), kind="training", seed=seed,
            phase=variant, elapsed_seconds=summary["task_seconds_this_attempt"],
            peak_reserved_mib=summary["peak_reserved_mib"], peak_allocated_mib=summary["peak_allocated_mib"],
            best_update=best["update"], **budget, exited=True, lock_released=True, bound_artifacts=len(bound),
            **summary["dynamics"]))

    @staticmethod
    def check_eval_summary(values, target):
        for dist, value in values.items():
            require(target[dist]["count"] == len(value), "evaluation count")
            close(target[dist]["mean"], value.mean(), "evaluation mean")
            close(target[dist]["p95"], np.quantile(value, .95), "evaluation p95")
            close(target[dist]["maximum"], value.max(), "evaluation maximum")
        close(target["nominal_mean"], values["uniform"].mean(), "nominal summary")
        close(target["ood_worst_group_mean"], max(values[d].mean() for d in DISTS[1:]), "OOD worst group")

    def no_leakage(self):
        overlap = {}
        names = list(self.roles)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                key = left+"__"+right
                overlap[key] = len(self.roles[left] & self.roles[right])
                require(overlap[key] == 0, f"actual canonical pin-hash leakage: {key}")
        return dict(unique_canonical_nets={k: len(v) for k, v in self.roles.items()}, intersections=overlap)


def contrasts(audit):
    main = [("D1", "region_mse", "uniform_mse"), ("D2", "region_rankdiff", "region_mse")]
    main += [(f"D{d}", f"direction{d}", "baseline") for d in range(3, 7)]
    secondary = [("D2_huber", "region_huber", "region_mse"),
                 ("no_move", "no_move", "baseline"), ("random_direction", "random_direction", "baseline")]
    rows, distributions, seed_rows = [], [], []
    for label, left, right in main+secondary:
        deltas = np.stack([np.stack([audit.fixed[s, d, left]["score"]-audit.fixed[s, d, right]["score"] for d in DISTS]) for s in SEEDS])
        add_contrast(rows, distributions, seed_rows, label, left, right, deltas, "higher", label in {x[0] for x in main})
    for variant in VARIANTS[1:]:
        delta = np.stack([np.stack([audit.training[s, variant, "best", d]-audit.training[s, VARIANTS[0], "best", d] for d in DISTS]) for s in SEEDS])
        add_contrast(rows, distributions, seed_rows, "D7_"+variant, variant, VARIANTS[0], delta, "lower", True)
    require(sum(r["primary"] for r in rows) == 9, "nine prespecified primary comparisons required")
    return rows, distributions, seed_rows


def add_contrast(rows, distributions, seed_rows, label, left, right, deltas, better, primary):
    result = paired_bootstrap(deltas)
    sign = 1 if better == "higher" else -1
    seed_values = np.asarray(result["seed_deltas"])
    all_better = bool(np.all(sign*seed_values > 0))
    sim_better = (result["simultaneous_low"] > 0 if sign > 0 else result["simultaneous_high"] < 0)
    sim_worse = (result["simultaneous_high"] < 0 if sign > 0 else result["simultaneous_low"] > 0)
    conclusion = "simultaneous_supported_improvement" if primary and sim_better and all_better else (
                 "simultaneous_supported_degradation" if primary and sim_worse and np.all(sign*seed_values < 0) else "inconclusive")
    rows.append(dict(contrast=label, treatment=left, control=right, metric="macro_penalized_score" if sign > 0 else "macro_regret_best_checkpoint",
        better=better, primary=primary, mean_delta=result["estimate"], ci95_low=result["ci95_low"], ci95_high=result["ci95_high"],
        simultaneous_low=result["simultaneous_low"] if primary else None,
        simultaneous_high=result["simultaneous_high"] if primary else None,
        all_seed_improve=all_better, conclusion=conclusion))
    for i, dist in enumerate(DISTS):
        distributions.append(dict(contrast=label, distribution=dist, better=better,
            mean_delta=result["distribution_estimate"][i], ci95_low=result["distribution_ci95_low"][i],
            ci95_high=result["distribution_ci95_high"][i], simultaneous_correction=False))
    for seed, value in zip(SEEDS, seed_values):
        seed_rows.append(dict(contrast=label, seed=seed, mean_delta=float(value), better=better, improves=bool(sign*value > 0)))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name+".partial")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def trajectory_diagnostics(audit):
    summaries, centers = [], []
    for control in ("no_move", "random_direction"):
        # Both arrays retain candidate order. Only the 63 paired trajectories,
        # excluding nominal candidate 0, enter center-level aggregation.
        differences = np.stack([np.stack([
            audit.fixed[s, d, "baseline"]["all_score"][:, 1:]-audit.fixed[s, d, control]["all_score"][:, 1:]
            for d in DISTS]) for s in SEEDS])
        best_differences = np.stack([np.stack([
            audit.fixed[s, d, "baseline"]["score"]-audit.fixed[s, d, control]["score"]
            for d in DISTS]) for s in SEEDS])
        per_center_mean = differences.mean(-1)
        per_center_positive = (differences > 0).mean(-1)
        for metric, values in (("mean_trajectory_score_delta", per_center_mean),
                               ("fraction_positive_trajectories", per_center_positive),
                               ("best64_score_delta", best_differences)):
            stat = paired_bootstrap(values)
            summaries.append(dict(treatment="baseline", control=control, metric=metric,
                estimate=stat["estimate"], ci95_low=stat["ci95_low"], ci95_high=stat["ci95_high"],
                seed101=stat["seed_deltas"][0], seed102=stat["seed_deltas"][1], seed103=stat["seed_deltas"][2],
                statistical_unit="center_after_63_trajectory_aggregation", simultaneous_correction=False))
        for i, seed in enumerate(SEEDS):
            for j, dist in enumerate(DISTS):
                for k in range(differences.shape[2]):
                    centers.append(dict(seed=seed, distribution=dist, center_index=k, control=control,
                        trajectory_count=63, mean_trajectory_score_delta=float(per_center_mean[i, j, k]),
                        fraction_positive_trajectories=float(per_center_positive[i, j, k]),
                        best64_score_delta=float(best_differences[i, j, k])))
    return summaries, centers


def figures(tables, destination):
    """Scientific figures derived exclusively from the just-published CSV tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    destination.mkdir(parents=True, exist_ok=True)
    def load(name):
        with (tables/(name+".csv")).open() as stream: return list(csv.DictReader(stream))
    outputs = []
    def save(fig, name):
        for extension in ("png", "pdf"):
            path = destination/(name+"."+extension)
            fig.savefig(path, bbox_inches="tight"); outputs.append(path)
        plt.close(fig)
    rows = load("contrasts")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, subset, xlabel in ((axes[0], [r for r in rows if not r["contrast"].startswith("D7_")], "Delta true penalized score (higher is better)"),
                              (axes[1], [r for r in rows if r["contrast"].startswith("D7_")], "Delta test regret (lower is better)")):
        labels = []
        for i, row in enumerate(subset):
            primary = row["primary"] == "True"
            low = float(row["simultaneous_low"] if primary else row["ci95_low"])
            high = float(row["simultaneous_high"] if primary else row["ci95_high"])
            mean = float(row["mean_delta"])
            color = "#185c91" if primary else "#b05b16"
            ax.plot([low, high], [i, i], color=color, linewidth=2)
            ax.scatter([mean], [i], color=color, s=25, zorder=3)
            labels.append(row["contrast"].replace("D7_", ""))
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis(); ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_title("Fixed actors: budget 64 / center")
    axes[1].set_title("Continued training: 128 updates; inner budget 4")
    fig.suptitle("Degree 30; 3 fixed seeds; 4-distribution macro paired differences\nBlue: Bonferroni 9-family intervals; orange: pointwise 95% intervals", fontsize=11)
    fig.tight_layout(); save(fig, "paired_effects")
    dynamics = load("training_dynamics")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    colors = dict(zip(VARIANTS, ("#185c91", "#cd742a", "#3d8c5c", "#ab4996")))
    for i, seed in enumerate(SEEDS):
        for variant in VARIANTS:
            data = sorted([r for r in dynamics if int(r["seed"]) == seed and r["variant"] == variant], key=lambda r:int(r["update"]))
            step = [int(r["update"]) for r in data]
            axes[i, 0].plot(step, [float(r["lambda_after"]) for r in data], color=colors[variant], label=variant)
            axes[i, 1].plot(step, [float(r["mean_cost"]) for r in data], color=colors[variant], alpha=.8, label=variant)
        rho = float(data[0]["rho"])
        axes[i, 1].axhline(rho, color="black", linestyle="--", label="rho")
        axes[i, 0].set_ylabel(f"seed {seed}: lambda")
        axes[i, 1].set_ylabel(f"seed {seed}: raw mean cost")
        for ax in axes[i]: ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel("Outer update")
    fig.suptitle("Direction 7: degree 30; 3 seeds; 128 updates; true inner budget 4 / center\nRaw transported cost is shown, even for EMA dual updates", fontsize=11)
    fig.tight_layout(); save(fig, "dual_and_raw_cost")
    metrics = load("training_metrics")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(VARIANTS)); width = .24
    for j, checkpoint in enumerate(("initial", "best", "latest")):
        for ax, field, title in ((axes[0], "mean_regret", "Macro mean regret"),
                                 (axes[1], "p95_regret", "Mean of per-distribution P95 regret")):
            values = [np.mean([float(r[field]) for r in metrics if r["variant"] == variant and r["checkpoint"] == checkpoint]) for variant in VARIANTS]
            ax.bar(x+(j-1)*width, values, width=width, label=checkpoint)
            ax.set_title(title); ax.set_xticks(x, [v.replace("_", "\n") for v in VARIANTS]); ax.grid(axis="y", alpha=.2)
            ax.set_ylabel("Regret (lower is better)")
    axes[0].legend()
    fig.suptitle("Degree 30; 3 seeds; 4 distributions; 256 independent test nets / distribution\nInitial, validation-selected best, and update-128 latest", fontsize=11)
    fig.tight_layout(); save(fig, "initial_best_latest")
    return outputs


def markdown_report(audit, rows, selection, complete, trajectories):
    translate = {"simultaneous_supported_improvement": "支持改善", "simultaneous_supported_degradation": "支持退化", "inconclusive": "证据不足"}
    lines = ["# tmp2 七方向实验结果与独立验收", "", "## 验收范围", "",
        "16 个正式任务完成：1 个搜索 validation、3 个固定模型 test、12 个继续训练任务。所有结果来自 degree30、ERM 父模型 seed 101/102/103。",
        "独立复核产物 SHA256、来源绑定、真实 regret/运输成本/候选 argmax、预算、λ/EMA、选模、实际数据 hash 不交叠及任务进程退出和锁释放。", "",
        "这是有限预算机制探索。测试攻击每中心64次真实评分；外层训练每分支128次更新、每中心4次内层评分。结果不提供连续空间或普适鲁棒保证。", "",
        "## 预先固定的主要比较", "",
        "差值均为实验变体减对照；D1–D6 的真实惩罚分数越高越好，D7 的 validation 选中模型 test regret 越低越好。四分布等权宏平均。", "",
        "| 对比 | 宏均值差 | 点态95% CI | 9项同时区间 | 三seed均改善 | 判断 |",
        "|---|---:|---|---|---|---|"]
    for row in rows:
        if not row["primary"]: continue
        lines.append(f"| {row['contrast']} | {row['mean_delta']:.7g} | [{row['ci95_low']:.7g}, {row['ci95_high']:.7g}] | [{row['simultaneous_low']:.7g}, {row['simultaneous_high']:.7g}] | {'是' if row['all_seed_improve'] else '否'} | {translate[row['conclusion']]} |")
    lines += ["", "D1 比较 region_mse−uniform_mse；D2 比较 region_rankdiff−region_mse；D3–D6 比较各 validation winner−baseline；D7 比较三种刷新/EMA变体−refresh32_raw。", "",
        "置信区间使用5,000次 paired bootstrap：条件于已固定的3个模型，在每个分布内重采样中心索引，同一次抽样对三个seed与成对方法共同使用。四分布独立抽样后等权平均。",
        "同时区间对9个主要宏平均对比采用 Bonferroni α=0.05/9；分布细表及次要比较仅给点态95%区间。极端分位数受5,000次Monte Carlo重采样精度限制。",
        "只有同时区间支持改善且三个seed均同向才标记“支持改善”。区间包含0不证明等价；这些区间不覆盖重新训练模型的随机性，3个seed不足以支撑普适训练结论。", "",
        "## 各研究方向", ""]
    narratives = {
        "D1": "相同新增1,024真实标签与6 epochs下，比较搜索区域数据对uniform刷新；区域标签采集额外使用梯度计算。",
        "D2": "相同搜索区域标签和初始参数下比较MSE与保留回归尺度的局部排序/差分目标；Huber为预声明次要对比。",
        "D3": "步长、步数及更新器由独立validation选择；计算行数与耗时在fixed_metrics.csv中单列。",
        "D4": "起点尺度/结构及广度深度配置由独立validation选择，真实评分预算保持64。",
        "D5": "终点与轨迹位置分配按validation选择；起点覆盖与轨迹观察之间的折中由真实评分检验。",
        "D6": "合法化、可微归一化或去整体平移/缩放方向按validation选择，运输成本定义未变。"}
    for key, description in narratives.items():
        row = next(r for r in rows if r["contrast"] == key)
        winner = f"选中配置 `{selection['selected']['direction'+key[1:]]}`。" if int(key[1:]) >= 3 else ""
        lines += [f"### {key}", "", description+winner,
                  f"宏均值差 {row['mean_delta']:.7g}；同时区间 [{row['simultaneous_low']:.7g}, {row['simultaneous_high']:.7g}]；判断：{translate[row['conclusion']]}。", ""]
    lines += ["### D7", "", "每个seed的四个分支共享初始模型与训练中心，ρ保持不变。初始、选中和第128步模型均在training_metrics.csv中报告；预算与代理误差逐步曲线在training_dynamics.csv中。", "",
        "| 分支−refresh32_raw | 最佳模型宏regret差 | 同时区间 | 判断 |", "|---|---:|---|---|"]
    for row in rows:
        if row["contrast"].startswith("D7_"):
            lines.append(f"| {row['treatment']} | {row['mean_delta']:.7g} | [{row['simultaneous_low']:.7g}, {row['simultaneous_high']:.7g}] | {translate[row['conclusion']]} |")
    lines += ["", "## 梯度方向诊断与次要比较", "", "| 对比−对照 | 宏分数差 | 点态95% CI |", "|---|---:|---|"]
    for row in rows:
        if not row["primary"]:
            lines.append(f"| {row['treatment']}−{row['control']} | {row['mean_delta']:.7g} | [{row['ci95_low']:.7g}, {row['ci95_high']:.7g}] |")
    lines += ["", "逐轨迹诊断保留63个起点的候选索引配对，先在每个中心内计算平均分数差/正向比例，再以中心为单位bootstrap；不将63条轨迹当成63个独立net。", "",
        "| 梯度baseline−对照 | 统计量 | 宏均值 | 点态95% CI |", "|---|---|---:|---|"]
    for r in trajectories:
        lines.append(f"| {r['control']} | {r['metric']} | {r['estimate']:.7g} | [{r['ci95_low']:.7g}, {r['ci95_high']:.7g}] |")
    lines += ["", "best-of64改善与平均轨迹改善可以不同，前者不能单独证明多数梯度方向有效。这些诊断不能用作重新调参或替换主要比较。no_move保留同尺度起点；random_direction替换移动方向但代码仍执行相同梯度前后向，因此其耗时不是无梯度优化器的效率估计。", "",
        "## 资源与实际预算", "", f"累计任务耗时（并行任务相加）{sum(r['elapsed_seconds'] for r in audit.task_rows):.1f}秒；最大Torch reserved显存 {max(r['peak_reserved_mib'] for r in audit.task_rows):.1f} MiB。",
        f"实际greedy请求 {sum(r['actual_greedy_requested'] for r in audit.task_rows):,}；已提交greedy预算 {sum(r['committed_greedy'] for r in audit.task_rows):,}；sampled请求 {sum(r['actual_sampled_requested'] for r in audit.task_rows):,}。",
        f"实际GeoSteiner miss完成 {sum(r['geosteiner_misses_completed'] for r in audit.task_rows):,}；padding行 {sum(r['actor_padding_rows_requested'] for r in audit.task_rows):,}。完整表含每任务耗时、显存、额外重做、搜索与校准开销。",
        "任务耗时字段是最终运行尝试；若恢复过，历史尝试开销需结合operations日志，不能将该字段视为完整端到端墙钟时间。GPU峰值是本进程Torch分配器记录，不等于整张显卡峰值。", "",
        "## 产物", "", "- `artifacts/tables/fixed_metrics.csv`：每seed/分布/方法的mean、P95、原始regret、Hungarian成本、计算行数与延迟。",
        "- `artifacts/tables/contrasts.csv`、`contrast_distributions.csv`、`contrast_seeds.csv`：主要/次要宏比较、分布点态区间和seed方向。",
        "- `artifacts/tables/training_metrics.csv`、`training_dynamics.csv`：initial/best/latest评价、λ、真实成本与代理误差。",
        "- `artifacts/tables/surrogate_metrics.csv`、`tasks.csv`、`namespaces.csv`：校准选模、预算资源及实际数据hash。",
        "- `artifacts/tables/trajectory_diagnostics.csv`、`trajectory_centers.csv`：按中心聚合的逐轨迹方向诊断。",
        "- `artifacts/figures/`：宏配对效果、逐seed λ/原始cost轨迹、initial/best/latest结果；每图含PNG和PDF。",
        "- `artifacts/audit/completion_audit.json`、`artifacts/summary.json`：独立验收及可机器读取结论。", ""]
    return "\n".join(lines)


def expected_tasks(exp):
    return [(exp/"fixed/validation_s101", "fixed")]+[(exp/"fixed"/f"test_s{s}", "fixed") for s in SEEDS]+[
        (exp/"training"/f"{v}_d30_s{s}", "training") for s in SEEDS for v in VARIANTS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="publish only task inventory, never completion")
    parser.add_argument("--smoke", action="store_true", help="audit separate smoke artifacts without publishing formal results")
    args = parser.parse_args(argv)
    EXP.mkdir(parents=True, exist_ok=True)
    tasks = expected_tasks(EXP)
    missing = [str(path) for path, _ in tasks if not (path/"complete.json").is_file()]
    if args.partial:
        atomic_json(EXP/"audit/partial_status.json", dict(status="incomplete" if missing else "ready_for_audit",
            expected_tasks=16, completed_files=16-len(missing), missing=missing, at=time.time(), performance_not_inspected=True))
        print(json.dumps(dict(status="partial_only", completed_files=16-len(missing))))
        return 0
    if args.smoke:
        tasks = [(EXP/"fixed/smoke_s101", "fixed"), (EXP/"training_smoke/refresh32_raw_d30_s101", "training")]
    elif missing:
        raise RuntimeError(f"formal audit requires all 16 completed tasks; missing {len(missing)}")
    audit = Audit(); started = time.monotonic()
    if not args.smoke:
        for binding in read_json(EXP/"freeze.json")["files"]: audit.binding(binding)
    for out, kind in tasks:
        print(json.dumps(dict(audit_task=str(out), kind=kind)), flush=True)
        (audit.fixed_task if kind == "fixed" else audit.training_task)(out, smoke=args.smoke)
    leakage = audit.no_leakage()
    if args.smoke:
        atomic_json(EXP/"audit/smoke_audit.json", dict(status="smoke_numerical_pass", formal_completion=False,
            tasks=audit.task_rows, leakage=leakage, seconds=time.monotonic()-started,
            historical_smoke_source_drift=audit.smoke_source_drift,
            report_source_sha256=sha256(__file__)))
        print("SMOKE AUDIT PASS; formal results not published")
        return 0
    selection = audit.selection()
    # Verify model pairing and repeated initial evaluations before bootstrapping.
    for dist in DISTS:
        for method in FIXED_METHODS:
            for seed in SEEDS[1:]:
                require(np.array_equal(audit.fixed[101, dist, method]["centers"], audit.fixed[seed, dist, method]["centers"]), "fixed cross-seed centers not shared")
        for seed in SEEDS:
            for variant in VARIANTS[1:]:
                require(np.array_equal(audit.training[seed, VARIANTS[0], "initial", dist], audit.training[seed, variant, "initial", dist]), "training initial parent evaluation differs")
        for family in ("validation", "evaluation"):
            expected_hash = audit.shared_inputs[101, VARIANTS[0], family, dist]
            require(all(audit.shared_inputs[s, v, family, dist] == expected_hash for s in SEEDS for v in VARIANTS), "training validation/test geometry not shared across seeds/variants")
    for seed in SEEDS:
        for step in range(1, 129):
            expected_hash = audit.shared_inputs[seed, VARIANTS[0], "train", f"u{step:06d}"]
            require(all(audit.shared_inputs[seed, v, "train", f"u{step:06d}"] == expected_hash for v in VARIANTS), "training centers differ across same-seed variants")
    rows, distribution_rows, seed_rows = contrasts(audit)
    trajectory_rows, trajectory_centers = trajectory_diagnostics(audit)
    queue = read_json(EXP/"operations/queue.json")
    queue_exit = read_json(EXP/"operations/exit.json")
    require(not queue["running"] and not queue["pending"] and not queue["failures"], "queue not successful/drained")
    require(not queue_exit["pending"] and not queue_exit["failures"], "scheduler exit record unsuccessful")
    # Scheduler invokes this report while owning queue.lock; worker locks are released.
    operations = dict(queue_drained=True, task_processes_exited=True, task_locks_released=True,
        scheduler_process_active=process_matches(read_json(EXP/"operations/launch.json")),
        scheduler_lock_released=lock_free(EXP/"operations/queue.lock"),
        scheduler_report_parent_may_still_own_queue_lock=True)
    tables = {"fixed_metrics": audit.fixed_rows, "training_metrics": audit.training_rows,
        "training_dynamics": audit.dynamics_rows, "surrogate_metrics": audit.surrogate_rows,
        "tasks": audit.task_rows, "namespaces": audit.namespace_rows,
        "contrasts": rows, "contrast_distributions": distribution_rows, "contrast_seeds": seed_rows,
        "trajectory_diagnostics": trajectory_rows, "trajectory_centers": trajectory_centers}
    for name, values in tables.items(): csv_write(EXP/"tables"/(name+".csv"), values)
    plot_outputs = figures(EXP/"tables", EXP/"figures")
    summary = dict(status="pass", research_scope="degree30_fixed_three_parents_conditional_exploratory",
        tasks=16, selection=selection, primary_family_size=9, bootstrap_replicates=5000,
        statistical_unit="paired shared centers within distribution, conditional on three fixed model seeds",
        contrasts=rows, leakage=leakage, operations=operations,
        total_actual_greedy=sum(r["actual_greedy_requested"] for r in audit.task_rows),
        total_committed_greedy=sum(r["committed_greedy"] for r in audit.task_rows),
        total_sampled=sum(r["actual_sampled_requested"] for r in audit.task_rows),
        audit_seconds=time.monotonic()-started, report_source_sha256=sha256(__file__))
    atomic_json(EXP/"summary.json", summary)
    result_path = ROOT/"RESULTS.md"
    temporary = result_path.with_name(result_path.name+".partial")
    temporary.write_text(markdown_report(audit, rows, selection, summary, trajectory_rows)); temporary.replace(result_path)
    outputs = [EXP/"summary.json", result_path]+list((EXP/"tables").glob("*.csv"))+plot_outputs
    unique_bindings = {r["path"]: r for r in audit.artifacts}
    completion = dict(status="pass", formal_tasks=16, at=time.time(), leakage=leakage,
        operations=operations, report_source_sha256=sha256(__file__),
        audited_artifacts=list(unique_bindings.values()),
        published_outputs=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in outputs],
        acceptance="all numerical tasks and independent audit passed; scheduler lifecycle recorded separately")
    atomic_json(EXP/"audit/completion_audit.json", completion)
    print(json.dumps(dict(status="pass", tasks=16, results=str(result_path), audit_seconds=summary["audit_seconds"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
