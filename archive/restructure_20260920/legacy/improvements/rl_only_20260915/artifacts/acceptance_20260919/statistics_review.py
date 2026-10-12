"""Independent CPU-only numerical review; never imports training/report/model code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

SCRIPT = Path(__file__).resolve()
DEFAULT_OUT = SCRIPT.parent
ROOT = DEFAULT_OUT.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outdir", default=str(DEFAULT_OUT))
parser.add_argument("--require-complete", action="store_true",
                    help="refuse to begin final statistics unless every formal run is complete")
args = parser.parse_args()
OUT = Path(args.outdir).resolve()
if not OUT.is_relative_to(ROOT):
    parser.error("--outdir must remain inside the RL-only branch")
if OUT == DEFAULT_OUT and (OUT / "statistics_review.json").exists():
    parser.error("preserved partial snapshot already exists; select a new --outdir")
P = json.loads((ROOT / "protocol.json").read_text())
METHODS, SEEDS, GROUPS = P["methods"], P["seeds"], P["distributions"]
if args.require_complete:
    unfinished = [f"{m}_s{s}" for m in METHODS for s in SEEDS
                  if not (ROOT / "artifacts/runs" / f"{m}_s{s}" / "complete.json").is_file()]
    if unfinished:
        print(json.dumps({"status": "PARTIAL", "required_runs": len(METHODS) * len(SEEDS),
                          "missing": unfinished, "error": "--require-complete gate refused"}))
        raise SystemExit(2)
OUT.mkdir(parents=True, exist_ok=True)
errors, bindings, records, arrays = [], {}, {}, {}
groups, attacks, run_rows, missing = [], [], [], []

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(condition, message):
    if not condition: errors.append(message)

def own_length(pins, topology):
    """Independent edge-by-edge accumulation of the REST terminal segment cost."""
    x = np.asarray(pins, dtype=np.float64)
    t = np.asarray(topology)
    rows = np.arange(len(x))
    min_x, max_x = x[:, :, 0].copy(), x[:, :, 0].copy()
    min_y, max_y = x[:, :, 1].copy(), x[:, :, 1].copy()
    for edge in range(x.shape[1] - 1):
        u, v = t[:, 2 * edge], t[:, 2 * edge + 1]
        yy, xx = x[rows, v, 1], x[rows, u, 0]
        min_y[rows, u] = np.minimum(min_y[rows, u], yy)
        max_y[rows, u] = np.maximum(max_y[rows, u], yy)
        min_x[rows, v] = np.minimum(min_x[rows, v], xx)
        max_x[rows, v] = np.maximum(max_x[rows, v], xx)
    return ((max_x - min_x) + (max_y - min_y)).sum(1)

def ci(delta, seed=20260916):
    # [training_seed, group, net]; resample seeds and paired nets within each group.
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim == 2: d = d[:, None, :]
    rng = np.random.default_rng(seed)
    values = np.empty(2000)
    for i in range(len(values)):
        s = rng.integers(len(d), size=len(d))
        values[i] = np.mean([d[s][:, g, rng.integers(d.shape[2], size=d.shape[2])].mean()
                             for g in range(d.shape[1])])
    lo, hi = np.quantile(values, [.025, .975])
    sd = d.mean((1, 2))
    return dict(mean_difference=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
                seed_differences=sd.tolist(), all_seed_negative=bool(np.all(sd < 0)),
                all_seed_positive=bool(np.all(sd > 0)))

snapshot = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for method in METHODS:
    for seed in SEEDS:
        name = f"{method}_s{seed}"
        folder = ROOT / "artifacts/runs" / name
        complete = folder / "complete.json"
        if not complete.exists():
            hb = json.loads((folder / "heartbeat.json").read_text()) if (folder / "heartbeat.json").exists() else {}
            missing.append(dict(run=name, heartbeat=hb, final_results_present=(folder / "results.npz").exists()))
            continue
        c = json.loads(complete.read_text())
        with np.load(folder / "results.npz", allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        records[method, seed], arrays[method, seed] = c, a
        bindings[name] = {file: digest(folder / file) for file in ("complete.json", "results.npz", "progress.jsonl")}
        events = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]
        updates = [row for row in events if row["type"] == "update"]
        check([row["step"] for row in updates] == list(range(1, P["updates"] + 1)), name + ": update sequence")
        initial_means, final_means = [], []
        for group in GROUPS:
            old, new = a[f"initial__{group}__length"], a[f"final__{group}__length"]
            initial_means.append(float(old.mean())); final_means.append(float(new.mean()))
            for stage in ("initial", "final"):
                prefix = f"{stage}__{group}__"
                length, pins, topology = a[prefix + "length"], a[prefix + "pins"], a[prefix + "topology"]
                check(len(length) == P["test_per_distribution"], name + ": test sample count")
                check(np.isfinite(length).all() and np.all(length > 0), name + ": positive finite length")
                check(np.allclose(own_length(pins, topology), length, rtol=0, atol=1e-12), name + ": independent length")
                groups.append(dict(method=method, seed=seed, distribution=group, stage=stage,
                                   count=len(length), mean=float(length.mean()), p95=float(np.quantile(length, .95))))
                check(abs(c[stage + "_groups"][group]["mean_length"] - length.mean()) < 1e-12, name + ": summary mean mismatch")
                check(abs(c[stage + "_groups"][group]["p95_length"] - np.quantile(length, .95)) < 1e-12, name + ": summary p95 mismatch")
            check(np.array_equal(a[f"initial__{group}__pins"], a[f"final__{group}__pins"]), name + ": paired coordinates")
            prefix = f"attack__{group}__"
            pins, length, topology = a[prefix + "pins"], a[prefix + "length"], a[prefix + "topology"]
            n, k = length.shape
            check((n, k) == (P["attack_per_distribution"], P["attack_candidates"]), name + ": attack count")
            check(np.allclose(own_length(pins.reshape(-1, P["degree"], 2), topology.reshape(-1, 2 * (P["degree"] - 1))).reshape(n, k),
                              length, rtol=0, atol=1e-12), name + ": attack independent lengths")
            direct = np.abs(pins.astype(np.float64) - pins[:, :1].astype(np.float64)).sum(-1).mean(-1)
            check(np.allclose(a[prefix + "cost"], direct, rtol=0, atol=1e-12), name + ": attack direct costs")
            index = np.argmax(length, axis=1)
            check(np.array_equal(index, a[prefix + "selected_index"]), name + ": attack selection")
            chosen, chosen_cost = length[np.arange(n), index], direct[np.arange(n), index]
            within = np.max(np.where(direct <= P["rho"], length, -np.inf), axis=1)
            attacks.append(dict(method=method, seed=seed, distribution=group, centers=n, candidates=k,
                mean_center=float(length[:, 0].mean()), mean_attacked=float(chosen.mean()),
                p95_attacked=float(np.quantile(chosen, .95)), mean_lift=float((chosen - length[:, 0]).mean()),
                mean_selected_cost=float(chosen_cost.mean()), selected_over_rho_fraction=float((chosen_cost > P["rho"]).mean()),
                diagnostic_within_rho_candidate_max_mean=float(within.mean())))
        cost = np.array([r["mean_selected_cost"] for r in updates])
        lam = np.array([r["lambda"] for r in updates])
        if method.startswith("wdro_"):
            before = np.r_[P["lambda_init"], lam[:-1]]
            check(np.allclose(lam, np.maximum(0, before + P["lambda_lr"] * (cost - P["rho"])), atol=1e-12, rtol=0),
                  name + ": dual recurrence")
            gain = np.array([r["search_greedy_length"] - r["search_center_length"] for r in updates])
            check(np.all(gain - before * cost >= -1e-10), name + ": center dominance")
        selected = min((r for r in events if r["type"] == "validation"), key=lambda r: (r["worst_group_mean_length"], r["step"]))
        check(selected["step"] == c["best_step"], name + ": earliest best selection")
        run_rows.append(dict(method=method, seed=seed, initial_mean=float(np.mean(initial_means)),
            final_mean=float(np.mean(final_means)), delta_mean=float(np.mean(final_means) - np.mean(initial_means)),
            reduction_percent=float(100 * (1 - np.mean(final_means) / np.mean(initial_means))),
            final_worst_group_mean=max(final_means), best_step=c["best_step"],
            transport_applicable=method.startswith("wdro_"), mean_cost=float(cost.mean()),
            mean_cost_minus_rho=float(cost.mean() - P["rho"]) if method.startswith("wdro_") else None,
            last128_mean_cost=float(cost[-128:].mean()), over_rho_update_fraction=float((cost > P["rho"]).mean()),
            lambda_final=float(lam[-1]), mean_no_move_fraction=float(np.mean([r.get("search_no_move_fraction", 0.) for r in updates])) if method.startswith("wdro_") else None,
            last128_baseline_loss=float(np.mean([r["baseline_loss"] for r in updates[-128:]])),
            elapsed_seconds=c["elapsed_seconds"], sampled_calls=c["ledger"]["sampled_actor_scores"],
            search_calls=c["ledger"]["search_greedy_scores"], rollout_calls=c["ledger"]["rollout_greedy_scores"]))

reference = next(iter(arrays.values()), None)
for (method, seed), a in arrays.items():
    for group in GROUPS:
        for prefix in (f"initial__{group}__pins", f"final__{group}__pins", f"attack__{group}__pins"):
            check(np.array_equal(a[prefix], reference[prefix]), f"{method}/{seed}: common heldout coordinates {prefix}")
        anchor = arrays.get(("erm_critic", seed))
        if anchor is not None:
            check(np.array_equal(a[f"initial__{group}__length"], anchor[f"initial__{group}__length"]),
                  f"{method}/{seed}: common initialization output")

pairwise = []
for method in METHODS:
    for baseline in ("initial", "erm_critic", "wdro_random"):
        if method == baseline: continue
        seeds = [s for s in SEEDS if (method, s) in arrays and (baseline == "initial" or (baseline, s) in arrays)]
        if not seeds: continue
        delta = np.stack([np.stack([arrays[method, s][f"final__{g}__length"] -
                                   (arrays[method, s][f"initial__{g}__length"] if baseline == "initial" else arrays[baseline, s][f"final__{g}__length"])
                                   for g in GROUPS]) for s in seeds])
        for index, group in enumerate(["equal_weight_4_groups"] + GROUPS):
            pairwise.append(dict(method=method, baseline=baseline, distribution=group, seeds=seeds,
                interpretation="exploratory_partial_experiment" if missing else "three_seed_paired_experiment",
                **ci(delta if index == 0 else delta[:, index - 1])))
method_rows = []
for method in METHODS:
    rows = [r for r in run_rows if r["method"] == method]
    if not rows: continue
    aa = [r for r in attacks if r["method"] == method]
    method_rows.append(dict(method=method, completed_seeds=[r["seed"] for r in rows],
        initial_mean=float(np.mean([r["initial_mean"] for r in rows])),
        final_mean=float(np.mean([r["final_mean"] for r in rows])),
        final_seed_min=min(r["final_mean"] for r in rows), final_seed_max=max(r["final_mean"] for r in rows),
        mean_attacked=float(np.mean([r["mean_attacked"] for r in aa])),
        mean_attack_lift=float(np.mean([r["mean_lift"] for r in aa])),
        mean_attack_selected_cost=float(np.mean([r["mean_selected_cost"] for r in aa])),
        attack_selected_over_rho_fraction=float(np.mean([r["selected_over_rho_fraction"] for r in aa])),
        mean_training_cost=float(np.mean([r["mean_cost"] for r in rows])) if method.startswith("wdro_") else None))
warnings = [
    "Finite-run dual updates are not a hard transport-budget constraint; inspect each run's mean_cost_minus_rho.",
    "Random32 attack maximizes raw length at lambda=0 over a coordinate window; it does not enforce the rho transport budget.",
    "mix_gauss test examples share one generated centroid layout; bootstrap intervals are conditional on this layout.",
    "Three training seeds do not support broad stability claims; intervals have no multiple-comparison correction.",
    "REST terminal-segment accounting may double-count collinear overlap; no optimality gap is measured."]
if missing:
    warnings.insert(0, "The final 18-run experiment is incomplete; no complete six-method ranking is accepted.")
    warnings.append("Methods with missing seeds have different sample sets; only common-seed paired comparisons are comparable.")
result = dict(status="FAIL" if errors else ("PARTIAL" if missing else "NUMERICAL_REVIEW_PASS"),
    snapshot_utc=snapshot, completed_runs=len(records), expected_runs=len(METHODS) * len(SEEDS), missing=missing,
    errors=errors, warnings=warnings, method_summaries=method_rows, runs=run_rows, groups=groups, attacks=attacks,
    paired_comparisons=pairwise, source_bindings=bindings, protocol_sha256=digest(ROOT / "protocol.json"),
    reviewer_sha256=digest(SCRIPT), require_complete=bool(args.require_complete), output_directory=str(OUT),
    counts=dict(completed_final_test_predictions=len(records) * 4 * P["test_per_distribution"],
        completed_initial_test_predictions=len(records) * 4 * P["test_per_distribution"],
        completed_attack_centers=len(records) * 4 * P["attack_per_distribution"],
        completed_attack_candidate_predictions=len(records) * 4 * P["attack_per_distribution"] * P["attack_candidates"],
        unique_test_centers=4 * P["test_per_distribution"], unique_attack_centers=4 * P["attack_per_distribution"]),
    verification="Independent NumPy edge-by-edge length recomputation, costs, means/P95, paired coordinates, selection, dual recurrence and early-best selection; no model/GPU execution.")
(OUT / "statistics_review.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
for filename, rows in (("runs", run_rows), ("groups", groups), ("attacks", attacks), ("pairwise", pairwise)):
    with (OUT / f"statistics_review_{filename}.csv").open("w") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
title = "独立统计审阅（数值检查失败）" if errors else ("独立统计审阅（部分完成）" if missing else "独立统计审阅（完整冻结实验）")
scope = ("不接受完整六方法排名；以下为已完成种子的描述统计。" if missing else
         "已覆盖六种方法各三个冻结训练种子；比较解释仍受样本、预算和区间限制。")
completion_line = f"快照：{snapshot}；完成 {len(records)}/18。"
if missing:
    completion_line += "尚缺 " + ", ".join(x["run"] for x in missing) + "。"
lines = ["# " + title, "", completion_line, scope, "",
    "| 方法 | 完成种子 | 初始平均长度 | 最终平均长度 | 攻击后长度 | 训练平均成本 |", "|---|---|---:|---:|---:|---:|"]
for row in method_rows:
    cost = "不适用" if row["mean_training_cost"] is None else f'{row["mean_training_cost"]:.6f}'
    lines.append(f'| {row["method"]} | {row["completed_seeds"]} | {row["initial_mean"]:.6f} | {row["final_mean"]:.6f} | {row["mean_attacked"]:.6f} | {cost} |')
lines += ["", f"独立数值检查错误：{len(errors)}。", "", "## 解释边界", ""] + ["- " + x for x in warnings]
lines += ["", "均值对四个分布组及现有种子等权。长度由保存拓扑独立逐边重算；样本量与配对区间见同名前缀 JSON/CSV。"]
(OUT / "statistics_review.md").write_text("\n".join(lines) + "\n")
print(json.dumps({key: result[key] for key in ("status", "completed_runs", "errors", "method_summaries", "counts")}, indent=2))
raise SystemExit(1 if errors else (2 if missing else 0))
