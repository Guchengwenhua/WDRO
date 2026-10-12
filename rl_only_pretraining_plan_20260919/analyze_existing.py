"""Read accepted artifacts and summarize learning curves; no model/GPU execution."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "rl_only_20260915/artifacts/runs"
METHODS = ("erm_critic", "erm_rollout", "wdro_random", "wdro_es", "wdro_critic", "wdro_rl")
SEEDS = (101, 102, 103)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = HERE / "evidence"
    out.mkdir(exist_ok=True)
    inputs, curves, runs = {}, [], []
    for method in METHODS:
        for seed in SEEDS:
            folder = RUNS / f"{method}_s{seed}"
            complete = json.loads((folder / "complete.json").read_text())
            progress = folder / "progress.jsonl"
            expected = complete["artifacts_sha256"]["progress.jsonl"]
            if digest(progress) != expected:
                raise RuntimeError(f"Accepted progress hash mismatch: {progress}")
            for name in ("complete.json", "progress.jsonl"):
                path = folder / name
                inputs[str(path)] = digest(path)
            vals = {}
            for r in map(json.loads, progress.read_text().splitlines()):
                if r["type"] != "validation":
                    continue
                if r["step"] in vals:
                    raise RuntimeError(f"Duplicate validation: {folder}, {r['step']}")
                vals[r["step"]] = r
                curves.append(dict(method=method, seed=seed, step=r["step"],
                                   worst_group_mean=r["worst_group_mean_length"],
                                   mean=statistics.mean(g["mean_length"] for g in r["groups"].values())))
            if sorted(vals) != list(range(0, 2049, 128)):
                raise RuntimeError(f"Missing validation points: {folder}")
            selected = min(vals, key=lambda k: (vals[k]["worst_group_mean_length"], k))
            if selected != complete["best_step"]:
                raise RuntimeError(f"Checkpoint selection mismatch: {folder}")
            r = dict(method=method, seed=seed, best_step=selected,
                     seconds_per_update=complete["seconds_per_update"],
                     elapsed_seconds=complete["elapsed_seconds"])
            for step in (512, 1024, 1536, 2048):
                r[f"worst_{step}"] = vals[step]["worst_group_mean_length"]
            r["last512_reduction_pct"] = 100 * (r["worst_1536"] - r["worst_2048"]) / r["worst_1536"]
            runs.append(r)
    write_csv(out / "validation_curves.csv", curves)
    write_csv(out / "per_run_summary.csv", runs)
    methods = []
    for method in METHODS:
        mr = [r for r in runs if r["method"] == method]
        avg = {step: statistics.mean(r[f"worst_{step}"] for r in mr) for step in (512, 1024, 1536, 2048)}
        methods.append(dict(method=method, best_at_final=sum(r["best_step"] == 2048 for r in mr),
                            worst1536_seed_mean=avg[1536], worst2048_seed_mean=avg[2048],
                            last512_reduction_of_seed_mean_pct=100 * (avg[1536] - avg[2048]) / avg[1536],
                            min_seconds_per_update=min(r["seconds_per_update"] for r in mr),
                            max_seconds_per_update=max(r["seconds_per_update"] for r in mr)))
    write_csv(out / "method_summary.csv", methods)
    comparison = ROOT / "rl_only_vs_mainline_20260919/artifacts/report/summary.csv"
    inputs[str(comparison)] = digest(comparison)
    summaries = {(r["suite"], r["method"]): r for r in csv.DictReader(comparison.open())}
    gaps = []
    for suite in ("rl_test", "mainline_test", "rl_attack32"):
        old = float(summaries[(suite, "mainline:group_dro")]["mean"])
        new = float(summaries[(suite, "rl_only:erm_rollout")]["mean"])
        gaps.append(dict(suite=suite, groupdro=old, rollout=new, gap=new - old,
                         relative_gap_pct=100 * (new / old - 1),
                         halfway_length=old + .5 * (new - old)))
    write_csv(out / "observed_gaps.csv", gaps)
    for rel in ("../rl_only_20260915/protocol.json", "../rl_only_20260915/train.py",
                "../rl_only_20260915/geometry.py", "../rl_only_vs_mainline_20260919/TRAINING_DIFFERENCES.md"):
        path = (HERE / rel).resolve()
        inputs[str(path)] = digest(path)
    evidence = dict(status="PASS", scope="Read-only curve extraction; no training or new inference",
                    accepted_progress_hashes_verified=18, checkpoint_selections_recomputed=18,
                    validation_rows=len(curves), methods=methods, observed_gaps=gaps,
                    limitations=["Three training seeds", "Previously inspected validation data",
                                 "No extrapolated accuracy or convergence guarantee"],
                    input_sha256=inputs)
    (out / "summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")

    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, method, title in zip(axes, ("erm_rollout", "wdro_rl"), ("Self-critical ERM", "Learned-adversary WDRO")):
        steps = list(range(512, 2049, 128))
        series = []
        for seed in SEEDS:
            lookup = {r["step"]: r["worst_group_mean"] for r in curves if r["method"] == method and r["seed"] == seed}
            ys = [lookup[k] for k in steps]
            series.append(ys)
            ax.plot(steps, ys, linewidth=1, alpha=.65, label=f"Seed {seed}")
        ax.plot(steps, [statistics.mean(v) for v in zip(*series)], color="black", linewidth=2, label="Seed mean")
        ax.set_title(title)
        ax.set_xlabel("Training updates (batch 64)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Worst-group validation mean length")
    fig.suptitle("Existing degree-30 runs: last 1,536 updates; lower is better", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=180)
    fig.savefig(out / "learning_curves.pdf")
    print(json.dumps({k: evidence[k] for k in ("status", "accepted_progress_hashes_verified", "validation_rows", "methods")}, indent=2))


if __name__ == "__main__":
    main()
