"""Describe the audited B training histories without new model/oracle calls.

Requires the complete A30/B18 PASS manifest and verifies every consumed B
completion receipt against its hash binding before interpreting history.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/interpretation/research_review"
METHODS = ("random_top4", "critic_calibrated4", "rl_frozen4", "rl_joint4", "random64", "es64")
SEEDS = (101, 102, 103)
WINDOWS = {"all512": (0, 512), "first64": (0, 64), "last128": (384, 512)}
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path)
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "sha256": sha256(path), "bytes": path.stat().st_size}


def atomic_bytes(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_csv(path, rows):
    require(rows, f"empty output: {path}")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, handle.getvalue().encode())


def mean(values):
    return statistics.fmean(values)


def load_evidence():
    path = ROOT / "manifests/all_completion.json"
    require(path.is_file(), "all_completion.json is not yet available; no output published")
    complete = json.loads(path.read_text())
    require(complete.get("status") == "pass" and complete.get("phase") == "all"
            and complete.get("all_experiments_complete") is True,
            "all-experiment PASS gate is not satisfied")
    require(complete.get("provenance", {}).get("status") == "pass", "provenance did not pass")
    require(complete.get("phases") == {"fixed": {"tasks": 30, "cells": 1080},
                                      "joint": {"tasks": 18, "cells": 216}},
            "frozen A30/B18 coverage is not complete")
    bound = {}
    for record in complete["artifacts"]:
        resolved = str(Path(record["path"]).resolve())
        require(resolved not in bound or bound[resolved] == record["sha256"],
                f"conflicting input binding: {resolved}")
        bound[resolved] = record["sha256"]
    sources = [binding(path)]

    def verify(path):
        record = binding(path)
        require(bound.get(record["resolved_path"]) == record["sha256"],
                f"consumed source is missing or changed in all PASS manifest: {path}")
        sources.append(record)
        return record

    protocol_path = ROOT / "protocol.json"
    require(sha256(protocol_path) == complete["protocol_sha256"], "protocol digest mismatch")
    protocol = json.loads(protocol_path.read_text())
    require(protocol["degrees"] == [10, 20, 30, 40, 50]
            and protocol["seeds"] == list(SEEDS)
            and protocol["parents"] == ["erm", "wdro_rest"]
            and protocol["joint"]["methods"] == list(METHODS)
            and protocol["joint"]["updates"] == 512
            and protocol["joint"]["degree"] == 30, "frozen protocol grid mismatch")
    sources.append(binding(protocol_path))
    require(protocol["plan_sha256"] == sha256(ROOT / "EXPERIMENT_PLAN.md"), "frozen plan changed")
    sources.append(binding(ROOT / "EXPERIMENT_PLAN.md"))
    # Check exact receipt membership for both formal grids, without hashing A's
    # unused scientific artifacts again.
    expected = {str((ROOT / f"experiments/fixed/{p}_d{d}_s{s}/complete.json").resolve())
                for p in protocol["parents"] for d in protocol["degrees"] for s in SEEDS}
    expected |= {str((ROOT / f"experiments/joint/{m}_d30_s{s}/complete.json").resolve())
                 for m in METHODS for s in SEEDS}
    actual = {p for p in bound if Path(p).name == "complete.json"
              and Path(p).parent.parent in ((ROOT / "experiments/fixed").resolve(),
                                           (ROOT / "experiments/joint").resolve())}
    require(actual == expected, "all manifest receipt grid is missing, duplicated or expanded")
    histories = []
    # Scientific Python files are bound by task identity.source inside the
    # directly bound completion receipts, rather than necessarily appearing as
    # separate file entries in the all manifest.
    joint_source = binding(ROOT / "adv_v3/joint.py")
    joint_source["verified_via"] = "all_completion -> each B complete -> identity.source.joint.py"
    sources.append(joint_source)
    for method in METHODS:
        for seed in SEEDS:
            task = f"{method}_d30_s{seed}"
            task_path = ROOT / f"experiments/joint/{task}/complete.json"
            verify(task_path)  # Hash check happens before reading any history.
            done = json.loads(task_path.read_text())
            require(done.get("status") == "pass" and done.get("phase") == "joint"
                    and done.get("task") == task and done.get("method") == method
                    and done.get("seed") == seed and done.get("degree") == 30
                    and done.get("smoke") is False, f"unexpected task identity: {task}")
            require(done["identity"]["source"]["joint.py"] == joint_source["sha256"],
                    f"joint code source binding mismatch: {task}")
            checkpoints = {}
            for name in ("best.pt", "latest.pt"):
                target = (task_path.parent / name).resolve()
                checkpoint = [x for x in done["artifacts"] if Path(x["path"]).resolve() == target]
                require(len(checkpoint) == 1 and bound.get(str(target)) == checkpoint[0]["sha256"],
                        f"checkpoint binding disagreement: {target}")
                checkpoints[name] = checkpoint[0]
            histories.append((done, checkpoints))
    return histories, sources


def summarize_task(done, checkpoints):
    task, method, seed = done["task"], done["method"], done["seed"]
    history, config = done["history"], done["identity"]["config"]
    require(done["updates"] == config["updates"] == len(history) == 512,
            f"incomplete update count: {task}")
    require([r["update"] for r in history] == list(range(1, 513)), f"noncontiguous updates: {task}")
    require(config["batch_size"] == 64 and config["lambda_learning_rate"] == 1.0,
            f"unexpected dual-update settings: {task}")
    rho, price = float(done["rho"]), float(done["initial_price"])
    require(math.isfinite(rho) and rho > 0 and math.isfinite(price) and price >= 0,
            f"invalid initial dual state: {task}")
    rows = []
    for row in history:
        values = [row[k] for k in ("rho", "lambda_before", "lambda_after", "mean_cost",
                                  "mean_search_regret", "mean_search_score")]
        values += [row["sampled"][k] for k in ("mean_regret", "actor_loss", "critic_loss")]
        require(all(math.isfinite(v) for v in values), f"nonfinite history value: {task}")
        require(abs(row["rho"] - rho) <= TOLERANCE and row["mean_cost"] >= 0
                and row["lambda_before"] >= 0 and row["lambda_after"] >= 0,
                f"invalid nonnegative dual/cost state: {task}")
        continuity_error = abs(row["lambda_before"] - price)
        reconstructed = max(0.0, price + config["lambda_learning_rate"] * (row["mean_cost"] - rho))
        update_error = abs(reconstructed - row["lambda_after"])
        score_error = abs(row["mean_search_regret"] - row["lambda_before"] * row["mean_cost"]
                          - row["mean_search_score"])
        require(max(continuity_error, update_error, score_error) <= TOLERANCE,
                f"dual recurrence or greedy penalized score mismatch: {task} step {row['update']}")
        price = row["lambda_after"]
        rows.append({"task": task, "method": method, "seed": seed, "update": row["update"],
                     "rho": rho, "mean_cost": row["mean_cost"], "cost_over_rho": row["mean_cost"] / rho,
                     "lambda_before": row["lambda_before"], "lambda_after": price,
                     "lambda_after_is_zero": int(price == 0.0),
                     "lambda_recomputed": reconstructed, "lambda_update_abs_error": update_error,
                     "lambda_continuity_abs_error": continuity_error,
                     "greedy_score_identity_abs_error": score_error,
                     "greedy_selected_mean_regret": row["mean_search_regret"],
                     "sampled_selected_mean_regret": row["sampled"]["mean_regret"],
                     "sampled_minus_greedy_mean_regret": row["sampled"]["mean_regret"] - row["mean_search_regret"],
                     "sampled_actor_loss": row["sampled"]["actor_loss"],
                     "sampled_critic_loss": row["sampled"]["critic_loss"]})
    require(abs(price - done["final_price"]) <= TOLERANCE, f"final dual price mismatch: {task}")
    best_update = done["best_update"]
    require(best_update in range(0, 513, 64), f"invalid best validation checkpoint: {task}")
    best_price = done["initial_price"] if best_update == 0 else rows[best_update - 1]["lambda_after"]
    require(abs(best_price - done["best_price"]) <= TOLERANCE, f"best dual price mismatch: {task}")
    best_validation = done["initial_validation"] if best_update == 0 else history[best_update - 1]["validation"]
    require(best_validation == done["best_validation"], f"best validation metadata mismatch: {task}")
    summary = {"task": task, "method": method, "seed": seed, "degree": 30, "updates": 512,
               "rho": rho, "lambda_learning_rate": config["lambda_learning_rate"],
               "initial_lambda": done["initial_price"], "latest_lambda": done["final_price"],
               "best_update": best_update, "best_lambda": best_price,
               "best_is_latest": best_update == 512, "best_is_initial": best_update == 0,
               "max_lambda_update_abs_error": max(r["lambda_update_abs_error"] for r in rows),
               "max_lambda_continuity_abs_error": max(r["lambda_continuity_abs_error"] for r in rows),
               "max_greedy_score_identity_abs_error": max(r["greedy_score_identity_abs_error"] for r in rows)}
    for window, (start, stop) in WINDOWS.items():
        selected = rows[start:stop]
        for name in ("cost_over_rho", "lambda_before", "lambda_after", "lambda_after_is_zero",
                     "greedy_selected_mean_regret", "sampled_selected_mean_regret",
                     "sampled_minus_greedy_mean_regret", "sampled_actor_loss", "sampled_critic_loss"):
            field = "lambda_after_zero_fraction" if name == "lambda_after_is_zero" else "mean_" + name
            summary[f"{window}_{field}"] = mean(r[name] for r in selected)
    for label, validation in (("initial", done["initial_validation"]),
                              ("best", done["best_validation"]),
                              ("latest", history[-1]["validation"])):
        summary[f"{label}_validation_nominal_mean"] = validation["nominal_mean"]
        summary[f"{label}_validation_ood_worst_group_mean"] = validation["ood_worst_group_mean"]
    checkpoint_note = {"task": task, "best_update": best_update, "latest_update": 512,
                       "final_evaluation_uses": "best.pt", "training_trajectory_ends_at": "latest.pt",
                       "checkpoint_metadata_bindings": checkpoints,
                       "checkpoint_files_loaded_or_rehashed_here": False}
    return rows, summary, checkpoint_note


def rolling64(values):
    return [mean(values[end - 64:end]) for end in range(64, len(values) + 1)]


def make_figure(all_rows):
    os.environ["MPLCONFIGDIR"] = str(OUT / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ("#0072B2", "#D55E00", "#009E73")
    titles = ("Random top-4", "Calibrated critic-4", "Frozen RL-4", "Joint RL-4", "Random-64", "ES-64")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 6, figsize=(19, 7.5), sharex=True, sharey="row")
        for column, (method, title) in enumerate(zip(METHODS, titles)):
            for seed, color in zip(SEEDS, colors):
                selected = [r for r in all_rows if r["method"] == method and r["seed"] == seed]
                axes[0, column].plot(range(64, 513), rolling64([r["cost_over_rho"] for r in selected]),
                                     color=color, linewidth=1.7)
                axes[1, column].plot(range(64, 513), rolling64([r["lambda_after"] for r in selected]),
                                     color=color, linewidth=1.7)
            axes[0, column].axhline(1, color="#555555", linestyle="--", linewidth=1)
            axes[0, column].set_title(title)
            axes[1, column].set_xlabel("Main-policy update")
            for row in (0, 1):
                axes[row, column].set_xlim(64, 512)
                axes[row, column].set_xticks((64, 256, 512))
                axes[row, column].grid(alpha=0.18, linewidth=0.7)
        # Set shared limits only after all six panels have been populated;
        # setting a limit in the first panel disables later shared autoscaling.
        for row in (0, 1):
            all_plotted = [float(value) for axis in axes[row] for line in axis.lines
                           for value in line.get_ydata()]
            upper = max(all_plotted) * 1.06
            axes[row, 0].set_ylim(0, upper)
            for axis in axes[row]:
                low, high = axis.get_ylim()
                require(all(low <= float(value) <= high for line in axis.lines for value in line.get_ydata()),
                        "figure data would be clipped by shared y limits")
        axes[0, 0].set_ylabel("Mean transport cost / rho\n64-step rolling mean")
        axes[1, 0].set_ylabel("Lambda after update\n64-step rolling mean")
        fig.suptitle("B training budget and dual-price trajectories (degree 30; 512 updates)",
                     fontsize=16, y=0.99)
        legend = [Line2D([0], [0], color=c, linewidth=2, label=f"Seed {s}") for s, c in zip(SEEDS, colors)]
        legend.append(Line2D([0], [0], color="#555555", linestyle="--", label="Cost / rho = 1"))
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=False)
        fig.text(0.5, 0.025,
                 "512-update histories; full 64-update windows plotted from update 64. Final evaluation uses the validation-selected best checkpoint.\n"
                 "The dashed line is an average transport-budget reference, not a per-step feasibility guarantee.",
                 ha="center", va="bottom", fontsize=10, color="#444444")
        fig.subplots_adjust(left=0.062, right=0.99, bottom=0.15, top=0.865, wspace=0.15, hspace=0.22)
        outputs = []
        for suffix in ("png", "pdf"):
            path = OUT / f"joint_budget_trajectories.{suffix}"
            temporary = path.with_name(path.name + ".partial")
            fig.savefig(temporary, format=suffix, dpi=180,
                        metadata={"Title": "B training transport budget and dual-price trajectories"})
            temporary.replace(path)
            outputs.append(path)
        plt.close(fig)
    return outputs


def main():
    histories, sources = load_evidence()
    all_rows, tasks, checkpoint_notes = [], [], []
    for done, checkpoints in histories:
        rows, summary, note = summarize_task(done, checkpoints)
        all_rows.extend(rows)
        tasks.append(summary)
        checkpoint_notes.append(note)
    require(len(all_rows) == 18 * 512 and len(tasks) == 18, "B history coverage mismatch")
    methods = []
    for method in METHODS:
        selected = [r for r in tasks if r["method"] == method]
        result = {"method": method, "seed_count": 3,
                  "best_updates_in_seed_order": json.dumps([r["best_update"] for r in selected]),
                  "best_is_initial_count": sum(r["best_is_initial"] for r in selected),
                  "best_is_latest_count": sum(r["best_is_latest"] for r in selected)}
        for window in WINDOWS:
            for metric in ("mean_cost_over_rho", "lambda_after_zero_fraction", "mean_lambda_after",
                           "mean_greedy_selected_mean_regret", "mean_sampled_selected_mean_regret",
                           "mean_sampled_minus_greedy_mean_regret"):
                key = f"{window}_{metric}"
                result[key] = mean(r[key] for r in selected)
                result[key + "_seed_min"] = min(r[key] for r in selected)
                result[key + "_seed_max"] = max(r[key] for r in selected)
        methods.append(result)
    limitations = [
        "Descriptive summaries only; no new confidence intervals, significance tests or oracle calls.",
        "Equal-size batches and fixed rho within each task make the average of cost/rho equal to mean cost divided by rho.",
        "Window definitions: all512=updates 1-512; first64=1-64; last128=385-512.",
        "The zero fraction counts exactly zero lambda_after, not a numerical near-zero threshold.",
        "The dual update is lambda_next=max(0,lambda_before+eta*(mean_cost-rho)); eta=1.0. Recurrence tolerance is 1e-12.",
        "Rho is an average transport-budget parameter. A rolling mean near or below one does not certify a Wasserstein worst-case distribution or per-step feasibility.",
        "Greedy and sampled regret compare the same selected coordinates and pre-update Actor at each training step; sampled topology is stochastic.",
        "Greedy regret is the search evaluator loss; sampled actor_loss is a REINFORCE optimization surrogate and is not numerically comparable to regret.",
        "Final evaluation uses the validation-selected best checkpoint, which can be update 0. Training histories and latest_lambda extend to update 512.",
        "The authoritative all PASS audit establishes checkpoint contents. This consumer verifies their metadata binding agreement without reopening checkpoints.",
        "Methods have different real-score and training overhead budgets; these plots do not establish equal total compute or controlled speedup.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for name, rows in (("joint_budget_steps.csv", all_rows),
                       ("joint_budget_task_summary.csv", tasks),
                       ("joint_budget_method_summary.csv", methods)):
        path = OUT / name
        write_csv(path, rows)
        output_paths.append(path)
    summary_path = OUT / "joint_budget_summary.json"
    write_json(summary_path, {"status": "pass", "scope": "B18 audited descriptive training histories",
                             "tasks": tasks, "methods": methods, "checkpoint_notes": checkpoint_notes,
                             "history_rows": len(all_rows), "window_ranges": WINDOWS,
                             "recurrence_tolerance": TOLERANCE, "limitations": limitations,
                             "sources": sources, "script": binding(__file__),
                             "oracle_calls": 0, "inference_tests": 0})
    output_paths.append(summary_path)
    output_paths.extend(make_figure(all_rows))
    # Recheck consumed small source files after extraction to reject concurrent
    # republishing or edits during this observation.
    for record in sources:
        require(sha256(record["path"]) == record["sha256"], f"source changed during extraction: {record['path']}")
    receipt_path = OUT / "research_budget_completion.json"
    receipt = {"status": "pass", "created_at_unix": time.time(), "script": binding(__file__),
               "sources": sources, "outputs": [binding(p) for p in output_paths],
               "counts": {"fixed_coverage": 30, "joint_tasks": 18, "history_rows": len(all_rows),
                          "methods": 6, "seeds_per_method": 3, "figure_rows": 2, "figure_columns": 6},
               "all_lambda_recurrences_pass": True, "all_greedy_score_identities_pass": True,
               "checkpoint_best_latest_consistency_pass": True, "oracle_calls": 0,
               "inference_tests": 0, "limitations": limitations}
    write_json(receipt_path, receipt)
    print(json.dumps({"status": "pass", "receipt": binding(receipt_path), "counts": receipt["counts"]}))


if __name__ == "__main__":
    main()
