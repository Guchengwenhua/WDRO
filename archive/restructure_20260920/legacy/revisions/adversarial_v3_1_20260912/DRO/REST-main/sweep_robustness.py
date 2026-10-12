import os
import re
import json
import argparse
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def find_available_degrees(experiment: str, save_root: str = "save"):
    """
    Scan save/<experiment>/rsmt{degree}b.pt and return sorted degrees.
    Only supports degrees that have checkpoints.
    """
    exp_dir = os.path.join(save_root, experiment)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    degrees = []
    pat = re.compile(r"^rsmt(\d+)b\.pt$")
    for fn in os.listdir(exp_dir):
        m = pat.match(fn)
        if m:
            degrees.append(int(m.group(1)))
    degrees = sorted(set(degrees))
    if not degrees:
        raise RuntimeError(f"No checkpoints found under {exp_dir} with pattern rsmt{{degree}}b.pt")
    return degrees


def run_one(test_py: str, args_list: list, env=None):
    """
    Run: python test_robust.py <args...>
    """
    cmd = ["python", test_py] + args_list
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def read_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: str, header: list, rows: list):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_sweep_id(experiment: str, point_dist: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_exp{experiment}_dist{point_dist}"


def plot_lines(data, out_path: str, title: str):
    """
    data: dict degree -> list of (nonuniform, error)
    """
    plt.figure(figsize=(10, 6))
    for deg, pts in sorted(data.items(), key=lambda x: x[0]):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=f"deg={deg}")
    plt.xlabel("nonuniform (distribution shift strength)")
    plt.ylabel("REST percentage error (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(matrix, x_ticks, y_ticks, out_path: str, title: str, xlabel: str, ylabel: str):
    """
    matrix: shape (len(y_ticks), len(x_ticks))
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="REST percentage error (%)")
    plt.xticks(ticks=np.arange(len(x_ticks)), labels=[str(x) for x in x_ticks], rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(y_ticks)), labels=[str(y) for y in y_ticks])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    # sweep config
    parser.add_argument("--experiment", type=str, default="DAC21")
    parser.add_argument("--save_root", type=str, default="save")
    parser.add_argument("--test_py", type=str, default="test_robust.py")

    parser.add_argument("--point_dist", type=str, default="mix_gauss",
                        choices=["mix_gauss", "anisotropic_gauss", "hotspots_with_void", "edge_bias", "grid_cluster", "uniform"])
    parser.add_argument("--nonuniform_list", type=str, default="0,0.2,0.4,0.6,0.8,1.0",
                        help="comma separated list, e.g. 0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.04)

    # eval config
    parser.add_argument("--test_size", type=int, default=2000,
                        help="GeoSteiner is slow; start with 2000, then increase if needed")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--transformation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")

    # output
    parser.add_argument("--out_dir", type=str, default="results_sweep")
    parser.add_argument("--resume", type=str, default="true",
                        help="true: skip already existing runs in jsonl")
    parser.add_argument("--run_optimal", type=str, default="true",
                        help="keep true to compute GeoSteiner and percentage error")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    sweep_id = make_sweep_id(args.experiment, args.point_dist)
    sweep_jsonl = os.path.join(args.out_dir, f"sweep_{sweep_id}.jsonl")
    sweep_csv = os.path.join(args.out_dir, f"sweep_{sweep_id}.csv")

    degrees = find_available_degrees(args.experiment, args.save_root)
    print("Detected degrees from checkpoints:", degrees)

    nonuniforms = [float(x) for x in args.nonuniform_list.split(",") if x.strip() != ""]
    nonuniforms = [max(0.0, min(1.0, x)) for x in nonuniforms]

    # For resume: collect already finished (degree, nonuniform)
    done = set()
    if args.resume.lower() == "true" and os.path.exists(sweep_jsonl):
        prev = read_jsonl(sweep_jsonl)
        for r in prev:
            done.add((int(r["degree"]), float(r["nonuniform"])))

    # run sweep
    for deg in degrees:
        for nu in nonuniforms:
            if (deg, nu) in done:
                print(f"[resume] skip degree={deg}, nonuniform={nu}")
                continue

            # build args for test_robust.py
            run_args = [
                "--experiment", args.experiment,
                "--degree", str(deg),
                "--dimension", str(args.dimension),
                "--test_size", str(args.test_size),
                "--batch_size", str(args.batch_size),
                "--transformation", str(args.transformation),
                "--seed", str(args.seed),

                "--point_dist", args.point_dist,
                "--nonuniform", str(nu),
                "--clusters", str(args.clusters),
                "--sigma", str(args.sigma),

                "--run_optimal", args.run_optimal,
                "--plot_points", "false",       # sweep 时不画点（可大幅加速）
                "--plot_first", "false",        # sweep 时不画RSMT对比（需要时单独跑一个）
                "--device", args.device,
                "--out_dir", args.out_dir,
            ]

            # run
            run_one(args.test_py, run_args)

            # test_robust.py 会把记录 append 到 out_dir/runs.jsonl
            # 我们从最新 runs.jsonl 里找出匹配项，再写进 sweep_jsonl（统一汇总）
            runs_jsonl = os.path.join(args.out_dir, "runs.jsonl")
            runs = read_jsonl(runs_jsonl)
            # 取最后一条作为“本次运行”的记录（通常就是刚跑完的）
            last = runs[-1] if runs else None
            if last is None:
                raise RuntimeError("runs.jsonl is empty after running test_robust.py")

            # 记录 sweep 信息（避免依赖 run_id 的解析）
            rec = {
                "sweep_id": sweep_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "degree": deg,
                "nonuniform": nu,
                "point_dist": args.point_dist,
                "clusters": args.clusters,
                "sigma": args.sigma,
                "test_size": args.test_size,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": args.device,
                "rest_percentage_error": last["metrics"]["rest_percentage_error"],
                "rest_mean_length": last["metrics"]["rest_mean_length"],
                "geosteiner_mean_length": last["metrics"]["geosteiner_mean_length"],
                "geosteiner_time_sec": last["metrics"]["geosteiner_time_sec"],
                "run_id": last["run_id"],
            }

            with open(sweep_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            done.add((deg, nu))
            print(f"[saved] sweep record -> {sweep_jsonl}")

    # -------------------------
    # Visualization
    # -------------------------
    rows = read_jsonl(sweep_jsonl)
    if not rows:
        raise RuntimeError("No sweep records found. Nothing to plot.")

    # build per-degree data
    per_deg = {}
    for r in rows:
        deg = int(r["degree"])
        nu = float(r["nonuniform"])
        err = r["rest_percentage_error"]
        if err is None:
            continue
        err = float(err)
        per_deg.setdefault(deg, []).append((nu, err))

    # CSV export
    header = ["degree", "nonuniform", "rest_percentage_error", "rest_mean_length", "geosteiner_mean_length", "geosteiner_time_sec", "run_id"]
    csv_rows = []
    for r in rows:
        csv_rows.append([
            int(r["degree"]),
            float(r["nonuniform"]),
            r["rest_percentage_error"],
            r["rest_mean_length"],
            r["geosteiner_mean_length"],
            r["geosteiner_time_sec"],
            r["run_id"],
        ])
    write_csv(sweep_csv, header, csv_rows)
    print("Saved sweep CSV:", sweep_csv)

    # 1) lines: error vs nonuniform
    line_fig = os.path.join(args.out_dir, f"plot_error_lines_{sweep_id}.png")
    plot_lines(
        per_deg,
        line_fig,
        title=f"REST percentage error vs nonuniform ({args.point_dist})",
    )
    print("Saved line plot:", line_fig)

    # 2) heatmap: degree x nonuniform
    degs = sorted(per_deg.keys())
    nus = sorted(set([nu for d in degs for (nu, _) in per_deg[d]]))

    mat = np.full((len(degs), len(nus)), np.nan, dtype=float)
    for i, deg in enumerate(degs):
        pts = {nu: err for (nu, err) in per_deg[deg]}
        for j, nu in enumerate(nus):
            if nu in pts:
                mat[i, j] = pts[nu]

    heat_fig = os.path.join(args.out_dir, f"plot_error_heatmap_{sweep_id}.png")
    plot_heatmap(
        mat,
        x_ticks=nus,
        y_ticks=degs,
        out_path=heat_fig,
        title=f"Heatmap: REST percentage error (%) ({args.point_dist})",
        xlabel="nonuniform",
        ylabel="degree",
    )
    print("Saved heatmap:", heat_fig)

    # 3) degradation relative to baseline nu=0 (Δerror)
    # if nu=0 not present, skip
    if 0.0 in nus:
        delta_mat = np.full_like(mat, np.nan)
        for i in range(len(degs)):
            base = mat[i, nus.index(0.0)]
            if np.isnan(base):
                continue
            delta_mat[i, :] = mat[i, :] - base

        delta_fig = os.path.join(args.out_dir, f"plot_error_delta_from_uniform_{sweep_id}.png")
        plot_heatmap(
            delta_mat,
            x_ticks=nus,
            y_ticks=degs,
            out_path=delta_fig,
            title=f"ΔError vs uniform baseline (nu=0) ({args.point_dist})",
            xlabel="nonuniform",
            ylabel="degree",
        )
        print("Saved delta heatmap:", delta_fig)
    else:
        print("nu=0 not found in sweep -> skip Δerror plot")

    print("\nAll done.")
    print("Artifacts are in:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()