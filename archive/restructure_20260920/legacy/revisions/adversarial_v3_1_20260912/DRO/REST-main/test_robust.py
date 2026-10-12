# test_robust.py
# 用法示例：
#   1) 均匀分布（等价原 test.py 随机生成）
#      python test_robust.py --experiment DAC21 --degree 10 --test_size 10000 --point_dist uniform --nonuniform 0
#
#   2) 非均匀分布（混合高斯聚簇）
#      python test_robust.py --experiment DAC21 --degree 10 --test_size 10000 --point_dist mix_gauss --nonuniform 0.8 --clusters 6 --sigma 0.04
#
# 输出指标与原 test.py 一致，并额外：
#   - 保存点分布 4x4 图
#   - 保存结果日志 jsonl 便于追踪

import os
import time
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.actor_critic import Actor
from utils.rsmt_utils import *
from utils.log_utils import *


# -----------------------------
# Data generator (non-uniform)
# -----------------------------
def generate_terminals(
    n_nets: int,
    degree: int,
    dim: int,
    seed: int,
    point_dist: str = "uniform",
    nonuniform: float = 0.0,
    clusters: int = 4,
    sigma: float = 0.06,
):
    """
    Extended industrial-style terminal generator.
    All outputs in [0,1]^dim.
    nonuniform in [0,1] controls intensity of deviation from uniform.

    Supported point_dist:
      - uniform
      - mix_gauss
      - anisotropic_gauss
      - hotspots_with_void
      - edge_bias
      - grid_cluster
    """
    if not (0.0 <= nonuniform <= 1.0):
        raise ValueError("nonuniform must be in [0,1]")

    rng = np.random.default_rng(seed)

    # 1) uniform
    if point_dist == "uniform" or nonuniform == 0.0:
        return np.round(rng.random((n_nets, degree, dim)), 8)

    # 2) mix_gauss: mixture of uniform + gaussian hotspots
    if point_dist == "mix_gauss":
        if clusters <= 0 or sigma <= 0:
            raise ValueError("clusters and sigma must be > 0")
        centers = rng.random((clusters, dim))
        x = rng.random((n_nets, degree, dim))

        mask = rng.random((n_nets, degree)) < nonuniform
        cid = rng.integers(0, clusters, size=(n_nets, degree))
        c = centers[cid]
        gauss = c + rng.normal(0.0, sigma, size=(n_nets, degree, dim))
        x[mask] = gauss[mask]
        return np.round(np.clip(x, 0, 1), 8)

    # 3) anisotropic_gauss: elongated hotspots (corridor/stripe-like regions)
    if point_dist == "anisotropic_gauss":
        if clusters <= 0 or sigma <= 0:
            raise ValueError("clusters and sigma must be > 0")
        x = rng.random((n_nets, degree, dim))

        mask = rng.random((n_nets, degree)) < nonuniform
        centers = rng.random((clusters, dim))
        cid = rng.integers(0, clusters, size=(n_nets, degree))
        c = centers[cid]

        if dim < 2:
            stretch = np.array([sigma * 3.0])
        else:
            # x方向宽，y方向窄（其余维度按 sigma）
            stretch = np.ones(dim) * sigma
            stretch[0] = sigma * 4.0
            stretch[1] = sigma * 0.3

        gauss = c + rng.normal(0.0, stretch, size=(n_nets, degree, dim))
        x[mask] = gauss[mask]
        return np.round(np.clip(x, 0, 1), 8)

    # 4) hotspots_with_void: avoid a forbidden region + hotspots
    if point_dist == "hotspots_with_void":
        if clusters <= 0 or sigma <= 0:
            raise ValueError("clusters and sigma must be > 0")
        x = rng.random((n_nets, degree, dim))

        # A circular void in (0,1)^2 (for dim>=2); for dim<2 just fallback to mix_gauss
        if dim < 2:
            # fallback to mix_gauss behavior
            centers = rng.random((clusters, dim))
            mask = rng.random((n_nets, degree)) < nonuniform
            cid = rng.integers(0, clusters, size=(n_nets, degree))
            c = centers[cid]
            gauss = c + rng.normal(0.0, sigma, size=(n_nets, degree, dim))
            x[mask] = gauss[mask]
            return np.round(np.clip(x, 0, 1), 8)

        void_center = np.array([0.5, 0.5])
        void_radius = 0.15 + 0.2 * nonuniform

        # rejection sampling for all points to avoid the void
        for i in range(n_nets):
            for j in range(degree):
                while True:
                    p = rng.random(dim)
                    if np.linalg.norm(p[:2] - void_center) > void_radius:
                        x[i, j] = p
                        break

        # overlay hotspots for a subset
        centers = rng.random((clusters, dim))
        mask = rng.random((n_nets, degree)) < nonuniform
        cid = rng.integers(0, clusters, size=(n_nets, degree))
        c = centers[cid]
        gauss = c + rng.normal(0.0, sigma, size=(n_nets, degree, dim))
        x[mask] = gauss[mask]

        return np.round(np.clip(x, 0, 1), 8)

    # 5) edge_bias: biased to boundaries (I/O pin like)
    if point_dist == "edge_bias":
        x = rng.random((n_nets, degree, dim))
        bias_strength = nonuniform * 3.0

        # push towards 0 (then randomly flip to push towards 1)
        x = x ** (1.0 + bias_strength)
        flip_mask = rng.random((n_nets, degree, dim)) > 0.5
        x[flip_mask] = 1.0 - x[flip_mask]
        return np.round(np.clip(x, 0, 1), 8)

    # 6) grid_cluster: points align to several horizontal "rows" (standard-cell rows)
    if point_dist == "grid_cluster":
        if clusters <= 1:
            raise ValueError("clusters must be >= 2 for grid_cluster")
        if dim < 2:
            # if 1D, just fallback to edge_bias-ish
            x = rng.random((n_nets, degree, dim))
            bias_strength = nonuniform * 2.0
            x = x ** (1.0 + bias_strength)
            return np.round(np.clip(x, 0, 1), 8)

        x = np.zeros((n_nets, degree, dim))
        grid_y = np.linspace(0.1, 0.9, clusters)

        for i in range(n_nets):
            for j in range(degree):
                if rng.random() < nonuniform:
                    row = rng.choice(grid_y)
                    px = rng.random()
                    py = row + rng.normal(0.0, sigma * 0.3)
                    p = np.array([px, py] + [rng.random() for _ in range(dim - 2)], dtype=float)
                    x[i, j] = p
                else:
                    x[i, j] = rng.random(dim)

        return np.round(np.clip(x, 0, 1), 8)

    raise ValueError(f"Unknown point_dist: {point_dist}")


# -----------------------------
# Visualization (4x4)
# -----------------------------
def plot_nets_points_grid(test_cases: np.ndarray, save_path: str, n_show: int = 16):
    """
    Plot terminals for first n_show nets in a 4x4 grid.
    test_cases: (N, degree, dim)
    """
    n_show = min(n_show, len(test_cases))
    rows, cols = 4, 4
    fig = plt.figure(figsize=(12, 12))

    for idx in range(rows * cols):
        ax = plt.subplot(rows, cols, idx + 1)
        if idx >= n_show:
            ax.axis("off")
            continue

        pts = test_cases[idx]
        dim = pts.shape[1]

        if dim >= 2:
            ax.scatter(pts[:, 0], pts[:, 1], s=12)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.scatter(pts[:, 0], np.zeros(len(pts)), s=12)
            ax.set_ylim(-0.1, 0.1)

        ax.grid(True, alpha=0.2)
        ax.set_title(f"net {idx}", fontsize=10)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Safe checkpoint loader
# -----------------------------
def load_checkpoint_safely(ckp_path: str, device: torch.device):
    """
    PyTorch 2.6+ 可能触发 weights_only 行为变化；
    这里做一个尽量兼容、但仍以“你信任 checkpoint”为前提的加载方式。
    """
    try:
        return torch.load(ckp_path, weights_only=False, map_location=device)
    except TypeError:
        try:
            with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                return torch.load(ckp_path, map_location=device)
        except Exception:
            return torch.load(ckp_path, map_location=device)


# -----------------------------
# Result logging
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_run_id(args) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = (
        f"{ts}"
        f"_exp{args.experiment}"
        f"_deg{args.degree}"
        f"_N{args.test_size}"
        f"_dist{args.point_dist}"
        f"_nu{args.nonuniform}"
        f"_k{args.clusters}"
        f"_sig{args.sigma}"
        f"_T{args.transformation}"
        f"_seed{args.seed}"
    )
    # 文件名别太长：适当截断
    return rid[:220]


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # 原 test.py 参数
    parser.add_argument("--experiment", type=str, default="DAC21", help="experiment name")
    parser.add_argument("--degree", type=int, default=10, help="maximum degree of nets")
    parser.add_argument("--dimension", type=int, default=2, help="terminal representation dimension")
    parser.add_argument("--test_data", type=str, default="", help="test data")
    parser.add_argument("--test_size", type=int, default=10000, help="number of nets")
    parser.add_argument("--batch_size", type=int, default=1000, help="test batch size")
    parser.add_argument("--transformation", type=int, default=1, help="number of transformations for inference")
    parser.add_argument("--run_optimal", type=str, default="true", help="run GeoSteiner to generate optimal RSMT")
    parser.add_argument("--plot_first", type=str, default="true", help="plot the first result (gst vs rest)")
    parser.add_argument("--seed", type=int, default=7, help="random seed")

    # 新增：可控点分布参数
    parser.add_argument(
        "--point_dist",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "mix_gauss",
            "anisotropic_gauss",
            "hotspots_with_void",
            "edge_bias",
            "grid_cluster",
        ],
        help="terminal point distribution",
    )
    parser.add_argument(
        "--nonuniform",
        type=float,
        default=0.0,
        help="non-uniform degree in [0,1]. 0=uniform, 1=fully non-uniform",
    )
    parser.add_argument("--clusters", type=int, default=4, help="number of clusters (hotspots/rows)")
    parser.add_argument("--sigma", type=float, default=0.06, help="std of gaussian clusters")
    parser.add_argument("--plot_points", type=str, default="true", help="visualize terminals distribution")
    parser.add_argument("--plot_points_n", type=int, default=16, help="how many nets to plot (default 16)")

    # 保存与追踪
    parser.add_argument("--out_dir", type=str, default="results", help="output directory for logs and figures")

    # 可选：设备
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")

    args = parser.parse_args()

    device = torch.device(args.device)

    ensure_dir(args.out_dir)
    run_id = make_run_id(args)

    print()
    print("run_id                 ", run_id)
    print("experiment             ", args.experiment)
    print("degree                 ", args.degree)
    print("dimension              ", args.dimension)
    print("test_size              ", args.test_size)
    print("batch_size             ", args.batch_size)
    print("transformation         ", args.transformation)
    print("run_optimal            ", args.run_optimal)
    print("plot_first             ", args.plot_first)
    print("seed                   ", args.seed)
    print("point_dist             ", args.point_dist)
    print("nonuniform             ", args.nonuniform)
    print("clusters               ", args.clusters)
    print("sigma                  ", args.sigma)
    print("device                 ", device)
    print("out_dir                ", args.out_dir)
    print()

    base_dir = "save/"
    exp_dir = os.path.join(base_dir, args.experiment)
    ckp_path = os.path.join(exp_dir, f"rsmt{args.degree}b.pt")

    if not os.path.exists(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")

    checkpoint = load_checkpoint_safely(ckp_path, device)

    actor = Actor(args.degree, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    evaluator = Evaluator()

    # 读取或生成测试数据
    if args.test_data and os.path.exists(args.test_data):
        test_cases = read_data(args.test_data)
    else:
        test_cases = generate_terminals(
            n_nets=args.test_size,
            degree=args.degree,
            dim=args.dimension,
            seed=args.seed,
            point_dist=args.point_dist,
            nonuniform=args.nonuniform,
            clusters=args.clusters,
            sigma=args.sigma,
        )

    # -----------------------------
    # Plot terminals (4x4 grid)
    # -----------------------------
    points_fig_path = None
    if args.plot_points.lower() == "true":
        points_fig_path = os.path.join(args.out_dir, f"viz_points_{run_id}.png")
        plot_nets_points_grid(test_cases, points_fig_path, n_show=args.plot_points_n)
        print("Saved points visualization:", points_fig_path)
        print()

    # -----------------------------
    # Inference
    # -----------------------------
    num_batches = (args.test_size + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    if args.transformation <= 1:
        all_outputs = []
        for b in range(num_batches):
            test_batch = test_cases[b * args.batch_size : (b + 1) * args.batch_size]
            with torch.no_grad():
                outputs, _ = actor(test_batch, True)
            all_outputs.append(outputs.cpu().detach().numpy())

        inference_time = time.time() - start_time
        all_outputs = np.concatenate(all_outputs, 0)
        all_lengths = evaluator.eval_batch(test_cases, all_outputs, args.degree)

    else:
        inference_time = 0.0
        all_lengths = []
        all_outputs = []

        for b in range(num_batches):
            test_batch = test_cases[b * args.batch_size : (b + 1) * args.batch_size]

            best_lengths = [1e9 for _ in range(len(test_batch))]
            best_outputs = [[] for _ in range(len(test_batch))]

            for t in range(args.transformation):
                transformed_batch = transform_inputs(test_batch, t)

                ttime = time.time()
                with torch.no_grad():
                    outputs, _ = actor(transformed_batch, True)
                inference_time += time.time() - ttime

                outputs = outputs.cpu().detach().numpy()
                lengths = evaluator.eval_batch(transformed_batch, outputs, args.degree)

                if t >= 4:
                    outputs = np.flip(outputs, 1)

                for i in range(len(test_batch)):
                    if lengths[i] < best_lengths[i]:
                        best_lengths[i] = lengths[i]
                        best_outputs[i] = outputs[i]

            all_lengths.append(best_lengths)
            all_outputs.append(best_outputs)

        all_lengths = np.concatenate(all_lengths, 0)
        all_outputs = np.concatenate(all_outputs, 0)

    full_time = time.time() - start_time

    rest_mean = float(all_lengths.mean())
    rest_first = float(all_lengths[0])

    print("REST mean length       ", round(rest_mean, 6))
    print("inference time         ", round(inference_time, 3))
    print("   full   time         ", round(full_time, 3))
    print()

    # -----------------------------
    # GeoSteiner (optional)
    # -----------------------------
    gst_mean = None
    gst_time = None
    rest_pct_err = None

    if args.run_optimal.lower() == "true":
        gst_start_time = time.time()
        gst_lengths = []
        for test_case in test_cases:
            gst_length, _, _ = evaluator.gst_rsmt(test_case)
            gst_lengths.append(gst_length)

        gst_time = time.time() - gst_start_time
        gst_lengths = np.array(gst_lengths)

        gst_mean = float(gst_lengths.mean())
        rest_pct_err = float(((all_lengths / gst_lengths).mean() - 1.0) * 100.0)

        print("GeoSteiner mean length ", round(gst_mean, 6))
        print("GeoSteiner time        ", round(gst_time, 3))
        print()
        print("REST percentage error  ", f"{round(rest_pct_err, 3)}%")
        print()

    # -----------------------------
    # Plot first RSMT vs REST (optional)
    # -----------------------------
    rsmt_fig_path = None
    if args.plot_first.lower() == "true":
        rsmt_fig_path = os.path.join(args.out_dir, f"viz_rsmt_{run_id}.pdf")

        fig = plt.figure(figsize=(10, 4.6))
        plt.subplot(1, 2, 1)
        gst_length, sps, edges = evaluator.gst_rsmt(test_cases[0])
        plot_gst_rsmt(test_cases[0], sps, edges)
        plt.annotate("Optimal " + str(round(gst_length, 3)), (-0.04, -0.04))

        plt.subplot(1, 2, 2)
        plot_rest(test_cases[0], all_outputs[0])
        plt.annotate("REST " + str(round(rest_first, 3)), (-0.04, -0.04))

        fig.savefig(rsmt_fig_path)
        plt.close(fig)
        print("Saved comparison figure:", rsmt_fig_path)
        print()

    # -----------------------------
    # Save run record (jsonl)
    # -----------------------------
    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "metrics": {
            "rest_mean_length": rest_mean,
            "rest_first_length": rest_first,
            "inference_time_sec": float(inference_time),
            "full_time_sec": float(full_time),
            "geosteiner_mean_length": gst_mean,
            "geosteiner_time_sec": None if gst_time is None else float(gst_time),
            "rest_percentage_error": rest_pct_err,
        },
        "artifacts": {
            "points_fig": points_fig_path,
            "rsmt_fig": rsmt_fig_path,
        },
    }

    jsonl_path = os.path.join(args.out_dir, "runs.jsonl")
    append_jsonl(jsonl_path, record)
    print("Saved run log (append):", jsonl_path)
    print()


if __name__ == "__main__":
    main()