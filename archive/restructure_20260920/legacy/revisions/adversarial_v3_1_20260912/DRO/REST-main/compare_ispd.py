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

# 新增：tqdm 进度条，若未安装则回退
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_ispd_merged_file(path: str):
    """
    解析 benchmark/merged 下的文件格式，返回 numpy array (N, degree, dim)
    文件格式参见用户示例：第一行为 net 数量，随后若干 net 块：
      net <name> <id> <deg>
      x y
      ...
      END
    仅收集解析成功且点数等于 header 中 deg 的 net。
    """
    nets = []
    degs = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]
    if not lines:
        return np.zeros((0, 0, 2), dtype=np.float32)
    i = 0
    # 第一行可能是总数，跳过
    try:
        int(lines[0].split()[0])
        i = 1
    except Exception:
        i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line.startswith("net "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    deg = int(parts[-1])
                except ValueError:
                    # 找不到 degree，就跳过该 net
                    i += 1
                    continue
            else:
                i += 1
                continue
            i += 1
            pts = []
            while i < len(lines) and lines[i].strip() != "END":
                tok = lines[i].strip().split()
                if len(tok) >= 2:
                    try:
                        x = float(tok[0])
                        y = float(tok[1])
                        pts.append([x, y])
                    except Exception:
                        pass
                i += 1
            # 跳过 "END"
            i += 1
            if len(pts) == deg:
                nets.append(pts)
                degs.append(deg)
            else:
                # 如果点数不匹配，尝试跳过该网
                continue
        else:
            i += 1
    if not nets:
        return np.zeros((0, 0, 2), dtype=np.float32)
    arr = np.array(nets, dtype=np.float32)
    return arr


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_run_id(prefix: str, degree: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"{ts}_ispd_{prefix}_deg{degree}"
    return rid[:220]


def load_checkpoint_safely(ckp_path: str, device: torch.device):
    try:
        return torch.load(ckp_path, weights_only=False, map_location=device)
    except TypeError:
        try:
            with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                return torch.load(ckp_path, map_location=device)
        except Exception:
            return torch.load(ckp_path, map_location=device)


def plot_nets_points_grid(test_cases: np.ndarray, save_path: str, n_show: int = 16):
    n_show = min(n_show, len(test_cases))
    rows, cols = 4, 4
    fig = plt.figure(figsize=(12, 12))
    for idx in range(rows * cols):
        ax = plt.subplot(rows, cols, idx + 1)
        if idx >= n_show:
            ax.axis("off")
            continue
        pts = test_cases[idx]
        if pts.shape[1] >= 2:
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


def scale_translate_nets(nets: np.ndarray, low: float = 0.1, high: float = 0.9) -> np.ndarray:
    """
    对每个 net 单独做缩放和平移，使其最小坐标映射到 >= low，
    且最大坐标不超过 high。保持横纵比不变，保留相对位置。
    nets: (N, deg, 2)
    """
    if nets.size == 0:
        return nets
    out = []
    span = float(high) - float(low)
    for pts in nets:
        xs = pts[:, 0]
        ys = pts[:, 1]
        min_x, min_y = float(xs.min()), float(ys.min())
        max_x, max_y = float(xs.max()), float(ys.max())
        w = max_x - min_x
        h = max_y - min_y
        if w <= 0 and h <= 0:
            center = low + span * 0.5
            new = np.stack([np.full_like(xs, center), np.full_like(ys, center)], axis=1)
            out.append(new)
            continue
        scales = []
        if w > 0:
            scales.append(span / w)
        if h > 0:
            scales.append(span / h)
        scale = min(scales) if scales else 1.0
        norm = (pts - np.array([min_x, min_y])) * scale + low
        out.append(norm)
    return np.array(out, dtype=nets.dtype)


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_baseline_jsonl(path: str, prefer_point_dist: str = None):
    """
    解析 baseline jsonl，返回 dict degree -> rest_percentage_error
    选择 nonuniform == 0.0 的记录，若有多条按 prefer_point_dist 优先，否则取第一条。
    """
    if not os.path.exists(path):
        return {}
    best = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            deg = rec.get("degree")
            nu = rec.get("nonuniform", None)
            pd = rec.get("point_dist", None)
            err = rec.get("rest_percentage_error", None)
            if deg is None or err is None:
                continue
            if nu != 0.0:
                continue
            # prefer matching point_dist
            if deg not in best:
                best[deg] = (pd, err)
            else:
                cur_pd, cur_err = best[deg]
                if prefer_point_dist and pd == prefer_point_dist and cur_pd != prefer_point_dist:
                    best[deg] = (pd, err)
    # reduce to degree -> err
    return {d: v[1] for d, v in best.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", type=str, default="../benchmark/merged", help="merged folder")
    parser.add_argument("--degrees", type=str, default="5,10,15,20,25,30,35,40,45,50", help="comma separated degrees to test")
    parser.add_argument("--experiment", type=str, default="DAC21", help="experiment name (for checkpoint path)")
    parser.add_argument("--dimension", type=int, default=2, help="terminal representation dimension")
    parser.add_argument("--batch_size", type=int, default=100, help="test batch size")
    parser.add_argument("--transformation", type=int, default=1, help="number of transformations for inference")
    parser.add_argument("--run_optimal", type=str, default="true", help="run GeoSteiner to generate optimal RSMT")
    parser.add_argument("--plot_first", type=str, default="true", help="plot the first result (gst vs rest)")
    parser.add_argument("--plot_points", type=str, default="true", help="visualize terminals distribution")
    parser.add_argument("--plot_points_n", type=int, default=16, help="how many nets to plot")
    parser.add_argument("--out_dir", type=str, default="results_ispd", help="output directory for logs and figures")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--baseline_jsonl", type=str, default="", help="optional baseline jsonl (one-record-per-line)")
    parser.add_argument("--baseline_point_dist", type=str, default="mix_gauss", help="preferred point_dist in baseline")
    parser.add_argument("--scale_nets", action="store_true", help="per-net scale+translate into [padding,1-padding] box")
    parser.add_argument("--scale_padding", type=float, default=0.1, help="padding (low) when scaling nets; high = 1-padding")
    args = parser.parse_args()

    device = torch.device(args.device)
    ensure_dir(args.out_dir)

    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]

    # 使用用户提供的“完全 random (nonuniform=0.0)”数据作为 baseline，明确写出 nonuniform==0.0 的 rest_percentage_error
    baseline_map = {
        5: 0.014474156800825533,
        10: 0.22751388779964632,
        15: 0.4399038675264366,
        20: 0.5493583779770939,
        25: 0.6953123096313352,
        30: 0.768287684018043,
        35: 0.8286849597945567,
        40: 0.8642311616426035,
        45: 0.9805964124163458,
        50: 1.0132794000135004,
    }

    degrees_done = []
    industrial_errors = []
    baseline_errors = []

    # degrees 迭代增加进度显示
    for deg in tqdm(degrees, desc="degrees", unit="deg"):
        fname = os.path.join(args.benchmark_dir, f"degree_{deg}.txt")
        if not os.path.exists(fname):
            print(f"File not found, skip: {fname}")
            continue

        test_cases = parse_ispd_merged_file(fname)
        if test_cases.size == 0:
            print(f"No valid nets parsed in {fname}, skip")
            continue

        # 可选：对每个 net 单独做缩放和平移，使坐标落在 [padding, 1-padding] 区间内
        if getattr(args, "scale_nets", False):
            pad = float(getattr(args, "scale_padding", 0.1))
            low = pad
            high = 1.0 - pad
            test_cases = scale_translate_nets(test_cases, low=low, high=high)
            print(f"Applied per-net scale+translate with padding={pad}")

        test_size = len(test_cases)
        run_id = make_run_id(os.path.basename(args.benchmark_dir), deg)
        print()
        print("run_id     ", run_id)
        print("degree     ", deg)
        print("n_nets     ", test_size)
        print("device     ", device)
        print("out_dir    ", args.out_dir)
        print()

        # load model checkpoint
        base_dir = "save/"
        exp_dir = os.path.join(base_dir, args.experiment)
        ckp_path = os.path.join(exp_dir, f"rsmt{deg}b.pt")
        if not os.path.exists(ckp_path):
            print(f"Checkpoint not found: {ckp_path}, skip degree {deg}")
            continue

        checkpoint = load_checkpoint_safely(ckp_path, device)
        actor = Actor(deg, device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()
        evaluator = Evaluator()

        # plot points
        points_fig_path = None
        if args.plot_points.lower() == "true":
            points_fig_path = os.path.join(args.out_dir, f"viz_points_{run_id}.png")
            plot_nets_points_grid(test_cases, points_fig_path, n_show=args.plot_points_n)
            print("Saved points visualization:", points_fig_path)

        # inference
        num_batches = (test_size + args.batch_size - 1) // args.batch_size
        start_time = time.time()
        all_outputs = []
        inference_time = 0.0

        if args.transformation <= 1:
            # batch 级别进度条
            for b in tqdm(range(num_batches), desc=f"deg{deg} batches", unit="batch"):
                batch = test_cases[b * args.batch_size : (b + 1) * args.batch_size]
                with torch.no_grad():
                    outputs, _ = actor(batch, True)
                all_outputs.append(outputs.cpu().detach().numpy())
            inference_time = time.time() - start_time
            all_outputs = np.concatenate(all_outputs, 0)
            all_lengths = evaluator.eval_batch(test_cases, all_outputs, deg)
        else:
            all_lengths = []
            all_outputs = []
            # outer批次进度
            for b in tqdm(range(num_batches), desc=f"deg{deg} batches", unit="batch"):
                batch = test_cases[b * args.batch_size : (b + 1) * args.batch_size]
                best_lengths = [1e9 for _ in range(len(batch))]
                best_outputs = [[] for _ in range(len(batch))]
                for t in range(args.transformation):
                    transformed = transform_inputs(batch, t)
                    t0 = time.time()
                    with torch.no_grad():
                        outputs, _ = actor(transformed, True)
                    inference_time += time.time() - t0
                    outputs = outputs.cpu().detach().numpy()
                    lengths = evaluator.eval_batch(transformed, outputs, deg)
                    if t >= 4:
                        outputs = np.flip(outputs, 1)
                    for i in range(len(batch)):
                        if lengths[i] < best_lengths[i]:
                            best_lengths[i] = lengths[i]
                            best_outputs[i] = outputs[i]
                all_lengths.extend(best_lengths)
                all_outputs.extend(best_outputs)
            all_lengths = np.array(all_lengths)
            all_outputs = np.array(all_outputs)

        full_time = time.time() - start_time
        rest_mean = float(all_lengths.mean())
        rest_first = float(all_lengths[0])
        print("REST mean length       ", round(rest_mean, 6))
        print("inference time         ", round(inference_time, 3))
        print("full time              ", round(full_time, 3))

        # optional GeoSteiner comparison
        gst_mean = None
        gst_time = None
        rest_pct_err = None
        per_case_err_mean = None
        per_case_err_std = None
        gst_lengths = None
        if args.run_optimal.lower() == "true":
            gst_start = time.time()
            gst_lengths = []
            # 对每个 net 增加进度
            for tc in tqdm(test_cases, desc=f"deg{deg} GST", unit="net"):
                gst_len, _, _ = evaluator.gst_rsmt(tc)
                gst_lengths.append(gst_len)
            gst_time = time.time() - gst_start
            gst_lengths = np.array(gst_lengths)
            gst_mean = float(gst_lengths.mean())
            # 计算每个 test case 的相对误差 (rest - gst)/gst * 100%，然后取平均（用户要求）
            with np.errstate(divide="ignore", invalid="ignore"):
                per_case_err = (all_lengths - gst_lengths) / gst_lengths * 100.0
                # 对于 gst_lengths == 0，标记为 nan
                per_case_err = np.where(np.isfinite(per_case_err), per_case_err, np.nan)
            per_case_err_mean = float(np.nanmean(per_case_err))
            per_case_err_std = float(np.nanstd(per_case_err))
            rest_pct_err = per_case_err_mean
            print("GeoSteiner mean length ", round(gst_mean, 6))
            print("GeoSteiner time        ", round(gst_time, 3))
            print("REST mean pct error    ", f"{round(rest_pct_err,3)}% (std {round(per_case_err_std,3)})")
        else:
            print("GeoSteiner comparison skipped (run_optimal != true).")

        # optionally plot first net comparison
        rsmt_fig_path = None
        if args.plot_first.lower() == "true" and gst_lengths is not None:
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

        # save record
        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "degree": deg,
            "n_nets": test_size,
            "metrics": {
                "rest_mean_length": rest_mean,
                "rest_first_length": rest_first,
                "inference_time_sec": float(inference_time),
                "full_time_sec": float(full_time),
                "geosteiner_mean_length": gst_mean,
                "geosteiner_time_sec": None if gst_time is None else float(gst_time),
                # 主要按 per-case average error 保存
                "rest_percentage_error": rest_pct_err,
                "rest_percentage_error_std": per_case_err_std,
            },
            "artifacts": {"points_fig": points_fig_path, "rsmt_fig": rsmt_fig_path},
        }
        jsonl_path = os.path.join(args.out_dir, "ispd_runs.jsonl")
        append_jsonl(jsonl_path, record)
        print("Saved run log (append):", jsonl_path)
        print()

        # collect for final plot
        degrees_done.append(deg)
        industrial_errors.append(rest_pct_err if rest_pct_err is not None else np.nan)
        baseline_errors.append(baseline_map.get(deg, np.nan))

    # 绘制 degree vs error 比较图
    if degrees_done:
        order = np.argsort(degrees_done)
        xs = np.array(degrees_done)[order]
        ys_ind = np.array(industrial_errors)[order].astype(float)
        ys_base = np.array(baseline_errors)[order].astype(float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys_ind, marker="o", label="industrial (merged)")
        if np.any(~np.isnan(ys_base)):
            ax.plot(xs, ys_base, marker="s", label="random (nonuniform=0.0, provided)")
        ax.set_xlabel("degree")
        ax.set_ylabel("REST average % error vs GeoSteiner")
        ax.set_title("Degree vs REST % error")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_plot = os.path.join(args.out_dir, "degree_vs_error.png")
        fig.savefig(out_plot, dpi=200)
        plt.close(fig)
        print("Saved degree vs error plot:", out_plot)
    else:
        print("No degrees processed; skip final plot.")


if __name__ == "__main__":
    main()