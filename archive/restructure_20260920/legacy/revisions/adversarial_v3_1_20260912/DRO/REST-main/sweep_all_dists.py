import os
import argparse
import subprocess


DISTS = [
    "mix_gauss",
    "anisotropic_gauss",
    "hotspots_with_void",
    "edge_bias",
    "grid_cluster",
]


def run(cmd: list):
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str, default="DAC21")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default="results_sweep_all")

    # sweep settings
    parser.add_argument("--nonuniform_list", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.04)

    # eval settings
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--transformation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)

    # behavior
    parser.add_argument("--resume", type=str, default="true")
    parser.add_argument("--run_optimal", type=str, default="true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for dist in DISTS:
        # 为不同分布单独放一个子目录，文件不会混在一起
        dist_out = os.path.join(args.out_dir, dist)
        os.makedirs(dist_out, exist_ok=True)

        # anisotropic 通常 sigma 稍大一点更明显；你也可以不想改就删掉下面这段
        sigma = args.sigma
        if dist == "anisotropic_gauss":
            sigma = max(args.sigma, 0.05)

        cmd = [
            "python",
            "sweep_robustness.py",
            "--experiment", args.experiment,
            "--point_dist", dist,
            "--nonuniform_list", args.nonuniform_list,
            "--clusters", str(args.clusters),
            "--sigma", str(sigma),
            "--test_size", str(args.test_size),
            "--batch_size", str(args.batch_size),
            "--dimension", str(args.dimension),
            "--transformation", str(args.transformation),
            "--seed", str(args.seed),
            "--device", args.device,
            "--out_dir", dist_out,
            "--resume", args.resume,
            "--run_optimal", args.run_optimal,
        ]
        run(cmd)

    print("\nAll distributions finished.")
    print("Check outputs under:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()