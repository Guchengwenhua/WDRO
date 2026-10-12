"""Train the adaptive Selector-Conductor model from scratch."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.evaluate import evaluate_batch
from models.selector_conductor import Actor, Critic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="adaptive-hub-selection")
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=100_000)
    parser.add_argument("--max-hubs", type=int, default=5)
    parser.add_argument(
        "--objective",
        type=float,
        default=2,
        help="1/2/3 for the paper presets, or the wirelength weight in [0,1]",
    )
    parser.add_argument("--learning-rate", type=float, default=9e-6)
    parser.add_argument("--reward-gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--log-interval", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.degree < 2:
        raise ValueError("degree must be at least 2")
    if not 1 <= args.max_hubs <= args.degree - 1:
        raise ValueError("max-hubs must be between 1 and degree - 1")

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{args.experiment}-d{args.degree}-h{args.max_hubs}-"
        f"obj{args.objective:g}-seed{args.seed}"
    )
    checkpoint_path = args.output_dir / f"{run_name}.pt"
    writer = SummaryWriter(args.output_dir / "tensorboard" / run_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    actor = Actor(args.degree, device, args.reward_gamma)
    critic = Critic(args.degree, device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )
    mse_loss = torch.nn.MSELoss()
    eval_cases = np.random.rand(args.eval_size, args.degree, 2)
    best_eval = math.inf
    start_time = time.time()

    for batch_index in range(1, args.num_batches + 1):
        actor.train()
        critic.train()
        input_batch = np.random.rand(args.batch_size, args.degree, 2)
        selected, log_probs, intermediate_reward = actor(
            input_batch, args.max_hubs
        )
        predictions = critic(input_batch)
        _, _, values = evaluate_batch(
            input_batch, selected.detach().cpu().numpy(), args.objective
        )
        train_mean = float(values.mean())
        value_tensor = torch.as_tensor(values, device=device)

        with torch.no_grad():
            disadvantage = value_tensor - predictions - intermediate_reward
        actor_loss = torch.mean(disadvantage * log_probs)
        critic_loss = mse_loss(predictions, value_tensor)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()

        writer.add_scalar("train/actor_loss", actor_loss.item(), batch_index)
        writer.add_scalar("train/critic_loss", critic_loss.item(), batch_index)
        writer.add_scalar("train/objective", train_mean, batch_index)

        if batch_index % args.log_interval != 0:
            continue

        actor.eval()
        eval_values = []
        for start in range(0, args.eval_size, args.batch_size):
            eval_batch = eval_cases[start : start + args.batch_size]
            with torch.no_grad():
                selected, _, _ = actor(eval_batch, args.max_hubs, True)
            _, _, values = evaluate_batch(
                eval_batch, selected.cpu().numpy(), args.objective
            )
            eval_values.extend(values.tolist())

        eval_mean = float(np.mean(eval_values))
        writer.add_scalar("eval/objective", eval_mean, batch_index)
        print(
            f"batch={batch_index} train={train_mean:.6f} "
            f"eval={eval_mean:.6f} elapsed={time.time() - start_time:.1f}s"
        )
        if eval_mean < best_eval:
            best_eval = eval_mean
            torch.save(
                {
                    "batch_index": batch_index,
                    "best_eval": best_eval,
                    "args": vars(args),
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"saved {checkpoint_path}")

    writer.close()


if __name__ == "__main__":
    main()
