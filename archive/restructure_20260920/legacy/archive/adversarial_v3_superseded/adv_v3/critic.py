"""Regret surrogates and input-gradient proposals for adversarial v3.

The raw adapter exactly preserves the legacy computation except for the input
copy that detached coordinates.  A calibrated adapter uses a linear final head;
its predictions are proposals, never a replacement for the real oracle score.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from torch import nn


def _common():
    return importlib.import_module("adv_v3.common")


class InputCritic(nn.Module):
    """Independent legacy Critic copy with an input-autograd-preserving entry."""

    def __init__(self, parent_critic, calibrated=False):
        super().__init__()
        source = parent_critic.critic if isinstance(parent_critic, InputCritic) else parent_critic
        self.critic = copy.deepcopy(source)
        self.calibrated = bool(calibrated)
        self.eval()

    def forward(self, inputs, deterministic=False):
        parameter = next(self.parameters())
        if torch.is_tensor(inputs):
            points = inputs.to(device=parameter.device, dtype=parameter.dtype)
        else:
            points = torch.as_tensor(inputs, device=parameter.device, dtype=parameter.dtype)
        encoded = self.critic.crit_encoder(self.critic.crit_embedder(points), None)
        hidden = self.critic.relu(self.critic.critic_l1(self.critic.glimpse(encoded)))
        prediction = self.critic.critic_l2(hidden).squeeze(-1)
        return prediction if self.calibrated else self.critic.relu(prediction)


def matched_l1(centers, points):
    """Exact Hungarian value with the current matching's Torch L1 branch.

    The matching choice is deliberately outside autograd.  This is the local
    derivative where the active branch is unique and coordinate signs are fixed;
    ties/zero differences use the selected branch and Torch's abs convention.
    """
    if centers.shape != points.shape or points.ndim != 3:
        raise ValueError("matched_l1 expects equal [B,n,2] tensors")
    left = centers.detach().cpu().numpy().astype(np.float64)
    right = points.detach().cpu().numpy().astype(np.float64)
    distances = np.abs(left[:, :, None] - right[:, None, :]).sum(-1)
    columns = []
    for matrix in distances:
        rows, cols = linear_sum_assignment(matrix)
        if not np.array_equal(rows, np.arange(len(rows))):
            raise RuntimeError("unexpected Hungarian row ordering")
        columns.append(cols)
    indices = torch.as_tensor(np.asarray(columns), device=points.device, dtype=torch.long)
    selected = points.gather(1, indices[..., None].expand(-1, -1, 2))
    return (centers - selected).abs().sum(-1).mean(-1)


def gradient_diagnostics(model, pins, batch_size=128):
    """Input-gradient magnitudes in deterministic eval mode, without training."""
    was_training = model.training
    model.eval()
    parameter = next(model.parameters())
    norms, values = [], []
    try:
        for start in range(0, len(pins), batch_size):
            z = torch.tensor(pins[start:start + batch_size], device=parameter.device,
                             dtype=parameter.dtype, requires_grad=True)
            with torch.enable_grad():
                prediction = model(z)
                grad = torch.autograd.grad(prediction.sum(), z)[0]
            if not torch.isfinite(grad).all() or not torch.isfinite(prediction).all():
                raise RuntimeError("nonfinite surrogate diagnostic")
            norms.extend(torch.linalg.vector_norm(grad.flatten(1), dim=1).detach().cpu().tolist())
            values.extend(prediction.detach().cpu().tolist())
    finally:
        model.train(was_training)
    norms, values = np.asarray(norms), np.asarray(values)
    return {"nets": len(norms), "zero_gradient_fraction": float(np.mean(norms == 0)),
            "zero_prediction_fraction": float(np.mean(values == 0)),
            "gradient_l2_mean": float(norms.mean()), "gradient_l2_min": float(norms.min()),
            "gradient_l2_max": float(norms.max())}


def propose(model, centers, price, count, seed):
    """Return count restart endpoints, without prepending X or calling an oracle.

    Each restart uses eight RMS-normalized steps of 0.01, followed by the common
    legal map, with pre-normalization displacement bounded by 0.08.  The map can
    change final displacement; it is not a per-net transport-budget constraint.
    """
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 3 or centers.shape[-1] != 2 or int(count) < 1:
        raise ValueError("invalid proposal shape/count")
    count = int(count)
    prices = np.broadcast_to(np.asarray(price, dtype=np.float64), (len(centers),))
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("price must be finite and nonnegative")
    common = _common()
    rng = np.random.default_rng(seed)
    # Canonical search coordinates make the fixed-seed trajectories equivariant
    # to pin permutations, including the Gaussian restart allocation.
    orders = [np.lexsort((center[:, 1], center[:, 0])) for center in centers]
    canonical = np.stack([center[order] for center, order in zip(centers, orders)])
    repeated = np.repeat(canonical, count, axis=0)
    repeated_price = np.repeat(prices, count)
    initial = np.clip(rng.normal(0, 0.02, size=repeated.shape), -0.08, 0.08)
    result = np.empty_like(repeated)
    was_training = model.training
    requires = [p.requires_grad for p in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter = next(model.parameters())
    try:
        # Bound attention activation memory independently of the restart count.
        for start in range(0, len(repeated), 128):
            x_np = repeated[start:start + 128]
            x = torch.as_tensor(x_np, device=parameter.device, dtype=parameter.dtype)
            lam = torch.as_tensor(repeated_price[start:start + 128], device=parameter.device,
                                  dtype=parameter.dtype)
            candidate = common.legalize(x_np, initial[start:start + 128])
            for _ in range(8):
                z = torch.tensor(candidate, device=parameter.device,
                                 dtype=parameter.dtype, requires_grad=True)
                with torch.enable_grad():
                    score = model(z) - lam * matched_l1(x, z)
                    gradient = torch.autograd.grad(score.sum(), z)[0]
                if not torch.isfinite(gradient).all() or not torch.isfinite(score).all():
                    raise RuntimeError("nonfinite Critic input gradient")
                rms = gradient.square().mean(dim=(1, 2), keepdim=True).sqrt()
                direction = gradient / rms.clamp_min(1e-12)
                moved = (z + 0.01 * direction).detach().cpu().numpy()
                delta = np.clip(moved - x_np, -0.08, 0.08)
                candidate = common.legalize(x_np, delta)
            result[start:start + len(x_np)] = candidate
    finally:
        for p, flag in zip(model.parameters(), requires):
            p.requires_grad_(flag)
        model.train(was_training)
    canonical_result = result.reshape(len(centers), count, *centers.shape[1:])
    restored = np.empty_like(canonical_result)
    for index, order in enumerate(orders):
        restored[index][:, order] = canonical_result[index]
    return restored


def _state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _validation(model, pins, targets, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(pins), batch_size):
            predictions.extend(model(pins[start:start + batch_size]).cpu().tolist())
    prediction = np.asarray(predictions)
    if not np.isfinite(prediction).all():
        raise RuntimeError("nonfinite surrogate validation")
    rank_p, rank_y = rankdata(prediction), rankdata(targets)
    correlation = (float(np.corrcoef(rank_p, rank_y)[0, 1])
                   if rank_p.std() > 0 and rank_y.std() > 0 else None)
    return {"mse": float(np.mean(np.square(prediction - targets))), "spearman": correlation}


def train_surrogate(oracle, parent_critic, degree, seed, outdir, config,
                    resume=True, heartbeat=None):
    """Calibrate a separate linear-headed Critic using real greedy labels.

    Label chunks are atomically retained.  Training resumes at a completed epoch
    with optimizer/RNG state; the oracle's persistent actual ledger is never
    rolled back.  No evaluation/shift labels participate in model selection.
    """
    common = _common()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    settings = {"train_count": 8192, "validation_count": 1024, "epochs": 16,
                "batch_size": 128, "learning_rate": 1e-4, "label_microbatch": 128}
    settings.update({key: config[key] for key in settings if key in config})
    if "lr" in config:
        settings["learning_rate"] = config["lr"]
    if any(int(settings[key]) <= 0 for key in settings if key != "learning_rate"):
        raise ValueError("surrogate counts/epochs/batches must be positive")
    namespace = str(config.get("namespace", "adversarial_v3_critic"))
    identity = {"schema_version": 1, "degree": int(degree), "seed": int(seed),
                "namespace": namespace, "settings": settings,
                "parent_critic_sha256": _state_hash(parent_critic),
                "label_actor_sha256": (_state_hash(oracle.actor) if hasattr(oracle, "actor") else None),
                "output_head": "linear"}
    labels_path, latest_path, best_path = (outdir / name for name in
                                          ("labels.pt", "latest.pt", "best.pt"))
    if not resume and any(p.exists() for p in (labels_path, latest_path, best_path)):
        raise FileExistsError("refusing to overwrite an existing surrogate run")
    if labels_path.exists():
        labels = torch.load(labels_path, map_location="cpu", weights_only=False)
        if labels["identity"] != identity:
            raise ValueError("surrogate label identity mismatch")
    else:
        labels = {"identity": identity, "completed": 0, "label_requests": 0}
        for split, count_key in (("train", "train_count"), ("validation", "validation_count")):
            pins = common.make_data(degree, int(settings[count_key]), "uniform",
                                    namespace=f"{namespace}_{split}_d{degree}_s{seed}", seed=seed)
            labels[f"{split}_pins"] = torch.from_numpy(np.asarray(pins, dtype=np.float32))
        total = int(settings["train_count"]) + int(settings["validation_count"])
        labels["regret"] = torch.full((total,), float("nan"), dtype=torch.float64)
        common.atomic_torch(labels_path, labels)
    all_pins = torch.cat((labels["train_pins"], labels["validation_pins"])).numpy()
    microbatch = int(settings["label_microbatch"])
    for start in range(int(labels["completed"]), len(all_pins), microbatch):
        stop = min(start + microbatch, len(all_pins))
        answer = oracle.evaluate(all_pins[start:stop])
        regret = np.asarray(answer["regret"], dtype=np.float64)
        if regret.shape != (stop - start,) or not np.isfinite(regret).all():
            raise RuntimeError("invalid greedy surrogate labels")
        labels["regret"][start:stop] = torch.from_numpy(regret)
        labels["completed"] = stop
        labels["label_requests"] += stop - start
        labels["oracle_ledger_snapshot"] = dict(oracle.ledger)
        common.atomic_torch(labels_path, labels)
        if heartbeat:
            heartbeat()
    ntrain = int(settings["train_count"])
    train_pins, val_pins = all_pins[:ntrain], all_pins[ntrain:]
    train_y, val_y = labels["regret"][:ntrain].numpy(), labels["regret"][ntrain:].numpy()
    model = InputCritic(parent_critic, calibrated=True).to(oracle.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    start_epoch, best_mse, history = 0, float("inf"), []
    if latest_path.exists():
        state = torch.load(latest_path, map_location=oracle.device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("surrogate checkpoint identity mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _restore_rng(state["rng_state"])
        start_epoch, best_mse, history = state["epoch"], state["best_mse"], state["history"]
    else:
        torch.manual_seed(int(seed))
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(int(seed))
    batch_size = int(settings["batch_size"])
    for epoch in range(start_epoch, int(settings["epochs"])):
        model.train()
        order = np.random.default_rng(int(seed) + 1000003 * (epoch + 1)).permutation(ntrain)
        total_loss = 0.0
        for start in range(0, ntrain, batch_size):
            indices = order[start:start + batch_size]
            target = torch.as_tensor(train_y[indices], device=oracle.device, dtype=torch.float32)
            prediction = model(train_pins[indices])
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite surrogate training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
            if heartbeat:
                heartbeat()
        validation = _validation(model, val_pins, val_y, batch_size)
        history.append({"epoch": epoch + 1, "train_mse": total_loss / ntrain, **validation})
        improved = validation["mse"] < best_mse
        best_mse = min(best_mse, validation["mse"])
        state = {"identity": identity, "epoch": epoch + 1,
                 "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "rng_state": _rng_state(), "best_mse": best_mse, "history": history,
                 "label_requests": labels["label_requests"], "oracle_ledger_snapshot": dict(oracle.ledger)}
        # Publish best before latest, so a published latest always has a best.
        if improved:
            common.atomic_torch(best_path, state)
        common.atomic_torch(latest_path, state)
    best = torch.load(best_path, map_location=oracle.device, weights_only=False)
    if best["identity"] != identity:
        raise ValueError("surrogate best identity mismatch")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    raw = InputCritic(parent_critic, calibrated=False).to(oracle.device)
    summary = {"status": "complete", "identity": identity, "best_epoch": best["epoch"],
               "validation": _validation(model, val_pins, val_y, batch_size),
               "raw_validation": _validation(raw, val_pins, val_y, batch_size),
               "raw_gradient": gradient_diagnostics(raw, val_pins, batch_size),
               "calibrated_gradient": gradient_diagnostics(model, val_pins, batch_size),
               "label_requests": labels["label_requests"], "label_distribution": "uniform_normalized",
               "oracle_ledger_snapshot": dict(oracle.ledger), "history": history,
               "best_path": str(best_path), "latest_path": str(latest_path), "labels_path": str(labels_path)}
    common.atomic_json(outdir / "summary.json", summary)
    return model, summary


@torch.enable_grad()
def calibrate_online(model, optimizer, pins, regret):
    """One deterministic minibatch update from already-paid real greedy labels."""
    if isinstance(model, InputCritic) and not model.calibrated:
        raise ValueError("online calibration requires the independent calibrated surrogate")
    parameter = next(model.parameters())
    targets = torch.as_tensor(regret, device=parameter.device, dtype=parameter.dtype)
    if targets.shape != (len(pins),) or not torch.isfinite(targets).all():
        raise ValueError("invalid online regret labels")
    # Keep running statistics and dropout fixed during the short online update.
    model.eval()
    prediction = model(pins)
    loss = (prediction - targets).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite online surrogate loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = (model(pins) - targets).square().mean()
    return {"nets": len(pins), "updates": 1, "mse_before": float(loss.detach().cpu()),
            "mse_after": float(after.cpu()), "real_oracle_calls": 0}
