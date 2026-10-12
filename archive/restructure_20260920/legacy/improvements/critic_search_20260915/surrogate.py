"""Isolated, auditable data/target ablations for tmp2 directions 1 and 2.

The actor remains fixed. Every new target is a real greedy-regret label. The
linear-headed v3.1 InputCritic is copied rather than changing legacy source.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parents[1] / "revisions/adversarial_v3_1_20260912/Codex"
NAMESPACE = "critic_search_20260915"


def _legacy():
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))
    return importlib.import_module("adv_v3.common"), importlib.import_module("adv_v3.critic")


def _hash_model(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _hash_arrays(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _actor_hash(oracle):
    return _hash_model(oracle.actor) if hasattr(oracle, "actor") else None


def regression_loss(prediction, target, groups, kind="mse", *, huber_delta=.01,
                    difference_weight=.25, ranking_weight=.1, ranking_temperature=.01):
    """A scalar loss in native regret units; local comparisons never cross centers.

    Local loss = MSE + .25 difference-MSE + .1 tau^2 logistic-ranking.
    The absolute MSE anchors the output scale used by C(Z) - lambda*c(X,Z).
    """
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("prediction and target must be matching one-dimensional tensors")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite regression values")
    mse = (prediction - target).square().mean()
    if kind == "mse":
        return mse
    if kind == "huber":
        if huber_delta <= 0:
            raise ValueError("positive Huber delta required")
        return F.huber_loss(prediction, target, delta=float(huber_delta))
    if kind not in ("local", "rankdiff"):
        raise ValueError(f"unknown loss {kind}")
    if ranking_temperature <= 0 or difference_weight < 0 or ranking_weight < 0:
        raise ValueError("invalid local loss weights/temperature")
    group = torch.as_tensor(groups, device=prediction.device)
    if group.shape != prediction.shape:
        raise ValueError("one group identifier per prediction required")
    same = torch.triu(group[:, None] == group[None, :], diagonal=1)
    if not same.any():
        return mse
    delta_p = (prediction[:, None] - prediction[None, :])[same]
    delta_y = (target[:, None] - target[None, :])[same]
    difference = (delta_p - delta_y).square().mean()
    ordered = delta_y.abs() > 1e-12
    ranking = prediction.sum() * 0.
    if ordered.any():
        ranking = F.softplus(-delta_y[ordered].sign() * delta_p[ordered]
                             / ranking_temperature).mean() * ranking_temperature ** 2
    return mse + difference_weight * difference + ranking_weight * ranking


def _trajectory_points(model, centers, price, seed, heartbeat=None):
    """Four points per center: X, random start, step 4, step 8 of one trajectory."""
    common, critic = _legacy()
    centers = np.asarray(centers, dtype=np.float32)
    orders = [np.lexsort((x[:, 1], x[:, 0])) for x in centers]
    canonical = np.stack([x[order] for x, order in zip(centers, orders)])
    rng = np.random.default_rng(seed)
    initial = np.clip(rng.normal(0, .02, size=canonical.shape), -.08, .08)
    result = np.empty((len(centers), 4, *centers.shape[1:]), dtype=np.float32)
    parameter = next(model.parameters())
    was_training, flags = model.training, [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before_geometry = dict(common.GEOMETRY_LEDGER)
    started = time.monotonic()
    try:
        for start in range(0, len(centers), 128):
            x_np = canonical[start:start + 128]
            x = torch.as_tensor(x_np, device=parameter.device, dtype=parameter.dtype)
            z_np = common.legalize(x_np, initial[start:start + 128])
            result[start:start + len(x_np), 0] = x_np
            result[start:start + len(x_np), 1] = z_np
            for step in range(1, 9):
                z = torch.tensor(z_np, device=parameter.device, dtype=parameter.dtype,
                                 requires_grad=True)
                with torch.enable_grad():
                    score = model(z) - float(price) * critic.matched_l1(x, z)
                    gradient = torch.autograd.grad(score.sum(), z)[0]
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("nonfinite calibration trajectory gradient")
                rms = gradient.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
                moved = (z + .01 * gradient / rms).detach().cpu().numpy()
                z_np = common.legalize(x_np, np.clip(moved - x_np, -.08, .08))
                if step in (4, 8):
                    result[start:start + len(x_np), 2 if step == 4 else 3] = z_np
                if heartbeat:
                    heartbeat()
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(was_training)
    restored = np.empty_like(result)
    for i, order in enumerate(orders):
        restored[i][:, order] = result[i]
    return restored, {"surrogate_forward_rows": len(centers) * 8,
                      "surrogate_input_backward_rows": len(centers) * 8,
                      "trajectory_seconds": time.monotonic() - started,
                      "geometry": {k: common.GEOMETRY_LEDGER[k] - before_geometry[k]
                                   for k in before_geometry},
                      "sources": ["center", "random_start", "gradient_step4", "gradient_step8"]}


def _label_pool(oracle, initial_model, degree, seed, outdir, mode, namespace,
                count, price, heartbeat=None):
    """Cache proposals and labels with request-before-execution audit records."""
    common, _ = _legacy()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    identity = dict(schema=1, namespace=str(namespace), mode=mode, count=int(count),
                    degree=int(degree), seed=int(seed), price=float(price),
                    initial_model_sha256=_hash_model(initial_model),
                    label_actor_sha256=_actor_hash(oracle),
                    trajectory=dict(std=.02, maximum_motion=.08, step=.01, steps=8))
    data_path, manifest_path = outdir / "labels.npz", outdir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity or common.sha256(data_path) != manifest["labels_sha256"]:
            raise ValueError("cached surrogate labels identity/hash mismatch")
        if common.sha256(outdir / "label_requests.jsonl") != manifest["request_ledger_sha256"]:
            raise ValueError("cached surrogate request ledger hash mismatch")
        with np.load(data_path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        if _hash_arrays(arrays) != manifest["arrays_sha256"]:
            raise ValueError("cached surrogate array digest mismatch")
        return arrays, manifest
    state_path = outdir / "label_progress.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("surrogate label progress identity mismatch")
    else:
        if mode == "uniform":
            pins = common.make_data(degree, count, "uniform", namespace, seed)
            groups, sources, diagnostics = np.arange(count), np.zeros(count, dtype=np.int64), {}
        elif mode == "search_region":
            if count % 4:
                raise ValueError("search-region pool must contain four labels per center")
            centers = common.make_data(degree, count // 4, "uniform", namespace, seed)
            grouped, diagnostics = _trajectory_points(
                initial_model, centers, price, common.derived_seed(namespace, seed, "trajectory"), heartbeat)
            pins = grouped.reshape(count, degree, 2)
            groups, sources = np.repeat(np.arange(count // 4), 4), np.tile(np.arange(4), count // 4)
        else:
            raise ValueError(f"unknown data mode {mode}")
        state = dict(identity=identity, pins=np.asarray(pins, dtype=np.float32),
                     groups=groups, sources=sources, diagnostics=diagnostics, completed=0,
                     regret=np.full(count, np.nan), length=np.full(count, np.nan),
                     lstar=np.full(count, np.nan))
        common.atomic_torch(state_path, state)
    log_path = outdir / "label_requests.jsonl"
    for start in range(int(state["completed"]), count, 128):
        stop = min(start + 128, count)
        request_id = f"{start}:{stop}:{time.time_ns()}"
        common.append_json(log_path, dict(event="requested", request_id=request_id,
                                         start=start, stop=stop, count=stop-start,
                                         oracle_before=dict(getattr(oracle, "ledger", {}))))
        answer = oracle.evaluate(state["pins"][start:stop])
        for key in ("regret", "length", "lstar"):
            values = np.asarray(answer[key], dtype=np.float64)
            if values.shape != (stop-start,) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid oracle {key}")
            state[key][start:stop] = values
        if np.any(state["lstar"][start:stop] <= 0):
            raise RuntimeError("nonpositive reference length")
        state["completed"] = stop
        common.atomic_torch(state_path, state)
        common.append_json(log_path, dict(event="completed", request_id=request_id,
                                         start=start, stop=stop, count=stop-start,
                                         oracle_after=dict(getattr(oracle, "ledger", {}))))
        if heartbeat:
            heartbeat()
    arrays = {k: state[k] for k in ("pins", "groups", "sources", "regret", "length", "lstar")}
    if not np.allclose(arrays["regret"], arrays["length"] / arrays["lstar"] - 1,
                       rtol=0, atol=1e-12):
        raise RuntimeError("regret labels do not match stored real lengths")
    common.atomic_npz(data_path, **arrays)
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    requested = sum(row["count"] for row in events if row["event"] == "requested")
    completed = sum(row["count"] for row in events if row["event"] == "completed")
    manifest = dict(identity=identity, status="complete", labels_path=str(data_path),
                    labels_sha256=common.sha256(data_path), arrays_sha256=_hash_arrays(arrays),
                    request_ledger_sha256=common.sha256(log_path),
                    label_count=count, actual_requested=requested, logged_completed=completed,
                    request_overhead=requested-count, diagnostics=state["diagnostics"],
                    oracle_ledger_snapshot=dict(getattr(oracle, "ledger", {})))
    common.atomic_json(manifest_path, manifest)
    return arrays, manifest


def build_refresh_data(initial_model, oracle, degree, seed, outdir, mode, namespace,
                       price, train_count=1024, validation_centers=128, heartbeat=None):
    """Create one train pool and the shared 128-center local validation pool.

    Pass sibling outdirs for uniform and search_region to share validation labels.
    The validation count counts independent centers, each with four scored points.
    """
    if validation_centers < 1 or validation_centers > 128:
        raise ValueError("validation_centers must be between 1 and 128")
    outdir = Path(outdir)
    train, train_manifest = _label_pool(
        oracle, initial_model, degree, seed, outdir, mode,
        f"{namespace}/train/{mode}/d{degree}/s{seed}", train_count, price, heartbeat)
    val, val_manifest = _label_pool(
        oracle, initial_model, degree, seed, outdir.parent / "shared_validation",
        "search_region", f"{namespace}/validation/local/d{degree}/s{seed}",
        validation_centers * 4, price, heartbeat)
    train_hashes = {hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() for p in train["pins"]}
    val_hashes = {hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() for p in val["pins"]}
    if train_hashes & val_hashes:
        raise RuntimeError("exact train/validation pin overlap")
    dataset = {**{f"train_{k}": v for k, v in train.items()},
               **{f"validation_{k}": v for k, v in val.items()}}
    dataset["identity"] = dict(train_sha256=train_manifest["arrays_sha256"],
                               validation_sha256=val_manifest["arrays_sha256"])
    return dataset, {"train": train_manifest, "validation": val_manifest,
                     "exact_train_validation_overlap": 0}


def _metrics(model, pins, labels, groups, batch_size):
    model.eval()
    with torch.no_grad():
        predicted = np.concatenate([model(pins[start:start + batch_size]).cpu().numpy()
                                    for start in range(0, len(pins), batch_size)])
    if not np.isfinite(predicted).all():
        raise RuntimeError("nonfinite validation prediction")
    rp, ry = rankdata(predicted), rankdata(labels)
    spearman = float(np.corrcoef(rp, ry)[0, 1]) if rp.std() and ry.std() else None
    local, agreement, differences = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        p, y = predicted[mask], labels[mask]
        if len(p) < 2:
            continue
        pr, yr = rankdata(p), rankdata(y)
        if pr.std() and yr.std():
            local.append(float(np.corrcoef(pr, yr)[0, 1]))
        rows, cols = np.triu_indices(len(p), 1)
        dp, dy = p[rows] - p[cols], y[rows] - y[cols]
        differences.extend((dp-dy)**2)
        ordered = np.abs(dy) > 1e-12
        agreement.extend((np.sign(dp[ordered]) == np.sign(dy[ordered])).astype(float))
    return dict(mse=float(np.mean((predicted-labels)**2)),
                mae=float(np.mean(np.abs(predicted-labels))), spearman=spearman,
                local_spearman=float(np.mean(local)) if local else None,
                local_pair_accuracy=float(np.mean(agreement)) if agreement else None,
                local_difference_mse=float(np.mean(differences)) if differences else None,
                prediction_mean=float(predicted.mean()), label_mean=float(np.mean(labels)),
                prediction_std=float(predicted.std()), label_std=float(np.std(labels)))


def _group_batches(groups, batch_size, seed):
    """Shuffle whole center groups so local losses see all candidates together."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    batch = []
    for group in unique:
        indices = np.flatnonzero(groups == group).tolist()
        if len(indices) > batch_size:
            raise ValueError("one local group exceeds batch size")
        if batch and len(batch) + len(indices) > batch_size:
            yield np.asarray(batch)
            batch = []
        batch.extend(indices)
    if batch:
        yield np.asarray(batch)


def train_refresh(initial_model, dataset, seed, outdir, loss="mse", epochs=6,
                  batch_size=128, lr=1e-4, heartbeat=None):
    """Fit one exact-copy initialization; choose by shared validation MSE only."""
    common, _ = _legacy()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("positive epochs, batch size and learning rate required")
    settings = dict(loss=loss, epochs=epochs, batch_size=batch_size, lr=lr,
                    huber_delta=.01, difference_weight=.25, ranking_weight=.1,
                    ranking_temperature=.01, selection="validation_mse")
    identity = dict(schema=1, initial_model_sha256=_hash_model(initial_model),
                    seed=int(seed), dataset=dataset["identity"], settings=settings)
    model = copy.deepcopy(initial_model)
    parameter = next(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    latest_path, best_path = outdir / "latest.pt", outdir / "best.pt"
    history, start_epoch, best_mse = [], 0, float("inf")
    if latest_path.exists():
        state = torch.load(latest_path, map_location=parameter.device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("surrogate training resume identity mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history, start_epoch, best_mse = state["history"], state["epoch"], state["best_mse"]
    train_pins, train_y, groups = [dataset[f"train_{k}"] for k in ("pins", "regret", "groups")]
    for epoch in range(start_epoch, epochs):
        # Epoch-local RNG makes a restart reproduce dropout without replaying prior epochs.
        torch.manual_seed(int(seed) + 1000003 * (epoch+1))
        if parameter.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed) + 1000003 * (epoch+1))
        model.train()
        total_loss, seen = 0., 0
        started = time.monotonic()
        for indices in _group_batches(groups, batch_size, int(seed) + 1000003 * (epoch+1)):
            targets = torch.as_tensor(train_y[indices], device=parameter.device, dtype=parameter.dtype)
            predictions = model(train_pins[indices])
            objective = regression_loss(predictions, targets, groups[indices], kind=loss)
            if not torch.isfinite(objective):
                raise RuntimeError("nonfinite surrogate objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            total_loss += float(objective.detach().cpu()) * len(indices)
            seen += len(indices)
            if heartbeat:
                heartbeat()
        elapsed = time.monotonic() - started
        validation = _metrics(model, dataset["validation_pins"], dataset["validation_regret"],
                              dataset["validation_groups"], batch_size)
        history.append(dict(epoch=epoch+1, train_objective=total_loss/seen,
                            train_rows=seen, training_seconds=elapsed, validation=validation))
        improved = validation["mse"] < best_mse
        best_mse = min(best_mse, validation["mse"])
        state = dict(identity=identity, epoch=epoch+1, model_state_dict=model.state_dict(),
                     optimizer_state_dict=optimizer.state_dict(), best_mse=best_mse, history=history)
        if improved:
            common.atomic_torch(best_path, state)
        common.atomic_torch(latest_path, state)
    best = torch.load(best_path, map_location=parameter.device, weights_only=False)
    if best["identity"] != identity:
        raise ValueError("best checkpoint identity mismatch")
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    summary = dict(status="complete", identity=identity, best_epoch=best["epoch"], history=history,
                   validation=_metrics(model, dataset["validation_pins"], dataset["validation_regret"],
                                       dataset["validation_groups"], batch_size),
                   best_path=str(best_path), best_sha256=common.sha256(best_path),
                   latest_path=str(latest_path), latest_sha256=common.sha256(latest_path),
                   model_sha256=_hash_model(model), fresh_labels_during_training=0,
                   training_forward_rows=len(train_pins)*epochs,
                   training_backward_rows=len(train_pins)*epochs,
                   validation_forward_rows=len(dataset["validation_pins"])*(epochs+1))
    common.atomic_json(outdir / "summary.json", summary)
    return model, summary


def train_variants(oracle, parent_critic, base_model, seed, outdir, propose_fn=None, *,
                   degree=30, price=0., namespace=NAMESPACE, heartbeat=None):
    """Prepare the four predeclared variants and return models plus audit summaries.

    `parent_critic` is retained for worker compatibility; base_model is the
    already calibrated v3.1 independent InputCritic used by every variant.
    `propose_fn` is unused: collecting steps 4 and 8 requires one shared trajectory.
    """
    common, _ = _legacy()
    outdir = Path(outdir)
    if hasattr(base_model, "calibrated") and not base_model.calibrated:
        raise ValueError("base_model must use a linear calibrated output head")
    base_hash = _hash_model(base_model)
    datasets, data_summaries = {}, {}
    for mode in ("uniform", "search_region"):
        datasets[mode], data_summaries[mode] = build_refresh_data(
            base_model, oracle, degree, seed, outdir / "data" / mode, mode,
            namespace, price, heartbeat=heartbeat)
    models, summaries = {"old": base_model}, {}
    for name, mode, objective in (("uniform_mse", "uniform", "mse"),
                                  ("region_mse", "search_region", "mse"),
                                  ("region_huber", "search_region", "huber"),
                                  ("region_rankdiff", "search_region", "rankdiff")):
        models[name], summaries[name] = train_refresh(
            base_model, datasets[mode], seed, outdir / "models" / name,
            loss=objective, heartbeat=heartbeat)
    if _hash_model(base_model) != base_hash:
        raise RuntimeError("base surrogate changed during calibration")
    summary = dict(status="complete", seed=int(seed), degree=int(degree), price=float(price),
                   initial_model_sha256=base_hash, variants=summaries, datasets=data_summaries,
                   unique_new_training_labels=2048, shared_validation_labels=512,
                   logical_training_labels_per_variant=1024,
                   shared_validation_centers=128, total_unique_real_labels=2560,
                   selection="independent shared local validation MSE", test_labels_used=0)
    common.atomic_json(outdir / "summary.json", summary)
    return {"models": models, "summary": summary}
