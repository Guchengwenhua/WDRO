"""Recoverable, alternating main-model experiments for protocol v3, part B.

Search scores remain greedy. Main updates use sampled REINFORCE with the
original sampled-regret Critic baseline, and carry separate evaluator counters.
The latest checkpoint is the transaction boundary for every outer update;
actual Oracle counters are monotone and are never restored from a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .common import (
    ROOT, EXP, GEOMETRY_LEDGER, Oracle, atomic_json, atomic_npz, atomic_torch,
    derived_seed, json_default, load_parent, make_data, parent_model,
    parent_price, sha256,
)
from .search import BUDGETS, attack
from src.geometry import pin_set_sha256


METHODS = (
    "random_top4", "critic_calibrated4", "rl_frozen4", "rl_joint4",
    "random64", "es64",
)
DISTRIBUTIONS = ("uniform", "mix_gauss", "edge_bias", "grid_cluster")
DEFAULTS = {
    "updates": 512, "batch_size": 64, "learning_rate": 1e-5,
    "gradient_clip": 1.0, "lambda_learning_rate": 1.0,
    "validation_interval": 64, "validation_count": 128,
    "evaluation_count": 512, "independent_attack_count": 64,
    "evaluation_batch_size": 64, "rl_interval": 4, "critic_interval": 32,
    "rl_learning_rate": 3e-4, "critic_learning_rate": 1e-4,
    "nominal_degradation_max": 0.002,
    "initial_checkpoint_is_candidate": True,
    "optimizer_initialization": "fresh_for_main_policy_and_search_surrogate",
}


class _Paused(Exception):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=json_default, allow_nan=False).encode()).hexdigest()


def _array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def _model_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def _checked_artifact(binding):
    path = Path(binding.get("path", binding.get("logical_path", "")))
    if not path.is_file() or sha256(path) != binding["sha256"]:
        raise RuntimeError(f"bound artifact changed: {path}")
    return path


def _settings(smoke):
    settings = dict(DEFAULTS)
    if smoke:
        settings.update(updates=4, batch_size=4, validation_interval=2,
                        validation_count=4, evaluation_count=8,
                        independent_attack_count=2, evaluation_batch_size=4,
                        critic_interval=2)
    return settings


def _fixed_directory(args):
    return EXP / ("smoke" if args.smoke else "fixed") / f"erm_d30_s{args.seed}"


def _build_identity(args, settings):
    protocol = next((p for p in (ROOT / "protocol.json", ROOT / "manifests" / "protocol.json")
                     if p.exists()), None)
    if protocol is None:
        raise FileNotFoundError("freeze protocol.json before starting joint experiments")
    modules = ["common.py", "search.py", "joint.py"]
    dependencies = {}
    for method in ("erm", "wdro_rest"):
        best = parent_model(method, args.degree, args.seed)["best"]
        dependencies[f"parent_{method}"] = _artifact(_checked_artifact(best))
    fixed = _fixed_directory(args)
    # Formal B follows accepted A. Its per-parent task completion is a durable
    # dependency; the root scheduler additionally gates all 30 A tasks.
    if not args.smoke:
        complete = fixed / "complete.json"
        if not complete.exists():
            raise FileNotFoundError("joint task requires its completed fixed-opponent A task")
        finished = json.loads(complete.read_text())
        if finished.get("status") != "pass":
            raise RuntimeError("fixed-opponent parent task did not pass")
        for artifact in finished["artifacts"]:
            _checked_artifact(artifact)
        dependencies["fixed_opponent_completion"] = _artifact(complete)
    if args.method.startswith("rl_"):
        modules.append("rl.py")
        dependencies["learned_policy"] = _artifact(fixed / "rl" / "best.pt")
    if args.method == "critic_calibrated4":
        modules.append("critic.py")
        dependencies["learned_surrogate"] = _artifact(fixed / "critic" / "best.pt")
    return {
        "schema_version": 1, "experiment": "adversarial_v3_joint",
        "method": args.method, "degree": args.degree, "seed": args.seed,
        "smoke": bool(args.smoke), "config": settings,
        "protocol": _artifact(protocol), "dependencies": dependencies,
        "source": {name: sha256(ROOT / "adv_v3" / name) for name in modules},
        "data_namespaces": {
            "train": "joint-training-u{update:06d}",
            "validation": "joint-validation", "evaluation": "joint-evaluation",
            "independent_attack": "joint-independent-attack",
            "rl_online": "joint-attacker-training-u{update:06d}",
        },
    }


def _configure(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _delta(before, after):
    return {key: int(after.get(key, 0) - before.get(key, 0))
            for key in set(before) | set(after)}


def _assert_greedy_budget(delta, count):
    for key in ("actor_requested", "actor_completed", "rest_requested", "rest_completed", "reference_requests"):
        if delta.get(key, 0) != count:
            raise RuntimeError(f"true-score accounting mismatch: {key}={delta.get(key, 0)}, expected {count}")


def _load_learned(args, main_critic, device, settings):
    policy = policy_optimizer = surrogate = surrogate_optimizer = None
    fixed = _fixed_directory(args)
    if args.method.startswith("rl_"):
        from .rl import Policy
        saved = torch.load(fixed / "rl" / "best.pt", map_location=device, weights_only=False)
        config = saved["identity"]["config"]
        if int(saved["identity"]["degree"]) != args.degree:
            raise RuntimeError("A policy degree mismatch")
        policy = Policy(args.degree, hidden_size=int(config["hidden_size"]),
                        initial_std=float(config["initial_std"]),
                        maximum_motion=float(config["maximum_motion"])).to(device)
        policy.load_state_dict(saved["policy_state_dict"], strict=True)
        policy.eval()
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=settings["rl_learning_rate"])
    if args.method == "critic_calibrated4":
        from .critic import InputCritic
        saved = torch.load(fixed / "critic" / "best.pt", map_location=device, weights_only=False)
        if int(saved["identity"]["degree"]) != args.degree:
            raise RuntimeError("A surrogate degree mismatch")
        surrogate = InputCritic(main_critic, calibrated=True).to(device)
        surrogate.load_state_dict(saved["model_state_dict"], strict=True)
        surrogate.eval()
        surrogate_optimizer = torch.optim.Adam(surrogate.parameters(), lr=settings["critic_learning_rate"])
    return policy, policy_optimizer, surrogate, surrogate_optimizer


def _pack(update, price, actor, critic, optimizer, policy, policy_optimizer,
          surrogate, surrogate_optimizer, oracle, validation=None):
    value = {
        "update": int(update), "price": float(price),
        "actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "policy_state_dict": None if policy is None else policy.state_dict(),
        "policy_optimizer_state_dict": None if policy_optimizer is None else policy_optimizer.state_dict(),
        "surrogate_state_dict": None if surrogate is None else surrogate.state_dict(),
        "surrogate_optimizer_state_dict": None if surrogate_optimizer is None else surrogate_optimizer.state_dict(),
        "rng_state": _rng_state(), "oracle_ledger_snapshot": dict(oracle.ledger),
        "geometry_ledger_snapshot": dict(GEOMETRY_LEDGER), "validation": validation,
    }
    return _cpu_copy(value)


def _restore(saved, actor, critic, optimizer, policy, policy_optimizer,
             surrogate, surrogate_optimizer, *, rng=True):
    for model, key in ((actor, "actor_state_dict"), (critic, "critic_state_dict"),
                       (policy, "policy_state_dict"), (surrogate, "surrogate_state_dict")):
        if model is not None:
            model.load_state_dict(saved[key], strict=True)
        elif saved[key] is not None:
            raise RuntimeError(f"checkpoint contains unexpected {key}")
    for optim, key in ((optimizer, "optimizer_state_dict"), (policy_optimizer, "policy_optimizer_state_dict"),
                       (surrogate_optimizer, "surrogate_optimizer_state_dict")):
        if optim is not None:
            optim.load_state_dict(saved[key])
    if rng:
        _restore_rng(saved["rng_state"])


def _input(path, *, degree, count, distribution, namespace, seed, identity_hash):
    pins = make_data(degree, count, distribution, namespace, seed=seed)
    spec = {"degree": degree, "count": count, "distribution": distribution,
            "namespace": namespace, "seed": seed, "identity_hash": identity_hash}
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if json.loads(str(old["spec_json"])) != spec:
                raise RuntimeError(f"input identity mismatch: {path}")
            np.testing.assert_array_equal(old["pins"], pins)
    else:
        atomic_npz(path, pins=pins, spec_json=np.asarray(json.dumps(spec, sort_keys=True)))
    return pins


def _saved_chunk(path, centers, identity_hash, actor_hash, update, expected_calls):
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if str(arrays["identity_hash"]) != identity_hash or str(arrays["actor_sha256"]) != actor_hash:
        raise RuntimeError(f"result identity/model mismatch: {path}")
    if int(arrays["update"]) != update:
        raise RuntimeError(f"result checkpoint mismatch: {path}")
    np.testing.assert_array_equal(arrays["centers"], centers)
    np.testing.assert_array_equal(arrays["calls"], np.full(len(centers), expected_calls))
    for key in ("regret", "lstar", "length", "cost", "score", "original_regret", "original_score"):
        if arrays[key].shape != (len(centers),) or not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"invalid result {key}: {path}")
    if np.any(arrays["lstar"] <= 0) or np.any(arrays["score"] < arrays["original_score"] - 1e-12):
        raise RuntimeError(f"invalid result reference or best score: {path}")
    np.testing.assert_allclose(arrays["regret"], arrays["length"] / arrays["lstar"] - 1, atol=1e-12, rtol=1e-12)
    _assert_greedy_budget(json.loads(str(arrays["ledger_delta_json"])), len(centers) * expected_calls)
    return arrays


def _evaluate(actor, oracle, datasets, out, *, identity_hash, update, kind,
              batch_size, degree, seed, state, pause):
    actor_hash = _model_hash(actor)
    summary = {}
    for distribution, centers in datasets.items():
        parts = []
        for start in range(0, len(centers), batch_size):
            if pause.is_set():
                raise _Paused()
            x = centers[start:start + batch_size]
            if kind == "validation":
                path = out / "validation" / f"u{update:06d}" / f"{distribution}_{start:05d}.npz"
            elif kind == "cem64":
                path = out / "attacks" / distribution / f"cem64_{start:05d}.npz"
            else:
                path = out / "evaluation" / distribution / f"{kind}_{start:05d}.npz"
            count = 64 if kind == "cem64" else 1
            state.update(phase=kind, distribution=distribution, offset=start, update=update)
            if path.exists():
                arrays = _saved_chunk(path, x, identity_hash, actor_hash, update, count)
            else:
                before, geometry_before = dict(oracle.ledger), dict(GEOMETRY_LEDGER)
                began = time.monotonic()
                if kind == "cem64":
                    result = attack("cem64", x, oracle, 0.0,
                                    derived_seed("joint-independent-cem", degree, distribution, start))
                    arrays = {key: result[key] for key in (
                        "pins", "regret", "lstar", "length", "cost", "score", "original_regret", "original_score")}
                    diagnostics = result["diagnostics"]
                else:
                    result = oracle.evaluate(x)
                    arrays = {**result, "pins": x.copy(), "cost": np.zeros(len(x)),
                              "score": result["regret"].copy(), "original_regret": result["regret"].copy(),
                              "original_score": result["regret"].copy()}
                    diagnostics = {}
                delta = _delta(before, oracle.ledger)
                _assert_greedy_budget(delta, len(x) * count)
                arrays.update(centers=x, price=np.zeros(len(x)), calls=np.full(len(x), count, dtype=np.int32),
                              update=np.asarray(update), identity_hash=np.asarray(identity_hash),
                              actor_sha256=np.asarray(actor_hash), elapsed_seconds=np.asarray(time.monotonic() - began),
                              ledger_delta_json=np.asarray(json.dumps(delta, sort_keys=True)),
                              geometry_delta_json=np.asarray(json.dumps(_delta(geometry_before, GEOMETRY_LEDGER), sort_keys=True)),
                              diagnostics_json=np.asarray(json.dumps(diagnostics, default=json_default, sort_keys=True)))
                atomic_npz(path, **arrays)
                arrays = _saved_chunk(path, x, identity_hash, actor_hash, update, count)
            parts.append(arrays["regret"])
        regret = np.concatenate(parts)
        summary[distribution] = {"count": len(regret), "mean": float(regret.mean()),
                                 "p95": float(np.quantile(regret, 0.95)), "maximum": float(regret.max())}
    summary["ood_worst_group_mean"] = max(summary[key]["mean"] for key in DISTRIBUTIONS[1:])
    summary["nominal_mean"] = summary["uniform"]["mean"]
    return summary


def _is_better(validation, initial_validation, best_validation, tolerance=0.002):
    feasible = validation["nominal_mean"] <= initial_validation["nominal_mean"] + tolerance
    # Strict improvement keeps the earliest checkpoint on ties; update zero is
    # explicitly a candidate, so worse trained checkpoints never displace it.
    better = feasible and validation["ood_worst_group_mean"] < best_validation["ood_worst_group_mean"]
    return bool(feasible), bool(better)


def sampled_update(actor, critic, optimizer, oracle, pins, lstar, clip=1.0):
    """One main-policy update; reference lengths are reused from true search."""
    pins = np.asarray(pins, dtype=np.float32)
    reference = np.asarray(lstar, dtype=np.float64)
    if reference.shape != (len(pins),) or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("invalid reused reference lengths")
    before = dict(oracle.ledger)
    actor.train()
    critic.train()
    oracle.add("sampled_actor_requested", len(pins))
    oracle.persist()
    topology, log_probs = actor(pins, False)
    oracle.add("sampled_actor_completed", len(pins))
    oracle.add("sampled_rest_requested", len(pins))
    oracle.persist()
    length = oracle.rest.lengths(pins, topology.detach().cpu().numpy())
    oracle.add("sampled_rest_completed", len(pins))
    oracle.add("sampled_reference_reuses", len(pins))
    oracle.persist()
    regret = np.asarray(length, dtype=np.float64) / reference - 1
    if not np.isfinite(regret).all() or np.any(regret < -1e-7):
        raise RuntimeError("invalid sampled relative regret")
    target = torch.as_tensor(regret, dtype=torch.float32, device=next(actor.parameters()).device)
    prediction = critic(pins)
    if prediction.shape != target.shape or log_probs.shape != target.shape:
        raise RuntimeError("main Actor/Critic output shape mismatch")
    actor_loss = ((target - prediction).detach() * log_probs).mean()
    critic_loss = (prediction - target).square().mean()
    loss = actor_loss + critic_loss
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite sampled Actor/Critic loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), clip)
    critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), clip)
    if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
        raise RuntimeError("nonfinite main-model gradient")
    optimizer.step()
    if not all(torch.isfinite(p).all() for model in (actor, critic) for p in model.parameters()):
        raise RuntimeError("nonfinite main-model parameters")
    delta = _delta(before, oracle.ledger)
    for key in ("sampled_actor_requested", "sampled_actor_completed", "sampled_rest_requested",
                "sampled_rest_completed", "sampled_reference_reuses"):
        if delta.get(key) != len(pins):
            raise RuntimeError(f"sampled accounting mismatch: {key}")
    if any(delta.get(key, 0) for key in ("actor_requested", "rest_requested", "reference_requests")):
        raise RuntimeError("sampled update unexpectedly charged greedy/reference calls")
    return {"actor_loss": float(actor_loss.detach().cpu()), "critic_loss": float(critic_loss.detach().cpu()),
            "mean_regret": float(regret.mean()), "p95_regret": float(np.quantile(regret, 0.95)),
            "actor_grad_norm_before_clip": float(actor_norm.cpu()),
            "critic_grad_norm_before_clip": float(critic_norm.cpu()), "ledger_delta": delta}


def _write_history(out, history):
    for record in history:
        path = out / "updates" / f"u{record['update']:06d}.json"
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise RuntimeError("committed update log differs from checkpoint history")
        else:
            atomic_json(path, record)
    destination = out / "training.jsonl"
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("w") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True, default=json_default, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _check_data_disjoint(out):
    families = {key: set() for key in ("train", "validation", "evaluation", "independent_attack", "rl_online")}
    for path in sorted((out / "inputs").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            family = path.name.split("__", 1)[0]
            if family not in families:
                raise RuntimeError(f"unknown input role: {path}")
            keys = {pin_set_sha256(x) for x in data["pins"]}
            if len(keys) != len(data["pins"]):
                raise RuntimeError(f"duplicate net within {path}")
            families[family].update(keys)
    names = list(families)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if families[left] & families[right]:
                raise RuntimeError(f"data leakage between {left} and {right}")
    return {key: len(value) for key, value in families.items()}


def _heartbeat(path, stop, state):
    # Exactly one writer owns heartbeat.json and its atomic temporary path.
    while not stop.is_set():
        atomic_json(path, {**state, "pid": os.getpid(), "at": time.time()})
        stop.wait(5)


def run(args, *, stop_after=None):
    """Run/resume one B task. stop_after is an internal recovery-test limit."""
    if args.degree != 30 or args.seed not in (101, 102, 103) or args.method not in METHODS:
        raise ValueError("joint protocol requires degree30, seeds101/102/103 and a declared method")
    settings = _settings(bool(args.smoke))
    identity = _build_identity(args, settings)
    identity_hash = _digest(identity)
    ident = f"{args.method}_d30_s{args.seed}"
    out = EXP / ("joint_smoke" if args.smoke else "joint") / ident
    out.mkdir(parents=True, exist_ok=True)
    with (out / "task.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = out / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise RuntimeError("joint identity changed; refusing mixed-code/config resume")
        else:
            atomic_json(identity_path, identity)
        if (out / "complete.json").exists():
            complete = json.loads((out / "complete.json").read_text())
            if complete["identity_hash"] != identity_hash:
                raise RuntimeError("joint completion identity mismatch")
            for binding in complete["artifacts"]:
                _checked_artifact(binding)
            return complete
        pause, heartbeat_stop = threading.Event(), threading.Event()
        state = {"task": ident, "phase": "load", "update": 0, "smoke": bool(args.smoke),
                 "physical_gpu_index": args.physical_gpu_index}
        atomic_json(out / "launch.json", {**state, "pid": os.getpid(), "at": time.time(),
                    "process_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                    "command": list(os.sys.argv), "identity_hash": identity_hash})
        heartbeat = threading.Thread(target=_heartbeat, args=(out / "heartbeat.json", heartbeat_stop, state), daemon=True)
        heartbeat.start()
        previous_signals = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_signals[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: pause.set())
        oracle = None
        began = time.monotonic()
        try:
            _configure(derived_seed("joint-main", args.degree, args.seed))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            actor, critic, _, parent = load_parent("erm", args.degree, args.seed, device)
            initial_price, rho = parent_price(args.degree, args.seed)
            if not np.isfinite([initial_price, rho]).all() or initial_price < 0 or rho <= 0:
                raise RuntimeError("B requires the selected positive parent rho and finite nonnegative price")
            optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=settings["learning_rate"])
            policy, policy_optimizer, surrogate, surrogate_optimizer = _load_learned(args, critic, device, settings)
            oracle = Oracle(actor, device, out / "oracle", microbatch=256)
            objects = (actor, critic, optimizer, policy, policy_optimizer, surrogate, surrogate_optimizer)
            validation_data = {
                distribution: _input(out / "inputs" / f"validation__{distribution}.npz", degree=args.degree,
                    count=settings["validation_count"], distribution=distribution, namespace="joint-validation",
                    seed=0, identity_hash=identity_hash) for distribution in DISTRIBUTIONS
            }
            latest_path = out / "latest.pt"
            if latest_path.exists():
                saved = torch.load(latest_path, map_location="cpu", weights_only=False)
                if saved["identity_hash"] != identity_hash:
                    raise RuntimeError("joint checkpoint identity mismatch")
                current, best, history = saved["current"], saved["best"], saved["history"]
                initial_validation = saved["initial_validation"]
                _restore(current, *objects)
                update, price = int(current["update"]), float(current["price"])
                if [r["update"] for r in history] != list(range(1, update + 1)):
                    raise RuntimeError("joint checkpoint has incomplete update history")
                atomic_json(out / "recovery" / f"{time.time_ns()}.json", {
                    "checkpoint_update": update, "at": time.time(),
                    "oracle_actual": dict(oracle.ledger), "checkpoint_ledger": current["oracle_ledger_snapshot"],
                    "uncommitted_or_later_phase_delta": _delta(current["oracle_ledger_snapshot"], oracle.ledger),
                    "accounting": "actual counters retained; no rollback",
                })
                _write_history(out, history)
            else:
                update, price, history = 0, initial_price, []
                initial_validation = _evaluate(actor, oracle, validation_data, out, identity_hash=identity_hash,
                    update=0, kind="validation", batch_size=settings["evaluation_batch_size"],
                    degree=args.degree, seed=args.seed, state=state, pause=pause)
                current = _pack(0, price, *objects, oracle, validation=initial_validation)
                best = copy.deepcopy(current)
                atomic_torch(out / "initial.pt", current)
                atomic_torch(latest_path, {"identity_hash": identity_hash, "current": current, "best": best,
                                          "initial_validation": initial_validation, "history": history})
            atomic_torch(out / "best.pt", best)
            search_method = "rl4" if args.method.startswith("rl_") else args.method
            for step in range(update + 1, settings["updates"] + 1):
                if pause.is_set() or (stop_after is not None and step > stop_after):
                    raise _Paused()
                state.update(phase="training", update=step)
                x = _input(out / "inputs" / f"train__u{step:06d}.npz", degree=args.degree,
                    count=settings["batch_size"], distribution="uniform", namespace=f"joint-training-u{step:06d}",
                    seed=args.seed, identity_hash=identity_hash)
                proposer = None
                if policy is not None:
                    from .rl import propose
                    proposer = lambda c, p, k, s: propose(policy, c, p, k, s)
                if surrogate is not None:
                    from .critic import propose
                    proposer = lambda c, p, k, s: propose(surrogate, c, p, k, s)
                before, geometry_before = dict(oracle.ledger), dict(GEOMETRY_LEDGER)
                tick, price_before = time.monotonic(), price
                result = attack(search_method, x, oracle, price,
                                derived_seed("joint-search", args.degree, args.seed, step), proposer=proposer)
                search_delta = _delta(before, oracle.ledger)
                _assert_greedy_budget(search_delta, len(x) * BUDGETS[search_method])
                # Extra attacker updates do not shift the next main sampler's
                # random stream. All six branches share these per-step seeds.
                torch.manual_seed(derived_seed("joint-main-sampled", args.degree, args.seed, step))
                sampled = sampled_update(actor, critic, optimizer, oracle, result["pins"], result["lstar"], settings["gradient_clip"])
                price = max(0.0, price + settings["lambda_learning_rate"] * (float(result["cost"].mean()) - rho))
                if not np.isfinite(price):
                    raise RuntimeError("nonfinite updated transport price")
                online = None
                if args.method == "rl_joint4" and step % settings["rl_interval"] == 0:
                    from .rl import online_update
                    online_x = _input(out / "inputs" / f"rl_online__u{step:06d}.npz", degree=args.degree,
                        count=settings["batch_size"], distribution="uniform", namespace=f"joint-attacker-training-u{step:06d}",
                        seed=args.seed, identity_hash=identity_hash)
                    online_before = dict(oracle.ledger)
                    online = online_update(policy, policy_optimizer, oracle, online_x, price)
                    online["ledger_delta"] = _delta(online_before, oracle.ledger)
                    _assert_greedy_budget(online["ledger_delta"], 2 * len(online_x))
                    online.update(kind="rl", opponent_update=step, optimizer_initialization="fresh_at_B_start")
                if surrogate is not None and step % settings["critic_interval"] == 0:
                    from .critic import calibrate_online
                    online_before = dict(oracle.ledger)
                    online = calibrate_online(surrogate, surrogate_optimizer,
                        result["evaluated_pins"].reshape(-1, args.degree, 2), result["evaluated_regrets"].reshape(-1))
                    online.update(kind="critic", label_actor_update=step - 1,
                                  label_source="this_update_true_search_candidates", ledger_delta=_delta(online_before, oracle.ledger))
                    _assert_greedy_budget(online["ledger_delta"], 0)
                validation = None
                feasible = improved = False
                if step % settings["validation_interval"] == 0 or step == settings["updates"]:
                    validation = _evaluate(actor, oracle, validation_data, out, identity_hash=identity_hash,
                        update=step, kind="validation", batch_size=settings["evaluation_batch_size"],
                        degree=args.degree, seed=args.seed, state=state, pause=pause)
                    feasible, improved = _is_better(validation, initial_validation, best["validation"], settings["nominal_degradation_max"])
                record = {
                    "update": step, "input_sha256": _array_hash(x), "selected_sha256": _array_hash(result["pins"]),
                    "lambda_before": price_before, "lambda_after": price, "rho": rho,
                    "mean_cost": float(result["cost"].mean()), "mean_search_score": float(result["score"].mean()),
                    "mean_search_regret": float(result["regret"].mean()),
                    "mean_search_improvement": float((result["score"] - result["original_score"]).mean()),
                    "search_method": search_method, "search_calls_per_center": result["calls_per_center"],
                    "search_ledger_delta": search_delta, "search_diagnostics": result["diagnostics"],
                    "sampled": sampled, "online": online, "validation": validation,
                    "validation_feasible": feasible, "best_improved": improved,
                    "ledger_delta": _delta(before, oracle.ledger),
                    "geometry_delta": _delta(geometry_before, GEOMETRY_LEDGER),
                    "wall_seconds": time.monotonic() - tick,
                }
                history.append(record)
                current = _pack(step, price, *objects, oracle, validation=validation)
                if improved:
                    best = copy.deepcopy(current)
                payload = {"identity_hash": identity_hash, "current": current, "best": best,
                           "initial_validation": initial_validation, "history": history}
                atomic_torch(latest_path, payload)  # only now is this outer step committed
                atomic_json(out / "updates" / f"u{step:06d}.json", record)
                if improved:
                    atomic_torch(out / "best.pt", best)
                if step % settings["validation_interval"] == 0:
                    atomic_torch(out / "checkpoints" / f"u{step:06d}.pt", payload)
                update = step
            _write_history(out, history)
            initial = torch.load(out / "initial.pt", map_location="cpu", weights_only=False)
            evaluation_data = {
                distribution: _input(out / "inputs" / f"evaluation__{distribution}.npz", degree=args.degree,
                    count=settings["evaluation_count"], distribution=distribution, namespace="joint-evaluation",
                    seed=0, identity_hash=identity_hash) for distribution in DISTRIBUTIONS
            }
            actor.load_state_dict(initial["actor_state_dict"], strict=True)
            initial_evaluation = _evaluate(actor, oracle, evaluation_data, out, identity_hash=identity_hash,
                update=0, kind="initial", batch_size=settings["evaluation_batch_size"],
                degree=args.degree, seed=args.seed, state=state, pause=pause)
            _restore(best, *objects, rng=False)
            best_evaluation = _evaluate(actor, oracle, evaluation_data, out, identity_hash=identity_hash,
                update=int(best["update"]), kind="final", batch_size=settings["evaluation_batch_size"],
                degree=args.degree, seed=args.seed, state=state, pause=pause)
            attack_data = {
                distribution: _input(out / "inputs" / f"independent_attack__{distribution}.npz", degree=args.degree,
                    count=settings["independent_attack_count"], distribution=distribution, namespace="joint-independent-attack",
                    seed=0, identity_hash=identity_hash) for distribution in DISTRIBUTIONS
            }
            attack_summary = _evaluate(actor, oracle, attack_data, out, identity_hash=identity_hash,
                update=int(best["update"]), kind="cem64", batch_size=settings["evaluation_batch_size"],
                degree=args.degree, seed=args.seed, state=state, pause=pause)
            data_audit = _check_data_disjoint(out)
            oracle.persist()
            excluded = {"complete.json", "heartbeat.json", "launch.json", "exit.json", "failure.json", "paused.json"}
            artifacts = [_artifact(path) for path in sorted(out.rglob("*"))
                         if path.is_file() and path.suffix in (".json", ".jsonl", ".npz", ".pt")
                         and path.name not in excluded]
            complete = {
                "status": "pass", "phase": "joint", "task": ident, "method": args.method,
                "degree": args.degree, "seed": args.seed, "parent": "erm", "smoke": bool(args.smoke),
                "identity": identity, "identity_hash": identity_hash,
                "updates": update, "best_update": int(best["update"]), "no_training_improvement": int(best["update"]) == 0,
                "initial_price": initial_price, "final_price": price, "best_price": float(best["price"]), "rho": rho,
                "initial_validation": initial_validation, "best_validation": best["validation"],
                "initial_evaluation": initial_evaluation, "final_evaluation": best_evaluation,
                "independent_attack": attack_summary, "history": history, "data_audit": data_audit,
                "ledger": dict(oracle.ledger), "geometry_ledger": dict(GEOMETRY_LEDGER),
                "task_seconds_this_attempt": time.monotonic() - began,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0,
                "artifacts": artifacts,
                "accounting": "greedy search/evaluation and sampled main updates separate; actual attempts never rolled back",
            }
            atomic_json(out / "complete.json", complete)
            return complete
        except _Paused:
            result = {**state, "status": "paused", "at": time.time(), "identity_hash": identity_hash}
            atomic_json(out / "paused.json", result)
            return result
        except Exception as error:
            atomic_json(out / "failure.json", {**state, "error": repr(error), "at": time.time()})
            raise
        finally:
            if oracle is not None:
                oracle.close()
            heartbeat_stop.set()
            heartbeat.join(timeout=6)
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)
            atomic_json(out / "exit.json", {**state, "pid": os.getpid(), "at": time.time(),
                        "complete": (out / "complete.json").exists()})
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=30, choices=[30])
    parser.add_argument("--seed", type=int, required=True, choices=[101, 102, 103])
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "task", "best_update") if key in result}))


if __name__ == "__main__":
    main()
