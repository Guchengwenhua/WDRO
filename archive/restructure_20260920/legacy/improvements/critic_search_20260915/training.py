"""Candidate 7: isolated 2x2 surrogate-refresh / dual-cost-smoothing experiment.

No legacy files are modified. Actual scorer counters are durable and never
rolled back; latest.pt is the outer-update transaction boundary.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
LEGACY = PROJECT / "revisions/adversarial_v3_1_20260912/Codex"
sys.path.insert(0, str(LEGACY))
from adv_v3 import common as C
from adv_v3 import joint as J
from adv_v3.critic import InputCritic, calibrate_online, propose
from adv_v3.search import attack

VARIANTS = {
    "refresh32_raw": (32, False),
    "refresh8_raw": (8, False),
    "refresh32_ema": (32, True),
    "refresh8_ema": (8, True),
}
DEFAULTS = dict(updates=128, batch_size=16, learning_rate=1e-5,
                gradient_clip=1.0, lambda_learning_rate=1.0,
                ema_beta=0.9, critic_learning_rate=1e-4,
                validation_interval=32, validation_count=64,
                evaluation_count=256, evaluation_batch_size=64,
                nominal_degradation_max=0.002, greedy_microbatch=256,
                torch_threads=1, reference_workers=3, gpu_memory_fraction=0.10,
                true_search_budget_per_center=4)


def settings_for(variant, smoke=False):
    settings = dict(DEFAULTS)
    interval, smoothing = VARIANTS[variant]
    settings.update(critic_interval=interval, lambda_cost_smoothing=smoothing)
    if smoke:
        settings.update(updates=4, batch_size=2, validation_interval=2,
                        validation_count=2, evaluation_count=4,
                        evaluation_batch_size=4,
                        critic_interval=2 if interval == 32 else 1)
    return settings


def update_price(price, cost, rho, ema, *, smoothing, beta=0.9, learning_rate=1.0):
    """EMA is initialized from the first observed cost (no zero-start bias)."""
    values = [price, cost, rho, beta, learning_rate]
    if not np.isfinite(values).all() or min(price, cost, rho, learning_rate) < 0:
        raise ValueError("invalid price/cost configuration")
    if not 0 <= beta < 1:
        raise ValueError("EMA beta must be in [0,1)")
    ema = float(cost) if ema is None else beta * float(ema) + (1 - beta) * float(cost)
    signal_cost = ema if smoothing else float(cost)
    return max(0.0, float(price) + learning_rate * (signal_cost - rho)), ema, signal_cost


def _identity(args, settings):
    surrogate = LEGACY / f"experiments/fixed/erm_d30_s{args.seed}/critic/best.pt"
    deps = {"surrogate": J._artifact(surrogate)}
    for method in ("erm", "wdro_rest"):
        deps[method] = J._artifact(J._checked_artifact(C.parent_model(method, 30, args.seed)["best"]))
    code = {"training.py": C.sha256(__file__)}
    for name in ("common.py", "joint.py", "critic.py", "search.py"):
        code[f"adv_v3/{name}"] = C.sha256(LEGACY / "adv_v3" / name)
    return dict(schema_version=1, experiment="tmp2_candidate07_refresh_dual_2x2",
                degree=30, seed=args.seed, variant=args.variant, smoke=bool(args.smoke),
                settings=settings, dependencies=deps, source_sha256=code,
                theory=J._artifact(ROOT / "THEORY_07.md"), protocol=J._artifact(ROOT / "protocol.json"),
                model_selection="minimum validation OOD worst-group mean subject to nominal degradation <=0.002; initial eligible; earliest ties",
                data_namespaces=dict(train="tmp2-candidate07-train-u{step:06d}",
                    validation="tmp2-candidate07-validation", evaluation="tmp2-candidate07-test"),
                label_freshness="current Actor before outer update; current batch only; no regret replay",
                optimizer_initialization="fresh Adam for Actor plus baseline Critic and independent surrogate",
                fixed_inference_batch=256)


def _surrogate_error(model, pins, targets):
    model.eval()
    with torch.no_grad():
        pred = model(pins).detach().cpu().numpy().astype(np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(pred).all():
        raise RuntimeError("nonfinite surrogate predictions")
    error = pred - targets
    from scipy.stats import rankdata
    rp, ry = rankdata(pred), rankdata(targets)
    corr = float(np.corrcoef(rp, ry)[0, 1]) if rp.std() > 0 and ry.std() > 0 else None
    return dict(nets=len(pins), mse=float(np.mean(error ** 2)),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()), spearman=corr)


def _pack(step, price, ema, objects, oracle, validation=None):
    saved = J._pack(step, price, *objects, oracle, validation=validation)
    saved["ema_cost"] = ema
    return saved


def _summary(history, rho):
    costs = np.asarray([row["mean_cost"] for row in history])
    lambdas = np.asarray([history[0]["lambda_before"]] + [row["lambda_after"] for row in history])
    errors = np.asarray([row["surrogate_error"]["mse"] for row in history])
    return dict(mean_cost=float(costs.mean()), std_cost=float(costs.std()),
                rho=float(rho), mean_cost_minus_rho=float(costs.mean() - rho),
                mean_absolute_cost_minus_rho=float(np.abs(costs - rho).mean()),
                fraction_cost_above_rho=float(np.mean(costs > rho)),
                cost_total_variation=float(np.abs(np.diff(costs)).sum()),
                lambda_total_variation=float(np.abs(np.diff(lambdas)).sum()),
                mean_surrogate_mse=float(errors.mean()),
                total_search_seconds=sum(row["search_seconds"] for row in history),
                total_calibration_seconds=sum(row["calibration_seconds"] for row in history),
                total_policy_update_seconds=sum(row["policy_update_seconds"] for row in history),
                calibration_updates=sum(row["online"] is not None for row in history),
                committed_search_calls=sum(row["search_ledger_delta"]["actor_completed"] for row in history),
                committed_sampled_calls=sum(row["sampled"]["ledger_delta"]["sampled_actor_completed"] for row in history),
                last32_mean_cost=float(costs[-32:].mean()),
                last32_std_cost=float(costs[-32:].std()),
                last32_mean_surrogate_mse=float(errors[-32:].mean()))


def run(args):
    settings = settings_for(args.variant, args.smoke)
    os.environ["ADV_V3_REFERENCE_WORKERS"] = str(settings["reference_workers"])
    identity = _identity(args, settings)
    identity_hash = J._digest(identity)
    task = f"{args.variant}_d30_s{args.seed}"
    out = Path(args.outdir) if args.outdir else ROOT / "artifacts" / ("training_smoke" if args.smoke else "training") / task
    out = out.absolute()
    allowed = (PROJECT.resolve(), (PROJECT / "experiments").resolve())
    if not out.is_relative_to(PROJECT.absolute()) or not any(out.resolve().is_relative_to(p) for p in allowed):
        raise ValueError("all outputs must remain under DRO/Codex")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "task.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ipath = out / "identity.json"
        if ipath.exists() and json.loads(ipath.read_text()) != identity:
            raise RuntimeError("source/config/model identity changed; refusing resume")
        if not ipath.exists():
            C.atomic_json(ipath, identity)
        if (out / "complete.json").exists():
            complete = json.loads((out / "complete.json").read_text())
            if complete["identity_hash"] != identity_hash:
                raise RuntimeError("completion identity changed")
            for binding in complete["artifacts"]:
                J._checked_artifact(binding)
            return complete
        stop, heartbeat_stop = threading.Event(), threading.Event()
        state = dict(task=task, phase="load", update=0, physical_gpu_index=args.physical_gpu_index)
        C.atomic_json(out / "launch.json", {**state, "pid": os.getpid(), "at": time.time(),
            "process_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
            "command": sys.argv, "identity_hash": identity_hash})
        hb = threading.Thread(target=J._heartbeat, args=(out / "heartbeat.json", heartbeat_stop, state), daemon=True)
        hb.start()
        old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: stop.set())
        oracle = None
        began = time.monotonic()
        try:
            J._configure(C.derived_seed("tmp2-candidate07-main", args.seed))
            torch.set_num_threads(settings["torch_threads"])
            if not torch.cuda.is_available() and not args.allow_cpu:
                raise RuntimeError("GPU validation requires CUDA; --allow-cpu is only for diagnostics")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.cuda.set_per_process_memory_fraction(settings["gpu_memory_fraction"], device)
            actor, critic, _, _ = C.load_parent("erm", 30, args.seed, device)
            price, rho = C.parent_price(30, args.seed)
            initial_price = price
            if not np.isfinite([price, rho]).all() or price < 0 or rho <= 0:
                raise ValueError("invalid inherited lambda/rho")
            optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=settings["learning_rate"])
            surrogate = InputCritic(critic, calibrated=True).to(device)
            saved_surrogate = torch.load(identity["dependencies"]["surrogate"]["path"], map_location="cpu", weights_only=False)
            expected_surrogate_seed = C.derived_seed("critic", 30, "erm", args.seed)
            if saved_surrogate["identity"]["degree"] != 30 or saved_surrogate["identity"]["seed"] != expected_surrogate_seed:
                raise RuntimeError("surrogate parent mismatch")
            surrogate.load_state_dict(saved_surrogate["model_state_dict"], strict=True)
            surrogate.eval()
            surrogate_optimizer = torch.optim.Adam(surrogate.parameters(), lr=settings["critic_learning_rate"])
            objects = (actor, critic, optimizer, None, None, surrogate, surrogate_optimizer)
            oracle = C.Oracle(actor, device, out / "oracle", microbatch=256)
            def data(family, distribution, count, step=None):
                name = f"{family}__{distribution}" if step is None else f"train__u{step:06d}"
                namespace = identity["data_namespaces"][family]
                if step is not None:
                    namespace = namespace.format(step=step)
                return J._input(out / "inputs" / (name + ".npz"), degree=30, count=count,
                    distribution=distribution, namespace=namespace,
                    seed=args.seed if family == "train" else 0, identity_hash=identity_hash)
            val_data = {d: data("validation", d, settings["validation_count"]) for d in J.DISTRIBUTIONS}
            def evaluate(datasets, step, kind):
                return J._evaluate(actor, oracle, datasets, out, identity_hash=identity_hash,
                    update=step, kind=kind, batch_size=settings["evaluation_batch_size"],
                    degree=30, seed=args.seed, state=state, pause=stop)
            latest = out / "latest.pt"
            if latest.exists():
                saved = torch.load(latest, map_location="cpu", weights_only=False)
                if saved["identity_hash"] != identity_hash:
                    raise RuntimeError("checkpoint identity mismatch")
                current, best, history = saved["current"], saved["best"], saved["history"]
                initial_val = saved["initial_validation"]
                J._restore(current, *objects)
                step, price, ema = int(current["update"]), float(current["price"]), current["ema_cost"]
                if [r["update"] for r in history] != list(range(1, step + 1)):
                    raise RuntimeError("incomplete checkpoint history")
                C.atomic_json(out / "recovery" / f"{time.time_ns()}.json", dict(update=step,
                    actual_ledger=oracle.ledger, checkpoint_ledger=current["oracle_ledger_snapshot"],
                    uncommitted_or_later_phase_delta=J._delta(current["oracle_ledger_snapshot"], oracle.ledger),
                    accounting="actual counters retained without rollback"))
                J._write_history(out, history)
            else:
                step, ema, history = 0, None, []
                initial_val = evaluate(val_data, 0, "validation")
                current = _pack(0, price, ema, objects, oracle, initial_val)
                best = copy.deepcopy(current)
                C.atomic_torch(out / "initial.pt", current)
                C.atomic_torch(latest, dict(identity_hash=identity_hash, current=current,
                    best=best, history=history, initial_validation=initial_val))
            C.atomic_torch(out / "best.pt", best)
            for step in range(step + 1, settings["updates"] + 1):
                if stop.is_set() or (args.stop_after is not None and step > args.stop_after):
                    raise J._Paused()
                state.update(phase="training", update=step)
                x = data("train", "uniform", settings["batch_size"], step)
                before, geometry_before = dict(oracle.ledger), dict(C.GEOMETRY_LEDGER)
                tick, price_before = time.monotonic(), price
                actor_before = J._model_hash(actor)
                result = attack("critic_calibrated4", x, oracle, price,
                    C.derived_seed("tmp2-candidate07-search", args.seed, step),
                    proposer=lambda c, p, k, s: propose(surrogate, c, p, k, s))
                search_seconds = time.monotonic() - tick
                search_delta = J._delta(before, oracle.ledger)
                J._assert_greedy_budget(search_delta, len(x) * 4)
                pins = result["evaluated_pins"].reshape(-1, 30, 2)
                regrets = result["evaluated_regrets"].reshape(-1)
                prediction_error = _surrogate_error(surrogate, pins, regrets)
                calibration_started = time.monotonic()
                online = None
                if step % settings["critic_interval"] == 0:
                    labels_before = dict(oracle.ledger)
                    online = calibrate_online(surrogate, surrogate_optimizer, pins, regrets)
                    online.update(label_actor_update=step - 1, label_actor_sha256=actor_before,
                        label_source="this_step_paid_search_candidates", before_actor_update=True,
                        ledger_delta=J._delta(labels_before, oracle.ledger))
                    J._assert_greedy_budget(online["ledger_delta"], 0)
                calibration_seconds = time.monotonic() - calibration_started
                C.atomic_npz(out / "search" / f"u{step:06d}.npz", centers=x,
                    evaluated_pins=result["evaluated_pins"], evaluated_regrets=result["evaluated_regrets"],
                    selected_pins=result["pins"], selected_lstar=result["lstar"], selected_length=result["length"],
                    selected_regret=result["regret"], selected_cost=result["cost"], selected_index=result["selected_index"],
                    original_score=result["original_score"], score=result["score"],
                    actor_sha256=np.asarray(actor_before), identity_hash=np.asarray(identity_hash),
                    update=np.asarray(step), price=np.asarray(price))
                torch.manual_seed(C.derived_seed("tmp2-candidate07-sampled", args.seed, step))
                policy_started = time.monotonic()
                sampled = J.sampled_update(actor, critic, optimizer, oracle,
                    result["pins"], result["lstar"], settings["gradient_clip"])
                policy_seconds = time.monotonic() - policy_started
                cost = float(result["cost"].mean())
                price, ema, signal_cost = update_price(price, cost, rho, ema,
                    smoothing=settings["lambda_cost_smoothing"], beta=settings["ema_beta"],
                    learning_rate=settings["lambda_learning_rate"])
                validation, feasible, improved = None, False, False
                if step % settings["validation_interval"] == 0 or step == settings["updates"]:
                    validation = evaluate(val_data, step, "validation")
                    feasible, improved = J._is_better(validation, initial_val,
                        best["validation"], settings["nominal_degradation_max"])
                record = dict(update=step, input_sha256=J._array_hash(x), label_actor_sha256=actor_before,
                    lambda_before=price_before, lambda_after=price, rho=rho, mean_cost=cost,
                    ema_cost=ema, dual_signal_cost=signal_cost, mean_search_regret=float(result["regret"].mean()),
                    mean_search_score=float(result["score"].mean()),
                    mean_search_improvement=float((result["score"] - result["original_score"]).mean()),
                    search_calls_per_center=4, search_ledger_delta=search_delta,
                    search_seconds=search_seconds, calibration_seconds=calibration_seconds,
                    policy_update_seconds=policy_seconds, surrogate_error=prediction_error,
                    sampled=sampled, online=online, validation=validation,
                    validation_feasible=feasible, best_improved=improved,
                    ledger_delta=J._delta(before, oracle.ledger),
                    geometry_delta=J._delta(geometry_before, C.GEOMETRY_LEDGER),
                    wall_seconds=time.monotonic() - tick)
                history.append(record)
                current = _pack(step, price, ema, objects, oracle, validation)
                if improved:
                    best = copy.deepcopy(current)
                payload = dict(identity_hash=identity_hash, current=current, best=best,
                    history=history, initial_validation=initial_val)
                C.atomic_torch(latest, payload)
                C.atomic_json(out / "updates" / f"u{step:06d}.json", record)
                if improved:
                    C.atomic_torch(out / "best.pt", best)
                if step % settings["validation_interval"] == 0:
                    C.atomic_torch(out / "checkpoints" / f"u{step:06d}.pt", payload)
            J._write_history(out, history)
            test_data = {d: data("evaluation", d, settings["evaluation_count"]) for d in J.DISTRIBUTIONS}
            J._restore(current, *objects, rng=False)
            latest_eval = evaluate(test_data, settings["updates"], "last")
            initial = torch.load(out / "initial.pt", map_location="cpu", weights_only=False)
            actor.load_state_dict(initial["actor_state_dict"], strict=True)
            initial_eval = evaluate(test_data, 0, "initial")
            J._restore(best, *objects, rng=False)
            final_eval = evaluate(test_data, int(best["update"]), "final")
            data_audit = J._check_data_disjoint(out)
            summary = dict(status="pass", candidate=7, task=task, variant=args.variant,
                degree=30, seed=args.seed, smoke=bool(args.smoke), updates=settings["updates"],
                best_update=int(best["update"]), initial_validation=initial_val,
                best_validation=best["validation"], initial_evaluation=initial_eval,
                final_evaluation=final_eval, latest_evaluation=latest_eval, initial_price=initial_price,
                final_price=price, best_price=float(best["price"]), rho=rho,
                dynamics=_summary(history, rho), data_audit=data_audit,
                ledger=dict(oracle.ledger), history=history,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0,
                task_seconds_this_attempt=time.monotonic() - began,
                inference_device=str(device), evidence_scope="independent 128-update mechanism experiment; no continuous robustness certificate")
            oracle.close()
            oracle = None
            C.atomic_json(out / "summary.json", summary)
            exclude = {"complete.json", "heartbeat.json", "launch.json", "exit.json", "failure.json", "paused.json"}
            artifacts = [J._artifact(p) for p in sorted(out.rglob("*"))
                if p.is_file() and p.suffix in (".json", ".jsonl", ".npz", ".pt") and p.name not in exclude]
            complete = {**summary, "identity_hash": identity_hash, "identity": identity, "artifacts": artifacts}
            C.atomic_json(out / "complete.json", complete)
            state.update(phase="complete", update=settings["updates"])
            return complete
        except J._Paused:
            paused = dict(status="paused", **state, identity_hash=identity_hash, at=time.time())
            C.atomic_json(out / "paused.json", paused)
            return paused
        except Exception as error:
            C.atomic_json(out / "failure.json", dict(**state, error=repr(error), at=time.time()))
            raise
        finally:
            if oracle is not None:
                oracle.close()
            heartbeat_stop.set()
            hb.join(timeout=6)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            C.atomic_json(out / "exit.json", dict(**state, pid=os.getpid(), at=time.time(),
                complete=(out / "complete.json").exists()))
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=[101, 102, 103])
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--outdir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != str(args.physical_gpu_index):
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly --physical-gpu-index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_index)
    answer = run(args)
    print(json.dumps({key: answer[key] for key in ("status", "task", "best_update") if key in answer}))


if __name__ == "__main__":
    main()
