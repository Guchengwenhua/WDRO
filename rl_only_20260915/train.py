"""REST reinforcement learning using only its own action lengths (GPU acceptance)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

import geometry
from geometry import (atomic_json, atomic_npz, atomic_torch, build_actor,
                      derived_seed, lengths, make_data, sha256)
from search import CoordinateSearcher, ProposalPolicy, SetRegressor


HERE = Path(__file__).resolve().parent
METHODS = ("erm_critic", "erm_rollout", "wdro_random", "wdro_es", "wdro_critic", "wdro_rl")
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def proc_start():
    return Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def as_tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def write_event(path, event):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def protocol_defaults():
    return dict(degree=30, updates=2048, batch_size=64, actor_lr=1e-4,
                baseline_lr=1e-4, surrogate_lr=1e-4, proposal_lr=1e-4,
                candidates=8, epsilon=.08, search_sigma=.04, es_step_size=.04,
                critic_steps=4, critic_step_size=.01, lambda_init=1., lambda_lr=10.,
                validation_every=128, checkpoint_every=32, val_per_distribution=256,
                test_per_distribution=1024, attack_per_distribution=128,
                attack_candidates=32, greedy_microbatch=64,
                distributions=["uniform", "mix_gauss", "edge_bias", "grid_cluster"])


class Trainer:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.outdir).resolve()
        if not self.out.is_relative_to(HERE):
            raise ValueError("All run outputs must stay inside the RL-only branch")
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / "run.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write(json.dumps({"pid": os.getpid(), "start_ticks": proc_start(), "time": now()}))
        self.lock.flush()
        self.p = protocol_defaults()
        self.p.update(json.loads(Path(args.protocol).read_text()))
        if "rho" not in self.p:
            raise ValueError("Protocol must freeze the direct coordinate budget rho")
        if args.smoke:
            self.p.update(updates=2, batch_size=2, validation_every=1,
                          checkpoint_every=1, val_per_distribution=4,
                          test_per_distribution=4, attack_per_distribution=2)
        if int(self.p["candidates"]) != 8:
            raise ValueError("The frozen matched-budget design requires K=8")
        if int(self.p["greedy_microbatch"]) != 64:
            raise ValueError("Greedy microbatch is frozen to 64, including padded tails")
        if not math.isfinite(float(self.p["rho"])) or float(self.p["rho"]) < 0:
            raise ValueError("Invalid direct coordinate budget rho")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda" and not args.allow_cpu:
            raise RuntimeError("CUDA unavailable: CPU runs require explicit --allow-cpu")
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(.125, device=0)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.actor = build_actor(int(self.p["degree"]), self.device)
        self.initial_actor_sha256 = state_digest(self.actor)
        self.baseline = SetRegressor().to(self.device)
        self.surrogate = SetRegressor().to(self.device)
        self.proposal = ProposalPolicy(self.p["epsilon"], self.p["search_sigma"]).to(self.device)
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=self.p["actor_lr"]),
            "baseline": torch.optim.Adam(self.baseline.parameters(), lr=self.p["baseline_lr"]),
            "surrogate": torch.optim.Adam(self.surrogate.parameters(), lr=self.p["surrogate_lr"]),
            "proposal": torch.optim.Adam(self.proposal.parameters(), lr=self.p["proposal_lr"]),
        }
        self.networks = {"actor": self.actor, "baseline": self.baseline,
                         "surrogate": self.surrogate, "proposal": self.proposal}
        self.searcher = CoordinateSearcher(self.p, self.surrogate, self.proposal, self.device)
        dependencies = geometry.dependencies()
        self.binding = {"method": args.method, "seed": seed, "smoke": bool(args.smoke),
                        "protocol_sha256": sha256(Path(args.protocol)),
                        "code_sha256": {str(HERE / name): sha256(HERE / name)
                                        for name in ("geometry.py", "search.py", "train.py")},
                        "dependencies": dependencies, "effective_protocol": self.p}
        self.step = 0
        self.lam = float(self.p["lambda_init"])
        self.best_metric = float("inf")
        self.best_step = 0
        self.elapsed_prior = 0.
        self.update_seconds_total = 0.
        self.started = time.monotonic()
        self.ledger = {"search_greedy_scores": 0, "search_padded_forward_rows": 0,
                       "evaluation_greedy_scores": 0, "evaluation_padded_forward_rows": 0,
                       "rollout_greedy_scores": 0, "rollout_padded_forward_rows": 0,
                       "sampled_actor_scores": 0, "oracle_calls": 0}
        self.identity = {"pid": os.getpid(), "start_ticks": proc_start(), "time": now(),
                         "physical_gpu_index": args.physical_gpu_index,
                         "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "device": str(self.device), "torch_version": torch.__version__,
                         "numpy_version": np.__version__}
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.identity.update(gpu_name=props.name, gpu_total_memory=props.total_memory)
        latest = self.out / "latest.pt"
        if latest.exists():
            if not args.resume:
                raise RuntimeError("Existing checkpoint: pass --resume or use a fresh outdir")
            ckpt = torch.load(latest, map_location=self.device, weights_only=False)
            if ckpt["binding"] != self.binding:
                raise RuntimeError("Resume refused: protocol, method, seed, or source binding changed")
            for name, network in self.networks.items():
                network.load_state_dict(ckpt["networks"][name])
                self.optimizers[name].load_state_dict(ckpt["optimizers"][name])
            self.step = int(ckpt["step"])
            self.lam = float(ckpt["lambda"])
            self.best_metric = float(ckpt["best_metric"])
            self.best_step = int(ckpt["best_step"])
            self.elapsed_prior = float(ckpt["elapsed_seconds"])
            self.update_seconds_total = float(ckpt["update_seconds_total"])
            self.ledger = ckpt["ledger"]
            self.initial_actor_sha256 = ckpt["initial_actor_sha256"]
            set_rng_state(ckpt["rng"])
            for filename, digest in ckpt["checkpoint_artifacts"].items():
                if not (self.out / filename).exists() or sha256(self.out / filename) != digest:
                    if filename == "best.pt" and not (self.out / "complete.json").exists():
                        temporary = self.out / "best.pt.restore.partial"
                        with temporary.open("wb") as stream:
                            stream.write(ckpt["best_checkpoint_bytes"])
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, self.out / filename)
                    if sha256(self.out / filename) != digest:
                        raise RuntimeError(f"Resume artifact mismatch: {filename}")
            if not (self.out / "complete.json").exists():
                with (self.out / "progress.jsonl").open("r+b") as stream:
                    stream.truncate(ckpt["progress_bytes"])
        actual_path = self.out / "actual_calls.jsonl"
        if actual_path.exists() and not (self.out / "complete.json").exists():
            raw = actual_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                end = raw.rfind(b"\n") + 1
                (self.out / f"actual_calls_incomplete_tail_{time.time_ns()}.txt").write_bytes(raw[end:])
                with actual_path.open("r+b") as stream:
                    stream.truncate(end)
        if not (self.out / "complete.json").exists():
            atomic_json(self.out / "run.json", {"binding": self.binding, "identity": self.identity,
                                                "initial_actor_sha256": self.initial_actor_sha256})
        self.heartbeat("running")

    def elapsed(self):
        return self.elapsed_prior + time.monotonic() - self.started

    def heartbeat(self, status, **extra):
        seconds_per_update = self.update_seconds_total / self.step if self.step >= 2 else None
        atomic_json(self.out / "heartbeat.json", dict(self.identity, status=status,
                    updated=now(), step=self.step, updates=self.p["updates"],
                    phase=status, updates_completed=self.step, updates_target=self.p["updates"],
                    elapsed_seconds=self.elapsed(), seconds_per_update=seconds_per_update,
                    estimated_remaining_seconds=(seconds_per_update * (self.p["updates"] - self.step)
                                                 if seconds_per_update else None), **extra))

    def call_event(self, phase, count, padded, event, identifier=None):
        identifier = identifier or f"{os.getpid()}-{time.time_ns()}"
        write_event(self.out / "actual_calls.jsonl", {"id": identifier, "event": event,
                    "phase": phase, "real_rows": count, "forward_rows": padded,
                    "time": now(), "step": self.step})
        return identifier

    def score(self, pins, phase="evaluation"):
        count = len(pins)
        microbatch = int(self.p["greedy_microbatch"])
        call = self.call_event(phase, count, math.ceil(count / microbatch) * microbatch, "start")
        was_training = self.actor.training
        self.actor.eval()
        all_lengths = []
        all_topologies = []
        with torch.no_grad():
            for start in range(0, count, microbatch):
                batch = np.asarray(pins[start:start + microbatch], dtype=np.float32)
                keep = len(batch)
                if keep < microbatch:
                    batch = np.concatenate((batch, np.repeat(batch[-1:], microbatch - keep, 0)), 0)
                topology, _ = self.actor(batch, True)
                topology = topology[:keep].cpu().numpy()
                all_topologies.append(topology)
                all_lengths.append(lengths(batch[:keep], topology))
                self.ledger[f"{phase}_padded_forward_rows"] += microbatch
        self.actor.train(was_training)
        length = np.concatenate(all_lengths)
        if not np.isfinite(length).all() or (length <= 0).any():
            raise AssertionError("Invalid own-action length")
        self.ledger[f"{phase}_greedy_scores"] += count
        self.call_event(phase, count, math.ceil(count / microbatch) * microbatch, "complete", call)
        return {"length": length,
                "topology": np.concatenate(all_topologies)}

    def dataset(self, count, distribution, namespace):
        return make_data(count, int(self.p["degree"]), distribution, namespace, seed=0)

    def evaluate(self, namespace, count):
        arrays, summaries = {}, {}
        for distribution in self.p["distributions"]:
            pins = self.dataset(count, distribution, namespace + "/" + distribution)
            values = self.score(pins)
            arrays[distribution + "__pins"] = pins
            arrays.update({distribution + "__" + key: value for key, value in values.items()})
            summaries[distribution] = {"mean_length": float(values["length"].mean()),
                                       "p95_length": float(np.quantile(values["length"], .95))}
        worst = max(item["mean_length"] for item in summaries.values())
        return arrays, summaries, worst

    def validate(self):
        _, groups, worst = self.evaluate("validation", int(self.p["val_per_distribution"]))
        if worst < self.best_metric:
            self.best_metric = worst
            self.best_step = self.step
            atomic_torch(self.out / "best.pt", {"actor": self.actor.state_dict(),
                         "step": self.step, "metric": worst, "binding": self.binding})
        event = {"type": "validation", "time": now(), "step": self.step,
                 "groups": groups, "worst_group_mean_length": worst,
                 "best_metric": self.best_metric, "best_step": self.best_step,
                 "elapsed_seconds": self.elapsed()}
        write_event(self.out / "progress.jsonl", event)
        print(json.dumps(event), flush=True)

    def checkpoint(self):
        artifacts = {name: sha256(self.out / name) for name in ("best.pt", "initial.npz", "initial.json")
                     if (self.out / name).exists()}
        atomic_torch(self.out / "latest.pt", {"binding": self.binding, "step": self.step,
                     "networks": {key: value.state_dict() for key, value in self.networks.items()},
                     "optimizers": {key: value.state_dict() for key, value in self.optimizers.items()},
                     "lambda": self.lam, "best_metric": self.best_metric,
                     "best_step": self.best_step, "elapsed_seconds": self.elapsed(),
                     "update_seconds_total": self.update_seconds_total,
                     "ledger": self.ledger, "initial_actor_sha256": self.initial_actor_sha256,
                     "rng": rng_state(), "checkpoint_artifacts": artifacts,
                     "best_checkpoint_bytes": (self.out / "best.pt").read_bytes(),
                     "progress_bytes": (self.out / "progress.jsonl").stat().st_size})
        self.heartbeat("running")

    def update(self):
        update_started = time.monotonic()
        step = self.step + 1
        centers = make_data(int(self.p["batch_size"]), int(self.p["degree"]), "uniform",
                            f"train/{step}", seed=int(self.args.seed))
        search_result = None
        search_rng = np.random.default_rng(derived_seed("search", self.args.seed, step))
        if self.args.method.startswith("wdro_"):
            search_result = self.searcher.search(self.args.method, centers, self.lam,
                                               lambda pins: self.score(pins, "search"), search_rng)
            pins = search_result.selected
            mean_cost = float(search_result.selected_cost.mean())
            length_label = search_result.scores["length"]
            self.optimizers["surrogate"].zero_grad(set_to_none=True)
            prediction = self.surrogate(as_tensor(search_result.candidates.reshape(-1, self.p["degree"], 2), self.device))
            surrogate_loss = nn.functional.mse_loss(prediction, as_tensor(length_label.reshape(-1), self.device))
            surrogate_loss.backward()
            nn.utils.clip_grad_norm_(self.surrogate.parameters(), 1.)
            self.optimizers["surrogate"].step()
            if search_result.proposal_loss is not None:
                self.optimizers["proposal"].zero_grad(set_to_none=True)
                search_result.proposal_loss.backward()
                nn.utils.clip_grad_norm_(self.proposal.parameters(), 1.)
                self.optimizers["proposal"].step()
        else:
            pins = centers
            mean_cost = 0.
            surrogate_loss = torch.tensor(0.)
        # The rollout is generated by the current learned Actor itself. It is
        # evaluated without gradients and in eval mode, before sampling actions.
        # Consequently it is an action-independent REINFORCE control variate.
        rollout_baseline = None
        if self.args.method == "erm_rollout":
            rollout_baseline = as_tensor(self.score(pins, "rollout")["length"], self.device)
        # Identical per-update action RNG across methods; proposal sampling cannot
        # shift the paired Actor action stream. Full RNG states also checkpointed.
        action_seed = derived_seed("actor_action", self.args.seed, step)
        torch.manual_seed(action_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(action_seed)
        self.actor.train()
        call = self.call_event("sampled_training", len(pins), len(pins), "start")
        topology, log_prob = self.actor(pins, False)
        sampled_length = lengths(pins, topology.detach().cpu().numpy())
        self.call_event("sampled_training", len(pins), len(pins), "complete", call)
        own_length = as_tensor(sampled_length, self.device)
        prediction = rollout_baseline if rollout_baseline is not None else self.baseline(as_tensor(pins, self.device))
        actor_loss = ((own_length - prediction).detach() * log_prob).mean()
        baseline_loss = (torch.zeros((), device=self.device) if rollout_baseline is not None
                         else nn.functional.mse_loss(prediction, own_length.detach()))
        if not torch.isfinite(actor_loss + baseline_loss):
            raise FloatingPointError("Nonfinite training loss")
        self.optimizers["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        self.optimizers["actor"].step()
        if rollout_baseline is None:
            self.optimizers["baseline"].zero_grad(set_to_none=True)
            baseline_loss.backward()
            nn.utils.clip_grad_norm_(self.baseline.parameters(), 1.)
            self.optimizers["baseline"].step()
        if search_result is not None:
            self.lam = max(0., self.lam + float(self.p["lambda_lr"]) * (mean_cost - float(self.p["rho"])))
        self.ledger["sampled_actor_scores"] += len(pins)
        self.step = step
        self.update_seconds_total += time.monotonic() - update_started
        event = {"type": "update", "time": now(), "step": step,
                 "sampled_length": float(own_length.mean()), "actor_loss": float(actor_loss.detach()),
                 "mean_baseline_prediction": float(prediction.detach().mean()),
                 "baseline_loss": float(baseline_loss.detach()),
                 "surrogate_loss": float(surrogate_loss.detach()), "lambda": self.lam,
                 "mean_selected_cost": mean_cost, "elapsed_seconds": self.elapsed()}
        if search_result is not None:
            event["search_greedy_length"] = float(search_result.scores["length"][np.arange(len(pins)), search_result.selected_index].mean())
            event["search_center_length"] = float(search_result.scores["length"][:, 0].mean())
            event["search_no_move_fraction"] = float((search_result.selected_index == 0).mean())
        write_event(self.out / "progress.jsonl", event)
        if step <= 3 or step % 32 == 0:
            print(json.dumps(event), flush=True)
        self.heartbeat("running", last_update_seconds=event["elapsed_seconds"])

    def finish(self):
        self.heartbeat("evaluating")
        best = torch.load(self.out / "best.pt", map_location=self.device, weights_only=False)
        self.actor.load_state_dict(best["actor"])
        final_arrays, final_groups, final_worst = self.evaluate("test", int(self.p["test_per_distribution"]))
        with np.load(self.out / "initial.npz", allow_pickle=False) as initial:
            result_arrays = {"initial__" + key: initial[key] for key in initial.files}
        result_arrays.update({"final__" + key: value for key, value in final_arrays.items()})
        attacks = {}
        for distribution in self.p["distributions"]:
            pins = self.dataset(int(self.p["attack_per_distribution"]), distribution,
                                "attack/" + distribution)
            rng = np.random.default_rng(derived_seed("heldout_attack", distribution, 0))
            found = self.searcher.search("random", pins, 0., self.score, rng,
                                        candidate_count=int(self.p["attack_candidates"]))
            prefix = "attack__" + distribution + "__"
            result_arrays[prefix + "pins"] = found.candidates
            result_arrays[prefix + "cost"] = found.transport
            result_arrays[prefix + "selected_index"] = found.selected_index
            result_arrays.update({prefix + name: value for name, value in found.scores.items()})
            selected = found.scores["length"][np.arange(len(pins)), found.selected_index]
            attacks[distribution] = {"mean_center_length": float(found.scores["length"][:, 0].mean()),
                                     "mean_attacked_length": float(selected.mean()),
                                     "mean_lift": float((selected - found.scores["length"][:, 0]).mean())}
        atomic_npz(self.out / "results.npz", **result_arrays)
        expected_search = int(self.p["updates"]) * int(self.p["batch_size"]) * int(self.p["candidates"]) if self.args.method.startswith("wdro_") else 0
        if self.ledger["search_greedy_scores"] != expected_search:
            raise AssertionError(f"Search budget mismatch: {self.ledger['search_greedy_scores']} != {expected_search}")
        if self.ledger["sampled_actor_scores"] != int(self.p["updates"]) * int(self.p["batch_size"]):
            raise AssertionError("Sampled Actor budget mismatch")
        expected_rollout = int(self.p["updates"]) * int(self.p["batch_size"]) if self.args.method == "erm_rollout" else 0
        if self.ledger["rollout_greedy_scores"] != expected_rollout:
            raise AssertionError("Self-critical rollout budget mismatch")
        forbidden_markers = ("geosteiner", "libeval", "scipy.optimize", "networkx.algorithms.tree.mst")
        forbidden_modules = sorted(name for name in sys.modules
                                   if any(marker in name.lower() for marker in forbidden_markers))
        if forbidden_modules:
            raise AssertionError(f"Forbidden oracle modules loaded: {forbidden_modules}")
        initial_info = json.loads((self.out / "initial.json").read_text())
        attempts, completed = {}, set()
        with (self.out / "actual_calls.jsonl").open() as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["event"] == "start":
                    attempts[entry["id"]] = entry
                else:
                    completed.add(entry["id"])
        actual = {}
        for phase in ("search", "evaluation", "rollout", "sampled_training"):
            entries = [item for item in attempts.values() if item["phase"] == phase]
            actual[phase] = {"completed_real_rows": sum(item["real_rows"] for item in entries if item["id"] in completed),
                             "attempted_real_rows_upper_bound": sum(item["real_rows"] for item in entries),
                             "completed_forward_rows": sum(item["forward_rows"] for item in entries if item["id"] in completed),
                             "attempted_forward_rows_upper_bound": sum(item["forward_rows"] for item in entries)}
        acceptance = ("gpu_smoke" if self.device.type == "cuda" else "cpu_smoke") if self.args.smoke else ("accepted_gpu" if self.device.type == "cuda" else "cpu_only_complete")
        summary = {"binding": self.binding, "identity": self.identity,
                   "status": "complete", "acceptance": acceptance,
                   "step": self.step, "best_step": self.best_step,
                   "best_validation_worst_group_length": self.best_metric,
                   "initial_actor_sha256": self.initial_actor_sha256,
                   "final_actor_sha256": state_digest(self.actor),
                   "initial_groups": initial_info["groups"], "final_groups": final_groups,
                   "final_test_worst_group_length": final_worst, "attacks": attacks,
                   "lambda_final": self.lam, "ledger": self.ledger,
                   "reward_source": "raw length of current learned Actor's own sampled topology",
                   "control_variate": ("current learned Actor greedy rollout" if self.args.method == "erm_rollout"
                                        else "independent learned value network fitted to own sampled lengths"),
                   "adversarial_score": "own greedy Actor length minus lambda times direct indexwise mean L1",
                   "ledger_semantics": "Committed work; actual_calls.jsonl and actual_call_bounds include replay after interruption",
                   "actual_call_bounds": actual,
                   "elapsed_seconds": self.elapsed(), "finished": now(),
                   "update_seconds_total": self.update_seconds_total,
                   "seconds_per_update": self.update_seconds_total / self.step,
                   "forbidden_modules_loaded": forbidden_modules,
                   "oracle_guard": "geometry.install_oracle_guard active before model construction",
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0,
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.device.type == "cuda" else 0,
                   "finite_result_validation": True,
                   "artifacts": {name: sha256(self.out / name) for name in
                                 ("latest.pt", "best.pt", "initial.npz", "initial.json", "results.npz", "progress.jsonl", "actual_calls.jsonl", "run.json")}}
        summary["artifacts_sha256"] = summary["artifacts"]
        atomic_json(self.out / "complete.json", summary)
        self.heartbeat(summary["status"])
        print(json.dumps({"status": summary["status"], "outdir": str(self.out),
                          "elapsed_seconds": self.elapsed(), "final_groups": final_groups}), flush=True)

    def run(self):
        if (self.out / "complete.json").exists():
            complete = json.loads((self.out / "complete.json").read_text())
            if complete["binding"] != self.binding:
                raise RuntimeError("Completed run binding mismatch")
            for name, digest in complete["artifacts"].items():
                if sha256(self.out / name) != digest:
                    raise RuntimeError(f"Completed artifact changed: {name}")
            self.heartbeat(complete["status"])
            print(json.dumps({"already_complete": str(self.out), "status": complete["status"]}), flush=True)
            return 0
        if self.step == 0 and not (self.out / "latest.pt").exists():
            arrays, groups, _ = self.evaluate("test", int(self.p["test_per_distribution"]))
            atomic_npz(self.out / "initial.npz", **arrays)
            atomic_json(self.out / "initial.json", {"groups": groups,
                        "initial_actor_sha256": self.initial_actor_sha256})
            self.validate()
            self.checkpoint()
        while self.step < int(self.p["updates"]):
            if STOP_REQUESTED:
                self.checkpoint()
                self.heartbeat("paused", reason="signal")
                return 143
            self.update()
            if self.step % int(self.p["validation_every"]) == 0 or self.step == int(self.p["updates"]):
                self.validate()
            if self.step % int(self.p["checkpoint_every"]) == 0:
                self.checkpoint()
            if self.args.stop_after is not None and self.step >= self.args.stop_after:
                self.checkpoint()
                self.heartbeat("paused", reason="stop_after")
                return 0
        self.checkpoint()
        self.finish()
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--protocol", default=str(HERE / "protocol.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.physical_gpu_index is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_index) if args.physical_gpu_index >= 0 else ""
    geometry.install_oracle_guard()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    trainer = None
    try:
        trainer = Trainer(args)
        return trainer.run()
    except Exception as error:
        if trainer is not None:
            trainer.heartbeat("failed", error=repr(error))
            atomic_json(trainer.out / "FAILED.json", {"time": now(), "step": trainer.step,
                        "error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        if trainer is not None:
            fcntl.flock(trainer.lock, fcntl.LOCK_UN)
            trainer.lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
