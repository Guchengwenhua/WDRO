"""Synthetic-validation-only loading and deterministic REST evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .accounting import CallLedger
from .datasets import load_dataset
from .metrics import relative_regret, summarize_groups, summarize_regret
from .rest import RestTreeEvaluator


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "cache" / "datasets" / "protocol_v1"
DEFAULT_LSTAR_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "cache" / "lstar" / "protocol_v1"


def load_validation_group(
    protocol: dict[str, Any],
    *,
    degree: int,
    distribution: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    lstar_root: str | Path = DEFAULT_LSTAR_ROOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    count = protocol["data"]["sizes_per_degree_distribution"]["validation"]
    seed = protocol["data"]["seeds"]["validation"]
    dataset_path = Path(dataset_root) / "validation" / f"{distribution}_d{degree}_s{seed}_n{count}.npz"
    arrays, metadata = load_dataset(dataset_path, verify=True)
    if metadata["split"] != "validation" or metadata["base_seed"] != seed:
        raise ValueError("validation loader refused a non-validation dataset")
    if metadata["distribution"] != distribution or metadata["degree"] != degree:
        raise ValueError("validation dataset identity mismatch")
    if not all(item.decode("utf-8").startswith("validation:") for item in arrays["net_ids"]):
        raise ValueError("validation net ID namespace mismatch")
    lstar_path = Path(lstar_root) / f"{metadata['array_bundle_sha256']}.npz"
    lstar_arrays, lstar_metadata = load_dataset(lstar_path, verify=True)
    from .geosteiner import reference_identity
    if lstar_metadata.get("reference_identity") != reference_identity():
        raise ValueError("reference precision identity mismatch")
    if lstar_metadata["dataset_array_bundle_sha256"] != metadata["array_bundle_sha256"]:
        raise ValueError("L_star cache refers to a different dataset")
    if not np.array_equal(arrays["net_ids"], lstar_arrays["net_ids"]):
        raise ValueError("L_star cache net IDs differ from validation data")
    return arrays["pins"], lstar_arrays["lstar"], arrays["net_ids"], metadata


def evaluate_validation(
    actor: torch.nn.Module,
    rest_evaluator: RestTreeEvaluator,
    protocol: dict[str, Any],
    *,
    degree: int,
    device: torch.device,
    batch_size: int,
    ledger: CallLedger,
) -> dict[str, Any]:
    actor.eval()
    grouped_regrets: list[np.ndarray] = []
    grouped_labels: list[str] = []
    per_group: dict[str, Any] = {}
    for distribution in protocol["data"]["distributions"]:
        pins, lstar, net_ids, metadata = load_validation_group(
            protocol, degree=degree, distribution=distribution
        )
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pins), batch_size):
                batch = pins[start : start + batch_size]
                topology, _ = actor(batch, True)
                outputs.append(topology.detach().cpu().numpy())
                ledger.add("actor_forward_nets", len(batch))
        all_outputs = np.concatenate(outputs, axis=0)
        lengths = rest_evaluator.lengths(pins, all_outputs)
        ledger.add("exact_rest_evaluator_nets", len(pins))
        regrets = relative_regret(lengths, lstar)
        if not np.isfinite(regrets).all():
            raise RuntimeError("non-finite validation regret")
        per_group[distribution] = {
            "summary": summarize_regret(regrets),
            "dataset_array_bundle_sha256": metadata["array_bundle_sha256"],
            "first_net_id": net_ids[0].decode("utf-8"),
            "last_net_id": net_ids[-1].decode("utf-8"),
        }
        grouped_regrets.append(regrets)
        grouped_labels.extend([distribution] * len(regrets))
    all_regrets = np.concatenate(grouped_regrets)
    aggregate = summarize_groups(all_regrets, grouped_labels)
    ood_names = protocol["training"]["selection"]["worst_group_scope"]
    ood_worst_group = max(ood_names, key=lambda name: per_group[name]["summary"]["mean"])
    result = {
        "split": "validation",
        "seed": protocol["data"]["seeds"]["validation"],
        "degree": degree,
        "inference": {
            "mode": "deterministic_greedy",
            "symmetry_transforms": 1,
            "batch_size": batch_size,
        },
        "groups": per_group,
        "aggregate": aggregate,
        "selection": {
            "objective": "minimum_worst_group_relative_regret",
            "scope": ood_names,
            "worst_group": ood_worst_group,
            "worst_group_mean": per_group[ood_worst_group]["summary"]["mean"],
            "nominal_mean": per_group["uniform"]["summary"]["mean"],
            "test_or_ispd_used": False,
        },
    }
    actor.train()
    return result


def validation_manifest_hash(result: dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
