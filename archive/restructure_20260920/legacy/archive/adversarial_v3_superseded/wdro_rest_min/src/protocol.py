"""Load, hash, and validate the frozen WDRO-REST protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocol_v1.json"


class ProtocolError(ValueError):
    """Raised when a protocol file violates the frozen schema invariants."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def protocol_sha256(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)
    return protocol


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Validate invariants that must not drift between experiment runs."""

    _require(protocol.get("schema_version") == 1, "schema_version must be 1")
    _require(protocol.get("status") == "frozen", "protocol status must be frozen")

    scope = protocol["scope"]
    _require(
        scope["methods"] == ["erm", "mixed_erm", "hard_mining", "group_dro", "wdro_rest"],
        "the five primary methods changed",
    )
    _require(scope["degrees"] == [10, 20, 30, 40, 50], "degree grid changed")

    data = protocol["data"]
    _require(data["seeds"]["training"] == [101, 102, 103], "training seeds changed")
    _require(data["seeds"]["validation"] == 202, "validation seed changed")
    _require(data["seeds"]["test"] == 303, "test seed changed")
    _require(
        len({*data["seeds"]["training"], data["seeds"]["validation"], data["seeds"]["test"]}) == 5,
        "training, validation, and test seeds must be disjoint",
    )
    _require(data["sizes_per_degree_distribution"]["validation"] == 2000, "validation size changed")
    _require(data["sizes_per_degree_distribution"]["test"] == 10000, "test size changed")
    expected_distributions = {
        "uniform": {"nonuniform": 0.0},
        "mix_gauss": {"nonuniform": 0.8, "clusters": 6, "sigma": 0.04},
        "edge_bias": {"nonuniform": 0.8, "clusters": 4, "sigma": 0.06},
        "grid_cluster": {"nonuniform": 0.8, "clusters": 8, "sigma": 0.03},
    }
    _require(data["distributions"] == expected_distributions, "distribution parameters changed")

    preprocessing = protocol["preprocessing"]
    _require(preprocessing["normalized_interval"] == [0.1, 0.9], "normalization interval changed")
    _require(preprocessing["preserve_aspect_ratio"], "aspect ratio preservation is required")
    _require(preprocessing["inverse_transform_required"], "inverse transform is required")

    training = protocol["training"]
    _require(training["policy_updates"] == 2000, "policy update budget changed")
    _require(training["batch_size"] == 256, "batch size changed")
    _require(
        training["selection"]["nominal_mean_regret_degradation_max_absolute"] == 0.002,
        "nominal degradation constraint changed",
    )
    _require(not training["selection"]["test_data_used_for_selection"], "test leakage is forbidden")
    _require(not training["selection"]["ispd_used_for_selection"], "ISPD selection leakage is forbidden")

    metrics = protocol["loss_and_metrics"]
    _require(metrics["formula"] == "(L_REST - L_star) / L_star", "relative regret formula changed")
    _require(metrics["reference"] == "GeoSteiner", "reference evaluator changed")
    _require(metrics["confidence_level"] == 0.95, "confidence level changed")

    inference = protocol["inference"]
    _require(inference["symmetry_transforms"] == 1, "inference T must equal 1")
    _require(inference["mode"] == "deterministic_greedy", "inference must be deterministic greedy")

    wdro = protocol["method_contracts"]["wdro_rest"]
    _require(wdro["candidate_count"] == 32, "main WDRO K changed")
    _require(wdro["ground_cost"] == "hungarian_l1_mean", "main WDRO cost changed")
    _require(wdro["lambda"] == "learned_nonnegative", "main WDRO lambda changed")
    _require(wdro["include_nominal_candidate"], "nominal candidate must be included")
    _require(
        wdro["selection_unit"] == "one_argmax_per_nominal_center",
        "WDRO must select independently per nominal center",
    )

    ispd = protocol["ispd"]
    _require(ispd["protocol"] == "A", "ISPD protocol changed")
    _require(not ispd["researcher_blind_claim_allowed"], "researcher-blind claim is not allowed")

    runtime = protocol["runtime"]
    _require(runtime["allowed_gpu_indices"] == [0, 1, 2, 3], "allowed GPU set changed")
    _require(runtime["forbidden_gpu_indices"] == [4, 5, 6, 7], "forbidden GPU set changed")
    _require(runtime["minimum_free_disk_gib_for_batch"] >= 100, "batch disk floor is below 100 GiB")
