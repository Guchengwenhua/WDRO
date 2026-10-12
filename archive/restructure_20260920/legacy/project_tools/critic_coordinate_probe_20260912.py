"""Read-only CPU diagnostic; prints JSON without editing models or caches."""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "revisions/reference_precision_v2/Codex/wdro_rest_min"
sys.path.insert(0, str(CODE))
from src.geometry import normalize_net
from src.rest import build_critic


def main():
    torch.set_num_threads(1)
    models = json.loads(
        (CODE / "manifests/stage6_models_frozen_v1.json").read_text()
    )["models"]
    records = []
    for method, degree, seed in [
        ("erm", 10, 101), ("wdro_rest", 10, 101), ("wdro_rest", 50, 101)
    ]:
        model = next(
            m for m in models
            if (m["method"], m["degree"], m["seed"]) == (method, degree, seed)
        )
        path = Path(model["best"]["logical_path"])
        # These are the project's trusted local checkpoints.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        critic = build_critic(degree, torch.device("cpu"))
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        critic.eval()
        rng = np.random.default_rng(43217 + degree)
        pins = np.stack([
            normalize_net(rng.uniform(size=(degree, 2)))[0] for _ in range(8)
        ]).astype(np.float32)
        z = torch.tensor(pins, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            native = critic(z)
        native_grad = torch.autograd.grad(native.sum(), z, allow_unused=True)[0]
        encoded = critic.crit_encoder(critic.crit_embedder(z), None)
        inner = critic.relu(critic.critic_l1(critic.glimpse(encoded)))
        pre = critic.critic_l2(inner).squeeze(-1)
        prediction = critic.relu(pre)
        grad = torch.autograd.grad(prediction.sum(), z)[0]
        records.append({
            "method": method, "degree": degree, "seed": seed,
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_update": checkpoint["update"],
            "input_seed": 43217 + degree,
            "nets": len(pins),
            "native_input_gradient_is_none": native_grad is None,
            "forward_bypass_exactly_matches": bool(torch.equal(native, prediction)),
            "predictions": prediction.tolist(),
            "pre_relu": pre.tolist(),
            "coord_grad_l2_per_net": torch.linalg.vector_norm(
                grad.flatten(1), dim=1
            ).tolist(),
            "coord_grad_finite": bool(torch.isfinite(grad).all()),
            "parameters": sum(p.numel() for p in critic.parameters()),
            "batchnorm_layers": sum(
                isinstance(n, torch.nn.BatchNorm1d) for n in critic.modules()
            ),
            "dropout_probabilities": [
                n.p for n in critic.modules() if isinstance(n, torch.nn.Dropout)
            ],
        })
    print(json.dumps({
        "date": "2026-09-12",
        "kind": "coordinate_gradient_diagnostic_not_attack_evaluation",
        "device": "cpu", "torch_threads": 1,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "checkpoint_net_checks": 24, "distinct_geometries": 16,
        "actor_calls": 0, "rest_calls": 0, "geosteiner_calls": 0,
        "parameter_updates": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_generation": "default_rng(43217+degree); 8 uniform [n,2] nets; normalize_net; float32",
        "records": records,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
