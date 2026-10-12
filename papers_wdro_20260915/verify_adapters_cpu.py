"""Bounded CPU integration checks for the two official routing adapters."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

# Exercise coexistence with the frozen mainline's src namespace as in runner.
import core
from adapter_timing import Adapter as Timing
from adapter_hub import Adapter as Hub


def main():
    root = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    pins = np.random.default_rng(82913).random((8, 10, 2), dtype=np.float32)
    records = []
    for name, cls, params in (("timing", Timing, {"alpha": .5}), ("hub", Hub, {"beta": .5, "policy_bonus_weight": 0.0})):
        torch.manual_seed(711)
        model = cls(10, "cpu", params)
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            cost, logp, metrics = model.evaluate(pins, True)
            structure = (model.last_edges if name == "timing" else model.last_hubs).copy()
            cost2, _, metrics2 = model.evaluate(pins, True)
        assert torch.equal(cost, cost2)
        assert metrics["legal"].all()
        assert np.isfinite(metrics["raw_objective"]).all()
        assert np.allclose(cost.numpy(), metrics["raw_objective"] / 18)
        assert np.all(metrics["policy_reward"] == 0)
        assert all(v.shape == (len(pins),) for v in metrics.values())
        if name == "timing":
            assert np.array_equal(structure, model.last_edges)
            expected = .5 * (metrics["wirelength"] + metrics["max_path"])
        else:
            assert np.array_equal(structure, model.last_hubs)
            expected = .5 * (metrics["wirelength"] + metrics["skew"])
            # Empty hub sets are an intentional legal stop action in release.
            lengths, skews, values = model._evaluator.evaluate_batch(pins[:1], np.zeros((1,5),dtype=int), objective=.5)
            assert np.isfinite(values).all() and (lengths >= 0).all() and (skews >= 0).all()
        assert np.allclose(expected, metrics["raw_objective"])
        model.train()
        cost, logp, _ = model.evaluate(pins, False)
        assert not cost.requires_grad and logp.requires_grad
        (cost * logp).mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        grad_l1 = sum(float(g.abs().sum()) for g in gradients)
        assert grad_l1 > 0
        records.append({"adapter": name, "batch":8,"degree":10,"greedy_deterministic":True,"all_structures_legal":True,"objective_identity":True,"detached_cost_differentiable_logp":True,"gradient_finite_nonzero":True,"gradient_l1":grad_l1,"seconds":time.perf_counter()-start})
    evidence = {"status":"PASS","scope":"CPU integration, not GPU acceptance or trained quality","checks":records,"files":{str(root/n):hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ("adapter_timing.py","adapter_hub.py","verify_adapters_cpu.py")}}
    out = root / "artifacts" / "adapters_cpu_checks.json"
    out.write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(evidence,indent=2))


if __name__ == "__main__":
    main()
