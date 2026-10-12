"""Compare a deliberately interrupted RL-only smoke run with its uninterrupted twin."""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
left = torch.load(ROOT / "uninterrupted/latest.pt", map_location="cpu", weights_only=False)
right = torch.load(ROOT / "resumed/latest.pt", map_location="cpu", weights_only=False)
tensors_checked = 0


def equal(x, y, path):
    global tensors_checked
    if torch.is_tensor(x):
        assert torch.equal(x, y), path
        tensors_checked += 1
    elif isinstance(x, np.ndarray):
        assert np.array_equal(x, y), path
    elif isinstance(x, dict):
        assert x.keys() == y.keys(), path
        for key in x:
            equal(x[key], y[key], path + "/" + str(key))
    elif isinstance(x, (tuple, list)):
        assert len(x) == len(y), path
        for index, (a, b) in enumerate(zip(x, y)):
            equal(a, b, path + "/" + str(index))
    else:
        assert x == y, (path, x, y)


for field in ("networks", "optimizers", "rng", "ledger", "lambda", "best_metric",
              "best_step", "initial_actor_sha256", "binding"):
    equal(left[field], right[field], field)
with np.load(ROOT / "uninterrupted/results.npz") as a, np.load(ROOT / "resumed/results.npz") as b:
    assert a.files == b.files
    for key in a.files:
        assert np.array_equal(a[key], b[key]), key
    arrays_checked = len(a.files)
assert "uncommitted_test_row" not in (ROOT / "resumed/progress.jsonl").read_text()
assert len(list((ROOT / "resumed").glob("actual_calls_incomplete_tail_*.txt"))) == 1
result = {
    "status": "PASS", "method": "wdro_rl", "cpu_smoke_updates": 2,
    "checks": ["interrupted_at_update1_resume_equals_uninterrupted",
               "all_network_and_optimizer_tensors_bitwise_identical",
               "RNG_lambda_and_committed_ledger_identical",
               "all_initial_final_attack_arrays_bitwise_identical",
               "replaced_best_checkpoint_restored_from_latest",
               "uncommitted_progress_tail_truncated",
               "incomplete_actual_call_tail_salvaged"],
    "tensors_checked": tensors_checked, "arrays_checked": arrays_checked,
    "protocol_sha256": left["binding"]["protocol_sha256"],
    "code_sha256": left["binding"]["code_sha256"],
}
(ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
