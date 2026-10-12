"""Check accounting boundaries, CUDA-only calibration and incomplete logs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("papers_eta", ROOT / "eta.py")
eta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eta)


class EtaTests(unittest.TestCase):
    def test_cpu_measurements_do_not_calibrate_cuda(self):
        protocol = dict(batch_size=64, greedy_microbatch=64, distributions={"uniform": {}}, validation_count=256)
        record = dict(cuda=False, identity=dict(family="rest", method="wdro", scope="smoke"), progress=[{"seconds": 1}], calls=[], metrics={})
        self.assertIsNone(eta.calibration([record], "rest", "wdro", protocol))

    def test_repeated_microbatches_pair_without_crossing_attempts(self):
        rows = []
        for start in (10, 12):
            rows += [dict(event="started", attempt="a", update=0, category="validation_uniform", time=start, nets=8),
                     dict(event="completed", attempt="a", update=0, category="validation_uniform", time=start+1, nets=8)]
        rows += [dict(event="started", attempt="old", update=0, category="attack_uniform", time=20, nets=8),
                 dict(event="completed", attempt="new", update=0, category="attack_uniform", time=22, nets=8)]
        self.assertEqual([r["seconds"] for r in eta.call_durations(rows)], [1, 1])

    def test_accounting_includes_initial_validation_two_tests_and_33_attack(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/operations") as temp:
            root = Path(temp)
            protocol = dict(families=["rest"], methods=["wdro"], seeds=[101], updates=2048,
                            batch_size=64, greedy_microbatch=64, validation_every=128,
                            distributions={str(i): {} for i in range(4)}, validation_count=256,
                            test_count=1024, attack_count=128, search=dict(candidate_count=32))
            (root / "protocol.json").write_text(json.dumps(protocol))
            report = eta.build_report(root)
            task = report["tasks"][0]
            self.assertEqual(task["validation_rounds_remaining"], 17)
            self.assertEqual(task["remaining_scored_nets"], dict(validation=17408, test=8192, attack=16896))
            self.assertIsNone(report["queue_wait_seconds"])
            self.assertFalse(report["fully_estimated"])
            self.assertEqual(report["queue_status"], "not_started")

    def test_partial_jsonl_and_batch_padding(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/operations") as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text('{"update":1}\n{"update":')
            self.assertEqual(eta.jsonl(path), [{"update": 1}])
        self.assertEqual(eta.scored_nets(65, 64), 128)

    def test_smoke64_has_six_updates_and_its_own_eval_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/operations") as temp:
            root = Path(temp)
            protocol = dict(families=["rest"], methods=["wdro"], seeds=[101], updates=2048,
                            batch_size=64, greedy_microbatch=64, validation_every=128,
                            distributions={str(i): {} for i in range(4)}, validation_count=256,
                            test_count=1024, attack_count=128, search=dict(candidate_count=32))
            (root / "protocol.json").write_text(json.dumps(protocol))
            task = dict(id="gpu_smoke_rest", command=["python", "runner.py", "--family", "rest", "--method", "wdro", "--smoke", "--smoke-batch-size", "64"],
                        complete="artifacts/runs/gpu_smoke_rest/complete.json")
            (root / "queue.json").write_text(json.dumps({"tasks": [task]}))
            row = eta.build_report(root)["tasks"][0]
            self.assertEqual(row["total_updates"], 6)
            self.assertEqual(row["validation_rounds_remaining"], 4)
            self.assertEqual(row["remaining_scored_nets"], dict(validation=1024, test=512, attack=4352))


if __name__ == "__main__":
    unittest.main(verbosity=2)
