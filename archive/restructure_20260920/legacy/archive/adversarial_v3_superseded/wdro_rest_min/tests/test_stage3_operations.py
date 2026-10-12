import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_stage3 import effective_events
from run_stage3_baseline_queue import build_tasks, choose_gpu, recent_memory_floors


class Stage3OperationsTests(unittest.TestCase):
    def test_recent_memory_low_watermark_rejects_transient_capacity(self):
        snapshots=[{"captured_at_utc":"2026-09-05T03:00:00Z","gpus":[{"index":1,"memory_free_mib":500}]},
            {"captured_at_utc":"2026-09-05T03:05:00Z","gpus":[{"index":1,"memory_free_mib":30000}]}]
        from datetime import datetime
        now=datetime.fromisoformat("2026-09-05T03:05:00+00:00").timestamp()
        self.assertEqual(recent_memory_floors(snapshots,now),{1:500})
        self.assertEqual(recent_memory_floors(snapshots,now+301),{1:30000})
        self.assertEqual(recent_memory_floors(snapshots,now+601),{})

    def test_task_matrix_keeps_all_45_runs(self):
        tasks=build_tasks()
        expected={(m,d,s) for m in ("mixed_erm","group_dro","hard_mining") for d in (10,20,30,40,50) for s in (101,102,103)}
        self.assertEqual({(t["method"],t["degree"],t["seed"]) for t in tasks},expected)
        self.assertEqual(len({t["run_id"] for t in tasks}),45)

    def test_replay_discards_rolled_back_train_and_validation(self):
        events=[{"event":"train","update":1,"value":"committed"},
            {"event":"train","update":2,"value":"rolled_back"},
            {"event":"validation","update":2},
            {"event":"resume","from_update":1},
            {"event":"train","update":2,"value":"replay"}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"events.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events)+"\n")
            train, validation, raw=effective_events(path)
            self.assertEqual(train[2]["value"],"replay")
            self.assertEqual(validation,{})
            self.assertEqual(len(raw),5)

    def test_capacity_allows_shared_gpu_and_respects_current_exclusions(self):
        preflight={"manifest":{"preflight_passed":True,"eligible_gpu_indices":[4,7],"gpu_decisions":[
            {"index":4,"minimum_free_memory_mib":30000,"average_gpu_utilization_percent":10},
            {"index":7,"minimum_free_memory_mib":10000,"average_gpu_utilization_percent":50}]}}
        self.assertEqual(choose_gpu(preflight,{},set()),4)
        self.assertEqual(choose_gpu(preflight,{4:1},set()),7)
        self.assertEqual(choose_gpu(preflight,{}, {4}),7)
        self.assertIsNone(choose_gpu(preflight,{}, {4,7}))


if __name__=="__main__":unittest.main()
