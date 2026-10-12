"""Recovery regressions for actual shared-capacity failure boundaries."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import precision_queue_runtime as runtime


class CapacityRecoveryTests(unittest.TestCase):
    def test_fifteen_clean_capacity_pauses_do_not_exhaust_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'latest.pt'; p.write_bytes(b'checkpoint')
            result = dict(status='paused_for_resume', completed_updates=64,
                          latest_checkpoint=str(p), best_checkpoint=str(p),
                          latest_checkpoint_sha256=runtime.file_sha256(p),
                          best_checkpoint_sha256=runtime.file_sha256(p))
            attempts = 0
            for _ in range(15):
                attempts += 1
                if runtime.capacity_exit(dict(stopping=True, start_update=57), 0, result, ''):
                    attempts -= 1
            self.assertEqual(attempts, 0)

    def test_unexplained_success_exit_is_not_capacity(self):
        with self.assertRaises(RuntimeError):
            runtime.capacity_exit(dict(stopping=False), 0, dict(status='paused_for_resume'), '')

    def test_non_resource_errors_stop_even_after_a_pause_request(self):
        with self.assertRaises(RuntimeError):
            runtime.capacity_exit(dict(stopping=True), 1, None, 'AssertionError: wrong dual direction')

    def test_only_explicit_cuda_oom_is_retryable(self):
        self.assertTrue(runtime.capacity_exit({}, 1, None, 'torch.OutOfMemoryError: CUDA out of memory.'))
        with self.assertRaises(RuntimeError):
            runtime.capacity_exit({}, 1, None, 'ValueError: out of memory appears in a field')

    def test_completed_training_is_not_a_failure(self):
        self.assertFalse(runtime.capacity_exit({}, 0, dict(status='complete', completed_updates=2000), ''))

    def test_capacity_first_allows_a_second_tenant_on_a_roomier_gpu(self):
        rows = [dict(index=0, eligible=True, minimum_free_memory_mib=12000),
                dict(index=4, eligible=True, minimum_free_memory_mib=35000)]
        self.assertEqual(runtime.choose_gpu(rows, {'existing': dict(gpu=4)}, [], {}, 4702, 1000), 4)
        active = {'one': dict(gpu=4), 'two': dict(gpu=4)}
        self.assertEqual(runtime.choose_gpu(rows, active, [], {}, 4702, 1000), 0)

    def test_recent_saturation_or_memory_low_watermark_blocks_admission(self):
        rows = [dict(index=0, eligible=True, minimum_free_memory_mib=12000)]
        history = [dict(time=t, gpus=[dict(index=0, memory_free_mib=12000, utilization_gpu_percent=99)]) for t in (900, 920, 940)]
        self.assertIsNone(runtime.choose_gpu(rows, {}, history, {}, 4702, 1000))
        history = [dict(time=900, gpus=[dict(index=0, memory_free_mib=4000, utilization_gpu_percent=0)])]
        self.assertIsNone(runtime.choose_gpu(rows, {}, history, {}, 4702, 1000))
        self.assertEqual(runtime.choose_gpu(rows, {}, history, {}, 4702, 1600), 0)

    def test_peak_margin_is_strict_and_cooldown_is_respected(self):
        rows = [dict(index=4, eligible=True, minimum_free_memory_mib=4702)]
        self.assertIsNone(runtime.choose_gpu(rows, {}, [], {}, 4702, 1000))
        rows[0]['minimum_free_memory_mib'] = 30000
        self.assertIsNone(runtime.choose_gpu(rows, {}, [], {4: 1100}, 4702, 1000))

    def test_a_zombie_is_not_live_and_live_lock_is_preserved(self):
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        try:
            for _ in range(100):
                if not runtime.process_is_alive(child.pid): break
                time.sleep(.01)
            self.assertFalse(runtime.process_is_alive(child.pid))
        finally:
            child.wait()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'run.lock'; p.write_text(json.dumps(dict(pid=os.getpid())))
            with self.assertRaises(RuntimeError): runtime.cleanup_dead_lock(directory)
            self.assertTrue(p.exists())

    def test_signal_guard_requires_exact_script_and_run_id(self):
        self.assertFalse(runtime.process_matches(os.getpid(), 'not_a_project_run', ROOT / 'scripts/train_stage5.py'))

    def test_capture_names_are_unique_and_errors_propagate(self):
        result = subprocess.CompletedProcess(['capture'], 2, 'denied', '')
        with patch.object(runtime.subprocess, 'run', return_value=result) as run, \
             patch.object(runtime, 'append_jsonl'), \
             patch.object(runtime, 'classify_capture', return_value=(False, {})):
            self.assertIsNone(runtime.capture('first', 1630, ROOT.parent / 'experiments/logs'))
            self.assertIsNone(runtime.capture('second', 1630, ROOT.parent / 'experiments/logs'))
            commands = [c.args[0] for c in run.call_args_list]
            self.assertNotEqual(commands[0][-1], commands[1][-1])
            self.assertTrue(all(c[c.index('--samples') + 1] == '10' for c in commands))
        with patch.object(runtime.subprocess, 'run', return_value=result), \
             patch.object(runtime, 'append_jsonl'), \
             patch.object(runtime, 'classify_capture', side_effect=ValueError('bad manifest')):
            with self.assertRaises(ValueError): runtime.capture('bad', 1630, ROOT.parent / 'experiments/logs')


if __name__ == '__main__':
    unittest.main(verbosity=2)
