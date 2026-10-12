"""Exercise resource retry without launching training or changing real resources."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import precision_capacity_wait as runtime


class CapacityRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='capacity_test_', dir=str(runtime.CODEX / 'experiments/tmp'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.command = [sys.executable, str(ROOT / 'scripts/capture_host_runtime.py'), '--purpose', 'test']

    def result(self, name, *, gpu=True, disk=True, code=None):
        ready = gpu and disk
        value = dict(kind='host_runtime_preflight', research_protocol_override=False,
                     sampling=dict(count=3, samples=[{}, {}, {}]), eligible_gpu_indices=[2] if gpu else [],
                     checks=dict(experiments_disk_meets_floor=disk), preflight_passed=ready)
        path = self.root / (name + '.json')
        path.write_text(json.dumps(value))
        summary = dict(manifest=str(path), sha256=runtime.file_sha256(path), preflight_passed=ready)
        return subprocess.CompletedProcess(self.command, (0 if ready else 2) if code is None else code,
                                           stdout=json.dumps(summary), stderr='')

    def invoke(self, results):
        return patch.object(runtime.subprocess, 'run', side_effect=results)

    def test_gpu_denial_then_new_capture_success(self):
        denied = self.result('denied', gpu=False)
        ready = self.result('ready')
        with self.invoke([denied, ready]) as call, patch.object(runtime.time, 'sleep') as delay:
            got = runtime.wait_capture(self.command, context='test', log_root=self.root)
        self.assertIs(got, ready)
        self.assertEqual(call.call_count, 2)
        outputs = [c.args[0][c.args[0].index('--output') + 1] for c in call.call_args_list]
        self.assertEqual(len(set(outputs)), 2)
        self.assertTrue(all(Path(p).is_relative_to(ROOT / 'manifests') for p in outputs))
        delay.assert_called_once_with(15)
        events = [json.loads(x) for x in (self.root / 'attempts.jsonl').read_text().splitlines()]
        self.assertEqual([x['returncode'] for x in events], [2, 0])
        self.assertNotEqual(json.loads(events[0]['stdout'])['manifest'], json.loads(events[1]['stdout'])['manifest'])

    def test_disk_denial_also_waits(self):
        with self.invoke([self.result('disk_denied', disk=False), self.result('ready')]) as call, patch.object(runtime.time, 'sleep'):
            runtime.wait_capture(self.command, context='disk', log_root=self.root)
        self.assertEqual(call.call_count, 2)

    def test_other_exit_is_not_retried(self):
        error = subprocess.CompletedProcess(self.command, 1, stdout='', stderr='unrelated implementation failure')
        with self.invoke([error]) as call, patch.object(runtime.time, 'sleep') as delay:
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.wait_capture(self.command, context='error', log_root=self.root)
        self.assertEqual(call.call_count, 1)
        delay.assert_not_called()

    def test_exit_two_without_valid_capacity_evidence_is_not_retried(self):
        error = subprocess.CompletedProcess(self.command, 2, stdout='not a capacity manifest', stderr='')
        with self.invoke([error]) as call, patch.object(runtime.time, 'sleep') as delay:
            with self.assertRaises(json.JSONDecodeError):
                runtime.wait_capture(self.command, context='invalid', log_root=self.root)
        self.assertEqual(call.call_count, 1)
        delay.assert_not_called()

    def test_exit_and_status_mismatch_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'exit/status mismatch'):
            runtime.classify_capture(self.result('wrong_status', code=2))

    def test_tampered_manifest_rejected(self):
        result = self.result('tampered', gpu=False)
        path = Path(json.loads(result.stdout)['manifest'])
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(AssertionError, 'hash mismatch'):
            runtime.classify_capture(result)

    def test_non_preflight_command_never_executed(self):
        with patch.object(runtime.subprocess, 'run') as call:
            with self.assertRaises(AssertionError):
                runtime.wait_capture([sys.executable, str(ROOT / 'scripts/train_wdro.py')], context='wrong')
        call.assert_not_called()

    def test_foreign_manifest_path_is_not_read(self):
        result = subprocess.CompletedProcess(self.command, 2, stdout=json.dumps(dict(
            manifest='/tmp/untrusted_capacity_fixture.json', sha256='unused', preflight_passed=False)), stderr='')
        with self.assertRaisesRegex(AssertionError, 'outside correction workspace'):
            runtime.classify_capture(result)

    def test_reused_output_is_rejected_before_capture(self):
        with patch.object(runtime.subprocess, 'run') as call:
            with self.assertRaisesRegex(AssertionError, 'new unique output'):
                runtime.wait_capture([*self.command, '--output', str(self.root / 'reused.json')], context='reuse')
        call.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
