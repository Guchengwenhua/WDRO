"""CPU regression cases for delivery process identity and stale PID gates."""
import unittest
from unittest import mock

from diagnostics import delivery_audit as audit


PYTHON = '/remote-home/Guilin/miniconda3/envs/py311/bin/python3.11'


class DeliveryProcessIdentityTests(unittest.TestCase):
    def test_tmux_retained_launch_text_is_not_a_research_process(self):
        argv = ['tmux', '-S', 'adv_v3.sock', 'new-session', '-d',
                'bash run_env.sh python -u -m adv_v3.pipeline --share-busy-gpus']
        self.assertIsNone(audit.research_process_kind('/usr/bin/tmux', argv))

    def test_shell_launcher_is_not_the_python_process(self):
        argv = ['bash', '-c', 'python -m adv_v3.pipeline; echo multiprocessing.spawn']
        self.assertIsNone(audit.research_process_kind('/usr/bin/bash', argv))

    def test_all_actual_research_modules_are_identified(self):
        for module in ('worker', 'joint', 'queue', 'pipeline', 'monitor'):
            with self.subTest(module=module):
                target = 'adv_v3.' + module
                self.assertEqual(audit.research_process_kind(PYTHON, ['python', '-B', '-u', '-m', target]), target)
                self.assertEqual(audit.research_process_kind(PYTHON, ['python', '-m' + target]), target)

    def test_interpreter_option_values_are_skipped(self):
        argv = ['python', '-W', 'ignore', '-X', 'faulthandler', '-m', 'adv_v3.joint']
        self.assertEqual(audit.research_process_kind(PYTHON, argv), 'adv_v3.joint')

    def test_actual_multiprocessing_spawn_is_identified(self):
        code = 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=8, pipe_handle=12)'
        argv = ['python', '-c', code, '--multiprocessing-fork']
        self.assertEqual(audit.research_process_kind(PYTHON, argv), 'multiprocessing.spawn')
        alias = 'import multiprocessing.spawn as child; child.spawn_main(pipe_handle=12)'
        self.assertEqual(audit.research_process_kind(PYTHON, ['python', '-c', alias, '--multiprocessing-fork']), 'multiprocessing.spawn')

    def test_unrelated_python_commands_with_matching_text_are_ignored(self):
        commands = [
            ['python', '-m', 'other.module', '-m', 'adv_v3.pipeline'],
            ['python', 'other.py', '-m', 'adv_v3.worker'],
            ['python', '-c', 'print("multiprocessing.spawn")', '--multiprocessing-fork'],
            ['python', '-c', 'print("-m adv_v3.pipeline")'],
            ['python', '-c', 'from multiprocessing.spawn import spawn_main', '--multiprocessing-fork'],
            ['python', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main()', 'argument'],
            ['python', '--', '-m', 'adv_v3.pipeline'],
            ['python', '-', '-m', 'adv_v3.pipeline'],
        ]
        for argv in commands:
            with self.subTest(argv=argv):
                self.assertIsNone(audit.research_process_kind(PYTHON, argv))

    def test_non_python_executable_cannot_spoof_argv(self):
        self.assertIsNone(audit.research_process_kind('/usr/bin/sleep', ['python', '-m', 'adv_v3.pipeline']))

    def test_empty_or_incomplete_arguments_are_ignored(self):
        for argv in ([], ['python'], ['python', '-m'], ['python', '-c']):
            with self.subTest(argv=argv):
                self.assertIsNone(audit.research_process_kind(PYTHON, argv))

    def test_pid_start_time_and_zombie_gates_remain_enforced(self):
        launch = {'pid': 123, 'process_start_ticks': '900'}
        for state, ticks, active in [('S', '900', True), ('S', '901', False), ('Z', '900', False)]:
            fields = [state] + ['0'] * 18 + [ticks]
            with self.subTest(state=state, ticks=ticks), mock.patch.object(audit.Path, 'read_text', return_value='123 (python) ' + ' '.join(fields)):
                self.assertEqual(audit.process(launch)['active'], active)


if __name__ == '__main__':
    unittest.main()
