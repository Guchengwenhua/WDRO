"""CPU-only source-degree, rejection-accounting, and reference recovery tests.

All mutable fixtures live in temporary directories under this extension.
No frozen source/model, real input, GPU, or running queue is modified.
"""
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common
import prepare
import reference
import run_queue


class ExtractionContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pipeline_extract_', dir=ROOT/'tests')
        self.base = Path(self.tmp.name)
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(prepare, 'BASE', self.base))
        self.module = types.ModuleType('_external_pipeline_fixture')
        self.module.__file__ = str(Path(__file__).resolve())
        self.stack.enter_context(mock.patch.dict(sys.modules, {self.module.__name__: self.module}))
        self.source = self.base/'source.txt'
        self.source.write_text('fixture data version 1\n')
        self.design = dict(design_id='fixture_design', family='fixture_family', design_family='fixture_chip',
            path=str(self.source), parser=self.module.__name__+'.parse_design', sha256=common.sha(self.source))
        self.points = np.array([[i, (i*i)%11] for i in range(10)],dtype=np.float64)

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def records(self):
        repeated = self.points.copy()
        repeated[2] = repeated[1]
        return [
            dict(net_id='accepted',raw_degree=10,points=self.points,valid=True),
            dict(net_id='missing_coordinates',raw_degree=10,points=None,valid=True),
            dict(net_id='duplicate_coordinates',raw_degree=10,points=repeated,valid=True),
            dict(net_id='incomplete_twenty',raw_degree=20,points=self.points,valid=True),
            dict(net_id='unplaced_thirty',raw_degree=30,points=None,valid=False,reasons=['missing_placement']),
            dict(net_id='unsupported_two',raw_degree=2,points=self.points,valid=True),
        ]

    def test_raw_degree_and_none_do_not_silently_reclassify_nets(self):
        self.module.parse_design=lambda path:iter(self.records())
        result=prepare.prepare_design(self.design)
        groups={g['degree']:g for g in result['groups']}
        self.assertEqual(result['total_source_nets'],6)
        self.assertEqual(result['unsupported_degree_nets'],1)
        self.assertEqual((groups[10]['parsed_count'],groups[10]['valid_count'],groups[10]['invalid_count']),(3,1,2))
        self.assertEqual(groups[20]['invalid_reasons'],{'degree_changed_by_deduplication':1})
        self.assertEqual(groups[30]['invalid_reasons'],{'source_incomplete:missing_placement':1})
        self.assertEqual(groups[10]['invalid_reasons'],{'source_incomplete:missing_coordinates':1,'degree_changed_by_deduplication':1})
        self.assertNotIn('input',groups[40])
        data=common.load(groups[10]['input']['logical_path'])
        self.assertEqual(data['pins'].shape,(1,10,2))
        self.assertEqual(data['source_index'].tolist(),[0])
        self.assertEqual(data['net_ids'][0].decode(),'public:fixture_design:accepted:d10')

    def test_zero_degree_groups_resume_without_reparsing(self):
        self.module.parse_design=lambda path:iter(self.records())
        first=prepare.prepare_design(self.design)
        self.module.parse_design=mock.Mock(side_effect=AssertionError('cache must be reused'))
        second=prepare.prepare_design(self.design)
        self.assertEqual(first['groups'],second['groups'])
        self.module.parse_design.assert_not_called()

    def test_changed_source_bytes_rejected_before_cache_reuse(self):
        self.module.parse_design=lambda path:iter(self.records())
        prepare.prepare_design(self.design)
        self.source.write_text('fixture data changed after preparation\n')
        with self.assertRaises(AssertionError):
            prepare.prepare_design(self.design)

    def test_corrupted_cached_array_rejected(self):
        self.module.parse_design=lambda path:iter(self.records())
        result=prepare.prepare_design(self.design)
        path=Path(next(g['input']['logical_path'] for g in result['groups'] if 'input' in g))
        with path.open('ab') as stream:stream.write(b'fixture corruption')
        with self.assertRaises(AssertionError):
            prepare.prepare_design(self.design)


class ReferenceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pipeline_reference_',dir=ROOT/'tests')
        self.base=Path(self.tmp.name)
        self.stack=ExitStack()
        self.stack.enter_context(mock.patch.object(reference,'BASE',self.base))
        self.stack.enter_context(mock.patch.object(reference,'_INPUT_CACHE',{}))
        self.stack.enter_context(mock.patch.object(reference.gst,'reference_identity',lambda:{'reference_id':'fixture','scale':reference.gst.SCALE}))
        reference.gst.SCALE=2**27
        xy=np.array([[0.1+0.08*i,0.1+0.08*((i*3)%10)] for i in range(10)],dtype=np.float32)
        self.input_path=self.base/'input.npz'
        common.npz(self.input_path,dict(pins=xy[None],net_ids=np.array(['fixture_n1'],dtype='S32')))
        self.task=(10,0,1,str(self.input_path),common.sha(self.input_path))
        self.calls=[]
        outer=self
        class FakeEvaluator:
            def solve(self,pins,*,require_integral=False):
                outer.assertTrue(require_integral)
                outer.calls.append(reference.gst.SCALE)
                # A deterministic exact-binary fixture objective; this test
                # exercises publication/recovery, not GeoSteiner optimality.
                return float(np.ptp(pins.astype(np.float64),axis=0).sum())
        self.stack.enter_context(mock.patch.object(reference.gst,'GeoSteinerEvaluator',FakeEvaluator))

    def tearDown(self):
        reference.gst.SCALE=2**27
        self.stack.close()
        self.tmp.cleanup()

    def test_dual_scale_commit_and_idempotent_recovery_make_no_new_calls(self):
        first=reference.solve_task(self.task)
        self.assertEqual(self.calls,[2**27,2**28])
        self.assertEqual(reference.gst.SCALE,2**27)
        second=reference.solve_task(self.task)
        self.assertEqual(first,second)
        self.assertEqual(self.calls,[2**27,2**28])
        arrays=common.load(first['output']['logical_path'])
        np.testing.assert_array_equal(arrays['lstar'],arrays['verification_lstar'])
        self.assertEqual(arrays['net_ids'].tolist(),[b'fixture_n1'])

    def test_changed_input_binding_refuses_reference_reuse(self):
        reference.solve_task(self.task)
        with self.assertRaises(AssertionError):
            reference.solve_task((*self.task[:-1],'0'*64))
        self.assertEqual(self.calls,[2**27,2**28])

    def test_corrupted_reference_output_refuses_reuse(self):
        result=reference.solve_task(self.task)
        with Path(result['output']['logical_path']).open('ab') as stream:stream.write(b'fixture corruption')
        with self.assertRaises(AssertionError):
            reference.solve_task(self.task)
        self.assertEqual(self.calls,[2**27,2**28])

    def test_secondary_failure_resets_scale_and_leaves_no_committed_result(self):
        outer=self
        class FailsSecondary:
            def solve(self,pins,*,require_integral=False):
                if reference.gst.SCALE==2**28:raise RuntimeError('secondary interrupted')
                return 1.0
        with mock.patch.object(reference.gst,'GeoSteinerEvaluator',FailsSecondary):
            with self.assertRaisesRegex(RuntimeError,'secondary interrupted'):
                reference.solve_task(self.task)
        self.assertEqual(reference.gst.SCALE,2**27)
        shards=self.base/'cache/references/shards'
        self.assertFalse(list(shards.glob('*.npz')))
        self.assertFalse([p for p in shards.glob('*.json') if not p.name.endswith('.started.json')])
        self.assertEqual(len(list(shards.glob('*.started.json'))),1)
        result=reference.solve_task(self.task)
        self.assertEqual(result['status'],'pass')


class QueueOwnershipTests(unittest.TestCase):
    def test_only_locked_writer_with_matching_live_start_ticks_is_adopted(self):
        with tempfile.TemporaryDirectory(prefix='pipeline_writer_',dir=ROOT/'tests') as tmp:
            output=Path(tmp)
            fields=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()
            job=dict(process=dict(pid=os.getpid(),start_ticks=fields[19]),physical_gpu=999)
            common.write(output/'job.json',job)
            task=dict(id='fixture',output=str(output))
            self.assertIsNone(run_queue.existing_writer(task))
            with (output/'run.lock').open('a+') as stream:
                common.fcntl.flock(stream,common.fcntl.LOCK_EX|common.fcntl.LOCK_NB)
                observed=run_queue.existing_writer(task)
                self.assertEqual(observed['process'].pid,os.getpid())
                self.assertIsNone(observed['process'].poll())
                self.assertIsNone(observed['log'])
                job['process']['start_ticks']=str(int(fields[19])+1)
                common.write(output/'job.json',job)
                with self.assertRaisesRegex(AssertionError,'matching live process'):
                    run_queue.existing_writer(task)
                common.fcntl.flock(stream,common.fcntl.LOCK_UN)
            self.assertIsNone(run_queue.existing_writer(task))

    def test_reused_pid_identity_is_not_treated_as_a_live_worker(self):
        fields=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()
        observed=run_queue.ExistingProcess(dict(pid=os.getpid(),start_ticks=str(int(fields[19])+1)))
        self.assertIsNotNone(observed.poll())


if __name__=='__main__':
    unittest.main(verbosity=2)
