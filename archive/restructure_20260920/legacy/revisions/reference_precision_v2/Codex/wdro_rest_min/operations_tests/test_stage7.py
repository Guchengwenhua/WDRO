"""Independent scientific aggregation, topology and recovery failure cases."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage7.common import atomic_npz, artifact, append_jsonl, Lease, read
from stage7.delivery import design_statistics, reconcile_calls
from stage7.evaluation import read_shards, python_tree_lengths


class Stage7Tests(unittest.TestCase):
    def test_unequal_design_sizes_are_not_macro_weights(self):
        r=design_statistics([0,0,0,1],np.array(['a','a','a','b']))
        self.assertEqual(r['net_weighted_mean'],.25)
        self.assertEqual(r['design_macro_mean'],.5)
        self.assertEqual(r['worst_design_mean'],1)
        self.assertAlmostEqual(r['p95'],.85)
        self.assertEqual(r['design_count'],2)

    def test_tail_metric_precedes_seed_averaging(self):
        seeds=np.array([[0,0,1],[1,0,0]],dtype=float)
        per_seed=[design_statistics(v,np.array(['a','a','a']))['p95'] for v in seeds]
        self.assertAlmostEqual(np.mean(per_seed),.9)
        self.assertNotEqual(np.mean(per_seed),np.quantile(seeds.mean(axis=0),.95))

    def test_invalid_metric_values_rejected(self):
        with self.assertRaises(AssertionError):
            design_statistics([np.nan],['a'])

    def test_connected_segment_union_known_length(self):
        points=np.array([[[0.,0.],[1.,0.],[1.,1.]]])
        value=python_tree_lengths(points,np.array([[0,1,1,2]],dtype=np.int32))
        np.testing.assert_array_equal(value,[2.])

    def test_disconnected_topology_rejected(self):
        with self.assertRaises(AssertionError):
            python_tree_lengths(np.array([[[0.,0.],[1.,0.],[1.,1.]]]),np.array([[0,1,0,1]]))

    def test_topology_bounds_rejected_before_ctypes(self):
        with self.assertRaises(AssertionError):
            python_tree_lengths(np.zeros((1,3,2)),np.array([[0,1,1,3]]))

    def test_resume_shard_hash_and_continuity(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'shard.npz';atomic_npz(path,{'x':np.arange(2)})
            state=dict(complete_nets=2,shards=[dict(start=0,end=2,artifact=artifact(path))])
            np.testing.assert_array_equal(read_shards(state)['x'],[0,1])
            state['shards'][0]['start']=1
            with self.assertRaises(AssertionError):read_shards(state)
            state['shards'][0]['start']=0
            path.write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):read_shards(state)

    def test_unfinished_call_is_recorded_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'calls.jsonl'
            for row in [dict(id='crash',phase='started',start=0,end=2),
                        dict(id='retry',phase='started',start=0,end=2),
                        dict(id='retry',phase='completed',actor_calls=2,rest_calls=2)]:
                append_jsonl(path,row)
            r=reconcile_calls(path,dict(complete_nets=2,shards=[dict(attempt='retry')]))
            self.assertEqual(r['committed_nets'],2)
            self.assertEqual(r['completed_attempt_nets'],2)
            self.assertEqual(r['unfinished_attempt_nets_upper_bound'],2)
            with self.assertRaises(AssertionError):
                reconcile_calls(path,dict(complete_nets=2,shards=[dict(attempt='crash')]))

    def test_lock_rejects_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            with Lease(temp,'test'):
                with self.assertRaises(RuntimeError):
                    with Lease(temp,'duplicate'):pass
            with Lease(temp,'resumed'):pass
            self.assertTrue(read(Path(temp)/'job.json')['lock_released'])


if __name__=='__main__':
    unittest.main()
