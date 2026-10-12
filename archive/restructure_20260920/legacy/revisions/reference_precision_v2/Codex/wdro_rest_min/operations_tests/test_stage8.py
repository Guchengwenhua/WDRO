"""Behavioral gates for independent attack fairness, recovery and accounting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.datasets import build_synthetic_dataset
from src.geometry import pin_set_sha256
from stage5.search import cem_round
from stage7.evaluation import python_tree_lengths
from stage8.common import EXP, PLAN, read, atomic_npz, artifact, append_jsonl
from stage8.attack import adapt, independent_tree_lengths, restore_rounds
from stage8.audit import reconcile_calls
from stage6.statistics import bootstrap_cell
from stage8.statistics import ispd_bootstrap,cached_bootstrap


class Stage8Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=EXP/'tmp',prefix='stage8_test_')
        self.directory=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.config=read(PLAN)['attack']
        self.data,_=build_synthetic_dataset(protocol_id='test-stage8',split='attack',distribution='uniform',
            degree=10,base_seed=303,count=3,parameters={'nonuniform':0})

    def generate(self,centers=None,ids=None,rounds=16):
        centers=self.data['pins'] if centers is None else centers
        ids=self.data['net_ids'] if ids is None else ids
        batch=SimpleNamespace(pins=centers,net_ids=ids)
        mean=np.zeros_like(centers,dtype=np.float64);std=np.full_like(mean,.02);seen=[set() for _ in centers]
        result=[]
        for t in range(rounds):
            pins,repairs=cem_round(batch,update=1,training_seed=303,round_index=t,mean=mean,std=std,seen=seen,config=self.config)
            # A symmetric analytic objective, independent of implementation internals.
            scores=np.var(pins.astype(np.float64),axis=(2,3))
            mean,std,elite=adapt(centers,pins,scores,.001)
            result.append((pins,scores,mean,std,elite,repairs))
        return result

    def test_integer_evaluator_matches_independent_scalar_construction(self):
        pins=self.data['pins'];rng=np.random.default_rng(14)
        t=np.array([[v for j in range(1,10) for v in (j,int(rng.integers(j)))] for _ in pins],dtype=np.int16)
        np.testing.assert_array_equal(independent_tree_lengths(pins,t),python_tree_lengths(pins,t))

    def test_disconnected_topology_rejected(self):
        t=np.tile([0,1],(3,9))
        with self.assertRaises(AssertionError):independent_tree_lengths(self.data['pins'],t)

    def test_invalid_topology_rejected(self):
        t=np.tile([0,10],(3,9))
        with self.assertRaises(AssertionError):independent_tree_lengths(self.data['pins'],t)

    def test_exact_256_distinct_candidates_and_original(self):
        values=np.concatenate([r[0] for r in self.generate()],axis=1)
        self.assertEqual(values.shape,(3,256,10,2))
        np.testing.assert_array_equal(values[:,0],self.data['pins'])
        for row in values:self.assertEqual(len({pin_set_sha256(p) for p in row}),256)

    def test_batch_partition_does_not_change_search(self):
        whole=self.generate(rounds=3)
        parts=[self.generate(self.data['pins'][i:i+1],self.data['net_ids'][i:i+1],3) for i in range(3)]
        for t in range(3):
            np.testing.assert_array_equal(whole[t][0],np.concatenate([r[t][0] for r in parts]))

    def test_pin_permutation_cannot_change_candidates(self):
        order=np.array([5,1,9,0,4,2,6,3,8,7])
        a=self.generate(rounds=3)
        b=self.generate(self.data['pins'][:,order],self.data['net_ids'],3)
        for x,y in zip(a,b):
            for i in range(3):
                for j in range(16):self.assertEqual(pin_set_sha256(x[0][i,j]),pin_set_sha256(y[0][i,j]))

    def test_elites_use_raw_regret_with_stable_ties(self):
        pins=np.repeat(self.data['pins'][:,None],16,axis=1)
        score=np.zeros((3,16));score[:,[3,6,7,9,10]]=1
        _,_,elite=adapt(self.data['pins'],pins,score,.001)
        np.testing.assert_array_equal(elite,np.tile([3,6,7,9],(3,1)))

    def test_round_recovery_reconstructs_all_state(self):
        rounds=self.generate(rounds=3);rows=[]
        for t,(pins,scores,mean,std,elite,repairs) in enumerate(rounds[:2]):
            path=self.directory/f'round{t}.npz'
            atomic_npz(path,dict(pins=pins,regret=scores,next_mean=mean,next_std=std,elite=elite))
            rows.append(dict(round=t,artifact=artifact(path),repaired=repairs))
        mean,std,seen,_=restore_rounds(self.data['pins'],rows,self.config,self.data['net_ids'],reconstruct=True)
        actual,_=cem_round(SimpleNamespace(pins=self.data['pins'],net_ids=self.data['net_ids']),update=1,
            training_seed=303,round_index=2,mean=mean,std=std,seen=seen,config=self.config)
        np.testing.assert_array_equal(actual,rounds[2][0])

    def test_corrupt_round_is_rejected(self):
        path=self.directory/'round.npz';atomic_npz(path,dict(x=np.array([1])))
        row=dict(round=0,artifact=artifact(path));path.write_bytes(b'corrupt')
        with self.assertRaises(AssertionError):restore_rounds(self.data['pins'],[row],self.config,self.data['net_ids'])

    def test_unfinished_calls_are_bounds_not_completed_calls(self):
        events=[dict(id='a',phase='started',actor_planned=256,rest_planned=256,reference_planned=256),
                dict(id='a',phase='actor_completed',calls=256),dict(id='a',phase='rest_completed',calls=256)]
        for e in events:append_jsonl(self.directory/'calls.jsonl',e)
        r=reconcile_calls(self.directory,dict(batches=[]))
        self.assertEqual(r['actual_calls_lower_bound']['reference'],0)
        self.assertEqual(r['actual_calls_upper_bound']['reference'],256)
        self.assertEqual(r['completed_replay_extra']['actor'],256)

    def test_orphan_or_duplicate_receipt_rejected(self):
        append_jsonl(self.directory/'calls.jsonl',dict(id='a',phase='actor_completed',calls=256))
        with self.assertRaises(AssertionError):reconcile_calls(self.directory,dict(batches=[]))

    def test_paired_bootstrap_is_paired_and_preserves_seed_tails(self):
        x=np.array([[0.,0.,0.,4.],[3.,0.,0.,0.],[0.,2.,0.,0.]])
        a=bootstrap_cell(x,samples=50,seed=5);b=bootstrap_cell(x+.5,samples=50,seed=5)
        np.testing.assert_allclose(b['mean']-a['mean'],.5,rtol=0,atol=1e-15)
        np.testing.assert_allclose(b['p95']-a['p95'],.5,rtol=0,atol=1e-15)
        self.assertNotEqual(float(np.quantile(x,.95,axis=1).mean()),float(np.quantile(x.mean(axis=0),.95)))

    def test_frozen_scope_has_all_models_cells_and_no_selection(self):
        p=read(PLAN)
        self.assertEqual(len(p['models']),75);self.assertEqual(len(p['groups']),20)
        self.assertEqual(p['expected_search_calls'],153600000)
        self.assertFalse(p['model_selection']);self.assertFalse(p['training'])
        self.assertEqual(p['attack']['cem']['maximum_motion'],.08)

    def test_ispd_weighting_pooled_tail_and_fixed_strata(self):
        values=np.array([[0.,0.,0.,0.,10.],[1.,1.,1.,1.,11.],[2.,2.,2.,2.,12.]])
        designs=np.array(['a','a','a','a','b']);strata=np.array(['a10','a10','a20','a20','b10'])
        b=ispd_bootstrap(values,designs,strata,samples=16,seed=44)
        for k,expected in [('net_weighted_mean',[2,3,4]),('design_macro_mean',[5,6,7]),
                           ('worst_design_mean',[10,11,12]),('p95',[8,9,10])]:
            np.testing.assert_allclose(b[k],np.tile(expected,(16,1)),rtol=0,atol=1e-14)

    def test_bootstrap_resume_reuses_only_bound_inputs(self):
        path=self.directory/'bootstrap.npz';binding={'input':'a'}
        a,_=cached_bootstrap(path,binding,lambda:dict(mean=np.array([1.,2.])))
        def should_not_run():raise AssertionError('replayed completed bootstrap')
        b,_=cached_bootstrap(path,binding,should_not_run)
        np.testing.assert_array_equal(a['mean'],b['mean'])
        with self.assertRaises(AssertionError):cached_bootstrap(path,{'input':'changed'},should_not_run)


if __name__=='__main__':unittest.main()
