import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import torch
import geometry as G
from common import ROOT,read,training_batch,seed_all,digest

class ScalingTests(unittest.TestCase):
    def test_matched_initialization(self):
        for s,row in read(ROOT/'artifacts/inputs/lineage.json').items():
            seed_all(int(s)); a=G.build_actor(30,torch.device('cpu'))
            self.assertEqual(digest(a),row['initial_actor_sha256'])

    def test_data_pairing_and_historical_stream(self):
        x,_=training_batch('U',101,2049)
        np.testing.assert_array_equal(x,G.make_data(64,30,'uniform','train/2049',101))
        x,g=training_batch('M',101,9000)
        for arm in ('D','ME','MG','SG'):
            z,h=training_batch(arm,101,9000)
            np.testing.assert_array_equal(x,z); np.testing.assert_array_equal(g,h)
        self.assertEqual(np.bincount(g).tolist(),[16]*4)

    def test_old_group_weights_give_expected_policy_gradient(self):
        theta=torch.tensor([.1,.3,-.2,.5],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
        cost=torch.tensor([2.,5.],dtype=torch.float64)
        prob=torch.sigmoid(theta)
        expected=(q*(3*prob+2)).sum()
        grad,=torch.autograd.grad(expected,theta,retain_graph=True)
        loss=0.
        for g in range(4):
            p=torch.stack([1-prob[g],prob[g]])
            loss=loss+q[g]*(p.detach()*(cost-7)*p.log()).sum()
        estimate,=torch.autograd.grad(loss,theta)
        torch.testing.assert_close(grad,estimate)

    def test_training_guard_denies_test_and_external_weights(self):
        code="""import os, geometry as G
os.environ['RL_BUDGET_ROLE']='train'
G.install_oracle_guard()
for path in [str(G.ROOT/'artifacts/data/test.npz'), '/tmp/external.pt']:
 try: open(path,'rb')
 except RuntimeError: pass
 else: raise AssertionError(path)
print('PASS')
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_test_gate_rejects_incomplete_training(self):
        from worker import frozen_all_training
        with self.assertRaises((FileNotFoundError,RuntimeError)): frozen_all_training()

    def test_report_independent_length_and_paired_sign(self):
        from report import independent_span,compare
        rng=np.random.default_rng(42)
        x=rng.random((64,30,2)); t=np.array([[j,int(rng.integers(j))] for j in range(1,30)]).reshape(-1)
        t=np.repeat(t[None],64,axis=0)
        np.testing.assert_allclose(independent_span(x,t),G.lengths(x,t),rtol=0,atol=1e-13)
        x[:]=np.round(x*4)/4
        np.testing.assert_allclose(independent_span(x,t),G.lengths(x,t),rtol=0,atol=1e-13)
        base=np.ones((3,4,32))*2; candidate=base*.9
        for name in ('mean','worst'):
            c=compare(base,candidate,name,20,rng)
            self.assertAlmostEqual(c['baseline_value'],2)
            self.assertAlmostEqual(c['candidate_value'],1.8)
            self.assertAlmostEqual(c['improvement_percent'],10)
            self.assertGreater(c['simultaneous_ci99'][0],0)

if __name__=='__main__': unittest.main(verbosity=2)
