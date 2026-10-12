import subprocess
import sys
import unittest
import numpy as np
import torch
import geometry as G
import engine as E
from common import ROOT,read,training_batch,seed_all,digest

class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.p=read(ROOT/'protocol.json'); torch.set_num_threads(2)

    def test_batch64_unchanged(self):
        x,info=E.batch_for(self.p['arms']['E50'],101,12289,self.p)
        np.testing.assert_array_equal(x,training_batch('M',101,12289)[0])

    def test_batch256_exact_sample_pairing(self):
        x,info=E.batch_for(self.p['arms']['B256'],101,12289,self.p)
        target=np.concatenate([training_batch('M',101,s)[0].reshape(4,16,30,2) for s in range(12289,12293)],axis=1).reshape(256,30,2)
        np.testing.assert_array_equal(x,target)
        _,last=E.batch_for(self.p['arms']['B256'],101,14336,self.p)
        self.assertEqual(last['first_base_step']+last['chunks']-1,20480)

    def test_learning_rate_fixed_global_cosine(self):
        c=self.p['arms']['COS']
        self.assertAlmostEqual(E.learning_rate(c,12288,self.p),5e-5)
        self.assertAlmostEqual(E.learning_rate(c,49152,self.p),5e-6)
        self.assertGreater(E.learning_rate(c,20480,self.p),E.learning_rate(c,32768,self.p))

    def test_group_floor_and_normalization(self):
        c=self.p['arms']['GDF']; r=torch.tensor([-1000.,0.,-1000.,-1000.],dtype=torch.float64)
        q=E.group_weights(r,c); torch.testing.assert_close(q,torch.tensor([.05,.85,.05,.05],dtype=torch.float64))
        r=E.next_logr(torch.full((4,),-np.log(4),dtype=torch.float64),torch.tensor([1.,2.,3.,4.]),c)
        self.assertAlmostEqual(float(r.exp().sum()),1.)

    def test_relative_objective_gradient_matches_fixed_anchor(self):
        c=self.p['arms']['GDN']; lp=torch.tensor([1.,2.,3.,4.],requires_grad=True)
        a=torch.tensor([2.,3.,4.,5.]); anchor=torch.tensor([1.,2.,4.,8.],dtype=torch.float64)
        loss,q,scale=E.losses(a,lp,c,torch.full((4,),-np.log(4),dtype=torch.float64),anchor,1.,self.p)
        loss.backward(); torch.testing.assert_close(lp.grad,.25*a/anchor.float())

    def test_erm_exact_original_loss(self):
        lp=torch.arange(64,dtype=torch.float32,requires_grad=True); adv=torch.sin(lp.detach())
        loss,_,_=E.losses(adv,lp,self.p['arms']['E50'],torch.full((4,),-np.log(4),dtype=torch.float64),torch.ones(4),1.,self.p)
        self.assertTrue(torch.equal(loss,(adv*lp).mean()))

    def test_advantage_uses_previous_variance(self):
        c=self.p['arms']['ADV']; lp=torch.ones(64,requires_grad=True)
        a=torch.arange(64,dtype=torch.float32)
        loss,_,scale=E.losses(a,lp,c,torch.full((4,),-np.log(4),dtype=torch.float64),torch.ones(4),4.,self.p)
        self.assertEqual(scale,2.); self.assertEqual(float(loss),float(a.mean()/2))

    def test_default_architecture_historical_initialization(self):
        c=torch.load(ROOT/'artifacts/inputs/ME_s101.pt',map_location='cpu',weights_only=False)
        seed_all(101); model=G.build_actor(30,torch.device('cpu'))
        self.assertEqual(digest(model),c['initial_actor_sha256'])

    def test_capacity_increases_parameters(self):
        counts=[]
        for c in self.p['architectures'].values():
            model=G.build_actor(30,torch.device('cpu'),c['width'],c['depth'],c['inner'])
            counts.append(sum(x.numel() for x in model.parameters()))
        self.assertGreater(counts[1],counts[0]); self.assertGreater(counts[2],counts[0])

    def test_d4_preserves_supplied_topology_cost(self):
        x=G.make_data(8,30,'uniform','d4-test'); t=np.tile(np.column_stack([np.arange(1,30),np.zeros(29,dtype=int)]).reshape(-1),(8,1))
        np.testing.assert_allclose(G.lengths(x,t),G.lengths(1-x,t),atol=2e-6)
        np.testing.assert_allclose(G.lengths(x,t),G.lengths(x[:,:,::-1].copy(),t.reshape(8,29,2)[:,:,::-1].reshape(8,-1)),atol=1e-12)

    def test_promotion_rejects_mean_tradeoff_and_batch(self):
        b=[dict(mean=4.,worst=5.) for _ in range(3)]
        good=[dict(mean=3.9,worst=4.9) for _ in range(3)]
        bad=[dict(mean=5.,worst=4.5) for _ in range(3)]
        winners=E.select_variants(dict(E50=b,E20=good,GDF=bad,B256=[dict(mean=1.,worst=1.)]*3),self.p)
        self.assertEqual(winners,dict(mean='E20',worst='E20'))
        self.assertEqual(E.select_variants(dict(E50=b,E20=b),self.p),dict(mean=None,worst=None))

    def test_guard_blocks_all_test_degrees_and_external_weights(self):
        code='''import os,geometry as g
os.environ['RL_BUDGET_ROLE']='train'
g.install_oracle_guard()
for p in ['test.npz','test_degree20.npz','test_degree40.npz','/tmp/not_allowed.pt']:
 try: open(p,'rb')
 except RuntimeError: pass
 else: raise AssertionError(p)
'''
        r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__': unittest.main(verbosity=2)
