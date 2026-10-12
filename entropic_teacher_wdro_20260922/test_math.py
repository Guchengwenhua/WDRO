import unittest
import numpy as np
from transport import solve,distribution,sample,tail_loss
from report import signflip,holm
from pipeline import pilot_tasks,formal_tasks
import geometry as geo

class Checks(unittest.TestCase):
    def test_analytic_active_constraint(self):
        p,r=solve([[0,1]],[[0,1]],.1,.25)
        np.testing.assert_allclose(p,[[.75,.25]],atol=1e-12)
        self.assertAlmostEqual(r['lambda_'],1+.1*np.log(3),places=11)
        self.assertLess(abs(r['gap']),1e-11)
    def test_inactive_constraint(self):
        p,r=solve([[0,1]],[[0,.02]],.2,.04);self.assertEqual(r['lambda_'],0)
        self.assertLessEqual(r['expected_cost'],.04)
    def test_shared_budget_not_per_row(self):
        p,r=solve([[0,2],[0,.1]],[[0,.1],[0,.1]],.02,.04)
        self.assertGreater(p[0,1]*.1,.04);self.assertLessEqual(r['expected_cost'],.04)
    def test_all_duplicate(self):
        p,r=solve([[3,3,3]],[[0,0,0]],.02,.04)
        np.testing.assert_allclose(p,[[1/3]*3]);self.assertEqual(r['expected_cost'],0)
    def test_extreme_scores(self):
        p,r=solve([[0,1000,-1000]],[[0,.1,.03]],.0001,.04)
        self.assertTrue(np.isfinite(p).all());self.assertLessEqual(r['expected_cost'],.04+1e-12)
    def test_random_constraint_certificates(self):
        rng=np.random.default_rng(11)
        for _ in range(30):
            l=rng.normal(size=(8,16));c=rng.uniform(.01,.08,size=l.shape);c[:,0]=0
            p,r=solve(l,c,.02,.04);self.assertLessEqual(r['expected_cost'],.04+1e-12);self.assertLess(abs(r['gap']),1e-9)
            np.testing.assert_allclose(p.sum(1),1)
    def test_cost_monotonicity(self):
        l=np.array([[1.,2.,3.]]);c=np.array([[0.,.03,.1]])
        v=[float((distribution(l,c,.1,lam)*c).sum()) for lam in [0,1,10,100]]
        self.assertTrue(all(a>=b for a,b in zip(v,v[1:])))
    def test_invalid_anchor(self):
        with self.assertRaises(ValueError):solve([[0,1]],[[.1,.2]],.1,.04)
    def test_sampling_frequency(self):
        p=np.tile([.2,.8],(20000,1));idx=sample(p,np.random.default_rng(4));self.assertLess(abs(idx.mean()-.8),.01)
    def test_cvar_transform(self):
        x=np.array([1.,2.,3.,4.]);v=tail_loss(x,3.,.75,1.)
        self.assertEqual(v.mean(),4.)
    def test_statistics(self):
        self.assertEqual(signflip(np.ones(8)),1/256)
        np.testing.assert_allclose(holm([.01,.03,.2,.6]),[.04,.09,.4,.6])
    def test_task_matrix(self):
        tasks=pilot_tasks()+formal_tasks(.02);self.assertEqual(len(tasks),116)
        self.assertEqual(len({t['id'] for t in tasks}),116)
        self.assertEqual(sum(t['updates']*128 for t in formal_tasks(.02)),86507520)
    def test_legalized_cost(self):
        x=geo.make_data(8,20,'uniform','unit',1);z=geo.legalize(x,np.ones((8,4,20,2))*.08)
        self.assertTrue((z<=.9).all());self.assertTrue((z>=.1).all());self.assertTrue((geo.costs(x,z)<=.16+1e-7).all())
    def test_topology_rejects_cycle(self):
        x=np.array([[[.1,.1],[.5,.5],[.9,.9]]])
        with self.assertRaises(ValueError):geo.lengths(x,np.array([[0,1,1,0]]))

if __name__=='__main__':unittest.main(verbosity=2)
