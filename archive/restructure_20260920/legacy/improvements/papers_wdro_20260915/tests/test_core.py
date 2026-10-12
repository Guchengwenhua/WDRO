import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from core import costs,dataset,read_protocol,generate_support,shortlist_support
from runner import finite_risk

class Contracts(unittest.TestCase):
    def test_root_label_counterexample(self):
        x=np.array([[[.1,.1],[.9,.1],[.2,.8]]]);z=x[:,[1,0,2]]
        self.assertEqual(float(costs(x,z,False)[0,0]),0)
        self.assertGreater(float(costs(x,z,True)[0,0]),0)
    def test_sink_permutation(self):
        x=np.array([[[.1,.1],[.9,.1],[.2,.8]]]);z=x[:,[0,2,1]]
        self.assertEqual(float(costs(x,z,True)[0,0]),0)
    def test_original_and_four_true_scores(self):
        p=read_protocol();b=dataset(p,'contract','uniform',8181,4)
        s=generate_support(b,training_seed=8181,update=1,search=p['search']);ids,_=shortlist_support(s,4)
        self.assertEqual(s.pins.shape,(4,33,10,2));self.assertEqual(ids.shape,(4,4))
        np.testing.assert_array_equal(ids[:,0],0);np.testing.assert_array_equal(s.pins[:,0],b.pins)
        self.assertTrue((costs(b.pins,s.pins,True)>=0).all())
    def test_shared_budget_not_per_point_cap(self):
        v=np.array([[0.,10.],[0.,4.]]);c=np.array([[0.,.1],[0.,.1]])
        result,mass=finite_risk(v,c,.05)
        self.assertAlmostEqual(result['risk'],5);self.assertAlmostEqual(result['average_cost'],.05)
        np.testing.assert_allclose(mass,[[0,1],[1,0]])
    def test_zero_rho_nominal(self):
        v=np.array([[2.,10.],[3.,4.]]);c=np.array([[0.,.1],[0.,.1]])
        result,mass=finite_risk(v,c,0)
        self.assertAlmostEqual(result['risk'],2.5)

if __name__=='__main__':unittest.main()
