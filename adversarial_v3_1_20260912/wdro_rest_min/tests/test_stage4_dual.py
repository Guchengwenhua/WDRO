import json
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from stage4.dual import select_wdro, update_lambda, primal_dual_lp, transport_costs
from src.candidates import generate_support, shortlist_support
from src.geometry import pin_set_sha256
from src.protocol import load_protocol
from src.training_data import materialize_training_batch

class DualTests(unittest.TestCase):
    def test_per_center_penalty_and_stable_ties(self):
        ids=np.array([[0,4,8],[0,3,7]])
        loss=np.array([[.1,.9,.4],[.2,.3,.8]])
        cost=np.array([[0,.8,.1],[0,.1,.9]])
        chosen, _=select_wdro(ids,loss,cost,1,.2)
        np.testing.assert_array_equal(chosen,[8,0])
        np.testing.assert_array_equal(select_wdro(ids,loss,cost,0,.2)[0],[4,7])

    def test_zero_radius_exact_erm_and_permutation_zero_cost(self):
        loss=np.array([[.1,10],[.2,20.]])
        costs=np.array([[0,1e-12],[0,.1]])
        np.testing.assert_array_equal(select_wdro(np.array([[0,1],[0,1]]),loss,costs,0,0)[0],[0,0])

    def test_update_direction_projection_and_negative_inputs(self):
        self.assertAlmostEqual(update_lambda(1,.3,.1,2),1.4)
        self.assertAlmostEqual(update_lambda(1,.1,.3,2),.6)
        self.assertEqual(update_lambda(.1,0,1,1),0)
        for args in [(-1,.1,.1,1),(1,np.nan,.1,1),(1,.1,-1,1),(1,.1,.1,0)]:
            with self.assertRaises(ValueError): update_lambda(*args)
        with self.assertRaises(ValueError): select_wdro(np.array([[0,1]]),[[0,1]],[[0,-1]],0,.1)

    def test_lp_known_fractional_optimum_and_average_not_per_center(self):
        result=primal_dual_lp([[0,4],[0,1]],[[0,2],[0,2]],.5)
        self.assertEqual(result['status'],'pass')
        self.assertAlmostEqual(result['primal_value'],1.)
        np.testing.assert_allclose(result['conditional_mass'],[[.5,.5],[1,0]])

    def test_lp_zero_slack_negative_losses_and_monotonic_radius(self):
        losses=np.array([[-.2,.5,1],[-.4,.2,.8]])
        costs=np.array([[0,.1,.4],[0,.2,.5]])
        values=[]
        for rho in [0,.05,.2,1]:
            r=primal_dual_lp(losses,costs,rho)
            self.assertEqual(r['status'],'pass');values.append(r['primal_value'])
        self.assertAlmostEqual(values[0],-.3)
        self.assertAlmostEqual(values[-1],.9)
        self.assertTrue(np.all(np.diff(values)>=0))

    def test_cost_support_and_selection_pin_order_invariance(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/stage3_baselines_v1.json').read_text())
        batch=materialize_training_batch(protocol=load_protocol(),degree=30,training_seed=101,update=1,batch_size=3)
        a=generate_support(batch,training_seed=101,update=1,search=config['search'])
        other=replace(batch,pins=batch.pins[:,::-1].copy())
        b=generate_support(other,training_seed=101,update=1,search=config['search'])
        ca=transport_costs(batch.pins,a.pins); cb=transport_costs(other.pins,b.pins)
        np.testing.assert_allclose(ca,cb,atol=1e-16)
        ia,ra=shortlist_support(a);ib,rb=shortlist_support(b)
        np.testing.assert_array_equal(ia,ib)
        np.testing.assert_allclose(ra,rb,atol=1e-14)
        self.assertTrue(np.all(ca[:,0]==0))
        self.assertTrue(np.all(ca[:,1:]>0))
        for row in a.pins:
            self.assertEqual(len({pin_set_sha256(net) for net in row}),33)

    def test_stronger_perturbations_raise_distribution_cost(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/stage3_baselines_v1.json').read_text())
        batch=materialize_training_batch(protocol=load_protocol(),degree=30,training_seed=101,update=1,batch_size=8)
        search=dict(config['search']); search['amplitudes']=[.005]*4
        small=generate_support(batch,training_seed=101,update=1,search=search)
        search['amplitudes']=[.04]*4
        large=generate_support(batch,training_seed=101,update=1,search=search)
        self.assertGreater(transport_costs(batch.pins,large.pins)[:,1:].mean(),transport_costs(batch.pins,small.pins)[:,1:].mean()*3)

if __name__=='__main__': unittest.main()
