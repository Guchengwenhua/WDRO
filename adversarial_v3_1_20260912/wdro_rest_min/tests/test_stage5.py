import unittest
import numpy as np
from src.candidates import generate_support,shortlist_support
from src.training_data import materialize_training_batch
from src.protocol import load_protocol
from src.geometry import pin_set_sha256,has_spacing_violation
from src.datasets import array_bundle_sha256
from src.costs import hungarian_l1_mean,symmetric_chamfer_l1
from stage5.search import cem_round,adapt_cem,transport_costs
from stage5.configuration import base_config,ROOT
import json

class Stage5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch=materialize_training_batch(protocol=load_protocol(),degree=30,training_seed=101,update=11,batch_size=4)
        cls.search=dict(json.loads((ROOT/'configs/stage3_baselines_v1.json').read_text())['search'],**base_config()['search_overrides'],ground_cost='hungarian_l1_mean')
    def test_k_shortlist_counts_and_nominal(self):
        for k in [1,8,32]:
            cfg=dict(self.search,candidate_count=k);s=generate_support(self.batch,training_seed=101,update=11,search=cfg)
            ids,_=shortlist_support(s,min(4,k+1))
            self.assertEqual(ids.shape,(4,min(4,k+1)));np.testing.assert_array_equal(s.pins[:,0],self.batch.pins)
    def test_cost_dispatch(self):
        s=generate_support(self.batch,training_seed=101,update=11,search=dict(self.search,candidate_count=1))
        for kind,fn in [('hungarian_l1_mean',hungarian_l1_mean),('symmetric_chamfer_l1',symmetric_chamfer_l1)]:
            c=transport_costs(self.batch.pins,s.pins,kind)
            self.assertEqual(c[0,1],fn(self.batch.pins[0],s.pins[0,1]));self.assertTrue(np.all(c[:,0]==0))
    def cem(self,batch):
        mean=np.zeros((4,30,2));std=np.full_like(mean,.02);seen=[set() for _ in range(4)];allpins=[]
        for t in range(4):
            pins,_=cem_round(batch,update=11,training_seed=101,round_index=t,mean=mean,std=std,seen=seen,config=self.search)
            fake_scores=np.ptp(pins,axis=2).sum(axis=2)
            mean,std,elite=adapt_cem(batch,pins,fake_scores,.2,.01,self.search)
            self.assertEqual(elite.shape,(4,4));self.assertTrue(np.all(std>=.001));allpins.append(pins)
        return np.concatenate(allpins,axis=1)
    def test_cem_64_unique_legal_deterministic(self):
        a=self.cem(self.batch);np.testing.assert_array_equal(a,self.cem(self.batch))
        self.assertEqual(a.shape,(4,64,30,2));np.testing.assert_array_equal(a[:,0],self.batch.pins)
        for row in a:
            self.assertEqual(len({pin_set_sha256(p) for p in row}),64)
            self.assertFalse(any(has_spacing_violation(p) for p in row))
            self.assertTrue(np.isfinite(row).all() and row.min()>=np.float32(.1) and row.max()<=np.float32(.9))
    def test_cem_pin_order_equivariance(self):
        from dataclasses import replace
        order=np.random.default_rng(1).permutation(30)
        b=replace(self.batch,pins=self.batch.pins[:,order])
        np.testing.assert_array_equal(self.cem(self.batch)[:,:,order],self.cem(b))
    def test_random64_total_includes_nominal(self):
        s=generate_support(self.batch,training_seed=101,update=11,search=dict(self.search,candidate_count=63))
        self.assertEqual(s.pins.shape[1],64);self.assertEqual(s.generated,4*63)
        np.testing.assert_array_equal(s.pins[:,0],self.batch.pins)

if __name__=='__main__':unittest.main()
