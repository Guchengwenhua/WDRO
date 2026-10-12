import tempfile
import unittest
from pathlib import Path
import numpy as np
from common import data, costs
from search import attack
from adv_v3.search import antithetic_gradient
from resources import admissible
from design import initial_tasks
from report import sign_flip
from unittest.mock import patch
import design

class FakeOracle:
    def __init__(self):self.count=0
    def evaluate(self,pins):
        self.count+=len(pins)
        r=np.mean(np.sort(np.square(pins.astype(np.float64)).reshape(len(pins),-1),axis=1),axis=1)
        return dict(regret=r,lstar=np.ones(len(pins)),length=1+r)

class Tests(unittest.TestCase):
    def test_gradient_and_jump(self):
        rng=np.random.default_rng(4);eps=rng.normal(size=(1,20000,2,2));u=np.ones((1,2,2))*.2;mu=.02
        p=np.sum((u[:,None]+mu*eps)**2,axis=(2,3));m=np.sum((u[:,None]-mu*eps)**2,axis=(2,3))
        np.testing.assert_allclose(antithetic_gradient(p,m,eps,mu),2*u,atol=.015)
        positive=(eps[:,:,0,0]>0).astype(float);negative=(eps[:,:,0,0]<0).astype(float)
        self.assertGreater(antithetic_gradient(positive,negative,eps,mu)[0,0,0],10)
    def test_exact_budget_selection_and_permutation(self):
        x=data(20,2,'uniform','cpu-test')
        for k in (64,256):
            for method in ('es','random'):
                oracle=FakeOracle();r=attack(method,x,oracle,.2,3,budget=k)
                self.assertEqual(oracle.count,len(x)*k);self.assertTrue(np.all(r['score']>=r['original_score']))
                scores=r['evaluated_regrets']-.2*costs(x,r['evaluated_pins'])
                np.testing.assert_allclose(scores.max(axis=1),r['score'])
        a=attack('es',x,FakeOracle(),.2,3);perm=np.arange(20)[::-1]
        b=attack('es',x[:,perm],FakeOracle(),.2,3)
        np.testing.assert_allclose(a['score'],b['score'],atol=1e-12)
        np.testing.assert_array_equal(a['pins'][:,perm],b['pins'])
    def test_resource_gate(self):
        base=dict(load=20,disk_free_gib=50,mem_available_gib=100,gpus=[dict(index=0,uuid='test',free=12000,util=2)])
        self.assertEqual(len(admissible([base]*3)),1)
        self.assertEqual(admissible([base]*3,busy=[0]),[])
        for key,val in [('load',80),('disk_free_gib',3),('mem_available_gib',2)]:
            self.assertEqual(admissible([dict(base,**{key:val})]*3),[])
        hot=dict(base,gpus=[dict(index=0,uuid='test',free=12000,util=100)])
        self.assertEqual(admissible([hot]*3),[])
    def test_task_identity_and_statistics(self):
        tasks=initial_tasks();self.assertEqual(len(tasks),23)
        self.assertEqual(len({t['id'] for t in tasks}),len(tasks))
        self.assertAlmostEqual(sign_flip([-1,-1,-1,-1,-1]),1/32)
        self.assertEqual(sign_flip([0,0,0,0,0]),1)
    def test_graph_requires_training_before_test(self):
        tasks=initial_tasks()
        with patch.object(design,'search_selection',return_value=dict(mu=.02,search_step=.01)), \
             patch.object(design,'outer_selection',return_value=dict(lr=1e-5,nominal_fraction=0)), \
             patch.object(design,'done',side_effect=lambda ident: ident.startswith('pre_')):
            expanded=design.expand(tasks)
        self.assertEqual(sum(t.get('phase')=='post' for t in expanded),33)
        self.assertEqual(sum(t['kind']=='eval' for t in expanded),0)
        self.assertEqual({t['seed'] for t in expanded if t.get('phase')=='pilot'},{901})
        self.assertEqual(len({t['id'] for t in expanded}),len(expanded))

if __name__=='__main__':unittest.main()
