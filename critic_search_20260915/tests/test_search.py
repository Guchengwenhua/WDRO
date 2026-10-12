import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
import torch
from search import configurations, propose, evaluate_candidates, normalized_map
from adv_v3.common import make_data
from adv_v3.search import _check_pins

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x):return self.weight*x.square().mean((1,2))

class Oracle:
    def evaluate(self,x):
        v=np.square(x).mean((1,2));return dict(regret=v,lstar=np.ones(len(x)),length=1+v)

def test_every_variant_legal_budget_permutation_and_restore():
    x=make_data(4,1,'uniform','critic-search-test',1);model=Toy().train()
    for name,cfg in configurations().items():
        z,d=propose(model,x,.2,123,cfg)
        _check_pins(z,4)
        assert model.training and model.weight.requires_grad
        swapped,_=propose(model,x[:,::-1].copy(),.2,123,cfg)
        np.testing.assert_allclose(swapped[:,:,::-1],z,atol=2e-7,rtol=0)
        result=evaluate_candidates(x,z,Oracle(),.2)
        assert result['calls'].tolist()==[64]
        np.testing.assert_array_equal(result['score'],result['all_score'].max(1))
        assert (result['score']>=result['original_regret']).all()

def test_differentiable_normalization_finite_difference():
    x=torch.tensor([[[.1,.1],[.9,.4],[.4,.8]]],dtype=torch.float64)
    u=torch.tensor([[[.01,.02],[-.02,.01],[.01,-.01]]],dtype=torch.float64,requires_grad=True)
    f=normalized_map(x,u).square().sum();g=torch.autograd.grad(f,u)[0]
    h=1e-6;up=u.detach().clone();down=up.clone();up[0,2,0]+=h;down[0,2,0]-=h
    fd=(normalized_map(x,up).square().sum()-normalized_map(x,down).square().sum())/(2*h)
    torch.testing.assert_close(g[0,2,0],fd,rtol=1e-6,atol=1e-8)

def test_repeatable_starts():
    x=make_data(4,2,'uniform','critic-search-test',2);m=Toy()
    a,_=propose(m,x,0,123,configurations()['no_move'])
    b,_=propose(m,x,0,123,configurations()['no_move'])
    np.testing.assert_array_equal(a,b)
