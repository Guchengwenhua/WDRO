"""Acceptance gates for mathematical invariants, not evidence of scientific benefit."""
from exp_common import *
import torch
from net_models import Denoiser,losses
from structure import graph_noise,tile_project

def main():
    seed_all(9);results=[]
    for kind in ['set','graph']:
        m=Denoiser(kind,32);x=torch.rand(3,30,2)*.8+.1;s=torch.ones(3)*.04
        # Nonzero output is required to make equivariance test meaningful.
        torch.nn.init.normal_(m.out[-1].weight,std=.1)
        p=torch.randperm(30);a=m(x,s);b=m(x[:,p],s)
        torch.testing.assert_close(a[:,p],b,atol=1e-6,rtol=1e-5)
        loss,_,_=losses(a,x,s,.1);loss.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        results.append(kind+'_equivariant_finite_gradient')
    rng=np.random.default_rng(77);x=(.1+.8*rng.random((30,2))).astype(np.float32);noise=rng.normal(size=x.shape)*.04
    p=rng.permutation(30);np.testing.assert_allclose(graph_noise(x[p],noise[p]),graph_noise(x,noise)[p],atol=1e-9)
    lower=x-.01;upper=x+.01;y,f=tile_project(x,x+noise,lower,upper)
    assert (y>=lower-6e-8).all() and (y<=upper+6e-8).all();np.testing.assert_array_equal(y.max(0),x.max(0));np.testing.assert_array_equal(y.min(0),x.min(0))
    results.extend(['laplacian_permutation','tile_bounds_and_bbox'])
    # RNG+optimizer restore must reproduce exactly the same next update.
    m=Denoiser('set',32);o=torch.optim.Adam(m.parameters(),lr=.001)
    def step():
        z=torch.rand(4,30,2);out=m(z,torch.ones(4)*.04);loss=out.square().mean();o.zero_grad();loss.backward();o.step();return {k:v.detach().clone() for k,v in m.state_dict().items()}
    step();atomic_torch(ART/'tests/resume.pt',dict(m=m.state_dict(),o=o.state_dict(),rng=rng_state()));expected=step()
    cp=torch.load(ART/'tests/resume.pt',weights_only=False);m.load_state_dict(cp['m']);o.load_state_dict(cp['o']);set_rng(cp['rng']);actual=step()
    for k in actual:torch.testing.assert_close(actual[k],expected[k],atol=0,rtol=0)
    results.append('optimizer_rng_resume_exact')
    atomic_json(ART/'tests/unit.json',dict(status='pass',checks=results,time=utc()));print(results,flush=True)
if __name__=='__main__':main()
