"""Real-data CUDA/scorer/backprop pilot, saved as pilot evidence only."""
from exp_common import *
from net_models import Denoiser,losses
from src.rest import build_actor,build_critic,RestTreeEvaluator
from src.geosteiner import GeoSteinerEvaluator
from evaluation import greedy,regret
from structure import tile_project
import torch

def main():
    seed_all(101);data=load_data();ids=np.flatnonzero(data['split']=='A_train')[:64];x=data['pins'][ids];device=torch.device('cuda');start=time.monotonic()
    model=Denoiser('graph',160).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003);xx=torch.tensor(np.tile(x,(2,1,1)),device=device);s=torch.ones(128,device=device)*.04
    for step in range(100):
        y=xx+s[:,None,None]*torch.randn_like(xx);out=model(y,s);loss,_,_=losses(out,xx,s,1.);opt.zero_grad();loss.backward();opt.step()
    torch.cuda.synchronize();gensec=(time.monotonic()-start)/100
    physical=np.load(ART/'physical.npz');rng=np.random.default_rng(711);audit=[]
    # Actual physical samples including fixed extrema; mapped to original tile bounds.
    for i in np.flatnonzero(data['split']!='C_external')[::17]:
        orig=data['pins'][i];z,f=tile_project(orig,orig+rng.normal(size=(30,2))*.08,physical['lower'][i],physical['upper'][i]);check_pins(z);audit.append(f)
    actor=build_actor(30,device);critic=build_critic(30,device);cp=torch.load(CFG['parent_template'].format(seed=101),map_location='cpu',weights_only=False);actor.load_state_dict(cp['actor_state_dict']);critic.load_state_dict(cp['critic_state_dict'])
    rest=RestTreeEvaluator(CFG['library']);oracle=GeoSteinerEvaluator(CFG['library']);exact=np.array([oracle.solve(a,require_integral=True) for a in x[:4]])
    np.testing.assert_allclose(exact,data['lstar'][ids[:4]],rtol=1e-7,atol=1e-7)
    ledger={};length=greedy(actor,rest,x,ledger,'pilot');v=regret(length,data['lstar'][ids]);opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1e-5);start=time.monotonic()
    pool=np.load(ART/'pools/raw_wdro.npz');bank=pool['pins'][:64];refs=pool['lstar'][:64]
    for step in range(8):
        length=greedy(actor,rest,bank.reshape(-1,30,2),ledger,'pilot_candidate').reshape(64,8);r=regret(length,refs);slot=r.argmax(1);pins=bank[np.arange(64),slot];lstar=refs[np.arange(64),slot]
        actor.train();critic.train()
        for m in list(actor.modules())+list(critic.modules()):
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        topology,logp=actor(pins,False);v=regret(rest.lengths(pins,topology.detach().cpu().numpy()),lstar);target=torch.tensor(v,device=device,dtype=torch.float32);baseline=critic(pins);loss=((target-baseline).detach()*logp).mean()+(target-baseline).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);opt.step();assert torch.isfinite(loss)
    torch.cuda.synchronize();rsec=(time.monotonic()-start)/8
    atomic_json(ART/'tests/gpu_pilot.json',dict(status='PASS_STARTUP_ONLY',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),graph_w160_seconds_per_step=gensec,rest_wdro_seconds_per_update=rsec,max_torch_memory_mb=torch.cuda.max_memory_allocated()/1024**2,physical_net_checks=len(audit),tile_backtrack_fraction=float(np.mean(np.array(audit)<1)),exact_reference_checks=4,rest_updates=8,generator_updates=100,time=utc()))
    print(read_json(ART/'tests/gpu_pilot.json'),flush=True)
if __name__=='__main__':main()
