"""Independent fresh RAW K32 pressure test; never used for model selection."""
from exp_common import *
from candidates import perturb
from src.costs import hungarian_l1_mean
import argparse

def prepare():
    if (ART/'pools/attack32.geometry.npz').exists():return
    data=load_data();ids=[];rng=np.random.default_rng(8110922)
    for d in np.unique(data['design_ids'][data['split']=='B_test']):
        subset=np.flatnonzero((data['design_ids']==d)&(data['split']=='B_test'));ids.extend(np.sort(rng.choice(subset,min(32,len(subset)),replace=False)))
    ids=np.array(ids);bank=np.repeat(data['pins'][ids,None],32,axis=1);cost=np.zeros((len(ids),32));families=['jitter','sparse','correlated','axis_biased'];amps=[.01,.02,.04,.08]
    for i,x in enumerate(bank[:,0]):
        for k in range(1,32):
            y,_,_=perturb(x,families[(k-1)%4],amps[((k-1)//4)%4],rng);bank[i,k]=y;cost[i,k]=hungarian_l1_mean(x,y)
    atomic_npz(ART/'pools/attack32.geometry.npz',pins=bank,cost=cost,center_index=ids)
    mark_done('attack_geometry')

def evaluate(method,seed,generator):
    import torch
    from src.rest import build_actor,RestTreeEvaluator
    from evaluation import greedy,regret
    name=f'attack_{method}_g{generator}_s{seed}'
    if (ART/'done'/f'{name}.json').exists():return
    seed_all(seed);torch.cuda.set_per_process_memory_fraction(CFG['runtime']['gpu_memory_fraction']);actor=build_actor(30,torch.device('cuda'));rest=RestTreeEvaluator(CFG['library'])
    folder=ART/'runs'/f'{method}_g{generator}_s{seed}';cp=torch.load(folder/'val_best_2048.pt',map_location='cpu',weights_only=False);actor.load_state_dict(cp['actor_state_dict'])
    pool=np.load(ART/'pools/attack32.npz');ledger={};length=greedy(actor,rest,pool['pins'].reshape(-1,30,2),ledger,'independent_attack').reshape(-1,32)
    valid=pool['valid'];v=np.full_like(length,-np.inf);v[valid]=regret(length[valid],pool['lstar'][valid]);slot=v.argmax(1);idx=pool['center_index'];data=load_data()
    arrays=dict(net_ids=data['net_ids'][idx],design_ids=data['design_ids'][idx],worst_regret=v.max(1),nominal_regret=v[:,0],selected_cost=pool['cost'][np.arange(len(idx)),slot],slot=slot)
    atomic_npz(ART/'attack'/f'{name}.npz',**arrays);mark_done(name,method=method,seed=seed,generator=generator,mean=float(v.max(1).mean()),p95=float(np.quantile(v.max(1),.95)),ledger=ledger,lambda_value=0.,K=32,selected_update=cp['update'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--method');p.add_argument('--seed',type=int);p.add_argument('--generator',type=int,default=301);a=p.parse_args()
    if a.prepare:prepare()
    else:evaluate(a.method,a.seed,a.generator)
