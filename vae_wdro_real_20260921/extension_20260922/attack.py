"""Shared independent RAW-Random32, lambda=0, never used for selection."""
import argparse
import time
import numpy as np
import torch
from exp_common import *
from geometry_local import perturb,hungarian_l1_mean
from evaluation import greedy,regret,metrics
from src.rest import build_actor,RestTreeEvaluator

def prepare():
    if (ART/'done'/'attack_geometry.json').exists():return
    data=load_data();rng=np.random.default_rng(CFG['attack']['seed']);ids=[]
    for family in CFG['splits']['B_test']:
        available=np.flatnonzero((data['split']=='B_test')&(data['family_ids']==family))
        ids.extend(np.sort(rng.choice(available,32,replace=False)).tolist())
    ids=np.array(ids);center=data['pins'][ids];bank=np.empty((96,32,30,2),np.float32);bank[:,0]=center
    cost=np.zeros((96,32));repair=np.zeros((96,32),bool);disp=np.zeros((96,32));families=['jitter','sparse','correlated','axis_biased'];amp=[.01,.02,.04,.08]
    for i,x in enumerate(center):
        for slot in range(1,32):
            family=families[(slot-1)%4];amplitude=amp[((slot-1)//4)%4]
            y,r,d=perturb(x,family,amplitude,rng);bank[i,slot]=y;cost[i,slot]=hungarian_l1_mean(x,y);repair[i,slot]=r;disp[i,slot]=d
    unique=np.array([len({x.tobytes() for x in group}) for group in bank])
    atomic_npz(ART/'pools'/'attack_raw32.geometry.npz',pins=bank,cost=cost,center_index=ids,
        slot=np.tile(np.arange(32),(96,1)),repaired=repair,repair_max_displacement=disp,unique_K=unique)
    atomic_json(ART/'pools'/'attack_raw32.geometry.json',dict(centers=96,candidates=32,seed=CFG['attack']['seed'],
        families=CFG['splits']['B_test'],centers_per_family=32,objective='maximum true regret, lambda=0',not_used_for_selection=True))
    mark_done('attack_geometry',centers=96,candidates=32)

def evaluate():
    if (ART/'done'/'attack_evaluation.json').exists():return
    start=time.monotonic();seed_all(20260925);data=load_data()
    with np.load(ART/'pools'/'attack_raw32.npz') as f:pool={k:f[k] for k in f.files}
    actor=build_actor(30,torch.device('cpu'));rest=RestTreeEvaluator(CFG['library']);ledger={};summaries=[]
    out=ART/'attack';out.mkdir(exist_ok=True);cache=out/'prediction_cache';cache.mkdir(exist_ok=True)
    entries=[]
    for seed in CFG['seeds']:
        entries.append(dict(method='parent',seed=seed,budget=0,role='parent',path=CFG['parent_template'].format(seed=seed)))
        for budget in CFG['attack']['budgets']:
            for method in CFG['primary_methods']:
                for role in ['last','val_best']:
                    path=ART/'runs'/f'{method}_g0_s{seed}'/f'{role}_{budget}.pt'
                    if path.exists():entries.append(dict(method=method,seed=seed,budget=budget,role=role,path=str(path)))
    for entry in entries:
        cp=torch.load(entry['path'],map_location='cpu',weights_only=False)
        update=0 if entry['method']=='parent' else cp['update']
        identity=f'parent_s{entry["seed"]}' if update==0 else f'{entry["method"]}_s{entry["seed"]}_update{update}'
        saved=cache/f'{identity}.npz';reused=saved.exists()
        if reused:
            with np.load(saved) as f:length=f['length']
        else:
            actor.load_state_dict(cp['actor_state_dict'],strict=True)
            length=greedy(actor,rest,pool['pins'].reshape(-1,30,2),ledger,'attack').reshape(96,32)
            atomic_npz(saved,length=length)
        score=np.full((96,32),-np.inf);score[pool['valid']]=regret(length[pool['valid']],pool['lstar'][pool['valid']])
        slot=score.argmax(1);rows=np.arange(96);worst=score[rows,slot];idx=pool['center_index']
        name=f'{entry["method"]}_s{entry["seed"]}_u{entry["budget"]}_{entry["role"]}'
        arrays=dict(net_ids=data['net_ids'][idx],design_ids=data['design_ids'][idx],family_ids=data['family_ids'][idx],
            chosen_slot=slot,length=length[rows,slot],selected_lstar=pool['lstar'][rows,slot],selected_cost=pool['cost'][rows,slot],
            worst_regret=worst,nominal_regret=score[:,0],candidate_length=length,candidate_regret=score)
        atomic_npz(out/f'{name}.npz',**arrays)
        write_csv(out/f'{name}.csv',[{k:v[i].item() for k,v in arrays.items() if v.ndim==1} for i in range(96)])
        summaries.append(dict(**{k:entry[k] for k in ['method','seed','budget','role']},update=update,**metrics(worst,data['design_ids'][idx]),
            nominal_mean=float(score[:,0].mean()),reused_checkpoint_prediction=reused,path=str(out/f'{name}.npz')))
        write_csv(ROOT/'tables'/'independent_attack.csv',summaries)
        progress('attack_evaluation',completed=len(summaries),total=len(entries),ledger=ledger)
        print('ATTACK',name,summaries[-1]['mean'],flush=True)
    atomic_json(out/'summary.json',dict(models=len(summaries),ledger=ledger,seconds=time.monotonic()-start,rows=summaries,
        attack='shared RAW-Random32 lambda0',backend='cpu',selection_use=False))
    mark_done('attack_evaluation',seconds=time.monotonic()-start,models=len(summaries),ledger=ledger)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    if a.evaluate:evaluate()
