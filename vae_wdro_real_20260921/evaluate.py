"""Identical greedy batch/padding convention for candidate, validation and test."""
import time
import numpy as np
import torch
from exp_common import *
from src.rest import RestTreeEvaluator

def greedy_lengths(actor,rest,pins,ledger,category):
    actor.eval();result=[];batch=CFG['evaluation']['greedy_batch_size']
    with torch.no_grad():
        for begin in range(0,len(pins),batch):
            x=np.asarray(pins[begin:begin+batch],dtype=np.float32);n=len(x)
            if n<batch:x=np.concatenate([x,np.repeat(x[-1:],batch-n,axis=0)])
            topology,_=actor(x,True);topology=topology[:n].cpu().numpy()
            if topology.shape!=(n,58) or np.any(topology<0) or np.any(topology>=30): raise RuntimeError('invalid Actor topology')
            lengths=rest.lengths(x[:n],topology)
            ledger[category+'_requested']=ledger.get(category+'_requested',0)+n
            ledger[category+'_forward_rows']=ledger.get(category+'_forward_rows',0)+batch
            backend=next(actor.parameters()).device.type
            ledger[category+'_'+('gpu_rows' if backend=='cuda' else 'cpu_rows')]=ledger.get(category+'_'+('gpu_rows' if backend=='cuda' else 'cpu_rows'),0)+batch
            result.append(lengths)
    return np.concatenate(result)

def relative_regret(lengths,reference):
    values=np.asarray(lengths,dtype=np.float64)/reference-1
    if not np.isfinite(values).all() or values.min() < -1e-6:
        raise RuntimeError(f'invalid true regret: minimum={values.min()}')
    return values

def metrics(regret,design):
    return dict(nets=len(regret),mean_regret=float(regret.mean()),p95_regret=float(np.quantile(regret,.95)),
                design_macro_mean_regret=float(np.mean([regret[design==d].mean() for d in np.unique(design)])))

def validation(actor,rest,data,ledger):
    take=data['split']=='A_val';lengths=greedy_lengths(actor,rest,data['pins'][take],ledger,'validation')
    return metrics(relative_regret(lengths,data['lstar'][take]),data['design_ids'][take])

def evaluate_test(actor,rest,data,ledger,method,seed,role,update,path):
    start=time.monotonic();take=np.isin(data['split'],['B_test','C_external'])
    lengths=greedy_lengths(actor,rest,data['pins'][take],ledger,'test')
    regret=relative_regret(lengths,data['lstar'][take]);n=len(lengths)
    arrays={k:data[k][take] for k in ['net_ids','design_ids','family_ids','dataset_ids','split','lstar']}
    arrays.update(length=lengths,regret=regret,method=np.full(n,method),seed=np.full(n,seed),
                  checkpoint_role=np.full(n,role),update=np.full(n,update))
    atomic_npz(path,**arrays)
    write_csv(Path(path).with_suffix('.csv'),[{k:arrays[k][i].item() for k in arrays} for i in range(n)])
    return dict(seconds=time.monotonic()-start,nets=n,path=str(path))

def finish_tests(actor,rest,data,ledger,folder,method,seed):
    tests=[];parent_path=ART/'test'/f'parent_s{seed}.npz'
    if not parent_path.exists():
        cp=torch.load(CFG['parent_template'].format(seed=seed),map_location='cpu',weights_only=False)
        actor.load_state_dict(cp['actor_state_dict'],strict=True)
        tests.append(evaluate_test(actor,rest,data,ledger,'parent',seed,'parent',0,parent_path))
    for filename,role in [('last.pt','last_256'),('val_best.pt','val_best')]:
        path=ART/'test'/f'{method}_s{seed}_{role}.npz'
        cp=torch.load(folder/filename,map_location='cpu',weights_only=False)
        actor.load_state_dict(cp['actor_state_dict'],strict=True)
        tests.append(evaluate_test(actor,rest,data,ledger,method,seed,role,cp['update'],path))
    return tests
