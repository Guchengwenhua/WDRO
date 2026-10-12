import fcntl
import time
import numpy as np
import torch
from exp_common import *
from src.rest import RestTreeEvaluator

def greedy(actor,rest,pins,ledger,category):
    actor.eval();out=[]
    with torch.no_grad():
        for begin in range(0,len(pins),256):
            x=np.asarray(pins[begin:begin+256],np.float32);n=len(x)
            if n<256:x=np.concatenate([x,np.repeat(x[-1:],256-n,axis=0)])
            topology,_=actor(x,True);topology=topology[:n].cpu().numpy()
            if topology.shape!=(n,58) or (topology<0).any() or (topology>=30).any():raise RuntimeError('invalid topology')
            out.append(rest.lengths(x[:n],topology))
            for key,value in [(category+'_requested',n),(category+'_forward_rows',256),
                    (category+('_gpu_rows' if next(actor.parameters()).is_cuda else '_cpu_rows'),256)]:ledger[key]=ledger.get(key,0)+value
    return np.concatenate(out)

def regret(length,reference):
    out=length/reference-1
    if not np.isfinite(out).all() or out.min()<-1e-6:raise RuntimeError('nonfinite or negative true regret')
    return out

def metrics(values,designs):
    return dict(nets=len(values),mean=float(values.mean()),p95=float(np.quantile(values,.95)),
        macro=float(np.mean([values[designs==d].mean() for d in np.unique(designs)])))

def validation(actor,rest,data,ledger):
    take=data['split']=='A_val';length=greedy(actor,rest,data['pins'][take],ledger,'validation')
    return metrics(regret(length,data['lstar'][take]),data['design_ids'][take])

def output_test(actor,rest,data,ledger,path,method,seed,generator,budget,role,update):
    take=np.isin(data['split'],['B_test','C_external']);n=int(take.sum())
    length=greedy(actor,rest,data['pins'][take],ledger,'test');v=regret(length,data['lstar'][take])
    a={k:data[k][take] for k in ['net_ids','design_ids','family_ids','dataset_ids','split','lstar']}
    a.update(length=length,regret=v,method=np.full(n,method),seed=np.full(n,seed),generator=np.full(n,generator),
        budget=np.full(n,budget),role=np.full(n,role),update=np.full(n,update))
    atomic_npz(path,**a)
    write_csv(Path(path).with_suffix('.csv'),[{k:a[k][i].item() for k in a} for i in range(n)])
    return dict(path=str(path),nets=n,method=method,seed=seed,generator=generator,budget=budget,role=role,update=update)

def evaluate_budget(actor,rest,data,ledger,folder,method,seed,generator,budget):
    (ART/'test').mkdir(exist_ok=True);results=[]
    # Lock the parent cache so concurrent methods cannot duplicate the reference evaluation.
    with (ART/'test'/f'parent_s{seed}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ART/'test'/f'parent_s{seed}.npz'
        if not path.exists():
            cp=torch.load(CFG['parent_template'].format(seed=seed),map_location='cpu',weights_only=False)
            actor.load_state_dict(cp['actor_state_dict'],strict=True)
            results.append(output_test(actor,rest,data,ledger,path,'parent',seed,0,0,'parent',0))
    for role in ['last','val_best']:
        cp=torch.load(folder/f'{role}_{budget}.pt',map_location='cpu',weights_only=False)
        actor.load_state_dict(cp['actor_state_dict'],strict=True)
        path=ART/'test'/f'{method}_g{generator}_s{seed}_u{budget}_{role}.npz'
        results.append(output_test(actor,rest,data,ledger,path,method,seed,generator,budget,role,cp['update']))
    return results
