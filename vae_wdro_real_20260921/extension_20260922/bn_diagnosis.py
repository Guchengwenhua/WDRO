"""Validation-only parameter/BN-statistics swap diagnosis, not a trained model."""
import torch
from exp_common import *
from src.rest import build_actor,RestTreeEvaluator
from evaluation import validation

def main():
    if (ART/'done'/'bn_diagnosis.json').exists():return
    seed_all(20260922);data=load_data();actor=build_actor(30,torch.device('cpu'));rest=RestTreeEvaluator(CFG['library']);rows=[]
    keys=lambda k:k.endswith('running_mean') or k.endswith('running_var') or k.endswith('num_batches_tracked')
    for seed in [101,102]:
        parent=torch.load(CFG['parent_template'].format(seed=seed),map_location='cpu',weights_only=False)['actor_state_dict']
        for method in ['erm_a','raw_wdro']:
            last=torch.load(Path(CFG['previous_artifacts'])/'runs'/f'{method}_s{seed}'/'last.pt',map_location='cpu',weights_only=False)['actor_state_dict']
            for weights,stats in [('parent','parent'),('parent','last'),('last','parent'),('last','last')]:
                params=parent if weights=='parent' else last;buffers=parent if stats=='parent' else last
                state={k:(buffers[k] if keys(k) else v) for k,v in params.items()};actor.load_state_dict(state,strict=True);ledger={}
                val=validation(actor,rest,data,ledger)
                rows.append(dict(seed=seed,method=method,weights=weights,bn_statistics=stats,**val,**ledger))
                print(rows[-1],flush=True)
    write_csv(ROOT/'tables'/'bn_validation_diagnosis.csv',rows)
    mark_done('bn_diagnosis',rows=rows,interpretation='posthoc A_val diagnosis only; no test, training, or primary-model substitution')

if __name__=='__main__':main()
