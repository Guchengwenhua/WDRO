#!/usr/bin/env python3
"""Training-only diagnostic q50; no model selection or held-out metrics."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.candidates import generate_support
from src.datasets import array_bundle_sha256
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.protocol import load_protocol, protocol_sha256
from src.rest import file_sha256
from src.training_data import materialize_training_batch
from stage4.dual import transport_costs, primal_dual_lp

def main():
    protocol=load_protocol()
    config=json.loads((ROOT/'configs/stage3_baselines_v1.json').read_text())
    batch=materialize_training_batch(protocol=protocol,degree=30,training_seed=101,update=1,batch_size=256)
    support=generate_support(batch,training_seed=101,update=1,search=config['search'])
    costs=transport_costs(batch.pins,support.pins)
    q50=float(np.quantile(costs[:,1:],.5))
    target=ROOT.parent/'experiments/cache/stage4/calibration_d30_s101.json'
    value={'status':'pass','at_utc':utc_now(),'degree':30,'seed':101,'update':1,'batch_size':256,
           'source':'training_centers_only','model_calls':0,'test_or_ispd_used':False,
           'q50':q50,'q75':float(np.quantile(costs[:,1:],.75)),
           'nominal_zero_cost':bool(np.all(costs[:,0]==0)),
           'positive_distinct_perturbation_costs':bool(np.all(costs[:,1:]>0)),
           'transport_costs':costs.tolist(),'costs_sha256':array_bundle_sha256({'costs':costs}),
           'training_batch_sha256':batch.bundle_sha256,'support_sha256':support.sha256,
           'protocol_canonical_sha256':protocol_sha256(protocol),
           'baseline_config_file_sha256':file_sha256(ROOT/'configs/stage3_baselines_v1.json'),
           'artifact_paths':{'logical_path':str(target),'resolved_path':str(target.resolve())}}
    assert value['nominal_zero_cost'] and value['positive_distinct_perturbation_costs'] and q50>0
    if target.exists():
        old=json.loads(target.read_text())
        for key in value.keys()-{'at_utc'}:
            assert old[key]==value[key],key
    else:
        atomic_write_json(target,value,mode=0o444)
    cases=[]
    for rho in [0,.05,.5,2.]:
        cases.append(primal_dual_lp([[0,4],[0,1]],[[0,2],[0,2]],rho))
    assert all(c['status']=='pass' for c in cases)
    atomic_write_json(ROOT.parent/'experiments/logs/stage4/math_lp_toy.json',
                      {'status':'pass','cases':cases,'at_utc':utc_now(),'source_file_sha256':file_sha256(ROOT/'stage4/dual.py')})
    print(json.dumps({'calibration':str(target),'q50':q50,'q75':value['q75'],'toy_lp':'pass'}))

if __name__=='__main__': main()
