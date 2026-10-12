#!/usr/bin/env python3
"""Freeze implementation choices and training-only cost quantiles before sweeps."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stage5.configuration import initial_tasks,base_config,PLAN,EXP
from stage5.search import transport_costs
from src.candidates import generate_support
from src.training_data import materialize_training_batch
from src.protocol import load_protocol,protocol_sha256
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.datasets import array_bundle_sha256

def main():
    protocol=load_protocol();baseline=json.loads((ROOT/'configs/stage3_baselines_v1.json').read_text())
    batch=materialize_training_batch(protocol=protocol,degree=30,training_seed=101,update=1,batch_size=256)
    support=generate_support(batch,training_seed=101,update=1,search=baseline['search'])
    value={'status':'pass','source':'training_centers_only','degree':30,'seed':101,'update':1,'batch_size':256,
        'protocol_canonical_sha256':protocol_sha256(protocol),'baseline_config_file_sha256':file_sha256(ROOT/'configs/stage3_baselines_v1.json'),
        'training_batch_sha256':batch.bundle_sha256,'support_sha256':support.sha256,'model_calls':0,'test_or_ispd_used':False}
    for kind in ['hungarian_l1_mean','symmetric_chamfer_l1']:
        costs=transport_costs(batch.pins,support.pins,kind)
        assert np.all(costs[:,0]==0) and np.all(costs[:,1:]>0)
        value[kind]={'q50':float(np.quantile(costs[:,1:],.5)),'q75':float(np.quantile(costs[:,1:],.75)),
                     'costs_sha256':array_bundle_sha256({'costs':costs}),'transport_costs':costs.tolist()}
    old=json.loads((EXP/'cache/stage4/calibration_d30_s101.json').read_text())
    assert value['hungarian_l1_mean']['q50']==old['q50'] and value['hungarian_l1_mean']['q75']==old['q75']
    path=EXP/'cache/stage5/calibration_d30_s101.json'
    value['artifact_paths']={'logical_path':str(path),'resolved_path':str(path.resolve())}
    if path.exists():assert json.loads(path.read_text())==value
    else:atomic_write_json(path,value,mode=0o444)
    plan={'schema_version':1,'status':'frozen','stage':5,'degree':30,'training_seed':101,'policy_updates':2000,'batch_size':256,
          'test_or_ispd_used':False,'selection_source':'synthetic_validation_only','nominal_guard_absolute':0.002,
          'rho_order':['zero','q50','q75'],'tie_rule':'first in declared rho order; earliest checkpoint within run',
          'dependent_ablations':['lambda_fixed','k1','k8','chamfer','random_64','cem_64'],
          'minimum_total_unique_runs':9,'main_method_locked':{'searcher':'random_sobol','candidate_count':32,'ground_cost':'hungarian_l1_mean','lambda_mode':'learned'},
          'fixed_lambda_rule':'mean lambda_after over updates 1901-2000 of selected learned-rho run; frozen before fixed-lambda training',
          'radius_rule':'same selected quantile across K/searcher; Hungarian and Chamfer calibrated separately on same training K32 support',
          'equal_calls':'64 true Actor/REST/reference scoring requests per center per update; includes nominal once; Random=63 perturbations plus nominal; CEM=4x16 with nominal in round 1, elite4 after each round',
          'cem':base_config()['search_overrides']['cem'],'calibration':str(path),'calibration_sha256':file_sha256(path),
          'initial_tasks':initial_tasks(),'research_protocol_override':False}
    if PLAN.exists():assert json.loads(PLAN.read_text())==plan
    else:atomic_write_json(PLAN,plan,mode=0o444)
    print(json.dumps({'status':'pass','plan':str(PLAN),'quantiles':{k:{q:value[k][q] for q in ['q50','q75']} for k in ['hungarian_l1_mean','symmetric_chamfer_l1']}}))
if __name__=='__main__':main()
