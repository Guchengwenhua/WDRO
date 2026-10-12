#!/usr/bin/env python3
"""Recompute all update-1 shortlisted scores with fresh GeoSteiner calls."""
import argparse
import json
import os
import sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--physical-gpu-index',type=int,required=True);p.add_argument('--preflight',type=Path,required=True);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu_index)
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.candidates import generate_support,shortlist_support
from src.training_data import materialize_training_batch
from src.protocol import load_protocol
from src.trainer import configure_determinism,load_trainer_config,WORKSPACE_ROOT
from src.rest import build_actor,load_trusted_legacy_actor,RestTreeEvaluator,file_sha256
from src.geosteiner import GeoSteinerEvaluator
from src.metrics import relative_regret
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from stage4.dual import transport_costs,primal_dual_lp

def main():
    preflight=json.loads(args.preflight.read_text());assert preflight['preflight_passed'] and args.physical_gpu_index in preflight['eligible_gpu_indices']
    protocol=load_protocol();config=json.loads((ROOT/'configs/stage3_baselines_v1.json').read_text())
    configure_determinism(load_trainer_config());device=torch.device('cuda:0')
    actor=build_actor(30,device);legacy=WORKSPACE_ROOT/protocol['training']['initial_actor_checkpoint_template'].format(degree=30)
    load_trusted_legacy_actor(actor,legacy,map_location=device);actor.eval()
    batch=materialize_training_batch(protocol=protocol,degree=30,training_seed=101,update=1,batch_size=256)
    support=generate_support(batch,training_seed=101,update=1,search=config['search']);indices,_=shortlist_support(support)
    flat=support.pins[np.arange(256)[:,None],indices].reshape(-1,30,2)
    outputs=[]
    with torch.no_grad():
        for start in range(0,1024,256):
            out,_=actor(flat[start:start+256],True);outputs.append(out.cpu().numpy())
    lengths=RestTreeEvaluator().lengths(flat,np.concatenate(outputs))
    references=GeoSteinerEvaluator().lengths(flat)
    regrets=relative_regret(lengths,references).reshape(256,4)
    event_path=EXP/'logs/stage4/stage4_smoke_q50_continuous_v1/events.jsonl'
    event=next(json.loads(line) for line in event_path.read_text().splitlines() if json.loads(line).get('event')=='train')
    detail=event['method_detail'];np.testing.assert_array_equal(regrets,detail['exact_relative_regrets'])
    np.testing.assert_array_equal(references.reshape(256,4),detail['exact_lstar'])
    costs=transport_costs(batch.pins,flat.reshape(256,4,30,2));np.testing.assert_array_equal(costs,detail['exact_transport_costs'])
    lp=primal_dual_lp(regrets[:8],costs[:8],detail['rho']);assert lp['status']=='pass'
    output={'status':'pass','at_utc':utc_now(),'gpu':args.physical_gpu_index,'pid':os.getpid(),
        'preflight':str(args.preflight),'legacy_checkpoint_sha256':file_sha256(legacy),'source_events_sha256':file_sha256(event_path),
        'centers':256,'independent_actor_calls_nets':1024,'independent_rest_calls_nets':1024,'independent_geosteiner_calls_nets':1024,
        'maximum_regret_difference':0.,'maximum_lstar_difference':0.,'lp':lp,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'test_ispd_used':False}
    atomic_write_json(EXP/'logs/stage4/independent_initial_score_audit.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='lp'}))

if __name__=='__main__':main()
