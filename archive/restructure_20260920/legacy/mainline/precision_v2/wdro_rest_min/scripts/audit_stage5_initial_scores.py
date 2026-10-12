#!/usr/bin/env python3
"""Independently rerun initial real Actor/REST/GeoSteiner scores, including adaptive CEM."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--preflight',type=Path,required=True);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.protocol import load_protocol
from src.training_data import materialize_training_batch
from src.trainer import configure_determinism,load_trainer_config,WORKSPACE_ROOT
from src.rest import build_actor,RestTreeEvaluator,load_trusted_legacy_actor,file_sha256
from src.geosteiner import ParallelGeoSteinerEvaluator
from src.accounting import CallLedger
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from stage5.search import search_batch,transport_costs
from stage4.dual import select_wdro
from audit_stage3 import effective_events

class IndependentReference:
    def lengths(self,pins,evaluator):return evaluator.lengths(pins),len(pins)

def main():
    host=json.loads(args.preflight.read_text());assert host['preflight_passed'] and args.gpu in host['eligible_gpu_indices']
    rp=EXP/'logs/stage5'/args.run_id/'result.json';result=json.loads(rp.read_text());train,_,_=effective_events(result['event_log']);event=train[1];d=event['method_detail']
    protocol=load_protocol();configure_determinism(load_trainer_config());device=torch.device('cuda:0')
    actor=build_actor(30,device);load_trusted_legacy_actor(actor,WORKSPACE_ROOT/protocol['training']['initial_actor_checkpoint_template'].format(degree=30),map_location=device)
    cfg=result['run_identity']['ablation_config'];search=dict(json.loads((ROOT/'configs/stage3_baselines_v1.json').read_text())['search'],**cfg['search_overrides'],ground_cost=cfg['ground_cost'])
    batch=materialize_training_batch(protocol=protocol,degree=30,training_seed=101,update=1,batch_size=256)
    ledger=CallLedger();geo=ParallelGeoSteinerEvaluator(8)
    support,ids,regrets,refs,detail=search_batch(batch,update=1,training_seed=101,config=search,price=cfg['lambda_initial'],rho=result['run_identity']['rho'],
        actor=actor,rest=RestTreeEvaluator(),geosteiner=geo,cache=IndependentReference(),ledger=ledger)
    assert support.sha256==d['support_sha256']
    np.testing.assert_array_equal(ids,d['exact_shortlist_indices']);np.testing.assert_array_equal(regrets,d['exact_relative_regrets']);np.testing.assert_array_equal(refs,d['exact_lstar'])
    costs=transport_costs(batch.pins,support.pins[np.arange(256)[:,None],ids],cfg['ground_cost'])
    np.testing.assert_array_equal(costs,d['exact_transport_costs'])
    chosen,_=select_wdro(ids,regrets,costs,cfg['lambda_initial'],result['run_identity']['rho']);np.testing.assert_array_equal(chosen,d['selected_indices'])
    geo.close()
    out={'status':'pass','at_utc':utc_now(),'auditor_sha256':file_sha256(Path(__file__)),'run_id':args.run_id,'result_sha256':file_sha256(rp),'update':1,
         'real_regret_lstar_cost_selection_exact_equality':True,'searcher':search['searcher'],'actual_scoring_calls_per_center':len(ids[0]),
         'additional_independent_call_ledger':ledger.to_dict(),'reference_cache_reused':False,'geosteiner_workers':8,'training_reference_workers':result['geosteiner_workers'],'gpu':args.gpu,'preflight':str(args.preflight),'test_or_ispd_used':False}
    target=EXP/'logs/stage5_initial_score_audit'/f'{args.run_id}.json'
    if target.exists():
        atomic_write_json(target.parent/'history'/f'{args.run_id}_{file_sha256(target)[:12]}.json',json.loads(target.read_text()),mode=0o444)
    atomic_write_json(target,out);print(json.dumps(out))
if __name__=='__main__':main()
