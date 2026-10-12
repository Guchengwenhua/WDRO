#!/usr/bin/env python3
"""Reload the selected checkpoint and reproduce all 8,000 validation metrics."""
import argparse
import json
import os
import sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--preflight',type=Path,required=True);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
import torch
from src.accounting import CallLedger
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.protocol import load_protocol
from src.rest import build_actor,RestTreeEvaluator,file_sha256
from src.trainer import configure_determinism,load_trainer_config
from src.validation import evaluate_validation,validation_manifest_hash

def main():
    host=json.loads(args.preflight.read_text());assert host['preflight_passed'] and args.gpu in host['eligible_gpu_indices']
    run=EXP/'logs/stage5'/args.run_id/'result.json';result=json.loads(run.read_text())
    assert result['status']=='complete' and result['completed_updates']==2000
    ck=torch.load(result['best_checkpoint'],map_location='cpu',weights_only=False)
    assert file_sha256(result['best_checkpoint'])==result['best_checkpoint_sha256']
    trainer=load_trainer_config();configure_determinism(trainer);device=torch.device('cuda:0')
    actor=build_actor(30,device);actor.load_state_dict(ck['actor_state_dict'],strict=True);ledger=CallLedger()
    v=evaluate_validation(actor,RestTreeEvaluator(),load_protocol(),degree=30,device=device,
                          batch_size=trainer['validation_batch_size'],ledger=ledger)
    expected=None
    for line in Path(result['event_log']).read_text().splitlines():
        event=json.loads(line)
        if event.get('event')=='validation' and event['update']==ck['update']:expected=event['result']
    assert v==expected and validation_manifest_hash(v)==result['best']['validation_manifest_sha256']
    output={'status':'pass','at_utc':utc_now(),'selected_update':ck['update'],'checkpoint_sha256':file_sha256(result['best_checkpoint']),
      'result_sha256':file_sha256(run),'validation_manifest_sha256':validation_manifest_hash(v),
      'validation_nets':8000,'exact_metric_equality':True,'call_ledger':ledger.to_dict(),
      'gpu':args.gpu,'preflight':str(args.preflight),'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'test_ispd_used':False}
    atomic_write_json(EXP/'logs/stage5_best_validation'/f'{args.run_id}.json',output);print(json.dumps(output))

if __name__=='__main__':main()
