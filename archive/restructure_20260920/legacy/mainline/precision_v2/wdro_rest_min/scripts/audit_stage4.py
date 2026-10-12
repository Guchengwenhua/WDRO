#!/usr/bin/env python3
"""Full WDRO input/cost/dual replay, actual HM-prefix budgets, and LP audit."""
from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
CODEX=ROOT.parent
EXP=CODEX/'experiments'
sys.path.insert(0,str(ROOT))
from src.candidates import generate_support, shortlist_support
from src.datasets import array_bundle_sha256, load_dataset
from src.geometry import pin_set_sha256, has_spacing_violation
from src.io_utils import atomic_write_json
from src.job_control import JobLease, utc_now
from src.protocol import load_protocol, protocol_sha256
from src.rest import file_sha256
from src.stage2_audit_checks import all_finite, optimizer_checks, validation_checks
from src.trainer import canonical_sha256, load_trainer_config
from src.training_data import materialize_training_batch, EMPTY_STREAM_SHA256, extend_stream_hash
from src.validation import validation_manifest_hash
from stage4.dual import transport_costs, primal_dual_lp
from audit_stage3 import effective_events

AUDIT=EXP/'logs/stage4_audit'
BUDGET_FIELDS=('centers','perturbations','surrogate_nets','exact_rescore_nets','reference_requests')

def audit_run(run_id, stability=False, heartbeat=None):
    result_path=EXP/'logs/stage4'/run_id/'result.json'
    result=json.loads(result_path.read_text())
    train,validation,raw=effective_events(result['event_log'])
    updates=result['completed_updates'];rho=result['run_identity']['rho']
    assert sorted(train)==list(range(1,updates+1)), 'missing update log'
    if stability: assert updates==500 and result['status']=='complete'
    protocol=load_protocol();trainer=load_trainer_config()
    config=json.loads((ROOT/'configs/stage3_baselines_v1.json').read_text())
    wdro=json.loads((ROOT/'configs/stage4_wdro_v1.json').read_text())
    contract=json.loads((ROOT/'manifests/stage4_training_contract_v1.json').read_text())
    for path,digest in contract['files'].items():
        assert file_sha256(CODEX/path)==digest, f'frozen file changed: {path}'
    latest=torch.load(result['latest_checkpoint'],map_location='cpu',weights_only=False)
    best=torch.load(result['best_checkpoint'],map_location='cpu',weights_only=False)
    hm_path=EXP/'logs/stage3/formal_hard_mining_d30_s101_u2000/result.json'
    hm=json.loads(hm_path.read_text());hm_train,_,_=effective_events(hm['event_log'])
    assert hm['status']=='complete' and hm['completed_updates']==2000
    # Only held-out coordinates and hashes are accessed, never model predictions.
    heldout=set();data_hashes={}
    for split in ('validation','test'):
        n=protocol['data']['sizes_per_degree_distribution'][split]
        seed=protocol['data']['seeds'][split]
        for name in protocol['data']['distributions']:
            path=EXP/'cache/datasets/protocol_v1'/split/f'{name}_d30_s{seed}_n{n}.npz'
            arrays,metadata=load_dataset(path,verify=True)
            heldout.update(pin_set_sha256(pins) for pins in arrays['pins'])
            data_hashes[str(path)]=metadata['npz_sha256']
    target=AUDIT/'cursors'/f'{run_id}.json'
    binding={'result_sha256':file_sha256(result_path),'events_sha256':file_sha256(Path(result['event_log'])),
             'auditor_sha256':file_sha256(Path(__file__)),'hm_result_sha256':file_sha256(hm_path),
             'hm_events_sha256':file_sha256(Path(hm['event_log'])),'heldout_files':data_hashes,
             'training_contract_sha256':file_sha256(ROOT/'manifests/stage4_training_contract_v1.json')}
    state={'binding':binding,'completed_updates':0,'stream_hash':EMPTY_STREAM_SHA256,
           'support_stream_hash':EMPTY_STREAM_SHA256,'selection_stream_hash':EMPTY_STREAM_SHA256,
           'heldout_intersections':0,'nets_checked':0,'candidate_nets_checked':0,
           'lambda':wdro['lambda_initial'],'cache_values_checked':0,'hm_paired_updates':0,
           'budget':{},'status':'running'}
    if target.exists():
        state=json.loads(target.read_text()); assert state['binding']==binding,'stale audit cursor'
    cache_path=Path(result['reference_cache']['logical_path'])
    db=sqlite3.connect(f'file:{cache_path}?mode=ro',uri=True)
    assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    for u in range(state['completed_updates']+1,updates+1):
        event=train[u];detail=event['method_detail']
        batch=materialize_training_batch(protocol=protocol,degree=30,training_seed=101,update=u,batch_size=256)
        assert event['batch_bundle_sha256']==batch.bundle_sha256
        assert all(n.decode().startswith(f'train_u{u:06d}:uniform:') for n in batch.net_ids)
        support=generate_support(batch,training_seed=101,update=u,search=config['search'])
        assert support.sha256==detail['support_sha256']
        ids,_=shortlist_support(support)
        np.testing.assert_array_equal(ids,detail['exact_shortlist_indices'])
        assert array_bundle_sha256({'indices':ids})==detail['shortlist_sha256']
        candidates=support.pins[np.arange(256)[:,None],ids]
        costs=transport_costs(batch.pins,candidates)
        np.testing.assert_array_equal(costs,detail['exact_transport_costs'])
        assert np.all(costs[:,0]==0) and np.all(costs[:,1:]>0)
        scores=np.asarray(detail['exact_relative_regrets'])
        price=state['lambda'];assert detail['lambda_before']==price and detail['rho']==rho
        penalized=np.where(costs==0,scores,-np.inf) if rho==0 else scores-price*costs
        columns=np.argmax(penalized,axis=1)
        selected=ids[np.arange(256),columns]
        np.testing.assert_array_equal(selected,detail['selected_indices'])
        selected_costs=costs[np.arange(256),columns]
        np.testing.assert_array_equal(selected_costs,detail['selected_costs'])
        mean_cost=float(selected_costs.mean())
        assert mean_cost==detail['mean_selected_cost'] and detail['budget_gap']==mean_cost-rho
        state['lambda']=float(max(0.,price+wdro['lambda_learning_rate']*(mean_cost-rho)))
        assert state['lambda']==detail['lambda_after']
        history=[train[v]['method_detail']['mean_selected_cost'] for v in range(max(1,u-99),u+1)]
        assert float(np.mean(history))==detail['mean_cost_last100']
        pins=support.pins[np.arange(256),selected]
        selected_ids=np.asarray([f'{n.decode()}:candidate{int(j)}:block{support.refresh_block}'.encode() for n,j in zip(batch.net_ids,selected)])
        selected_hash=array_bundle_sha256({'pins':pins,'net_ids':selected_ids})
        assert selected_hash==detail['selected_bundle_sha256']
        # Every candidate: geometry, duplicate identity, and held-out firewall.
        for row in support.pins:
            hashes=[pin_set_sha256(p) for p in row]
            assert len(set(hashes))==33
            state['heldout_intersections']+=sum(h in heldout for h in hashes)
            assert np.isfinite(row).all() and row.min()>=np.float32(.1) and row.max()<=np.float32(.9)
            assert not any(has_spacing_violation(p) for p in row)
        assert state['heldout_intersections']==0,'candidate overlaps held-out content'
        references=np.asarray(detail['exact_lstar']).ravel()
        for net,length in zip(candidates.reshape(-1,30,2),references):
            row=db.execute('SELECT length FROM lstar WHERE hash=?',(pin_set_sha256(net),)).fetchone()
            assert row is not None and row[0]==length and length>0
        state['cache_values_checked']+=len(references)
        paired=hm_train[u]['method_detail']
        assert hm_train[u]['batch_bundle_sha256']==batch.bundle_sha256
        for key in ('support_sha256','shortlist_sha256','refresh_block',*BUDGET_FIELDS):
            assert detail[key]==paired[key], f'HM actual prefix mismatch {u}/{key}'
        state['hm_paired_updates']+=1
        for key in (*BUDGET_FIELDS,'reference_cache_misses','reference_cache_hits','repaired_candidates'):
            state['budget'][key]=state['budget'].get(key,0)+detail[key]
        state['stream_hash']=extend_stream_hash(state['stream_hash'],batch.bundle_sha256)
        state['support_stream_hash']=extend_stream_hash(state['support_stream_hash'],support.sha256)
        state['selection_stream_hash']=extend_stream_hash(state['selection_stream_hash'],selected_hash)
        assert state['stream_hash']==event['training_stream_hash']
        state['nets_checked']+=256;state['candidate_nets_checked']+=256*33
        state['completed_updates']=u
        if u%10==0 or u==updates:
            state['updated_at_utc']=utc_now();atomic_write_json(target,state)
            if heartbeat: heartbeat(u)
            print(json.dumps({'run_id':run_id,'audit_update':u,'of':updates}),flush=True)
    calls=db.execute('SELECT COUNT(*), SUM(requested), SUM(misses_started), SUM(misses_completed) FROM calls').fetchone()
    cached_count=db.execute('SELECT COUNT(*) FROM lstar').fetchone()[0];db.close()
    feasible=[v for _,v in sorted(validation.items()) if v['result']['selection']['nominal_mean']<=result['nominal_limit']]
    assert feasible,'no nominal-feasible validation checkpoint'
    selected_validation=min(feasible,key=lambda v:v['result']['selection']['worst_group_mean'])
    expected_calls={'actor_forward_nets':updates*256*5+len(validation)*8000+1,
      'exact_rest_evaluator_nets':updates*256*5+len(validation)*8000+2,
      'surrogate_evaluator_nets':updates*256*33,'geosteiner_nets':2+state['budget']['reference_cache_misses']}
    src=[*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'stage4').glob('*.py')),ROOT/'scripts/train_wdro.py']
    checks={'complete_updates':latest['update']==updates and (not stability or updates==500),
       'frozen_protocol':latest['run_identity']['protocol_canonical_sha256']==protocol_sha256(protocol),
       'frozen_trainer':latest['run_identity']['trainer_config_sha256']==canonical_sha256(trainer),
       'source_hash':latest['run_identity']['implementation_sha256']==canonical_sha256({str(p.relative_to(ROOT)):file_sha256(p) for p in src}),
       'wdro_configuration':latest['run_identity']['wdro_config_sha256']==canonical_sha256(wdro),
       'baseline_configuration':latest['run_identity']['baseline_config_sha256']==canonical_sha256(config),
       'checkpoint_identity':latest['run_identity']==best['run_identity']==result['run_identity'],
       'call_ledger':latest['call_ledger']==result['call_ledger']==expected_calls,
       'method_state':latest['method_state']==result['method_state'],
       'training_stream':state['stream_hash']==result['training_stream_hash']==latest['training_stream_hash'],
       'support_stream':state['support_stream_hash']==result['method_state']['support_stream_hash'],
       'selection_stream':state['selection_stream_hash']==result['method_state']['selection_stream_hash'],
       'lambda_recurrence':state['lambda']==result['method_state']['lambda'],
       'budget_sum':state['budget']==result['method_state']['budget'],
       'actual_hm_prefix_paired':state['hm_paired_updates']==updates,
       'full_heldout_firewall':state['heldout_intersections']==0,
       'all_finite':all_finite(latest) and all_finite(best) and all_finite(list(train.values())) and all_finite(list(validation.values())),
       'checkpoint_hashes':file_sha256(Path(result['latest_checkpoint']))==result['latest_checkpoint_sha256'] and file_sha256(Path(result['best_checkpoint']))==result['best_checkpoint_sha256'],
       'validation_selection':best['update']==selected_validation['update'] and best['best']==latest['best']==result['best'],
       'validation_hash':result['best']['validation_manifest_sha256']==validation_manifest_hash(selected_validation['result']),
       'nominal_guard':result['selection_feasible'] and result['best']['nominal_mean']<=result['nominal_limit'],
       'test_ispd_metrics_locked':result['test_or_ispd_used'] is False and all(v['result']['split']=='validation' for v in validation.values()),
       'inverse_length':result['inverse_length_audit']['status']=='pass',
       'only_latest_best':sorted(p.name for p in Path(result['latest_checkpoint']).parent.glob('*.pt'))==['best.pt','latest.pt'],
       'no_checkpoint_partial':not list(Path(result['latest_checkpoint']).parent.glob('*.partial')),
       'run_lock_released':not (EXP/'logs/jobs'/run_id/'run.lock').exists(),
       'reference_cache_receipts':calls[1]>=state['budget']['reference_requests'] and calls[2]==calls[3]==cached_count,
       'stored_references_replayed':state['cache_values_checked']==updates*1024}
    validation_protocol=json.loads(json.dumps(protocol));validation_protocol['training']['policy_updates']=500 if stability else 0
    vc=validation_checks(list(validation.values()),best,validation_protocol,trainer,30)
    vc.pop('selected_best_is_earliest_global_validation_minimum');checks.update(vc)
    checks.update({'latest_'+k:v for k,v in optimizer_checks(latest,trainer).items()})
    if best['update']: checks.update({'best_'+k:v for k,v in optimizer_checks(best,trainer).items()})
    costs=np.array([train[u]['method_detail']['mean_selected_cost'] for u in sorted(train)])
    prices=np.array([train[u]['method_detail']['lambda_after'] for u in sorted(train)])
    diagnostic={'rho':rho,'lambda_final':float(prices[-1]),'lambda_min':float(prices.min()),'lambda_max':float(prices.max()),'mean_cost':float(costs.mean())}
    if stability:
        window=wdro['stability_gate']['window_updates']
        windows=[float(costs[-window*i:len(costs)-window*(i-1) if i>1 else None].mean()) for i in [2,1]]
        diagnostic['tail_100_update_window_means']=windows
        checks['lambda_active_not_divergent']=bool(np.all(prices[-200:]>0) and prices.max()<wdro['stability_gate']['maximum_lambda'])
        checks['long_run_budget_stable']=all(abs(v-rho)<=rho*wdro['stability_gate']['relative_mean_cost_tolerance'] for v in windows)
    real_lp=[]
    for u in sorted(set([1,updates])):
        d=train[u]['method_detail']; loss=np.asarray(d['exact_relative_regrets'])[:8];cost=np.asarray(d['exact_transport_costs'])[:8]
        for radius in [0,rho,float(cost.max())]:
            lp=primal_dual_lp(loss,cost,radius);lp.update(update=u,regrets=loss.tolist(),costs=cost.tolist());real_lp.append(lp)
    checks['real_candidate_lp']=all(p['status']=='pass' for p in real_lp)
    state['status']='pass' if all(checks.values()) else 'fail';atomic_write_json(target,state)
    output={'status':state['status'],'run_id':run_id,'checks':checks,'at_utc':utc_now(),'binding':binding,
       'stream_cursor':str(target),'training_nets_checked':state['nets_checked'],'candidate_nets_checked':state['candidate_nets_checked'],
       'cache_values_checked':state['cache_values_checked'],'raw_cache_receipts':dict(zip(['receipts','requested','misses_started','misses_completed'],calls)),
       'actual_hard_mining_pair':{'run_id':hm['run_id'],'paired_updates':[1,updates],'comparison':'actual matching prefix, HM full run remains 2000 updates','budget':{k:state['budget'][k] for k in BUDGET_FIELDS}},
       'diagnostics':diagnostic,'real_lp':real_lp,'test_ispd_model_metrics_read':False}
    atomic_write_json(AUDIT/'runs'/f'{run_id}.json',output)
    assert output['status']=='pass',[k for k,v in checks.items() if not v]
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stability',action='store_true');args=p.parse_args()
    import shlex
    command=shlex.join([sys.executable,str(Path(__file__)),*sys.argv[1:]])
    with JobLease(AUDIT/'jobs'/args.run_id,run_id='audit_'+args.run_id,exact_command=command,resume_command=command,
        monitor_command=f'cat {AUDIT}/cursors/{args.run_id}.json',expected_updates=500 if args.stability else 12,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        result=audit_run(args.run_id,args.stability,lambda u:lease.update(update=u,status='running'))
        lease.finish(update=result['actual_hard_mining_pair']['paired_updates'][1],status='complete')
    print(json.dumps({'status':'pass','checks':len(result['checks']),'run_id':args.run_id}),flush=True)

if __name__=='__main__': main()
