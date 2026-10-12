#!/usr/bin/env python3
"""Independent readback of required cells, hash/path identities and frozen selection."""
import csv,json,sys,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.rest import file_sha256
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from stage5.configuration import select_rho
from audit_stage3 import effective_events

def main():
    source=ROOT/'manifests/stage5_completion_v1.json';data=json.loads(source.read_text());assert data['status']=='pass'
    checks={};artifacts=[]
    contract=json.loads((ROOT/'manifests/stage5_training_contract_v1.json').read_text())
    for path,digest in contract['files'].items():assert file_sha256(CODEX/path)==digest,path
    checks['all_frozen_input_hashes']=True
    def walk(value):
        if isinstance(value,dict):
            if {'logical_path','resolved_path','sha256'}<=value.keys():
                p=Path(value['logical_path']);assert p.is_relative_to(CODEX),str(p)
                assert str(p.resolve())==value['resolved_path'] and file_sha256(p)==value['sha256'],str(p)
                artifacts.append(value)
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    walk(data);checks['artifact_paths_and_hashes']=True
    tasks=json.loads((EXP/'logs/stage5_pipeline/all_tasks.json').read_text())['tasks']
    assert len(tasks)==len(data['models'])==9 and len({t['run_id'] for t in tasks})==9
    checks['nine_unique_runs']=True
    selected=select_rho(tasks[:3]);saved=json.loads((EXP/'logs/stage5_pipeline/rho_selection.json').read_text());assert selected==saved
    checks['independent_validation_selection']=True
    fp=ROOT/'configs/stage5_selected_wdro_v1.json';freeze=json.loads(fp.read_text())
    expected={'searcher':'random_sobol','candidate_count':32,'ground_cost':'hungarian_l1_mean','lambda_mode':'learned','lambda_initial':0.,'lambda_learning_rate':1.}
    for k,v in expected.items():assert freeze[k]==v,k
    assert freeze['selected_rho_mode']==selected['selected']['task']['rho_mode']
    assert stat.S_IMODE(fp.stat().st_mode)==0o444
    checks['frozen_main_constraints']=True
    rows=list(csv.DictReader((EXP/'tables/stage5_ablation.csv').open()));assert len(rows)==9
    long=list(csv.DictReader((EXP/'tables/stage5_validation_long.csv').open()));assert len(long)==9*21*4
    trajectory=list(csv.DictReader((EXP/'tables/stage5_budget_trajectory.csv').open()));assert len(trajectory)==18000
    assert len({(r['run_id'],r['update'],r['distribution']) for r in long})==len(long)
    assert len({(r['run_id'],r['update']) for r in trajectory})==18000
    checks['table_rows_and_unique_keys']=True
    for t in tasks:
        rid=t['run_id'];r=json.loads((EXP/'logs/stage5'/rid/'result.json').read_text())
        assert r['run_identity']['ablation_config']==json.loads(Path(t['config']).read_text())
        assert file_sha256(Path(t['config']))==t['config_sha256']
        assert r['status']=='complete' and r['completed_updates']==2000 and r['batch_size']==256
        assert r['degree']==30 and r['training_seed']==101 and not r['test_or_ispd_used']
        row=next(v for v in rows if v['run_id']==rid)
        assert float(row['validation_worst_group_mean'])==r['best']['selection_objective'] and float(row['nominal_mean'])==r['best']['nominal_mean']
        train,validation,_=effective_events(r['event_log'])
        assert sorted(validation)==list(range(0,2001,100))
        for v in long:
            if v['run_id']!=rid:continue
            u=int(v['update']);group=validation[u]['result']['groups'][v['distribution']]['summary']
            assert v['split']=='validation' and float(v['mean_regret'])==group['mean'] and float(v['p95_regret'])==group['p95']
            assert (v['selected']=='True')==(u==r['best']['update'])
        a=json.loads((EXP/'logs/stage5_audit/runs'/f'{rid}.json').read_text())
        assert a['status']=='pass' and all(a['checks'].values()) and a['completed_updates']==2000
        assert not (EXP/'logs/jobs'/rid/'run.lock').exists()
        assert not list(Path(r['latest_checkpoint']).parent.glob('*.partial'))
        checks[rid]=True
    for name in ['random_64','cem_64']:
        r=next(v for v in rows if v['searcher']==name)
        assert int(r['exact_rescore_nets'])==int(r['reference_requests'])==32768000 and int(r['surrogate_nets'])==0
    checks['equal_64_true_call_budget']=True
    assert not data['test_or_ispd_used'] and not data['stage6_started'] and not freeze['stage6_started']
    checks['stage_boundary']=True
    result={'status':'pass','at_utc':utc_now(),'checks':checks,'artifact_count':len(artifacts),
      'completion_sha256':file_sha256(source),'completion_logical_path':str(source),'completion_resolved_path':str(source.resolve()),'test_or_ispd_used':False}
    atomic_write_json(ROOT/'manifests/stage5_delivery_verification_v1.json',result)
    print(json.dumps({'status':'pass','checks':len(checks),'artifacts':len(artifacts)}))
if __name__=='__main__':main()
