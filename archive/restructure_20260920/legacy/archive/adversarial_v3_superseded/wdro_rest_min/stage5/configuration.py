"""Predeclared minimal sequential ablations and validation-only selection."""
import json
from pathlib import Path
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import utc_now
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments'
PLAN=ROOT/'configs/stage5_plan_v1.json'

def base_config():
    return {'schema_version':1,'stage':5,'status':'frozen','research_protocol_override':False,
      'degree':30,'training_seed':101,'updates':2000,'batch_size':256,
      'ground_cost':'hungarian_l1_mean','lambda_initial':0.0,'lambda_mode':'learned','lambda_learning_rate':1.0,
      'calibration_artifact':'experiments/cache/stage5/calibration_d30_s101.json',
      'stability_gate':{'maximum_lambda':100.0},
      'search_overrides':{'searcher':'random_sobol','candidate_count':32,'exact_rescore_top_k':4,
          'cem':{'rounds':4,'population':16,'elite':4,'initial_std':0.02,'minimum_std':0.001,'maximum_motion':0.08}}}

def write_config(name,rho_mode,**overrides):
    config=base_config();config.update(overrides);config['rho_mode']=rho_mode;config['config_id']=name
    path=ROOT/'configs/stage5'/f'{name}.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert json.loads(path.read_text())==config,'config drift'
    else:atomic_write_json(path,config,mode=0o444)
    return {'run_id':'stage5_'+name+'_d30_s101_u2000_v1','config':str(path),'config_sha256':file_sha256(path),'rho_mode':rho_mode}

def initial_tasks():return [write_config('rho_'+mode,mode) for mode in ['zero','q50','q75']]

def select_rho(tasks):
    records=[]
    for task in tasks:
        path=EXP/'logs/stage5'/task['run_id']/'result.json';r=json.loads(path.read_text())
        assert r['status']=='complete' and r['completed_updates']==2000 and not r['test_or_ispd_used']
        records.append({'task':task,'result':str(path),'result_sha256':file_sha256(path),
            'best':r['best'],'nominal_limit':r['nominal_limit']})
    feasible=[r for r in records if r['best']['feasible'] and r['best']['nominal_mean']<=r['nominal_limit']]
    assert feasible,'no rho configuration meets nominal guard'
    winner=min(feasible,key=lambda r:r['best']['selection_objective'])
    return {'status':'pass','source':'synthetic_validation_only','records':records,'selected':winner,'test_or_ispd_used':False}

def remaining_tasks(selection):
    import numpy as np
    selected=selection['selected'];mode=selected['task']['rho_mode']
    result=json.loads(Path(selected['result']).read_text())
    effective={}
    for line in Path(result['event_log']).read_text().splitlines():
        event=json.loads(line)
        if event['event']=='resume':effective={u:v for u,v in effective.items() if u<=event['from_update']}
        if event['event']=='train':effective[event['update']]=event
    fixed=float(np.mean([effective[u]['method_detail']['lambda_after'] for u in range(1901,2001)]))
    tasks=[write_config('lambda_fixed',mode,lambda_mode='fixed',lambda_initial=fixed,
                       fixed_lambda_source={'result_sha256':selected['result_sha256'],'updates':[1901,2000],'aggregation':'mean_lambda_after'})]
    for k in [1,8]:
        search=dict(base_config()['search_overrides'],candidate_count=k,exact_rescore_top_k=min(4,k+1))
        tasks.append(write_config('k'+str(k),mode,search_overrides=search))
    tasks.append(write_config('chamfer',mode,ground_cost='symmetric_chamfer_l1'))
    for searcher in ['random_64','cem_64']:
        search=dict(base_config()['search_overrides'],searcher=searcher,candidate_count=63,exact_rescore_top_k=64)
        tasks.append(write_config(searcher,mode,search_overrides=search))
    return tasks
