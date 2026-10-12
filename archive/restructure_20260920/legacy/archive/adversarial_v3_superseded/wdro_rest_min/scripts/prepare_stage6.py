"""Apply stage-5's frozen quantile transfer and bind all pre-test inputs."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; CODEX=ROOT.parent; EXP=CODEX/'experiments'
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.protocol import load_protocol, protocol_sha256
from src.training_data import materialize_training_batch
from src.candidates import generate_support
from stage5.search import transport_costs
from src.datasets import array_bundle_sha256
from src.job_control import utc_now

def record(p):
    p=Path(p);return dict(logical_path=str(p),resolved_path=str(p.resolve()),sha256=file_sha256(p))

def frozen(p,v):
    if p.exists(): assert json.loads(p.read_text())==v, str(p)
    else: atomic_write_json(p,v,mode=0o444)

def main():
    selected=ROOT/'configs/stage5_selected_wdro_v1.json'; chosen=json.loads(selected.read_text())
    completion=ROOT/'manifests/stage5_completion_v1.json'
    delivery=json.loads((ROOT/'manifests/stage5_delivery_verification_v1.json').read_text())
    assert delivery['status']=='pass' and delivery['completion_sha256']==file_sha256(completion)
    assert chosen['status']=='frozen' and chosen['selected_rho_mode'] in ('zero','q50','q75') and not chosen['test_or_ispd_used']
    mode=chosen['selected_rho_mode']
    files={}
    for name in ['stage4_training_contract_v1','stage5_training_contract_v1']:
        for p,h in json.loads((ROOT/'manifests'/f'{name}.json').read_text())['files'].items():
            assert file_sha256(CODEX/p)==h,p;files[p]=h
    for name in ['stage2_completion_20260905','stage3_completion_v1','stage3_delivery_verification_v1','stage5_completion_v1','stage5_delivery_verification_v1']:
        p=ROOT/'manifests'/f'{name}.json'; assert json.loads(p.read_text())['status'] in ['pass','complete'];files[str(p.relative_to(CODEX))]=file_sha256(p)
    protocol=load_protocol(); baseline_path=ROOT/'configs/stage3_baselines_v1.json';baseline=json.loads(baseline_path.read_text())
    base=json.loads(Path(chosen['selected_config']['logical_path']).read_text())
    tasks=[];calibrations=[]
    for degree in protocol['scope']['degrees']:
        batch=materialize_training_batch(protocol=protocol,degree=degree,training_seed=101,update=1,batch_size=256)
        support=generate_support(batch,training_seed=101,update=1,search=baseline['search'])
        costs=transport_costs(batch.pins,support.pins,'hungarian_l1_mean')
        assert np.all(costs[:,0]==0) and np.all(costs[:,1:]>0)
        q=0. if mode=='zero' else float(np.quantile(costs[:,1:],{'q50':.5,'q75':.75}[mode]))
        if degree==30: assert q==chosen['selected_degree30_rho']
        p=EXP/'cache/stage6'/f'calibration_d{degree}_s101.json'
        value=dict(status='pass',source='training_centers_only',degree=degree,seed=101,update=1,batch_size=256,
            protocol_canonical_sha256=protocol_sha256(protocol),baseline_config_file_sha256=file_sha256(baseline_path),
            training_batch_sha256=batch.bundle_sha256,support_sha256=support.sha256,model_calls=0,test_or_ispd_used=False,
            hungarian_l1_mean={mode:q,'costs_sha256':array_bundle_sha256({'costs':costs}),'transport_costs':costs.tolist()})
        frozen(p,value);calibrations.append(record(p))
        cfg=dict(base,stage=6,degree=degree,training_seed='per_run_protocol_seed',config_id=f'stage6_d{degree}_{mode}',calibration_artifact=str(p.relative_to(CODEX)))
        config=ROOT/'configs/stage6'/f'd{degree}_{mode}.json'; frozen(config,cfg)
        for seed in protocol['data']['seeds']['training']:
            reused=degree==30 and seed==101
            rid=f'stage5_rho_{mode}_d30_s101_u2000_v1' if reused else f'formal_wdro_d{degree}_s{seed}_u2000_v1'
            tasks.append(dict(run_id=rid,degree=degree,seed=seed,rho_mode=mode,rho=q,config=str(config),config_sha256=file_sha256(config),
                reused_stage5=reused,result=str(EXP/f'logs/{"stage5" if reused else "stage6"}'/rid/'result.json')))
    # Reuse only identities audited with the corrected validation freeze.
    reused=[]
    for method in protocol['scope']['methods']:
        for degree in protocol['scope']['degrees']:
            for seed in protocol['data']['seeds']['training']:
                if method=='wdro_rest':
                    if (degree,seed)!=(30,101):continue
                    p=Path(next(t['result'] for t in tasks if t['reused_stage5']))
                else:p=EXP/f'logs/{"stage2" if method=="erm" else "stage3"}'/f'formal_{method}_d{degree}_s{seed}_u2000'/'result.json'
                r=json.loads(p.read_text()); assert r['status']=='complete' and r['completed_updates']==2000 and r['batch_size']==256 and not r['test_or_ispd_used']
                assert r['method']==method and r['degree']==degree and r['training_seed']==seed
                for key in ['latest','best']: assert file_sha256(Path(r[f'{key}_checkpoint']))==r[f'{key}_checkpoint_sha256']
                reused.append(dict(method=method,degree=degree,seed=seed,result=record(p),best=record(r['best_checkpoint']),latest=record(r['latest_checkpoint'])))
    plan=dict(stage=6,status='frozen',research_protocol_override=False,selected_configuration=record(selected),tasks=tasks,
        calibration=calibrations,reused_models=reused,expected_models=75,new_training_runs=14,
        evaluation=dict(split='test',distributions=list(protocol['data']['distributions']),nets_per_cell=10000,
            batch_size=256,symmetry_transforms=1,mode='deterministic_greedy',hardware='NVIDIA RTX 6000 Ada Generation',
            expected_cells=300,expected_net_evaluations=3000000,checkpoint_choice='validation_selected_best_before_any_test'),
        statistics=dict(confidence_level=.95,bootstrap_samples=10000,seed=606,paired=True,
            unit='paired net within each degree/distribution, seeds averaged per net; per-seed CIs also retained',
            primary='mean, p95, OOD worst-group mean; stratified percentile bootstrap',
            limitation='conditional on three trained seeds; no claim of broad training-seed population significance'),
        stage7_started=False,test_used_for_selection=False)
    frozen(ROOT/'configs/stage6_plan_v1.json',plan)
    paths=[selected,ROOT/'configs/stage6_plan_v1.json',ROOT/'stage6/trainer.py',ROOT/'scripts/train_stage6.py',ROOT/'scripts/prepare_stage6.py',
           *list((ROOT/'configs/stage6').glob('*.json')),*[Path(c['logical_path']) for c in calibrations]]
    for p in paths:files[str(p.relative_to(CODEX))]=file_sha256(p)
    frozen(ROOT/'manifests/stage6_training_contract_v1.json',dict(status='frozen',stage=6,files=files,research_protocol_override=False))
    atomic_write_json(EXP/'logs/stage6_pipeline/input_gate.json',dict(status='pass',at_utc=utc_now(),reused_models=len(reused),new_training_runs=14,
        calibration={t['degree']:t['rho'] for t in tasks},contract=record(ROOT/'manifests/stage6_training_contract_v1.json')))
    print(json.dumps({'status':'pass','reused_models':len(reused),'new_training_runs':14,'rho':{t['degree']:t['rho'] for t in tasks}}))
if __name__=='__main__':main()
