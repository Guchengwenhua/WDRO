#!/usr/bin/env python3
"""Export all predefined ablations and freeze the eligible main WDRO configuration."""
import csv,io,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from audit_stage3 import effective_events
from export_stage4 import artifact,atomic_text,csv_file
from stage5.configuration import select_rho

def main():
    job=EXP/'logs/stage5_pipeline';tasks=json.loads((job/'all_tasks.json').read_text())['tasks'];assert len(tasks)==9
    selection=json.loads((job/'rho_selection.json').read_text());assert select_rho(tasks[:3])==selection
    summary=[];trajectory=[];validation_long=[];models=[];traces={}
    for t in tasks:
        rid=t['run_id'];rp=EXP/'logs/stage5'/rid/'result.json';r=json.loads(rp.read_text())
        ap=EXP/'logs/stage5_audit/runs'/f'{rid}.json';a=json.loads(ap.read_text())
        vp=EXP/'logs/stage5_best_validation'/f'{rid}.json';v=json.loads(vp.read_text())
        assert r['status']=='complete' and r['completed_updates']==2000 and a['status']==v['status']=='pass'
        assert a['binding']['result_sha256']==v['result_sha256']==file_sha256(rp)
        assert v['checkpoint_sha256']==r['best_checkpoint_sha256'] and not r['test_or_ispd_used']
        train,validation,_=effective_events(r['event_log']);assert len(train)==2000 and sorted(validation)==list(range(0,2001,100))
        cfg=r['run_identity']['ablation_config'];search=cfg['search_overrides'];best=validation[r['best']['update']]['result']
        row={'run_id':rid,'ablation':cfg['config_id'],'degree':30,'seed':101,'updates':2000,'batch_size':256,
             'rho_mode':t['rho_mode'],'rho':r['run_identity']['rho'],'lambda_mode':cfg['lambda_mode'],'lambda_initial':cfg['lambda_initial'],
             'searcher':search['searcher'],'perturbation_count':search['candidate_count'],'ground_cost':cfg['ground_cost'],
             'best_update':r['best']['update'],'nominal_mean':r['best']['nominal_mean'],'nominal_limit':r['nominal_limit'],
             'feasible':r['selection_feasible'],'validation_worst_group_mean':r['best']['selection_objective'],
             'selected_main':rid==selection['selected']['task']['run_id'],'split':'validation',
             'lambda_final':r['method_state']['lambda'],'tail_mean_cost':float(np.mean(r['method_state']['cost_history'])),
             **r['method_state']['budget'],**r['call_ledger'],
             'peak_reserved_mib':max(json.loads(p.read_text())['peak_memory_reserved_bytes']/2**20 for p in rp.parent.glob('segment_*.json')),
             'successful_segments_wall_seconds':sum(json.loads(p.read_text())['segment_wall_seconds'] for p in rp.parent.glob('segment_*.json')),
             'best_checkpoint_sha256':r['best_checkpoint_sha256'],'result_sha256':file_sha256(rp),'audit_sha256':file_sha256(ap)}
        summary.append(row);run_trace=[]
        for u,e in sorted(train.items()):
            d=e['method_detail'];tr={'run_id':rid,'ablation':cfg['config_id'],'update':u,
                **{k:d[k] for k in ['lambda_before','lambda_after','rho','mean_selected_cost','budget_gap','mean_cost_last100','legal_candidate_fraction','surrogate_nets','exact_rescore_nets','reference_requests']}}
            trajectory.append(tr);run_trace.append(tr)
        traces[rid]=run_trace
        for u,e in sorted(validation.items()):
            val=e['result']
            for group,metrics in val['groups'].items():
                validation_long.append({'run_id':rid,'ablation':cfg['config_id'],'degree':30,'seed':101,'update':u,'distribution':group,
                    'split':'validation','selected':u==r['best']['update'],'mean_regret':metrics['summary']['mean'],'p95_regret':metrics['summary']['p95'],
                    'nominal_feasible':val['selection']['nominal_mean']<=r['nominal_limit'],
                    'validation_manifest_sha256':r['best']['validation_manifest_sha256'] if u==r['best']['update'] else ''})
        models.append({'run_id':rid,'config':artifact(t['config']),'result':artifact(rp),'audit':artifact(ap),'best_validation_recompute':artifact(vp),
                       'latest_checkpoint':artifact(r['latest_checkpoint']),'best_checkpoint':artifact(r['best_checkpoint']),'event_log':artifact(r['event_log']),
                       'reference_cache':artifact(r['reference_cache']['logical_path'])})
    tables=EXP/'tables';figures=EXP/'figures'
    paths=[tables/'stage5_ablation.csv',tables/'stage5_budget_trajectory.csv',tables/'stage5_validation_long.csv']
    for p,rows in zip(paths,[summary,trajectory,validation_long]):csv_file(p,rows)
    # The searcher comparison has 64 *actual* true scoring requests per center.
    budget_rows=[r for r in summary if r['searcher'] in ('random_64','cem_64')]
    assert len(budget_rows)==2
    for r in budget_rows:assert r['exact_rescore_nets']==r['reference_requests']==2000*256*64 and r['surrogate_nets']==0
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=selection['selected']['task']['run_id'];trace=traces[selected]
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,constrained_layout=True)
    x=np.array([t['update'] for t in trace]);rho=trace[0]['rho'];cost=np.array([t['mean_selected_cost'] for t in trace]);ma=np.array([t['mean_cost_last100'] for t in trace])
    axes[0].plot(x,[t['lambda_after'] for t in trace]);axes[0].set_ylabel('Lambda');axes[0].set_title('Validation-selected main WDRO | degree 30, seed 101')
    axes[1].plot(x,cost,alpha=.3,lw=.7,label='Per-update mean cost');axes[1].plot(x,ma,label='Trailing 100-update mean');axes[1].axhline(rho,color='darkred',ls='--',label=f'rho={rho:.6f}');axes[1].set_ylabel('Hungarian-L1 cost');axes[1].legend()
    axes[2].plot(x,cost-rho,alpha=.3,lw=.7);axes[2].plot(x,ma-rho);axes[2].axhline(0,color='black',lw=.7);axes[2].set_ylabel('Mean cost - rho');axes[2].set_xlabel('Policy update')
    for ax in axes:ax.grid(alpha=.2)
    figpaths=[]
    for ext in ['png','pdf','svg']:
        p=figures/('stage5_budget_diagnostic.'+ext);tmp=p.with_suffix(p.suffix+'.partial');fig.savefig(tmp,format=ext,dpi=180);assert tmp.stat().st_size>1000;tmp.replace(p);figpaths.append(p)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for ax,r in zip(axes.flat,summary):
        tr=traces[r['run_id']];ax.plot([t['update'] for t in tr],[t['mean_cost_last100'] for t in tr],color='teal');ax.axhline(r['rho'],color='darkred',ls='--');ax.set_title(r['ablation']);ax.set_ylabel('Mean cost / rho')
        twin=ax.twinx();twin.plot([t['update'] for t in tr],[t['lambda_after'] for t in tr],color='slateblue',alpha=.6);twin.set_ylabel('Lambda')
    p=figures/'stage5_all_ablation_budgets.png';tmp=p.with_suffix('.png.partial');fig.savefig(tmp,format='png',dpi=160);tmp.replace(p);figpaths.append(p);plt.close(fig)
    freeze={'schema_version':1,'status':'frozen','stage':5,'frozen_at_utc':utc_now(),'source':'synthetic_validation_only',
        'selected_rho_mode':selection['selected']['task']['rho_mode'],'selected_degree30_rho':next(r['rho'] for r in summary if r['selected_main']),
        'searcher':'random_sobol','candidate_count':32,'include_nominal_candidate':True,'exact_rescore_top_k':4,
        'ground_cost':'hungarian_l1_mean','lambda_mode':'learned','lambda_initial':0.0,'lambda_learning_rate':1.0,
        'radius_transfer':'calibrate selected quantile separately for each degree using training update1/seed101/256 centers and unchanged K32 support; no further grid search',
        'calibration':artifact(EXP/'cache/stage5/calibration_d30_s101.json'),'selection':artifact(job/'rho_selection.json'),
        'selected_config':artifact(selection['selected']['task']['config']),'protocol':artifact(ROOT/'configs/protocol_v1.json'),
        'data_contract':artifact(ROOT/'configs/data_contract_v1.json'),'trainer':artifact(ROOT/'configs/stage2_trainer_v1.json'),
        'test_or_ispd_used':False,'stage6_started':False,'no_post_test_retuning':True}
    fp=ROOT/'configs/stage5_selected_wdro_v1.json'
    if fp.exists():
        old=json.loads(fp.read_text());freeze['frozen_at_utc']=old['frozen_at_utc'];assert old==freeze
    else:atomic_write_json(fp,freeze,mode=0o444)
    completion={'status':'pass','stage':5,'at_utc':utc_now(),'run_count':9,'total_updates':18000,'test_or_ispd_used':False,'stage6_started':False,
        'models':models,'tables':[artifact(p) for p in paths],'figures':[artifact(p) for p in figpaths],
        'selected_configuration':artifact(fp),'selection':artifact(job/'rho_selection.json'),'training_contract':artifact(ROOT/'manifests/stage5_training_contract_v1.json'),
        'smoke':artifact(EXP/'logs/stage5_smoke/audit.json'),'smoke_audit':artifact(EXP/'logs/stage5_audit/smoke_gate.json'),
        'limitation':'single degree and seed validation ablations; no test significance or WDRO superiority claim; retry efficiency is separate from effective trajectory'}
    atomic_write_json(ROOT/'manifests/stage5_completion_v1.json',completion)
    print(json.dumps({'status':'pass','runs':9,'selected_rho':freeze['selected_rho_mode'],'table':str(paths[0]),'figure':str(figpaths[0])}))
if __name__=='__main__':main()
