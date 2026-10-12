"""Complete synthetic tables and paired CIs; resume per method/degree/group."""
import csv,json,sys,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from stage6.statistics import bootstrap_cell,ci,metrics
from stage6.evaluation import atomic_npz,artifact
from src.io_utils import atomic_write_json
from src.job_control import JobLease,utc_now
from src.rest import file_sha256
METHODS=['erm','mixed_erm','hard_mining','group_dro','wdro_rest'];DEGREES=[10,20,30,40,50];SEEDS=[101,102,103]
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster'];JOB=EXP/'logs/stage6_statistics'

def write_csv(path,rows):
    temp=path.with_suffix('.csv.partial');path.parent.mkdir(parents=True,exist_ok=True)
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with temp.open() as f:assert len(list(csv.DictReader(f)))==len(rows)
    temp.replace(path);return artifact(path)

def main():
    plan=ROOT/'configs/stage6_plan_v1.json';index=ROOT/'manifests/stage6_models_frozen_v1.json'
    assert json.loads(index.read_text())['status']=='frozen'
    command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {Path(__file__)}'
    with JobLease(JOB,run_id='stage6_statistics',exact_command=command,resume_command=command,
        monitor_command=f'cat {JOB}/heartbeat.json',expected_updates=100,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        boot={};point={};raw={};ids={};long=[];degree_rows=[];main_rows=[];paired=[];prediction_records=[];bootstrap_records=[];counter=0
        for degree in DEGREES:
            for gi,group in enumerate(GROUPS):
                for method in METHODS:
                    vectors=[];bindings=[]
                    for seed in SEEDS:
                        directory=EXP/'logs/stage6_test'/f'{method}_d{degree}_s{seed}'/group
                        r=json.loads((directory/'result.json').read_text());assert r['status']=='complete' and r['completed_nets']==10000
                        pred=Path(r['predictions']['logical_path']);assert file_sha256(pred)==r['predictions']['sha256']
                        with np.load(pred,allow_pickle=False) as a:
                            vectors.append(a['regret']);net_ids=a['net_ids']
                        key=(degree,group)
                        if key in ids:np.testing.assert_array_equal(ids[key],net_ids)
                        else:ids[key]=net_ids
                        bindings.append(artifact(directory/'result.json'));prediction_records.append(bindings[-1])
                    values=np.stack(vectors);raw[(method,degree,group)]=values
                    bp=EXP/'cache/stage6_bootstrap'/f'{method}_d{degree}_{group}.npz';meta=bp.with_suffix('.json')
                    binding=dict(inputs=bindings,plan_sha256=file_sha256(plan),code_sha256=file_sha256(ROOT/'stage6/statistics.py'))
                    if meta.exists():
                        info=json.loads(meta.read_text());assert info['binding']==binding and info['artifact']['sha256']==file_sha256(bp)
                        with np.load(bp) as f:b={k:f[k] for k in f.files}
                    else:
                        b=bootstrap_cell(values,seed=606+degree*10+gi,
                            heartbeat=lambda n:lease.update(update=counter,status='bootstrap',detail={'method':method,'degree':degree,'group':group,'replicates':n}))
                        atomic_npz(bp,b);atomic_write_json(meta,dict(status='pass',binding=binding,artifact=artifact(bp)))
                    bootstrap_records.append(artifact(meta));boot[(method,degree,group)]={k:v.mean(axis=1) for k,v in b.items()};point[(method,degree,group)]=metrics(values)
                    for si,seed in enumerate(SEEDS):
                        row=dict(method=method,degree=degree,seed=seed,distribution=group,count=10000,
                            mean=float(values[si].mean()),p95=float(np.quantile(values[si],.95)))
                        for stat in ['mean','p95']:
                            for k,v in ci(b[stat][:,si]).items():row[stat+'_'+k]=v
                        row['result_sha256']=bindings[si]['sha256'];long.append(row)
                    row=dict(method=method,degree=degree,distribution=group,seeds=3,nets_per_seed=10000,**point[(method,degree,group)])
                    for stat in ['mean','p95']:
                        for k,v in ci(boot[(method,degree,group)][stat]).items():row[stat+'_'+k]=v
                    degree_rows.append(row);counter+=1;lease.update(update=counter,status='bootstrap')
                # Paired net differences and same-draw bootstrap preserve model pairing.
                for baseline in METHODS[:-1]:
                    diff=raw[('wdro_rest',degree,group)]-raw[(baseline,degree,group)]
                    path=EXP/'tables/stage6_paired_nets'/f'wdro_minus_{baseline}_d{degree}_{group}.npz'
                    atomic_npz(path,dict(net_ids=ids[(degree,group)],seeds=np.array(SEEDS),relative_regret_difference=diff))
                    for si,seed in enumerate(SEEDS):
                        # Per-seed paired mean CI uses the existing aligned bootstrap caches.
                        with np.load(EXP/'cache/stage6_bootstrap'/f'wdro_rest_d{degree}_{group}.npz') as x,np.load(EXP/'cache/stage6_bootstrap'/f'{baseline}_d{degree}_{group}.npz') as y:
                            for stat in ['mean','p95']:
                                estimate=float(diff[si].mean()) if stat=='mean' else float(np.quantile(raw[('wdro_rest',degree,group)][si],.95)-np.quantile(raw[(baseline,degree,group)][si],.95))
                                paired.append(dict(candidate='wdro_rest',baseline=baseline,degree=degree,seed=seed,distribution=group,metric=stat,difference=estimate,**ci(x[stat][:,si]-y[stat][:,si])))
                    for stat in ['mean','p95']:
                        paired.append(dict(candidate='wdro_rest',baseline=baseline,degree=degree,seed='mean3',distribution=group,metric=stat,
                            difference=point[('wdro_rest',degree,group)][stat]-point[(baseline,degree,group)][stat],
                            **ci(boot[('wdro_rest',degree,group)][stat]-boot[(baseline,degree,group)][stat])))
        aggregate={};aggregate_point={}
        for method in METHODS:
            for group in GROUPS:
                estimates={s:float(np.mean([point[(method,d,group)][s] for d in DEGREES])) for s in ['mean','p95']}
                draws={s:np.mean([boot[(method,d,group)][s] for d in DEGREES],axis=0) for s in ['mean','p95']}
                aggregate[(method,group)]=draws;aggregate_point[(method,group)]=estimates
                row=dict(method=method,distribution=group,degree_count=5,seeds=3,nets_per_degree_seed=10000,**estimates)
                for stat in ['mean','p95']:
                    for k,v in ci(draws[stat]).items():row[stat+'_'+k]=v
                main_rows.append(row)
            draws=np.max([aggregate[(method,g)]['mean'] for g in GROUPS[1:]],axis=0)
            value=max(aggregate_point[(method,g)]['mean'] for g in GROUPS[1:])
            aggregate[(method,'ood_worst_group')]={'mean':draws};aggregate_point[(method,'ood_worst_group')]={'mean':value}
        for baseline in METHODS[:-1]:
            for group in GROUPS+['ood_worst_group']:
                for stat in aggregate[('wdro_rest',group)]:
                    paired.append(dict(candidate='wdro_rest',baseline=baseline,degree='macro5',seed='mean3',distribution=group,metric=stat,
                        difference=aggregate_point[('wdro_rest',group)][stat]-aggregate_point[(baseline,group)][stat],
                        **ci(aggregate[('wdro_rest',group)][stat]-aggregate[(baseline,group)][stat])))
        worst=[]
        for method in METHODS:
            for degree in DEGREES:
                value=max(point[(method,degree,g)]['mean'] for g in GROUPS[1:]);draws=np.max([boot[(method,degree,g)]['mean'] for g in GROUPS[1:]],axis=0)
                worst.append(dict(method=method,degree=degree,metric='ood_worst_group_mean',value=value,**ci(draws)))
            worst.append(dict(method=method,degree='macro5',metric='ood_worst_group_mean',value=aggregate_point[(method,'ood_worst_group')]['mean'],**ci(aggregate[(method,'ood_worst_group')]['mean'])))
        tables=[]
        for name,rows in [('per_seed',long),('per_degree',degree_rows),('main',main_rows),('paired_ci',paired),('worst_group',worst)]:
            tables.append(write_csv(EXP/'tables'/f'stage6_{name}.csv',rows))
        atomic_write_json(JOB/'result.json',dict(status='pass',at_utc=utc_now(),tables=tables,rows=dict(per_seed=len(long),per_degree=len(degree_rows),main=len(main_rows),paired_ci=len(paired),worst_group=len(worst)),
            predictions=prediction_records,bootstrap=bootstrap_records,index=artifact(index),
            definitions=dict(mean='equal degree and training-seed macro mean',p95='mean of per-degree/per-seed P95; not pooled or P95 of seed-averaged nets',
                worst_group='max of three OOD group mean regrets',ci='10000 percentile paired net bootstraps, stratified by degree/distribution, identical net draws across seeds and methods',
                limitation='conditional on the three trained seeds; does not estimate a population of training seeds; no multiple-comparison adjustment')))
        lease.finish(update=100,status='complete')
if __name__=='__main__':main()
