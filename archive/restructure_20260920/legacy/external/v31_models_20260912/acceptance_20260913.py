"""Read-only reacceptance of frozen results; independent statistics in a new folder."""
import concurrent.futures
import csv
import fcntl
import io
import shutil
from datetime import datetime
import numpy as np
import finish
import stats_helpers as st
from run_queue import ExistingProcess
from common import *

REVIEW=OUT/'review_20260913'

def assert_close(actual,expected,label):
    ev.require(np.allclose(actual,expected,rtol=0,atol=2e-14),f'independent numerical mismatch: {label}')

def check_bound_artifacts():
    queue=[OUT/'STATUS.json',OUT/'completion_manifest.json',OUT/'models_frozen.json',OUT/'input_manifest.json',
           OUT/'runtime_frozen.json',OUT/'logs/queue/completion.json',OUT/'smoke/completion.json']
    queue.extend(sorted((OUT/'results').glob('*/cursor.json')))
    visited=set();bindings={}
    def walk(value):
        if isinstance(value,dict):
            logical=value.get('logical_path',value.get('path'))
            if isinstance(logical,str) and 'sha256' in value:
                path=ev.project_path(logical)
                key=(str(path),value['sha256'])
                if key not in bindings:
                    actual=artifact(path)
                    ev.require(actual['sha256']==value['sha256'],f'changed bound artifact: {path}')
                    if 'resolved_path' in value:
                        ev.require(actual['resolved_path']==value['resolved_path'],f'path target changed: {path}')
                    bindings[key]=actual
                    if path.suffix=='.json' and path.is_relative_to(OUT):queue.append(path)
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    while queue:
        path=queue.pop()
        if str(path) in visited:continue
        visited.add(str(path));walk(read(path))
    write(REVIEW/'artifact_bindings.json',list(bindings.values()))
    return list(bindings.values())

def check_lifecycle(models):
    records=[]
    directories=[OUT/'results'/task_id(m) for m in models]+[OUT/'logs'/name for name in ('pipeline','queue','finish')]
    for directory in directories:
        job=read(directory/'job.json')
        ev.require(job['status']=='exited' and job['outcome']=='complete' and job['lock_released'],f'not closed: {directory}')
        ev.require(ExistingProcess(job['process']).poll() is not None,f'live old process: {directory}')
        with (directory/'run.lock').open('r+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(stream,fcntl.LOCK_UN)
        records.append(dict(directory=str(directory),job=artifact(directory/'job.json'),lock_released=True,process_exited=True))
    return records

def independent_statistics(data,values,manifest):
    designs=st.string_vector(data['design_ids']);datasets=st.string_vector(data['dataset_ids']);families=st.string_vector(data['family_ids'])
    groups={('all','all'):np.ones(len(designs),dtype=bool)}
    groups.update({('dataset',s):datasets==s for s in sorted(set(datasets))})
    groups.update({('design',c['design_id']):designs==c['design_id'] for c in manifest['coverage']})
    table=list(csv.DictReader((OUT/'tables/metrics_by_seed.csv').open()))
    ev.require(len(table)==len(groups)*33,'missing per-seed rows')
    metric_values={}
    for row in table:
        key=(row['scope'],row['name'],row['method'],int(row['seed']))
        ev.require(key not in metric_values,'duplicate statistic row')
        mask=groups[key[:2]];regret=values[key[2:]][mask]
        distinct=np.unique(designs[mask]);family_count=len(np.unique(families[mask]))
        ev.require(int(row['count'])==len(regret) and int(row['design_count'])==len(distinct) and int(row['family_count'])==family_count,'statistic coverage mismatch')
        if not len(regret):
            ev.require(all(row[k]=='' for k in ('net_weighted_mean','p95','design_macro_mean','worst_design_mean','worst_design_id')),'empty design has fake metric')
            metric_values[key]={k:None for k in ('net_weighted_mean','p95','design_macro_mean','worst_design_mean')}
            continue
        per_design=np.array([np.mean(values[key[2:]][mask & (designs==d)]) for d in distinct])
        independent=dict(net_weighted_mean=float(sum(regret)/len(regret)),p95=float(np.percentile(regret,95)),
                         design_macro_mean=float(sum(per_design)/len(per_design)),worst_design_mean=float(max(per_design)))
        for metric,value in independent.items():assert_close(float(row[metric]),value,str(key)+metric)
        ev.require(row['worst_design_id']==distinct[int(np.argmax(per_design))],'worst design ID differs')
        metric_values[key]=independent
    summary=read(OUT/'summary.json')
    for row in summary['metrics_seed_mean']:
        for metric in ('net_weighted_mean','p95','design_macro_mean','worst_design_mean'):
            items=[metric_values[(row['scope'],row['name'],row['method'],seed)][metric] for seed in SEEDS]
            if items[0] is None:ev.require(row[metric] is None,'empty seed-mean has metric')
            else:assert_close(row[metric],np.mean(items),'seed-mean '+metric)
    # Independent implementation: directly sum each sampled cluster's numerator/denominator.
    ci_groups={}
    for row in summary['paired_comparisons']:
        ci_groups.setdefault((row['scope'],row['name'],row['method'],row['control']),[]).append(row)
    for (scope,name,method,control),rows in ci_groups.items():
        mask=groups[(scope,name)];d=designs[mask];f=families[mask]
        delta=np.array([values[(method,s)][mask]-values[(control,s)][mask] for s in SEEDS])
        names=np.unique(f);F=len(names);draw=np.random.default_rng(20260912).integers(0,F,size=(2000,F))
        for metric in ('net_weighted_mean','design_macro_mean'):
            if metric=='net_weighted_mean':
                units=f;matrix=delta
            else:
                unique_designs=np.unique(d)
                units=np.array([np.unique(f[d==design])[0] for design in unique_designs])
                matrix=np.array([delta[:,d==design].mean(axis=1) for design in unique_designs]).T
            numerator=np.array([matrix[:,units==family].sum(axis=1) for family in names])
            denominator=np.array([np.sum(units==family) for family in names])
            replicates=numerator[draw].sum(axis=1)/denominator[draw].sum(axis=1)[:,None]
            for row in (r for r in rows if r['metric']==metric):
                if row['training_seed']=='mean':point=matrix.mean(axis=1).mean();samples=replicates.mean(axis=1)
                else:
                    index=SEEDS.index(row['training_seed']);point=matrix[index].mean();samples=replicates[:,index]
                assert_close([row['mean_difference'],row['ci_low'],row['ci_high']],
                             [point,*np.quantile(samples,[.025,.975])],'paired family CI')
                ev.require(row['family_count']==F and row['count']==mask.sum(),'CI coverage mismatch')
    # Official regeneration must reproduce the delivered CSV and JSON exactly.
    generated=finish.build_tables(manifest,data,values,[])
    for name,rows in zip(('metrics_by_seed','metrics_seed_mean','paired_family_ci'),generated):
        path=REVIEW/f'{name}_recomputed.csv';st.write_csv(path,rows)
        ev.require(ev.sha256(path)==ev.sha256(OUT/'tables'/f'{name}.csv'),'official CSV regeneration mismatch')
    ev.require(generated[1]==summary['metrics_seed_mean'] and generated[2]==summary['paired_comparisons'],'summary differs from tables')
    coverage=list(csv.DictReader((OUT/'tables/coverage.csv').open()))
    ev.require(len(coverage)==49 and len({c['design_id'] for c in coverage})==49,'coverage table mismatch')
    for row,c in zip(coverage,manifest['coverage']):
        ev.require(row['design_id']==c['design_id'] and int(row['eligible_nets'])==int(np.sum(designs==c['design_id'])),'coverage count mismatch')
    return dict(per_seed_rows=len(table),seed_mean_rows=len(generated[1]),paired_ci_rows=len(generated[2]),
                coverage_rows=49,independent_mean_p95_macro_worst=True,independent_ci_replicates=True,
                regenerated_csv_byte_identical=True),summary

def supplemental_statistics(data,values):
    designs=st.string_vector(data['design_ids']);datasets=st.string_vector(data['dataset_ids'])
    families=st.string_vector(data['family_ids']);rows=[];comparisons=[]
    groups={'all':np.ones(len(designs),dtype=bool),'circuitops_all':np.char.startswith(datasets,'circuitops_'),
            'ispd2018_2019':np.char.startswith(datasets,'ispd201')}
    for name,mask in groups.items():
        for method in METHODS:
            metrics=[st.pooled_metrics(values[(method,s)][mask],designs[mask],families[mask]) for s in SEEDS]
            rows.append(dict(group=name,method=method,count=int(mask.sum()),designs=len(np.unique(designs[mask])),
                **{k:float(np.mean([x[k] for x in metrics])) for k in ('net_weighted_mean','p95','design_macro_mean')},
                seed_mean_min=min(x['net_weighted_mean'] for x in metrics),seed_mean_max=max(x['net_weighted_mean'] for x in metrics)))
        for method in NEW:
            for control in ('erm','group_dro','mixed_erm'):
                difference=np.array([values[(method,s)][mask]-values[(control,s)][mask] for s in SEEDS])
                for row in st.family_bootstrap(difference,designs[mask],families[mask]):
                    comparisons.append(dict(group=name,method=method,control=control,posthoc_exploratory=True,**row))
    winloss=[]
    for method,control in [(m,c) for m in NEW for c in ('erm','wdro_rest','group_dro','mixed_erm')]+[('es64','random64'),('rl_joint4','rl_frozen4')]:
        for metric in ('mean','p95'):
            differences=[]
            for d in sorted(set(designs)):
                mask=designs==d
                compute=np.mean if metric=='mean' else lambda x:np.quantile(x,.95)
                differences.append(np.mean([compute(values[(method,s)][mask])-compute(values[(control,s)][mask]) for s in SEEDS]))
            differences=np.array(differences)
            winloss.append(dict(method=method,control=control,metric=metric,designs=len(differences),
                wins=int(np.sum(differences < -1e-14)),ties=int(np.sum(np.abs(differences)<=1e-14)),losses=int(np.sum(differences>1e-14))))
    st.write_csv(REVIEW/'additional_group_statistics.csv',rows)
    st.write_csv(REVIEW/'additional_exploratory_ci.csv',comparisons)
    st.write_csv(REVIEW/'design_win_loss.csv',winloss)
    return rows,comparisons,winloss

def make_figures(summary,values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=sorted(METHODS,key=lambda m:np.mean([values[(m,s)].mean() for s in SEEDS]))
    labels={'random_top4':'RandomTop4','critic_calibrated4':'Critic4','rl_frozen4':'RL frozen4','rl_joint4':'RL joint4',
            'random64':'Random64','es64':'ES64','erm':'ERM','mixed_erm':'Mixed ERM','group_dro':'GroupDRO','hard_mining':'Hard Mining','wdro_rest':'WDRO-REST'}
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,title,metric in zip(axes,('Mean relative regret','P95 relative regret'),('net_weighted_mean','p95')):
        for i,m in enumerate(methods):
            row=next(r for r in summary['metrics_seed_mean'] if r['scope']=='all' and r['method']==m)
            color='#31688e' if m in NEW else '#9d6b28'
            ax.barh(i,row[metric]*100,color=color,alpha=.72)
            individual=[(values[(m,s)].mean() if metric=='net_weighted_mean' else np.quantile(values[(m,s)],.95))*100 for s in SEEDS]
            ax.scatter(individual,[i]*3,color='black',s=14,zorder=3)
        ax.set_title(title);ax.set_xlabel('Regret (%)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(methods)),[labels[m] for m in methods]);axes[0].invert_yaxis()
    fig.suptitle('All real degree-30 nets: 11,032 nets / model, 3 training seeds\nBlue: new training routes; brown: existing controls; dots: individual seeds',fontsize=11)
    fig.tight_layout();fig.savefig(REVIEW/'overall_statistics.png',dpi=180);fig.savefig(REVIEW/'overall_statistics.pdf');plt.close(fig)

def main():
    REVIEW.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    write(REVIEW/'STATUS.json',dict(status='running',at=now(),role='read_only_result_reacceptance'))
    write(REVIEW/'preflight.json',dict(at=now(),gpus=gpu_snapshot(),disk=dict(zip(('total','used','free'),shutil.disk_usage(OUT))),
          pid=ev.pid_identity(),new_gpu_evaluation_workers=0,new_geosteiner_calls=0))
    bindings=check_bound_artifacts()
    frozen=read(OUT/'models_frozen.json');models=frozen['models'];manifest=read(OUT/'input_manifest.json')
    ev.require(len(models)==33 and len({task_id(m) for m in models})==33,'model inventory differs')
    lifecycle=check_lifecycle(models)
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        audits=list(pool.map(finish.verify_model,models))
    print('PASS: all 33 model results, 364056 predictions, shards, bindings, and 36 closed jobs.',flush=True)
    import torch
    for model in models:
        ev.load_model(OUT/'models_frozen.json',model['method'],30,model['seed'])
        cp=torch.load(checked(model['best']),map_location='cpu',weights_only=False)
        ev.require(cp['update']==model['selected_update'],'checkpoint selected update mismatch')
    data=ev.load_npz(checked(manifest['input']))
    values={(m['method'],m['seed']):ev.load_npz(OUT/'results'/task_id(m)/'predictions.npz')['regret'] for m in models}
    statistics,summary=independent_statistics(data,values,manifest)
    extra,ci,winloss=supplemental_statistics(data,values)
    make_figures(summary,values)
    telemetry=[json.loads(line) for line in (OUT/'logs/queue/resources.jsonl').read_text().splitlines()]
    pjob=read(OUT/'logs/pipeline/job.json');qjob=read(OUT/'logs/queue/job.json')
    runtime=dict(physical_gpus=sorted({row['physical_gpu'] for m in models for row in [read(OUT/'results'/task_id(m)/'job.json')]}),
        max_concurrent_workers=max(len(t['active']) for t in telemetry),
        pipeline_seconds=(datetime.fromisoformat(pjob['finished_at_utc'])-datetime.fromisoformat(pjob['started_at_utc'])).total_seconds(),
        queue_seconds=(datetime.fromisoformat(qjob['finished_at_utc'])-datetime.fromisoformat(qjob['started_at_utc'])).total_seconds(),
        formal_worker_attempts=sum(len((OUT/'results'/task_id(m)/'attempts.jsonl').read_text().splitlines())//2 for m in models),
        committed_shards=sum(a['committed_shards'] for a in audits),unfinished_or_replayed_attempts=sum(a['unfinished_or_replayed_attempts'] for a in audits))
    # Confirm this audit did not modify any preexisting bound artifact.
    for rec in bindings:checked(rec)
    outputs=[artifact(p) for p in sorted(REVIEW.iterdir()) if p.is_file() and p.name not in ('STATUS.json','acceptance.json')]
    result=dict(status='pass',at=now(),model_tasks=33,predictions=sum(a['checked_nets'] for a in audits),
        input_nets=manifest['count'],designs=manifest['all_designs'],eligible_designs=manifest['eligible_designs'],
        bound_artifacts=len(bindings),lifecycle=lifecycle,model_audits=audits,statistics=statistics,runtime=runtime,
        new_gpu_evaluations=0,new_geosteiner_calls=0,frozen_artifacts_unchanged=True,outputs=outputs,
        implementation=artifact(Path(__file__)),wall_seconds=time.monotonic()-begin)
    write(REVIEW/'acceptance.json',result)
    write(REVIEW/'STATUS.json',dict(status='pass',at=now(),acceptance=artifact(REVIEW/'acceptance.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','lifecycle','model_audits')}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        write(REVIEW/'STATUS.json',dict(status='failed',at=now(),error=repr(exc),traceback=traceback.format_exc()))
        raise
