"""Final evidence tables, efficiency provenance, figures and scoped claims."""
import csv
import shutil

from stage8.common import *
from stage8.attack import task_id
from stage8.statistics import METHODS,DEGREES,SEEDS,GROUPS,attack_statistics,ispd_statistics
from stage7.delivery import write_csv

TABLES=EXP/'tables/stage8'


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def check_tree(value):
    """Validate recursively recorded artifact descriptors, without guessing paths."""
    count=0
    if isinstance(value,dict):
        if {'logical_path','resolved_path','sha256'}<=set(value):
            checked(value);return 1
        for v in value.values():count+=check_tree(v)
    elif isinstance(value,list):
        for v in value:count+=check_tree(v)
    return count


def reuse_tables():
    stage6=read(EXP/'logs/stage6_statistics/result.json')
    stage7=read(EXP/'logs/stage7/statistics.json')
    assert stage6['status']==stage7['status']=='pass'
    check_tree(stage6['tables']);check_tree(stage7['tables'])
    reused=[artifact(EXP/'logs/stage6_statistics/result.json'),artifact(EXP/'logs/stage7/statistics.json')]
    sources=list(EXP.glob('tables/stage6_*.csv'))+list(EXP.glob('tables/stage5_*.csv'))+list((EXP/'tables/stage7').glob('*.csv'))
    for source in sources:
        suffix='stage7_'+source.name if source.parent.name=='stage7' else source.name
        dest=TABLES/'accepted_inputs'/suffix
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            tmp=dest.with_suffix('.csv.partial');shutil.copyfile(source,tmp)
            assert file_sha256(tmp)==file_sha256(source);os.replace(tmp,dest)
        assert file_sha256(dest)==file_sha256(source)
        reused.extend([artifact(source),artifact(dest)])
    main=[]
    for r in read_csv(EXP/'tables/stage6_main.csv'):
        for metric in ('mean','p95'):
            main.append(dict(dataset='synthetic',method=r['method'],degree='macro5',distribution=r['distribution'],
                metric=metric,estimate=float(r[metric]),ci_low=float(r[metric+'_ci_low']),ci_high=float(r[metric+'_ci_high']),
                seed101=None,seed102=None,seed103=None))
    for r in read_csv(EXP/'tables/stage6_worst_group.csv'):
        if r['degree']=='macro5':
            main.append(dict(dataset='synthetic',method=r['method'],degree='macro5',distribution='ood_worst_group',metric='mean',
                estimate=float(r['value']),ci_low=float(r['ci_low']),ci_high=float(r['ci_high']),seed101=None,seed102=None,seed103=None))
    # Supply real per-seed directions for the reused synthetic aggregate points.
    raw=read_csv(EXP/'tables/stage6_per_seed.csv')
    for row in main:
        for seed in SEEDS:
            if row['distribution']=='ood_worst_group':
                value=max(np.mean([float(x['mean']) for x in raw if x['method']==row['method'] and int(x['seed'])==seed and x['distribution']==g]) for g in GROUPS[1:])
            else:
                value=np.mean([float(x[row['metric']]) for x in raw if x['method']==row['method'] and int(x['seed'])==seed and x['distribution']==row['distribution']])
            row['seed'+str(seed)]=float(value)
    return main,reused


def training_efficiency(plan):
    rows,records=[],[]
    for model in plan['models']:
        path=checked(model['result']);result=read(path)
        segments=[(p,read(p)) for p in sorted(path.parent.glob('segment_u*.json'))]
        assert segments
        covered=set()
        for p,r in segments:
            assert r['run_id']==result['run_id']
            covered.update(range(r['started_this_segment_at_update'],r['completed_updates']+1))
            records.append(artifact(p))
        events_path=path.parent/'events.jsonl'
        train=[]
        with events_path.open() as f:
            for line in f:
                e=json.loads(line)
                if e.get('event')=='train':train.append(dict(update=e['update'],at_utc=e['at_utc']))
        # ERM intentionally logs every 10 updates; resumed attempts may repeat
        # a logged update. Training completion comes from the accepted model
        # audit, not an invented requirement for 2000 telemetry rows.
        assert result['completed_updates']==2000 and train and max(e['update'] for e in train)==2000
        assert all(1<=e['update']<=2000 for e in train)
        def timestamp(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
        elapsed=(timestamp(result['finished_at_utc'])-timestamp(train[0]['at_utc'])).total_seconds()
        ledger=result['call_ledger']
        rows.append(dict(method=model['method'],degree=model['degree'],seed=model['seed'],updates=2000,batch_size=256,
            recorded_segment_wall_seconds=sum(r['segment_wall_seconds'] for _,r in segments),
            updates_covered_by_recorded_segments=len(covered),segment_wall_covers_all_updates=len(covered)==2000,
            first_recorded_update=min(e['update'] for e in train),recorded_train_events=len(train),
            unique_logged_updates=len({e['update'] for e in train}),
            elapsed_first_recorded_update_to_finish_seconds=elapsed,
            peak_reserved_mib=max(r['peak_memory_reserved_bytes']/2**20 for _,r in segments),
            effective_actor_calls=ledger['actor_forward_nets'],effective_rest_calls=ledger['exact_rest_evaluator_nets'],
            effective_geosteiner_calls=ledger['geosteiner_nets'],effective_surrogate_calls=ledger['surrogate_evaluator_nets'],
            failed_attempt_total_known=False,model_sha256=model['best']['sha256'],result_sha256=model['result']['sha256']))
        records.extend([model['result'],artifact(events_path)])
    return rows,records


def inference_efficiency(plan,include_attacks=True):
    rows,records=[],[]
    for model in plan['models']:
        key=f"{model['method']}_d{model['degree']}_s{model['seed']}"
        for dataset,group,path in [('synthetic',d,EXP/'logs/stage6_test'/key/d/'result.json') for d in GROUPS]+[
            ('ispd','ispd',EXP/'logs/stage7/evaluation'/key/'result.json')]:
            r=read(path);checked(r['predictions']);records.append(artifact(path))
            rows.append(dict(dataset=dataset,method=model['method'],degree=model['degree'],seed=model['seed'],distribution=group,
                count=r['completed_nets'],actor_seconds=r['actor_seconds'],rest_seconds=r['rest_seconds'],
                reference_seconds=0.,search_wall_seconds=r['actor_seconds']+r['rest_seconds'],verification_seconds=0.,
                peak_reserved_mib=r['peak_reserved_mib'],actor_calls=r['call_ledger']['actor_forward_nets'],
                rest_calls=r['call_ledger']['exact_rest_evaluator_nets'],search_geosteiner_calls=0,verification_geosteiner_calls=0,
                result_sha256=records[-1]['sha256']))
        if include_attacks:
            for group in GROUPS:
                path=LOG/'attacks'/task_id(model,group)/'result.json';r=read(path);records.append(artifact(path))
                rows.append(dict(dataset='attack',method=model['method'],degree=model['degree'],seed=model['seed'],distribution=group,
                    count=2000,actor_seconds=r['times']['actor_seconds'],rest_seconds=r['times']['rest_seconds'],
                    reference_seconds=r['times']['reference_seconds'],search_wall_seconds=r['times']['wall_seconds'],
                    verification_seconds=r['verification_seconds'],peak_reserved_mib=r['peak_reserved_mib'],actor_calls=512000,
                    rest_calls=512000,search_geosteiner_calls=512000,verification_geosteiner_calls=2000,
                    result_sha256=records[-1]['sha256']))
    return rows,records


def claim_rows(main,attack_pairs,ispd_pairs):
    def item(dataset,method,distribution,metric):
        return next(r for r in main if r['dataset']==dataset and r['method']==method and r['distribution']==distribution and r['metric']==metric)
    rows=[]
    for baseline in METHODS[:-1]:
        a=item('synthetic','wdro_rest','uniform','mean');b=item('synthetic','erm','uniform','mean')
        nominal_delta=a['estimate']-b['estimate'];nominal_ok=nominal_delta<=.002
        for dataset,pairs in [('attack',attack_pairs),('ispd',ispd_pairs)]:
            for r in pairs:
                if r['baseline']!=baseline or r['seed']!='mean3' or r['degree'] not in ('macro5','pooled'):continue
                supported=r['ci_high']<0 and r['all_three_seed_improve'] and nominal_ok
                rows.append(dict(dataset=dataset,baseline=baseline,distribution=r['distribution'],metric=r['metric'],
                    difference=r['difference'],ci_low=r['ci_low'],ci_high=r['ci_high'],
                    all_three_seed_improve=r['all_three_seed_improve'],nominal_delta_vs_erm=nominal_delta,
                    nominal_guard_pass=nominal_ok,supported_in_this_test=bool(supported),
                    wording='supports improvement for this metric and test' if supported else 'does not establish stable improvement for this metric and test'))
    return rows


def figures(main):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    artifacts=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,metric in zip(axes,('mean','p95')):
        for mi,method in enumerate(METHODS):
            rows=[next(r for r in main if r['dataset']=='attack' and r['method']==method and r['distribution']==g and r['metric']==metric) for g in GROUPS]
            x=np.arange(4)+(mi-2)*.14;y=np.array([r['estimate'] for r in rows])*100
            line=ax.plot(x,y,'o',label=method)[0]
            ax.vlines(x,[100*r['ci_low'] for r in rows],[100*r['ci_high'] for r in rows],color=line.get_color())
        ax.set_xticks(range(4),GROUPS,rotation=20);ax.set_ylabel('Relative regret (%)')
        ax.set_title('CEM-256: '+metric+' (degree / seed macro)');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
    for suffix in ('png','pdf','svg'):
        path=EXP/'figures'/f'stage8_attack_v2.{suffix}';tmp=path.with_suffix('.'+suffix+'.partial')
        fig.savefig(tmp,format=suffix,dpi=180);assert tmp.stat().st_size>1000;os.replace(tmp,path);artifacts.append(artifact(path))
    plt.close(fig)
    return artifacts+budget_figures()


def budget_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    artifacts=[]
    # Rebuild the required training diagnostic from the corrected frozen table.
    trajectory=read_csv(EXP/'tables/stage5_budget_trajectory.csv')
    selected=read(ROOT/'configs/stage5_selected_wdro_v1.json')
    selected_model=next(m for m in gate()['models'] if m['method']=='wdro_rest' and m['degree']==30 and m['seed']==101)
    run_id=read(checked(selected_model['result']))['run_id']
    rows=[r for r in trajectory if r.get('run_id')==run_id]
    assert len(rows)==2000
    # Column names are asserted by supporting-input smoke before any batch launch.
    cost_key='mean_selected_cost';lambda_key='lambda_after'
    if cost_key not in rows[0]:cost_key='mean_cost'
    if lambda_key not in rows[0]:lambda_key='lambda'
    assert cost_key in rows[0] and lambda_key in rows[0] and 'rho' in rows[0]
    x=[int(r['update']) for r in rows];cost=np.array([float(r[cost_key]) for r in rows]);rho=np.array([float(r['rho']) for r in rows])
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True);right=ax.twinx()
    ax.plot(x,cost,alpha=.35,label='mean selected cost');ax.plot(x,rho,label='rho',color='black')
    ax.plot(x,cost-rho,label='budget gap',alpha=.7);right.plot(x,[float(r[lambda_key]) for r in rows],color='C3',label='lambda')
    ax.set_xlabel('Policy update');ax.set_ylabel('Hungarian-L1 cost');right.set_ylabel('lambda')
    ax.set_title('Corrected WDRO d30 / seed101: frozen validation-selected radius')
    lines=ax.lines+right.lines
    ax.legend(lines,[l.get_label() for l in lines],loc='upper center',bbox_to_anchor=(.5,-.18),ncol=4,fontsize=9)
    for suffix in ('png','pdf','svg'):
        path=EXP/'figures'/f'stage8_budget_v2.{suffix}';tmp=path.with_suffix('.'+suffix+'.partial')
        fig.savefig(tmp,format=suffix,dpi=180);assert tmp.stat().st_size>1000;os.replace(tmp,path);artifacts.append(artifact(path))
    plt.close(fig)
    return artifacts


def supporting_statistics():
    plan=gate();preflight,_=capture('Stage8 final supporting statistics and efficiency (CPU)',peak=0)
    with Lease(LOG/'supporting_statistics','stage8_supporting_statistics',str(preflight)) as lease:
        synthetic,reused=reuse_tables()
        training,training_sources=training_efficiency(plan)
        inference,inference_sources=inference_efficiency(plan,include_attacks=False)
        ispd=ispd_statistics(lease)
        tables=[write_csv(TABLES/'training_efficiency.csv',training),write_csv(TABLES/'clean_inference_efficiency.csv',inference),
                write_csv(TABLES/'ispd_main_ci.csv',ispd['main']),write_csv(TABLES/'ispd_seed_ci.csv',ispd['per_seed']),
                write_csv(TABLES/'ispd_paired_ci.csv',ispd['paired'])]
        r=dict(status='pass',at_utc=utc_now(),plan=artifact(PLAN),synthetic_main=synthetic,ispd=ispd,tables=tables,
            reused=reused,training_sources=training_sources,inference_sources=inference_sources,
            row_counts=dict(training=75,clean_inference=375,ispd_main=20,ispd_seed=60,ispd_paired=64),
            limitations=['Training effective ledgers exclude unrecorded failed/replayed work.',
                'Recorded segment wall time is incomplete when its update coverage is below 2000.',
                'First-update-to-finish elapsed includes validation, capacity pauses and queue waits.',
                'Wall time reflects shared hardware occupancy; no controlled speedup claim.',
                'ISPD CIs condition on fixed design-degree counts and three trained seeds; no design-population inference.'])
        atomic_write_json(LOG/'supporting_statistics/result.json',r,mode=0o444)
        progress('8.3 已验收合成/ISPD统计与训练效率汇总完成，独立攻击结论仍待全量运行','DONE',
            str(LOG/'supporting_statistics/result.json'),'同一阶段内完成CEM-256全量攻击及最终表图交付')
        return r


def export():
    plan=gate();audit=read(LOG/'audit/result.json')
    assert audit['status']=='pass' and audit['plan']==artifact(PLAN)
    preflight,_=capture('Stage8 final statistical export (CPU)',peak=0)
    with Lease(LOG/'statistics','stage8_statistics',str(preflight)) as lease:
        support=read(LOG/'supporting_statistics/result.json') if (LOG/'supporting_statistics/result.json').exists() else supporting_statistics()
        assert support['plan']==artifact(PLAN) and support['status']=='pass'
        check_tree(support['tables']);check_tree(support['ispd']['bootstrap'])
        attack=attack_statistics(lease)
        main=support['synthetic_main']+support['ispd']['main']+attack['main']
        claims=claim_rows(main,attack['paired'],support['ispd']['paired'])
        inference,inf_sources=inference_efficiency(plan)
        rows={'attack_per_seed':attack['per_seed'],'attack_per_degree':attack['per_degree'],'attack_main':attack['main'],
              'attack_paired_ci':attack['paired'],'final_main_long':main,'inference_efficiency':inference,'claim_support':claims}
        tables=[write_csv(TABLES/(name+'.csv'),values) for name,values in rows.items()]
        raw=TABLES/'final_tables.json';atomic_write_json(raw,rows,mode=0o444)
        figs=figures(main)
        result=dict(status='pass',at_utc=utc_now(),plan=artifact(PLAN),audit=artifact(LOG/'audit/result.json'),
            supporting=artifact(LOG/'supporting_statistics/result.json'),tables=tables,raw_json=artifact(raw),figures=figs,
            bootstrap=attack['bootstrap'],paired_nets=attack['paired_nets'],inference_sources=inf_sources,
            row_counts={k:len(v) for k,v in rows.items()},definitions=dict(
                attack_p95='mean of per-degree per-seed P95',worst_group='maximum OOD mean after seed and degree averaging',
                ispd_p95='pooled across nets for each seed; averaged over seeds',
                ci='10000 paired net bootstrap draws; fixed three trained seeds and data strata; no multiple comparison adjustment',
                attack='256-call local CEM stress test; empirical costs, not a certificate or distribution coverage'))
        atomic_write_json(LOG/'statistics/result.json',result,mode=0o444)
        lease.update(300,'pass')
        return result


def verify():
    plan=gate();result=read(LOG/'statistics/result.json');assert result['status']=='pass'
    assert result['plan']==artifact(PLAN)
    checked_count=check_tree(result)
    support=read(checked(result['supporting']))
    assert support['status']=='pass' and support['plan']==artifact(PLAN)
    checked_count+=check_tree(support)
    raw=read(checked(result['raw_json']))
    assert len(raw['attack_per_seed'])==300 and len(raw['final_main_long'])==110
    assert len(raw['inference_efficiency'])==675
    assert {(r['method'],int(r['degree']),int(r['seed']),r['distribution']) for r in raw['attack_per_seed']}=={
        (m['method'],m['degree'],m['seed'],d) for m in plan['models'] for d in GROUPS}
    for r in raw['attack_per_seed']:
        assert r['count']==2000 and r['actor_calls']==r['rest_calls']==r['search_geosteiner_calls']==512000
        assert r['verification_geosteiner_calls']==2000 and r['legal_fraction']==1
    # Read back every CSV against its typed JSON source, including CI endpoints.
    for rec in result['tables']:
        path=checked(rec);name=path.stem;rows=read_csv(path)
        assert len(rows)==len(raw[name])
        for a,b in zip(rows,raw[name],strict=True):
            assert set(a)==set(b)
            for k,v in b.items():
                assert a[k]==('' if v is None else str(v)),(name,k)
    for row in raw['final_main_long']:
        assert all(np.isfinite(row[k]) for k in ('estimate','ci_low','ci_high')) and row['ci_low']<=row['ci_high']
    for m in plan['models']:
        checked(m['best']);checked(m['latest'])
    audit=read(checked(result['audit']))
    assert len(audit['records'])==300
    for rec in audit['records']:
        a=read(checked(rec));assert a['status']=='pass' and a['centers']==2000 and a['candidates']==512000
        checked_count+=check_tree(a['binding'])
        # Rehash all committed round artifacts after the numerical audit. An
        # unchanged cursor alone does not prove its referenced files survived.
        state=read(checked(a['binding']['state']))
        checked_count+=check_tree(state['batches'])
        checked_count+=check_tree(a['accounting'])
    expected_claims=claim_rows(raw['final_main_long'],raw['attack_paired_ci'],
        read(LOG/'supporting_statistics/result.json')['ispd']['paired'])
    assert expected_claims==raw['claim_support']
    completion=dict(status='pass',stage=8,at_utc=utc_now(),plan=artifact(PLAN),statistics=artifact(LOG/'statistics/result.json'),
        audit=artifact(LOG/'audit/result.json'),models=75,attack_cells=300,unique_centers=40000,
        model_center_attacks=600000,candidate_scores=153600000,verification_reference_calls=600000,
        nominal_guard=.002,stage9_started=False,claims=raw['claim_support'])
    dest=ROOT/'manifests/stage8_completion_v1.json';atomic_write_json(dest,completion,mode=0o444)
    report_path=CODEX/'阶段8_独立攻击效率与统计验收报告.md'
    rows=['# 阶段8 独立攻击、效率与最终统计验收','',
        '阶段8已通过工程与数值交付门禁。75个precision_v2冻结模型完成300格CEM-256攻击，',
        '覆盖40000个独立中心、600000次模型-中心攻击、153600000次真实候选评分；选中候选另有600000次双尺度参考核验。','',
        '攻击设置为16轮×16、每轮elite4，原样本计入预算。对所有方法最大化真实regret；运输cost为事后测量，未声称全局最坏分布或Wasserstein球内证书。','',
        '主表与完整CI见 `experiments/tables/stage8/final_main_long.csv`；逐degree/seed/分布及成本、合法率见 `attack_per_seed.csv`；',
        '逐网配对差值、效率表、原始CSV/JSON和PNG/PDF/SVG图均列入完成manifest。','',
        '置信区间使用10000次配对net bootstrap，条件于三个已训练seed及固定数据分层；未做多重比较校正。','',
        '训练耗时保留记录完整度：segment未覆盖2000 updates时其wall time不是总训练耗时；失败调用未完整计入旧有效轨迹账本。','',
        '| 比较基线 | 攻击后OOD worst-group差值(百分点) | 95% CI | 三seed同向改善 |','|---|---:|---|---|']
    for baseline in METHODS[:-1]:
        r=next(r for r in raw['claim_support'] if r['dataset']=='attack' and r['baseline']==baseline and r['distribution']=='ood_worst_group')
        rows.append(f"| {baseline} | {100*r['difference']:.5f} | [{100*r['ci_low']:.5f}, {100*r['ci_high']:.5f}] | {r['all_three_seed_improve']} |")
    supported=[r for r in raw['claim_support'] if r['supported_in_this_test']]
    rows+=['','各指标允许表述逐行列于 `claim_support.csv`。仅对应行通过方向、CI与nominal门槛时支持该测试中的改善；',
           '尾部改善不能扩写为mean全面领先，不跨分布或方法推广。正文同步属于阶段9，本次未执行。','',
           f'完成索引：`{dest}`。',f'复核入口：`{shlex.join(command("verify"))}`。','']
    tmp=report_path.with_suffix('.md.partial');tmp.write_text('\n'.join(rows));os.replace(tmp,report_path)
    delivery=dict(status='pass',at_utc=utc_now(),completion=artifact(dest),report=artifact(report_path),checked_artifact_bindings=checked_count,
        all_csv_json_rows_exact=True,all_models_preserved=True,all_attack_cells_preserved=True,stage9_started=False)
    atomic_write_json(ROOT/'manifests/stage8_delivery_verification_v1.json',delivery,mode=0o444)
    progress('8.1–8.3独立攻击、完整审计、效率、CI及表图交付全部通过','DONE',str(ROOT/'manifests/stage8_delivery_verification_v1.json'),
             '本次阶段8完成；阶段9未启动')
    p=TOP/'01_WDRO-REST_项目进度.md';s=p.read_text().replace('阶段8 `DOING`（独立攻击、效率与最终统计）','阶段8 `DONE`（独立攻击、效率与最终统计验收通过）')
    s=s.replace('| 8. 独立攻击/统计 | `DOING` |','| 8. 独立攻击/统计 | `DONE` |')
    s=s.replace('以阶段8子步骤门禁与心跳为准','阶段8无剩余阻塞')
    tmp=p.with_suffix('.md.partial');tmp.write_text(s);os.replace(tmp,p)
    return delivery
