"""Independent delivery checks, complete tables and automatic Chinese report."""
import concurrent.futures
import fcntl
import numpy as np
import stats_helpers as st
from common import *

def verify_model(model):
    directory=OUT/'results'/task_id(model)
    with (directory/'run.lock').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=read(directory/'result.json');cursor=read(directory/'cursor.json');job=read(directory/'job.json')
        ev.require(job['status']=='exited' and job['outcome']=='complete' and job['lock_released'],'worker not exited cleanly')
        from run_queue import ExistingProcess
        ev.require(ExistingProcess(job['process']).poll() is not None,'worker still alive')
        binding=result['binding'];ev.require(binding['model']==model,'model binding mismatch')
        for rec in binding['sources']:checked(rec)
        checked(model['best']);checked(binding['model_index']);checked(binding['canonical_frozen_manifest'])
        data=ev.load_npz(checked(binding['input']));ref=ev.load_npz(checked(binding['reference']))
        receipt=read(checked(binding['reference_manifest']))
        ev.require(receipt['input']==binding['input'] and receipt['reference']==binding['reference'],'reference receipt mismatch')
        ev.validate_inputs(data,30);ev.validate_reference(ref,data)
        n=len(data['pins']);ev.require(result['completed_nets']==cursor['complete_nets']==n,'incomplete result/cursor')
        ev.require(cursor['binding']==binding,'cursor binding changed')
        ev.require(binding['inference']['tail_padding']=='repeat_last_requested_point_then_discard','padding protocol mismatch')
        ev.require(result['call_ledger']==dict(actor_forward_nets=n,actor_forward_rows=((n+255)//256)*256,
            actor_padding_rows=((n+255)//256)*256-n,exact_rest_evaluator_nets=n,geosteiner_nets=0,
            independent_python_tree_nets=n),'call ledger mismatch')
        arrays=ev.load_npz(checked(result['predictions']));ev.audit_predictions(arrays,data,ref)
        shards=ev.read_shards(cursor,data,ref)
        ev.require(shards.keys()==arrays.keys() and all(np.array_equal(shards[k],arrays[k]) for k in arrays), 'published result differs from shards')
        ev.require(result['summary']==st.metrics(arrays['regret']),'summary mismatch')
        calls=[json.loads(line) for line in (directory/'calls.jsonl').read_text().splitlines()]
        for shard in cursor['shards']:
            events=[row for row in calls if row['id']==shard['attempt']]
            ev.require(len(events)==2 and {e['phase'] for e in events}=={'started','completed'},'unaccounted committed call')
            done=next(e for e in events if e['phase']=='completed')
            ev.require(done['actor_forward_rows']==256 and done['actor_padding_rows']==256-(shard['end']-shard['start']), 'padding accounting mismatch')
        fcntl.flock(stream,fcntl.LOCK_UN)
    return dict(task=task_id(model),result=artifact(directory/'result.json'),predictions=result['predictions'],
        checked_nets=n,committed_shards=len(cursor['shards']),attempt_events=len(calls),
        unfinished_or_replayed_attempts=len([e for e in calls if e['phase']=='started'])-len(cursor['shards']))

def build_tables(manifest,data,values,models):
    designs=st.string_vector(data['design_ids']);datasets=st.string_vector(data['dataset_ids']);families=st.string_vector(data['family_ids'])
    groups=[('all','all',np.ones(len(designs),dtype=bool))]
    groups += [('dataset',d,datasets==d) for d in sorted(set(datasets))]
    groups += [('design',c['design_id'],designs==c['design_id']) for c in manifest['coverage']]
    rows=[];means=[]
    for scope,name,mask in groups:
        for method in METHODS:
            selected=[]
            for seed in SEEDS:
                metric=st.pooled_metrics(values[(method,seed)][mask],designs[mask],families[mask])
                row=dict(scope=scope,name=name,method=method,seed=seed,**metric)
                rows.append(row);selected.append(row)
            means.append(dict(scope=scope,name=name,method=method,count=selected[0]['count'],
                design_count=selected[0]['design_count'],family_count=selected[0]['family_count'],
                **{k:float(np.mean([r[k] for r in selected])) if selected[0][k] is not None else None
                   for k in ('net_weighted_mean','p95','design_macro_mean','worst_design_mean')}))
    pairs=[(m,c) for m in NEW for c in ('erm','wdro_rest')]
    pairs += [('critic_calibrated4','random_top4'),('rl_frozen4','random_top4'),('rl_joint4','random_top4'),
              ('rl_joint4','rl_frozen4'),('es64','random64')]
    comparisons=[]
    for scope,name,mask in groups:
        if scope=='design' or not mask.any():continue
        for method,control in pairs:
            difference=np.array([values[(method,s)][mask]-values[(control,s)][mask] for s in SEEDS])
            for row in st.family_bootstrap(difference,designs[mask],families[mask]):
                comparisons.append(dict(scope=scope,name=name,method=method,control=control,**row))
    return rows,means,comparisons

def main():
    with ev.Lease(OUT/'logs/finish',-1) as lease:
        manifest=read(OUT/'input_manifest.json');models=read(OUT/'models_frozen.json')['models']
        ev.require(len(models)==33 and set((m['method'],m['seed']) for m in models)==set((m,s) for m in METHODS for s in SEEDS),'model coverage differs')
        for rec in manifest['dependencies']:checked(rec)
        for rec in read(OUT/'runtime_frozen.json')['artifacts']:checked(rec)
        queue=read(OUT/'logs/queue/completion.json')
        ev.require(queue['status']=='pass' and queue['completed_tasks']==33,'queue not complete')
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
            audits=[]
            for audit in pool.map(verify_model,models):
                audits.append(audit);lease.update(len(audits),'auditing_models')
        data=ev.load_npz(checked(manifest['input']))
        values={(m['method'],m['seed']):ev.load_npz(OUT/'results'/task_id(m)/'predictions.npz')['regret'] for m in models}
        rows,means,comparisons=build_tables(manifest,data,values,models)
        tables=OUT/'tables';tables.mkdir(exist_ok=True)
        st.write_csv(tables/'coverage.csv',manifest['coverage'])
        st.write_csv(tables/'metrics_by_seed.csv',rows)
        st.write_csv(tables/'metrics_seed_mean.csv',means)
        st.write_csv(tables/'paired_family_ci.csv',comparisons)
        summary=dict(status='complete',at=now(),degree=30,model_count=33,new_models=18,controls=15,
            nets_per_model=manifest['count'],predictions=manifest['expected_predictions'],coverage=manifest['coverage'],
            metrics_seed_mean=means,paired_comparisons=comparisons)
        write(OUT/'summary.json',summary)
        report=['# 新训练模型：全部现有真实数据评测','',
            f"全部 18 个新模型及 15 个原有对照均完成。每个模型评测 {manifest['count']:,} 条网络，共 {manifest['expected_predictions']:,} 次预测。",'',
            f"覆盖库中全部 {manifest['all_designs']} 个设计，{manifest['eligible_designs']} 个含合法 degree=30 网络；零样本设计仍保留在覆盖表中。",'',
            '新增模型只训练了 degree=30，因此结果限于这一引脚数。所有模型使用原合成验证集选出的 best checkpoint，真实数据没有参与训练或选择。',
            '固定 256 行 greedy 推理；尾批重复最后一行后丢弃补齐结果。所有 15 个对照按同一规则重测。GeoSteiner 参考值复用已核验的双尺度精确结果。','',
            '数值均为相对 regret，越小越好。先对每个训练种子计算，再取三个种子的均值；P95 是各个种子 P95 的均值。', '',
            '| 方法 | 全部网络 mean | P95 | 设计宏平均 | 最差设计均值（种子均值） |',
            '|---|---:|---:|---:|---:|']
        for row in means:
            if row['scope']=='all':report.append(f"| {row['method']} | {row['net_weighted_mean']:.6f} | {row['p95']:.6f} | {row['design_macro_mean']:.6f} | {row['worst_design_mean']:.6f} |")
        report += ['', '## 分数据集', '', '| 数据集 | 方法 | 网络数 | mean | P95 | 设计宏平均 |', '|---|---|---:|---:|---:|---:|']
        for row in means:
            if row['scope']=='dataset':report.append(f"| {row['name']} | {row['method']} | {row['count']} | {row['net_weighted_mean']:.6f} | {row['p95']:.6f} | {row['design_macro_mean']:.6f} |")
        report += ['', '## 逐对差异与解释边界', '', '下表为全体网络的配对 mean 差值（新方法减对照），负数有利于新方法。95% CI 按设计家族聚类自助法计算，已将共享完全相同归一化几何的设计连接到同一簇。', '',
                   '| 新方法 | 对照 | 差值 | 95% CI |', '|---|---|---:|---|']
        for row in comparisons:
            if row['scope']=='all' and row['training_seed']=='mean' and row['metric']=='net_weighted_mean':
                report.append(f"| {row['method']} | {row['control']} | {row['mean_difference']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] |")
        report += ['', '完整逐设计、逐种子、分数据集 CI 见 tables/。未对多重比较作校正；这些区间是探索性配对比较。',
            '数据量较大的 ISPD2008 主导网络加权总体值，因此需同时查看设计宏平均和 CircuitOps/各 ISPD 年份拆分。',
            f"保留原始协议的全部样本（重复归一化几何行数 {manifest['duplicate_geometry_rows']}）；不能把这些行视作独立芯片。",
            'CircuitOps 是开放芯片设计的物理实现数据，本评测不主张这些设计均已流片。',
            '这是原始真实坐标上的泛化评测；没有新运行对抗攻击，不据此给出 Wasserstein 鲁棒性证书。','',
            '## 验收','', '全部结果逐网验证树连通性、独立 Python 线长、输入身份、参考值、regret、分片及发布文件一致性；模型和依赖哈希保持不变，33 个工作进程退出且锁已释放。']
        with ev.atomic_target(OUT/'REPORT.md') as stream:stream.write(('\n'.join(report)+'\n').encode())
        outputs=[artifact(p) for p in sorted(tables.glob('*.csv'))]+[artifact(OUT/'summary.json'),artifact(OUT/'REPORT.md')]
        write(OUT/'delivery_verification.json',dict(status='pass',at=now(),audits=audits,outputs=outputs,
            checked_predictions=sum(a['checked_nets'] for a in audits),checks=dict(all_worker_locks_released=True,
            all_model_hashes_verified=True,all_tree_lengths_independently_verified=True,all_shards_equal_published=True)))
        write(OUT/'completion_manifest.json',dict(status='complete',at=now(),model_tasks=33,
            predictions=manifest['expected_predictions'],input_manifest=artifact(OUT/'input_manifest.json'),
            frozen_models=artifact(OUT/'models_frozen.json'),runtime=artifact(OUT/'runtime_frozen.json'),
            delivery=artifact(OUT/'delivery_verification.json'),outputs=outputs))
        lease.update(33,'complete')
        print('Scientific evaluation, statistics and independent delivery checks complete.',flush=True)

if __name__=='__main__':main()
