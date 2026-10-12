"""Independent prediction audit and complete design-aware external tables."""
from __future__ import annotations

import csv
import itertools

import numpy as np

from src.metrics import relative_regret, summarize_regret
from src.rest import RestTreeEvaluator
from stage7.common import *
from stage7.evaluation import reference_gate, model_id, python_tree_lengths, read_shards


def reconcile_calls(path, state, reference=False):
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    starts, ends = {}, {}
    for event in events:
        target = starts if event['phase'] == 'started' else ends
        assert event['id'] not in target
        target[event['id']] = event
    assert set(ends).issubset(starts)
    for key, end in ends.items():
        n = starts[key]['end'] - starts[key]['start']
        fields = ('primary_calls','verification_calls') if reference else ('actor_calls','rest_calls')
        assert all(end[f] == n for f in fields)
    committed = [s['attempt'] for s in state['shards']]
    assert len(set(committed)) == len(committed) and set(committed).issubset(ends)
    assert sum(starts[k]['end']-starts[k]['start'] for k in committed) == state['complete_nets']
    return dict(committed_nets=state['complete_nets'],
        completed_attempt_nets=sum(starts[k]['end']-starts[k]['start'] for k in ends),
        unfinished_attempt_nets_upper_bound=sum(starts[k]['end']-starts[k]['start'] for k in starts.keys()-ends.keys()),
        completed_attempts=len(ends), unfinished_attempts=len(starts.keys()-ends.keys()))


def audit():
    plan = gate()
    references = reference_gate()
    preflight, _ = capture('Stage7.3 independent full prediction and topology audit (CPU)')
    directory = LOG / 'audit'
    with Lease(directory, 'stage7_independent_audit', str(preflight)) as lease:
        records, total, hardware = [], 0, set()
        rest = RestTreeEvaluator()
        for group in plan['groups']:
            degree = group['degree']
            data = load_npz(checked(group['input']))
            rr = read(CACHE / 'references' / f'degree_{degree}/result.json')
            ref = load_npz(checked(rr['reference']))
            refcursor = read(CACHE / 'references' / f'degree_{degree}/cursor.json')
            refcalls = reconcile_calls(LOG / 'references' / f'degree_{degree}/calls.jsonl', refcursor, True)
            atomic_write_json(directory / f'reference_calls_d{degree}.json', refcalls)
            for model in [m for m in plan['models'] if m['degree'] == degree]:
                rid = model_id(model)
                out = LOG / 'evaluation' / rid
                result_path = out / 'result.json'
                r = read(result_path)
                assert r['status'] == 'complete' and r['completed_nets'] == group['count']
                assert r['binding']['model'] == model and r['binding']['input'] == group
                assert r['binding']['reference'] == rr['reference']
                assert r['binding']['batch_size'] == 256 and r['binding']['T'] == 1
                assert r['binding']['mode'] == 'deterministic_greedy' and r['binding']['role'] == 'external_test'
                assert r['binding']['evaluator_sha256'] == file_sha256(ROOT / 'stage7/evaluation.py')
                assert r['binding']['plan'] == artifact(PLAN) and not r['model_selection']
                for k in ('best', 'latest', 'result'):
                    checked(model[k])
                pred = load_npz(checked(r['predictions']))
                cursor = read(out / 'cursor.json')
                assert cursor['binding'] == r['binding']
                merged = read_shards(cursor)
                for k in pred:
                    np.testing.assert_array_equal(pred[k], merged[k])
                for k in ('net_ids', 'design_ids', 'source_index'):
                    np.testing.assert_array_equal(pred[k], data[k])
                np.testing.assert_array_equal(pred['lstar'], ref['lstar'])
                binding = dict(result=artifact(result_path), prediction=r['predictions'], input=group,
                    auditor_sha256=file_sha256(Path(__file__)), evaluator_sha256=file_sha256(ROOT/'stage7/evaluation.py'))
                receipt = directory / 'models' / f'{rid}.json'
                verified = receipt.exists() and read(receipt).get('binding') == binding
                if not verified:
                    lengths = rest.lengths(data['pins'], pred['topology'])
                    np.testing.assert_array_equal(lengths, pred['rest_lengths'])
                    independent = python_tree_lengths(data['pins'], pred['topology'])
                    np.testing.assert_array_equal(independent, pred['rest_lengths'])
                    # Check inverse lengths on every represented normalized input algebraically;
                    # independently reevaluate first/last net of each existing design as well.
                    chosen = sorted(set(i for name in np.unique(data['design_ids'])
                                        for i in (np.flatnonzero(data['design_ids']==name)[0],
                                                  np.flatnonzero(data['design_ids']==name)[-1])))
                    raw = ((data['pins'][chosen].astype(np.float64)-.1)/data['scales'][chosen,None,None]
                           + data['offsets'][chosen,None,:])
                    inverse = python_tree_lengths(raw, pred['topology'][chosen])
                    np.testing.assert_allclose(inverse, pred['rest_lengths_inverse'][chosen], rtol=1e-12, atol=1e-8)
                regret = relative_regret(pred['rest_lengths'], ref['lstar'])
                np.testing.assert_array_equal(regret, pred['regret'])
                assert np.isfinite(regret).all() and regret.min() >= -1e-12
                np.testing.assert_array_equal(pred['rest_lengths_inverse'], pred['rest_lengths']/data['scales'])
                np.testing.assert_array_equal(pred['lstar_inverse'], ref['lstar']/data['scales'])
                assert r['summary'] == summarize_regret(regret)
                n = len(regret)
                assert r['call_ledger'] == dict(actor_forward_nets=n, exact_rest_evaluator_nets=n, geosteiner_nets=0)
                calls = reconcile_calls(out / 'calls.jsonl', cursor)
                checks = ['frozen_model_and_input', 'shared_inference', 'shards_equal_prediction',
                          'design_net_and_source_ids', 'precision_reference_equal', 'c_evaluator_equal',
                          'independent_python_length_and_connected_tree', 'inverse_length', 'regret_arithmetic',
                          'finite_nonnegative', 'summary_recomputed', 'call_receipts']
                if not verified:
                    atomic_write_json(receipt, dict(status='pass', binding=binding, checks=checks,
                        audited_nets=n, calls=calls, at_utc=utc_now()), mode=0o444)
                else:
                    assert read(receipt)['status'] == 'pass' and read(receipt)['calls'] == calls
                records.append(artifact(receipt))
                total += n
                hardware.add(json.dumps(r['binding']['hardware'], sort_keys=True))
                lease.update(len(records), 'auditing', audited_nets=total)
        assert len(records) == 75 and total == 337470 and len(hardware) == 1
        report = dict(status='pass', at_utc=utc_now(), plan=artifact(PLAN), records=records,
            references=references, models=75, predictions=total, model_checks=900,
            checks=['all_75_models', 'all_337470_predictions', 'one_hardware_class', 'all_designs_and_degrees'],
            additional_rest_evaluator_nets=total, independent_python_tree_nets=total)
        atomic_write_json(directory / 'result.json', report, mode=0o444)
        lease.update(75, 'pass')
        progress('7.3 337470条预测的独立长度/连通性/reference/账目审计通过', 'DONE',
                 str(directory/'result.json'), '导出design-aware表格、逐net配对差值与交付复核')
        return report


def design_statistics(values, designs):
    values = np.asarray(values, dtype=np.float64)
    designs = np.asarray(designs)
    assert len(values) and len(values) == len(designs) and np.isfinite(values).all()
    means = {x.decode() if isinstance(x, bytes) else str(x): float(values[designs==x].mean())
             for x in np.unique(designs)}
    return dict(count=len(values), net_weighted_mean=float(values.mean()),
                design_macro_mean=float(np.mean(list(means.values()))),
                p95=float(np.quantile(values, .95)), worst_design_mean=max(means.values()),
                worst_design_id=max(means, key=means.get), design_count=len(means))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.partial')
    fields = list(rows[0])
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    with tmp.open() as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    os.replace(tmp, path)
    return artifact(path)


def arrays_by_model(plan):
    result = {}
    for model in plan['models']:
        r = read(LOG / 'evaluation' / model_id(model) / 'result.json')
        result[(model['method'],model['degree'],model['seed'])] = load_npz(checked(r['predictions']))
    return result


def export():
    plan = gate()
    audit_report = read(LOG / 'audit/result.json')
    assert audit_report['status'] == 'pass' and audit_report['plan'] == artifact(PLAN)
    values = arrays_by_model(plan)
    methods = ['erm','mixed_erm','hard_mining','group_dro','wdro_rest']
    degrees, seeds = [10,20,30,40,50], [101,102,103]
    tables = EXP / 'tables/stage7'
    rows = dict(counts=[], per_design_degree=[], per_degree=[], per_design=[], per_seed=[], main=[], inference=[])
    rows['counts'] = [dict(c, invalid_reasons=json.dumps(c['invalid_reasons'],sort_keys=True)) for c in plan['counts']]
    for model in plan['models']:
        method, degree, seed = model['method'], model['degree'], model['seed']
        a = values[(method,degree,seed)]
        rows['per_degree'].append(dict(method=method, degree=degree, seed=seed, **design_statistics(a['regret'], a['design_ids'])))
        r = read(LOG / 'evaluation' / model_id(model) / 'result.json')
        rows['inference'].append(dict(method=method, degree=degree, seed=seed, count=len(a['regret']),
            selected_update=r['selected_update'], model_sha256=model['best']['sha256'],
            actor_seconds=r['actor_seconds'], rest_seconds=r['rest_seconds'],
            peak_reserved_mib=r['peak_reserved_mib'], actor_calls=len(a['regret']),
            rest_calls=len(a['regret']), geosteiner_calls=0))
        for c in [x for x in plan['counts'] if x['degree']==degree]:
            sample = a['regret'][a['design_ids']==c['design_id'].encode()]
            assert len(sample) == c['valid_count']
            summary = dict(mean=float(sample.mean()), p95=float(np.quantile(sample,.95))) if len(sample) else dict(mean=None,p95=None)
            rows['per_design_degree'].append(dict(method=method, seed=seed, design_id=c['design_id'], degree=degree,
                valid_count=c['valid_count'], invalid_count=c['invalid_count'], source_missing=c['source_missing'],
                **summary, model_sha256=model['best']['sha256']))
    for method, seed in itertools.product(methods, seeds):
        v = np.concatenate([values[(method,d,seed)]['regret'] for d in degrees])
        names = np.concatenate([values[(method,d,seed)]['design_ids'] for d in degrees])
        rows['per_seed'].append(dict(method=method, seed=seed, **design_statistics(v,names)))
        for name in sorted(np.unique(names)):
            sample = v[names==name]
            rows['per_design'].append(dict(method=method, seed=seed, design_id=name.decode(), count=len(sample),
                mean=float(sample.mean()), p95=float(np.quantile(sample,.95))))
    for method in methods:
        source = [r for r in rows['per_seed'] if r['method']==method]
        row = dict(method=method, seeds=3, unique_nets=22498, designs=20)
        for k in ('net_weighted_mean','design_macro_mean','p95','worst_design_mean'):
            row[k] = float(np.mean([r[k] for r in source]))
            row[k+'_seed_sd'] = float(np.std([r[k] for r in source],ddof=1))
        rows['main'].append(row)
    artifacts = [write_csv(tables / (name+'.csv'), group) for name,group in rows.items()]
    paired = []
    for method, degree in itertools.product(methods[:-1], degrees):
        a = values[(method,degree,101)]
        differences = np.stack([values[('wdro_rest',degree,s)]['regret'] - values[(method,degree,s)]['regret'] for s in seeds])
        for s in seeds:
            np.testing.assert_array_equal(a['net_ids'], values[('wdro_rest',degree,s)]['net_ids'])
        path = tables / 'paired_nets' / f'wdro_minus_{method}_d{degree}.npz'
        atomic_npz(path, dict(net_ids=a['net_ids'], design_ids=a['design_ids'], seeds=np.array(seeds),
                               relative_regret_difference=differences))
        paired.append(artifact(path))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [x['design_id'] for x in rows['per_design'] if x['method']=='erm' and x['seed']==101]
    matrix = np.array([[np.mean([r['mean'] for r in rows['per_design'] if r['design_id']==name and r['method']==method])*100
                       for method in methods] for name in names])
    fig, axis = plt.subplots(figsize=(9,9), constrained_layout=True)
    im = axis.imshow(matrix, aspect='auto', cmap='YlOrRd')
    axis.set_xticks(range(5), ['ERM','Mixed ERM','Hard Mining','GroupDRO','WDRO'])
    axis.set_yticks(range(len(names)), names)
    axis.set_title('ISPD protocol A: mean relative regret by design\nMean of three frozen training seeds; all available degrees')
    fig.colorbar(im, ax=axis, label='Relative regret (%)')
    figures = []
    for suffix in ('png','pdf','svg'):
        path = EXP / 'figures' / f'stage7_design_mean.{suffix}'
        tmp = path.with_suffix('.'+suffix+'.partial')
        fig.savefig(tmp, format=suffix, dpi=160)
        assert tmp.stat().st_size > 1000
        os.replace(tmp,path)
        figures.append(artifact(path))
    plt.close(fig)
    report = dict(status='pass', at_utc=utc_now(), plan=artifact(PLAN), audit=artifact(LOG/'audit/result.json'),
        tables=artifacts, figures=figures, paired_net_differences=paired,
        row_counts={k:len(v) for k,v in rows.items()}, definitions=plan['statistics'], main=rows['main'])
    atomic_write_json(LOG/'statistics.json',report,mode=0o444)
    return report


def verify():
    plan = gate()
    preflight, _ = capture('Stage7.3 final independent table and artifact verification (CPU)')
    report = read(LOG/'statistics.json')
    assert report['status']=='pass' and report['plan']==artifact(PLAN)
    checked(report['audit'])
    a = read(report['audit']['logical_path'])
    for record in a['records']:
        checked(record)
    values = arrays_by_model(plan)
    seeds, degrees = [101,102,103], [10,20,30,40,50]
    checked_rows, artifacts = 0, []
    expected_counts = dict(counts=100,per_design_degree=1500,per_degree=75,per_design=300,per_seed=15,main=5,inference=75)
    points = {}
    def eq(x,y):
        assert np.isclose(float(x),float(y),rtol=1e-12,atol=1e-14), (x,y)
    for table in report['tables']:
        path = checked(table)
        rows = list(csv.DictReader(path.open()))
        name = path.stem
        assert len(rows)==expected_counts[name]
        unique = set()
        for row in rows:
            if name=='counts':
                key=(row['design_id'],int(row['degree']))
                c=next(c for c in plan['counts'] if (c['design_id'],c['degree'])==key)
                for k in ('valid_count','invalid_count','parsed_count'):
                    assert int(row[k])==c[k]
                assert row['source_missing']==str(c['source_missing'])
                assert json.loads(row['invalid_reasons'])==c['invalid_reasons']
            elif name=='inference':
                key=(row['method'],int(row['degree']),int(row['seed']))
                model=next(m for m in plan['models'] if (m['method'],m['degree'],m['seed'])==key)
                r=read(LOG/'evaluation'/model_id(model)/'result.json')
                for k in ('actor_seconds','rest_seconds','peak_reserved_mib','selected_update'):
                    eq(row[k],r[k])
                assert row['model_sha256']==model['best']['sha256']
                assert int(row['count'])==len(values[key]['regret'])
            elif name=='main':
                key=row['method']
                for metric in ('net_weighted_mean','design_macro_mean','p95','worst_design_mean'):
                    v=[points[(key,s)][metric] for s in seeds]
                    eq(row[metric],np.mean(v));eq(row[metric+'_seed_sd'],np.std(v,ddof=1))
                assert (int(row['seeds']),int(row['unique_nets']),int(row['designs']))==(3,22498,20)
            else:
                method,seed=row['method'],int(row['seed'])
                ds=[int(row['degree'])] if 'degree' in row else degrees
                v=np.concatenate([values[(method,d,seed)]['regret'] for d in ds])
                names=np.concatenate([values[(method,d,seed)]['design_ids'] for d in ds])
                if 'design_id' in row:
                    v=v[names==row['design_id'].encode()]
                    key=(method,seed,row['design_id'],tuple(ds))
                    expected=int(row.get('valid_count',row.get('count')))
                    assert len(v)==expected
                    if len(v):
                        eq(row['mean'],np.mean(v));eq(row['p95'],np.percentile(v,95))
                    else:
                        assert row['mean']==row['p95']==''
                    if name=='per_design_degree':
                        c=next(c for c in plan['counts'] if c['design_id']==row['design_id'] and c['degree']==ds[0])
                        assert int(row['invalid_count'])==c['invalid_count'] and row['source_missing']==str(c['source_missing'])
                        model=next(m for m in plan['models'] if m['method']==method and m['seed']==seed and m['degree']==ds[0])
                        assert row['model_sha256']==model['best']['sha256']
                else:
                    key=(method,seed,tuple(ds))
                    means={n.decode():float(np.mean(v[names==n])) for n in np.unique(names)}
                    independently=dict(net_weighted_mean=float(np.mean(v)), design_macro_mean=float(sum(means.values())/len(means)),
                        p95=float(np.percentile(v,95)),worst_design_mean=max(means.values()))
                    for k,x in independently.items():
                        eq(row[k],x)
                    assert int(row['count'])==len(v) and int(row['design_count'])==len(means)
                    assert row['worst_design_id']==max(means,key=means.get)
                    if name=='per_seed':
                        points[(method,seed)]=independently
            assert key not in unique, (name,key)
            unique.add(key)
            checked_rows+=1
        artifacts.append(table)
    assert checked_rows==2070
    for row in report['paired_net_differences']:
        path=checked(row)
        name=path.stem.removeprefix('wdro_minus_');method,ds=name.rsplit('_d',1);d=int(ds)
        paired=load_npz(path)
        np.testing.assert_array_equal(paired['seeds'],seeds)
        for k in ('net_ids','design_ids'):
            np.testing.assert_array_equal(paired[k],values[(method,d,101)][k])
        np.testing.assert_array_equal(paired['relative_regret_difference'],
            np.stack([values[('wdro_rest',d,s)]['regret']-values[(method,d,s)]['regret'] for s in seeds]))
    for row in report['figures']:
        checked(row)
    for model in plan['models']:
        for k in ('best','latest','result'):
            checked(model[k])
    assert not list((LOG/'evaluation').glob('**/*.partial'))
    completion=dict(status='pass',stage=7,at_utc=utc_now(),models=75,unique_nets=22498,
        predictions=337470,designs=20,design_degree_input_files=95,missing_degree_files=5,invalid_nets=0,
        plan=artifact(PLAN),audit=artifact(LOG/'audit/result.json'),statistics=artifact(LOG/'statistics.json'),
        tables=report['tables'],figures=report['figures'],paired=report['paired_net_differences'],
        checked_table_rows=checked_rows,checked_prediction_models=75,reference_gate=artifact(LOG/'references_gate.json'),
        inference_smoke=artifact(LOG/'smoke/evaluation_gate.json'),preflight=artifact(preflight),
        stage8_started=False,ispd_used_for_training_or_selection=False,
        researcher_blind_claim=False,definitions=plan['statistics'])
    dest=ROOT/'manifests/stage7_completion_v1.json'
    atomic_write_json(dest,completion,mode=0o444)
    delivery=dict(status='pass',stage=7,at_utc=utc_now(),completion=artifact(dest),
        checks_passed=2070+75+20+3+5,table_rows=2070,prediction_models=75,paired_arrays=20,figures=3,
        final_global_checks=['unchanged_models','reference_gate','smoke_gate','no_partial_predictions','frozen_stage7_scope'])
    atomic_write_json(ROOT/'manifests/stage7_delivery_verification_v1.json',delivery,mode=0o444)
    render_report(completion,report)
    return delivery


def render_report(completion, statistics):
    path=TOP/'阶段7_ISPD外部测试验收报告.md'
    lines=['# 阶段7 ISPD协议A外部测试验收报告','',
        f"完成时间：{completion['at_utc']}。阶段7 **DONE**。",'',
        '使用 precision_v2 阶段6已经冻结的75个模型；模型及超参数只来自合成训练和validation。',
        '全部20个基础design、95个既有design×degree文件、22498个有效net已评价；5个缺失degree文件和0个无效net均逐格保留。排除6个metal5重复版本。',
        '合计337470条预测，统一共享归一化、batch256、T=1 deterministic greedy和RTX6000 Ada硬件。','',
        '| 方法 | net-weighted mean (%) | design-macro mean (%) | P95 (%) | worst-design mean (%) |',
        '|---|---:|---:|---:|---:|']
    for row in statistics['main']:
        lines.append('| '+row['method']+' | '+' | '.join(f"{100*row[k]:.6f}" for k in
            ('net_weighted_mean','design_macro_mean','p95','worst_design_mean'))+' |')
    lines += ['', '表中先对每个训练seed计算指标，再平均三个seed；P95和worst-design不通过混合seed预测计算。各seed、design和degree完整数值见CSV。',
        '这些是冻结模型的外部测试描述统计；置信区间和独立攻击属于阶段8，尚未执行。可称模型零样本迁移，不能称研究者完全未见的blind test。', '',
        '## 验收与来源', '',
        '- 95个源文件hash及全部22498个源net共享归一化重建，与阶段1冻结数组逐项一致。',
        '- 全部参考长度以2^27精确尺度求解，逐解检查optimal、连通性、整数边长；2^28复算22498项精确一致。正式缓存构建共44996次GeoSteiner调用，恢复smoke的512次包含在该缓存轨迹内。',
        '- 评价smoke：513个net连续与分段恢复的全部预测字段精确相同；五degree显存实测已保存。smoke额外Actor/REST各2050次，独立Python拓扑检查1537条。',
        '- 新增9项聚合/拓扑/恢复/锁/中断账目测试及19项原协议、归一化、ISPD回归通过；日志为revision experiments/logs/stage7/{unit_tests_correct_cwd,shared_input_regression}.log。最初测试入口cwd错误的失败日志保留，修正入口后未修改测试断言。',
        '- 75模型共337470条正式预测；独立C evaluator重算及Python线段长度/全树连通性均通过。2070行表格、20份逐net配对差值、3个图文件和全部冻结模型hash再次核验。',
        '- 每个预测保留design/net ID、source index、模型hash、reference缓存绑定、拓扑、长度及regret；每批原子分片并保存游标。失败/重放的调用attempts独立记账，不混入有效推理耗时。',
        '- 首次准备使用旧benchmark根路径而失败；改为已有precision_v2只读legacy链接后源hash一致，未改数据或研究设置。失败日志为revision experiments/logs/stage7_prepare_front.log。', '',
        '## 产物与恢复', '',
        f'- 完成索引：`{ROOT}/manifests/stage7_completion_v1.json`。',
        f'- 独立交付复核：`{ROOT}/manifests/stage7_delivery_verification_v1.json`。',
        f'- 原始预测、审计、任务清单、PID/锁、心跳、精确启动命令：`{LOG}`。',
        f'- 表格：`{EXP}/tables/stage7`；图：`{EXP}/figures/stage7_design_mean.*`。',
        f'- 逻辑产物根：`{EXP}`；resolved：`{EXP.resolve()}`。',
        f'- 监控：`{shlex.join(command("status"))}`。',
        f'- 如有中断，恢复：`{shlex.join(command("launch"))}`；启动前重新动态预检，活动会话或锁会拒绝重复执行。',
        '训练没有启动，latest/best、optimizer/scheduler与RNG由原冻结checkpoint只读保留。评价使用确定性greedy与原子批游标恢复。', '',
        '阶段8–9保持未启动；本阶段不根据ISPD结果重新选模或调参。','']
    tmp=path.with_suffix('.md.partial');tmp.write_text('\n'.join(lines));os.replace(tmp,path)
    progress('7.1–7.3全部通过：75模型、337470预测、独立审计及完整表格交付', 'DONE',
             str(ROOT/'manifests/stage7_delivery_verification_v1.json'), '本次阶段7完成；阶段8–9未启动')
    progress_path=TOP/'01_WDRO-REST_项目进度.md'
    text=progress_path.read_text()
    text=text.replace('| 总体阶段 | 阶段7 `DOING`（ISPD协议A外部测试） |',
                      '| 总体阶段 | 阶段7 `DONE`（precision_v2 ISPD协议A外部测试验收通过） |')
    text=text.replace('| 批量实验阻塞 | 以阶段7子步骤门禁与心跳为准 |', '| 批量实验阻塞 | 阶段7无剩余阻塞 |')
    text=text.replace('| 7. ISPD 外部测试 | `DOING` |','| 7. ISPD 外部测试 | `DONE` |')
    section='### 阶段 7 ISPD协议A外部测试\n\n- `DONE` 75个precision_v2冻结模型、20design/22498有效net/337470预测、双尺度reference缓存、独立拓扑审计和2070行统计表全部通过。\n- 证据：[阶段7验收报告](阶段7_ISPD外部测试验收报告.md)及revision下`wdro_rest_min/manifests/stage7_completion_v1.json`、`stage7_delivery_verification_v1.json`。\n- 所有结果只作外部测试，无ISPD训练或调参；阶段8–9未启动。\n\n'
    text=text.replace('## 4. 阶段状态\n', section+'## 4. 阶段状态\n',1)
    text=text.replace('本次无剩余阶段6操作，阶段7–9保持未启动。','阶段6无需恢复；阶段7后续已按本次授权完成，见下条。')
    text=text.replace('除非用户改变优先级，新聊天应从第 1 条尚未完成的操作继续，不重新设计实验范围。',
        '8. `DONE` 阶段7 ISPD外部测试及独立交付完成，见[阶段7验收报告](阶段7_ISPD外部测试验收报告.md)。本次止于阶段7。\n\n除非用户改变优先级，新聊天应从第 1 条尚未完成的操作继续，不重新设计实验范围。',1)
    tmp=progress_path.with_suffix('.md.partial');tmp.write_text(text);os.replace(tmp,progress_path)
