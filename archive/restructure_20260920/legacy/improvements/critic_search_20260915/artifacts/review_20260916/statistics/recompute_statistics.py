"""Independent CPU-only statistical reacceptance from immutable NPZ artifacts.

No project statistical/aggregation/bootstrap function is imported. Source NPZ,
update JSON, and declared protocol are the numerical inputs; published CSVs are
used only as comparison targets. This does not rerun Actor or GeoSteiner.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path('/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915')
ART = ROOT/'artifacts'
OUT = ART/'review_20260916/statistics'
SEEDS = (101, 102, 103)
DISTS = ('uniform', 'mix_gauss', 'edge_bias', 'grid_cluster')
VARIANTS = ('refresh32_raw', 'refresh8_raw', 'refresh32_ema', 'refresh8_ema')
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260915
FAMILY = 9
ATOL = 1e-12
VAL_METHODS = ['baseline', 'no_move', 'random_direction'] + [
    f'step_{a:g}_{t}' for t in (4, 8, 16) for a in (.0025, .005, .01)] + [
    'decay16', 'adaptive16', 'adam16', 'mixed_scales', 'structured_starts',
    'wide126_short4', 'narrow21_long24', 'trajectory21_8', 'trajectory_diverse',
    'differentiable_normalization', 'remove_rigid']
FAMILIES = {
    'direction3': [m for m in VAL_METHODS if m.startswith('step_')] + ['decay16', 'adaptive16', 'adam16'],
    'direction4': ['mixed_scales', 'structured_starts', 'wide126_short4', 'narrow21_long24'],
    'direction5': ['trajectory21_8', 'narrow21_long24', 'trajectory_diverse'],
    'direction6': ['differentiable_normalization', 'remove_rigid', 'adaptive16'],
}
INPUT_FILES = {}
CHECKS = {}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def observe(path):
    path = Path(path)
    INPUT_FILES[str(path)] = file_sha(path)


def readj(path):
    observe(path)
    return json.loads(Path(path).read_text())


def readz(path):
    observe(path)
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def require(condition, name):
    CHECKS[name] = CHECKS.get(name, 0) + 1
    if not condition: raise AssertionError(name)


def numerical(a, b, name):
    require(np.allclose(a, b, rtol=0, atol=ATOL), name)


def array_identity(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()


def hungarian(centers, candidates):
    answer = np.empty(candidates.shape[:2])
    for i, (x, zs) in enumerate(zip(centers, candidates)):
        x = x.astype(np.float64)
        for j, z in enumerate(zs.astype(np.float64)):
            distances = np.abs(x[:, None]-z[None]).sum(-1)
            rows, columns = linear_sum_assignment(distances)
            answer[i, j] = np.mean(distances[rows, columns])
    return answer


def quantiles(values):
    return float(np.mean(values)), float(np.quantile(values, .95)), float(np.max(values))


def bootstrap(cube):
    """[fixed model seed, distribution, shared center] paired sampling.

    Averaging over the fixed models before drawing centers is algebraically
    equivalent to applying the identical sampled center indices to all models.
    Each distribution is independently resampled, with equal macro weights.
    """
    per_center = np.asarray(cube, dtype=np.float64).mean(axis=0)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    replicate_by_dist = []
    for center_values in per_center:
        draw = generator.integers(0, center_values.size,
                                  size=(BOOTSTRAP_REPLICATES, center_values.size))
        replicate_by_dist.append(center_values[draw].mean(axis=1))
    replicated = np.stack(replicate_by_dist, axis=1)
    macro = replicated.mean(axis=1)
    return {
        'mean': float(cube.mean()),
        'ci95': np.quantile(macro, [.025, .975]).tolist(),
        'bonferroni': np.quantile(macro, [.05/(2*FAMILY), 1-.05/(2*FAMILY)]).tolist(),
        'per_seed': cube.mean(axis=(1, 2)).tolist(),
        'per_dist': cube.mean(axis=(0, 2)).tolist(),
        'per_dist_ci': np.quantile(replicated, [.025, .975], axis=0).T.tolist(),
    }


def csv_rows(path):
    observe(path)
    with Path(path).open() as f: return list(csv.DictReader(f))


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def compare(name, computed, keys):
    old_rows = csv_rows(ART/'tables'/name)
    existing = {tuple(str(r[k]) for k in keys): r for r in old_rows}
    calculated = {tuple(str(r[k]) for k in keys): r for r in computed}
    require(len(existing) == len(old_rows), name+' original unique keys')
    require(len(calculated) == len(computed), name+' recomputed unique keys')
    require(set(existing) == set(calculated), name+' identical row coverage')
    exact = approximate = numeric_count = text_count = 0
    largest = 0.; mismatches = []
    for key, row in calculated.items():
        old = existing[key]
        for field, value in row.items():
            require(field in old, name+' target field exists')
            if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, (bool, np.bool_)):
                target = float(old[field]); actual = float(value); numeric_count += 1
                difference = abs(actual-target); largest = max(largest, difference)
                if actual == target: exact += 1
                elif difference <= ATOL: approximate += 1
                else: mismatches.append(dict(key=key, field=field, actual=actual, published=target))
            else:
                text_count += 1
                expected = '' if value is None else str(value)
                if old[field] != expected: mismatches.append(dict(key=key, field=field, actual=expected, published=old[field]))
    write_csv('recomputed_'+name, computed)
    require(not mismatches, name+' all recomputed values match')
    return dict(file=name, rows=len(computed), numeric_cells=numeric_count, exactly_equal_numeric_cells=exact,
                roundoff_only_cells=approximate, text_cells=text_count, maximum_absolute_difference=largest,
                tolerance=ATOL, mismatches=mismatches, status='pass')


def fixed_statistics(protocol):
    metrics, result, center_hashes, val_means = [], {}, {}, {}
    for phase, seed in [('validation', 101)]+[('test', s) for s in SEEDS]:
        methods = VAL_METHODS if phase == 'validation' else protocol['test']['methods']
        count = 32 if phase == 'validation' else 64
        task = ART/'fixed'/f'{phase}_s{seed}'
        for dist in DISTS:
            nominal = None
            for method in methods:
                score_parts, regret_parts, cost_parts, center_parts, orig_parts, allscore_parts = [], [], [], [], [], []
                diagnostics, elapsed = [], []
                for offset in range(0, count, 16):
                    a = readz(task/'evaluation'/f'{dist}_{method}_{offset:04d}.npz')
                    allregret = a['all_length']/a['all_lstar']-1
                    numerical(allregret, a['all_regret'], 'fixed actual regret identity')
                    allscore = allregret-a['price'][:, None]*a['all_cost']
                    numerical(allscore, a['all_score'], 'fixed penalty identity')
                    index = np.argmax(allscore, axis=1)
                    require(np.array_equal(index, a['selected_index']), 'fixed earliest argmax')
                    rows = np.arange(len(index))
                    cost = hungarian(a['centers'], a['support'][rows, index][:, None])[:, 0]
                    numerical(cost, a['cost'], 'fixed selected Hungarian recomputation')
                    regret, score = allregret[rows, index], allscore[rows, index]
                    numerical(regret, a['regret'], 'fixed selected regret')
                    numerical(score, a['score'], 'fixed selected score')
                    require(np.all(a['calls'] == 64), 'fixed score budget64')
                    score_parts.append(score); regret_parts.append(regret); cost_parts.append(cost)
                    orig_parts.append(allregret[:, 0]); allscore_parts.append(allscore); center_parts.append(a['centers'])
                    elapsed.append(float(a['elapsed_seconds']))
                    diagnostics.append(json.loads(str(a['diagnostics_json'])))
                score, regret, cost, original, centers, allscore = [np.concatenate(parts) for parts in
                    (score_parts, regret_parts, cost_parts, orig_parts, center_parts, allscore_parts)]
                center_hashes[phase, seed, dist, method] = array_identity(centers)
                if nominal is None: nominal = original
                require(np.array_equal(original, nominal), 'fixed same nominal outcome across methods')
                if phase == 'test': result[seed, dist, method] = dict(score=score, allscore=allscore)
                if phase == 'validation': val_means[dist, method] = float(score.mean())
                diagnostic_sum = lambda key: sum(d[key] for d in diagnostics)
                score_mean, score95, _ = quantiles(score)
                regret_mean, regret95, maximum = quantiles(regret)
                cost_mean, cost95, _ = quantiles(cost)
                metrics.append(dict(phase=phase, seed=seed, distribution=dist, method=method, centers=count,
                    mean_score=score_mean, p95_score=score95, mean_regret=regret_mean,
                    p95_regret=regret95, maximum_regret=maximum, mean_hungarian_cost=cost_mean,
                    p95_hungarian_cost=cost95, mean_score_gain=float((score-original).mean()),
                    elapsed_seconds=sum(elapsed), seconds_per_center=sum(elapsed)/count,
                    gradient_forward_rows=diagnostic_sum('forward_samples'), gradient_backward_rows=diagnostic_sum('backward_samples'),
                    clipped_fraction=diagnostic_sum('clipped_coordinates')/max(1, diagnostic_sum('total_coordinates')),
                    movement_cosine=diagnostic_sum('cosine_sum')/max(1, diagnostic_sum('cosine_count')),
                    repairs=diagnostic_sum('repairs'), fallbacks=diagnostic_sum('fallbacks'),
                    duplicate_fraction=float(np.mean([d['duplicate_fraction'] for d in diagnostics]))))
        print(f'fixed {phase} seed {seed}: complete', flush=True)
    for dist in DISTS:
        target = center_hashes['test', 101, dist, 'baseline']
        for seed in SEEDS:
            for method in protocol['test']['methods']:
                require(center_hashes['test', seed, dist, method] == target, 'fixed test shared centers across all seeds/methods')
    macro = {m: float(np.mean([val_means[d, m] for d in DISTS])) for m in VAL_METHODS}
    winners = {family: max(names, key=lambda n: (macro[n], -VAL_METHODS.index(n))) for family, names in FAMILIES.items()}
    selection = readj(ART/'selection.json')
    require(winners == selection['selected'], 'independent validation winner selection')
    for method, value in macro.items(): numerical(value, selection['validation_macro_scores'][method], 'validation macro means')
    return metrics, result, winners


def training_statistics():
    metrics, dynamics, task_summaries = [], [], []
    outcomes, shared_geometry, train_geometry = {}, {}, {}
    for seed in SEEDS:
        for variant in VARIANTS:
            task = ART/'training'/f'{variant}_d30_s{seed}'
            identity = readj(task/'identity.json')
            settings = identity['settings']
            for family in ('validation', 'evaluation'):
                for dist in DISTS:
                    data = readz(task/'inputs'/f'{family}__{dist}.npz')
                    shared_geometry[seed, variant, family, dist] = array_identity(data['pins'])
            for checkpoint, prefix in [('initial', 'initial'), ('best', 'final'), ('latest', 'last')]:
                for dist in DISTS:
                    chunks = [readz(p) for p in sorted((task/'evaluation'/dist).glob(prefix+'_*.npz'))]
                    require(len(chunks) == 4, 'four evaluation chunks')
                    for a in chunks:
                        numerical(a['regret'], a['length']/a['lstar']-1, 'training actual evaluation regret')
                        require(np.all(a['calls'] == 1), 'training nominal score budget1')
                    regrets = np.concatenate([a['length']/a['lstar']-1 for a in chunks])
                    centers = np.concatenate([a['centers'] for a in chunks])
                    require(array_identity(centers) == shared_geometry[seed, variant, 'evaluation', dist],
                            'training each checkpoint evaluated on declared shared test centers')
                    update = int(chunks[0]['update'])
                    require(all(int(a['update']) == update for a in chunks), 'training evaluation same checkpoint update')
                    if checkpoint == 'initial': require(update == 0, 'initial checkpoint update0')
                    if checkpoint == 'latest': require(update == 128, 'latest checkpoint update128')
                    mean, p95, maximum = quantiles(regrets)
                    metrics.append(dict(seed=seed, variant=variant, checkpoint=checkpoint, update=update,
                                        distribution=dist, count=len(regrets), mean_regret=mean,
                                        p95_regret=p95, maximum_regret=maximum))
                    outcomes[seed, variant, checkpoint, dist] = regrets
            costs, prices, mse, logs = [], [], [], []
            price = ema = None
            for step in range(1, 129):
                a = readz(task/'search'/f'u{step:06d}.npz')
                raw = readj(task/'updates'/f'u{step:06d}.json')
                centers = readz(task/'inputs'/f'train__u{step:06d}.npz')['pins']
                require(np.array_equal(centers, a['centers']), 'training search/input centers equal')
                train_geometry[seed, variant, step] = array_identity(centers)
                if price is None:
                    price = float(a['price']); prices.append(price)
                numerical(price, a['price'], 'independent price before matches search')
                allcost = hungarian(centers, a['evaluated_pins'])
                allscore = a['evaluated_regrets']-price*allcost
                selected = np.argmax(allscore, axis=1)
                require(np.array_equal(selected, a['selected_index']), 'training true-score argmax')
                mean_cost = float(allcost[np.arange(len(selected)), selected].mean())
                numerical(mean_cost, a['selected_cost'].mean(), 'training mean selected Hungarian cost')
                rho = float(raw['rho'])
                previous_price = price
                ema = mean_cost if ema is None else .9*ema+.1*mean_cost
                signal = ema if variant.endswith('_ema') else mean_cost
                price = max(0., price+(signal-rho))
                numerical(ema, raw['ema_cost'], 'independent EMA recurrence')
                numerical(price, raw['lambda_after'], 'independent lambda recurrence')
                require((raw['online'] is not None) == (step % (8 if variant.startswith('refresh8_') else 32) == 0), 'declared refresh interval')
                error = float(raw['surrogate_error']['mse'])
                dynamics.append(dict(seed=seed, variant=variant, update=step, mean_cost=mean_cost,
                    rho=rho, cost_minus_rho=mean_cost-rho, ema_cost=ema, dual_signal_cost=signal,
                    lambda_before=previous_price, lambda_after=price, surrogate_mse=error,
                    calibration_performed=raw['online'] is not None,
                    search_seconds=raw['search_seconds'], calibration_seconds=raw['calibration_seconds'],
                    policy_update_seconds=raw['policy_update_seconds']))
                costs.append(mean_cost); prices.append(price); mse.append(error); logs.append(raw)
            costs, prices, mse = np.asarray(costs), np.asarray(prices), np.asarray(mse)
            task_summaries.append(dict(task=f'training/{variant}_d30_s{seed}',
                mean_cost=float(costs.mean()), std_cost=float(costs.std()), rho=rho,
                mean_cost_minus_rho=float(costs.mean()-rho), mean_absolute_cost_minus_rho=float(np.abs(costs-rho).mean()),
                fraction_cost_above_rho=float(np.mean(costs>rho)), cost_total_variation=float(np.abs(np.diff(costs)).sum()),
                lambda_total_variation=float(np.abs(np.diff(prices)).sum()), mean_surrogate_mse=float(mse.mean()),
                total_search_seconds=sum(t['search_seconds'] for t in logs),
                total_calibration_seconds=sum(t['calibration_seconds'] for t in logs),
                total_policy_update_seconds=sum(t['policy_update_seconds'] for t in logs),
                calibration_updates=sum(t['online'] is not None for t in logs), committed_search_calls=8192,
                committed_sampled_calls=2048, last32_mean_cost=float(costs[-32:].mean()),
                last32_std_cost=float(costs[-32:].std()), last32_mean_surrogate_mse=float(mse[-32:].mean())))
            print(f'training {variant} seed {seed}: complete', flush=True)
    for family in ('validation', 'evaluation'):
        for dist in DISTS:
            target = shared_geometry[101, VARIANTS[0], family, dist]
            for seed in SEEDS:
                for variant in VARIANTS:
                    require(shared_geometry[seed, variant, family, dist] == target,
                            'training validation/test shared centers across all seeds/variants')
    for seed in SEEDS:
        for step in range(1, 129):
            target = train_geometry[seed, VARIANTS[0], step]
            for variant in VARIANTS:
                require(train_geometry[seed, variant, step] == target, 'same-seed all-variant shared training centers at every step')
        for dist in DISTS:
            target = outcomes[seed, VARIANTS[0], 'initial', dist]
            for variant in VARIANTS:
                require(np.array_equal(outcomes[seed, variant, 'initial', dist], target), 'same-seed initial results bitwise equal across variants')
    return metrics, dynamics, task_summaries, outcomes


def contrasts(fixed, trained):
    main_fixed = [('D1','region_mse','uniform_mse'), ('D2','region_rankdiff','region_mse')] + [
        (f'D{d}',f'direction{d}','baseline') for d in range(3,7)]
    secondary = [('D2_huber','region_huber','region_mse'), ('no_move','no_move','baseline'),
                 ('random_direction','random_direction','baseline')]
    specifications = [(name,a,b,'higher',name.startswith('D') and name!='D2_huber','fixed') for name,a,b in main_fixed+secondary]
    specifications += [('D7_'+variant,variant,VARIANTS[0],'lower',True,'training') for variant in VARIANTS[1:]]
    rows, dist_rows, seed_rows = [], [], []
    for name, left, right, better, primary, kind in specifications:
        if kind == 'fixed':
            cube = np.stack([np.stack([fixed[s,d,left]['score']-fixed[s,d,right]['score'] for d in DISTS]) for s in SEEDS])
        else:
            cube = np.stack([np.stack([trained[s,left,'best',d]-trained[s,right,'best',d] for d in DISTS]) for s in SEEDS])
        result = bootstrap(cube)
        sign = 1 if better == 'higher' else -1
        all_improve = bool(np.all(sign*np.asarray(result['per_seed']) > 0))
        lower, upper = result['bonferroni']
        interval_improves = lower>0 if sign==1 else upper<0
        interval_worse = upper<0 if sign==1 else lower>0
        all_worse = bool(np.all(sign*np.asarray(result['per_seed']) < 0))
        conclusion = ('simultaneous_supported_improvement' if primary and interval_improves and all_improve else
                      'simultaneous_supported_degradation' if primary and interval_worse and all_worse else 'inconclusive')
        rows.append(dict(contrast=name, treatment=left, control=right,
            metric='macro_penalized_score' if sign==1 else 'macro_regret_best_checkpoint', better=better, primary=primary,
            mean_delta=result['mean'], ci95_low=result['ci95'][0], ci95_high=result['ci95'][1],
            simultaneous_low=lower if primary else None, simultaneous_high=upper if primary else None,
            all_seed_improve=all_improve, conclusion=conclusion))
        for index, dist in enumerate(DISTS):
            dist_rows.append(dict(contrast=name, distribution=dist, better=better,
                mean_delta=result['per_dist'][index], ci95_low=result['per_dist_ci'][index][0],
                ci95_high=result['per_dist_ci'][index][1], simultaneous_correction=False))
        for seed, delta in zip(SEEDS, result['per_seed']):
            seed_rows.append(dict(contrast=name, seed=seed, mean_delta=delta, better=better, improves=bool(sign*delta>0)))
    require(sum(r['primary'] for r in rows) == 9, 'exact nine primary family')
    return rows, dist_rows, seed_rows


def trajectories(fixed):
    summaries, per_center = [], []
    for control in ('no_move', 'random_direction'):
        difference = np.stack([np.stack([fixed[s,d,'baseline']['allscore'][:,1:]-fixed[s,d,control]['allscore'][:,1:]
                                        for d in DISTS]) for s in SEEDS])
        best = np.stack([np.stack([fixed[s,d,'baseline']['score']-fixed[s,d,control]['score'] for d in DISTS]) for s in SEEDS])
        mean_candidate, fraction = difference.mean(axis=-1), (difference>0).mean(axis=-1)
        for metric, cube in [('mean_trajectory_score_delta',mean_candidate), ('fraction_positive_trajectories',fraction), ('best64_score_delta',best)]:
            result = bootstrap(cube)
            summaries.append(dict(treatment='baseline', control=control, metric=metric, estimate=result['mean'],
                ci95_low=result['ci95'][0], ci95_high=result['ci95'][1], seed101=result['per_seed'][0],
                seed102=result['per_seed'][1], seed103=result['per_seed'][2],
                statistical_unit='center_after_63_trajectory_aggregation', simultaneous_correction=False))
        for i, seed in enumerate(SEEDS):
            for j, dist in enumerate(DISTS):
                for k in range(64):
                    per_center.append(dict(seed=seed, distribution=dist, center_index=k, control=control,
                        trajectory_count=63, mean_trajectory_score_delta=float(mean_candidate[i,j,k]),
                        fraction_positive_trajectories=float(fraction[i,j,k]), best64_score_delta=float(best[i,j,k])))
    return summaries, per_center


def report_text(review, rows, trajectory_rows, task_summaries):
    primary = [r for r in rows if r['primary']]
    lines = ['# 2026-09-16 独立统计复核', '',
        '**结论：独立复算通过。旧报告与CSV未被修改。** 本脚本没有导入旧report.py的任何统计函数，只读取已保存NPZ、逐步JSON与冻结协议，使用NumPy/SciPy独立实现。没有训练、Actor推理或GeoSteiner新调用。', '',
        f"复核 {sum(t['rows'] for t in review['tables'])} 行、{sum(t['numeric_cells'] for t in review['tables'])} 个数值字段；其中 {sum(t['exactly_equal_numeric_cells'] for t in review['tables'])} 个数值字段以Python浮点比较完全相等，{sum(t['roundoff_only_cells'] for t in review['tables'])} 个仅有≤1e-12舍入差异。最大绝对差 {max(t['maximum_absolute_difference'] for t in review['tables']):.3g}。", '',
        '## 主要方向结论', '',
        '| 对比 | 宏均值差 | 95% CI | Bonferroni 9项同时区间 | 三seed均改善 | 复核结论 |',
        '|---|---:|---|---|---|---|']
    for r in primary:
        label = '支持改善' if r['conclusion']=='simultaneous_supported_improvement' else '证据不足'
        lines.append(f"| {r['contrast']} | {r['mean_delta']:.9g} | [{r['ci95_low']:.9g}, {r['ci95_high']:.9g}] | [{r['simultaneous_low']:.9g}, {r['simultaneous_high']:.9g}] | {'是' if r['all_seed_improve'] else '否'} | {label} |")
    lines += ['', 'D1：区域数据相对uniform刷新没有稳定收益。D2：局部排序/差分相对MSE在点态95%区间下呈退化，但9项校正区间跨0，不能按主要判据宣布稳定退化。',
        'D3：Adam 16步在本轮固定模型和64次真实评分预算下支持改善，三个seed均同向。它使用更多梯度步骤，不能据此宣称相同总计算下更高效。',
        'D4与D5独立validation选择了同一个narrow21_long24配置，因此两行完全相同；二者不是两项独立重复证据，Bonferroni仍按预先保留的9项处理。',
        'D6：去刚体方向的三个seed均改善，但多重校正区间跨0。D7：三种刷新/EMA设置均未建立稳定训练收益，置信区间跨0不证明方法等价。', '',
        '## 梯度方向诊断', '', '| baseline−对照 | 中心内统计量 | 均值 | 点态95% CI |', '|---|---|---:|---|']
    for r in trajectory_rows:
        lines.append(f"| {r['control']} | {r['metric']} | {r['estimate']:.9g} | [{r['ci95_low']:.9g}, {r['ci95_high']:.9g}] |")
    lines += ['', '原Critic梯度相对不移动的平均轨迹分数下降，正向轨迹比例不足50%；相对匹配尺度随机方向则有正收益。best-of64收益与多数轨迹的方向质量是不同问题。每中心先聚合63条轨迹，再按中心重采样，没有把轨迹伪装成独立net。', '',
        '## 训练与配对审查', '',
        '所有initial/best/latest的每分布mean、P95、maximum由原始length/Lstar重算。128步真实Hungarian成本、EMA和投影λ递推独立计算，校准间隔以及成本波动、总变差、最后32步统计均与已发布值一致。',
        '已显式确认：固定test所有seed/方法共享同分布中心；训练validation/test跨seed/variant共享中心；每个seed的四个variant逐步共享训练中心；同seed四分支初始test结果逐元素完全相等。',
        '代理MSE、时间等观测值从原始逐步日志读取后重新聚合；未重新运行代理预测或时钟测量。本复核不把日志聚合等同于重新执行模型。', '',
        '## 统计单位与结论边界', '',
        '样本量：degree30，3个固定父模型，4个分布；固定test每分布64中心，训练test每分布256中心。四分布等权、三模型等权。5,000次配对bootstrap使用seed20260915，在每个分布内重采样中心，同一次抽样索引应用于三个模型；分布之间独立抽样。',
        '宏均值9个主要对比采用α=0.05/9的Bonferroni校正，分布细表、Huber及方向诊断仅用点态95%区间。置信区间条件于已有3个模型，不覆盖重新训练的随机性；3个seed不能支持普适模型结论。极端分位数也有有限5,000次bootstrap的Monte Carlo误差。',
        '实验为有限候选搜索及128步继续训练，不是连续Wasserstein最坏风险证书；代理预测可微也不自动保证梯度对真实非光滑分数有效。', '',
        '## 复核产物', '',
        '- `recompute_statistics.py`：独立实现和复跑入口。',
        '- `review.json`：逐表比较计数、最大差、检查次数及独立结论。',
        '- `recomputed_*.csv`：本次独立重算结果。',
        '- `input_manifest.json`：实际读取的源文件及SHA256；`completion.json`绑定本次复核输出。', '']
    return '\n'.join(lines)


def main():
    began = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    protocol = readj(ROOT/'protocol.json')
    require(protocol['statistics']['bootstrap_replicates'] == BOOTSTRAP_REPLICATES, 'bootstrap frozen replicates')
    # A deterministic fixture proves that opposite fixed model trajectories cancel
    # before center resampling; models must not be independently bootstrapped.
    fixture = np.arange(32, dtype=float).reshape(4,8)
    check = bootstrap(np.stack((fixture, np.zeros_like(fixture), -fixture)))
    require(check['ci95'] == [0.,0.] and check['bonferroni'] == [0.,0.], 'shared-center bootstrap fixture')
    fixed_rows, fixed, winners = fixed_statistics(protocol)
    training_rows, dynamic_rows, task_rows, trained = training_statistics()
    contrast_rows, dist_rows, seed_rows = contrasts(fixed, trained)
    trajectory_rows, center_rows = trajectories(fixed)
    comparisons = [
        compare('fixed_metrics.csv', fixed_rows, ['phase','seed','distribution','method']),
        compare('training_metrics.csv', training_rows, ['seed','variant','checkpoint','distribution']),
        compare('training_dynamics.csv', dynamic_rows, ['seed','variant','update']),
        compare('contrasts.csv', contrast_rows, ['contrast']),
        compare('contrast_distributions.csv', dist_rows, ['contrast','distribution']),
        compare('contrast_seeds.csv', seed_rows, ['contrast','seed']),
        compare('trajectory_diagnostics.csv', trajectory_rows, ['control','metric']),
        compare('trajectory_centers.csv', center_rows, ['seed','distribution','center_index','control']),
    ]
    # tasks.csv contains four fixed tasks too; compare only the independently
    # reconstructed fields for its twelve training rows, preserving coverage.
    published_tasks = csv_rows(ART/'tables/tasks.csv')
    fixed_metadata_rows = [{k:r[k] for k in ('task',)} for r in published_tasks if r['kind']=='fixed']
    comparisons.append(compare('tasks.csv', fixed_metadata_rows+task_rows, ['task']))
    review = dict(status='pass', independence='no import of report.py or project statistics; NumPy/SciPy CPU-only recomputation',
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        primary_family=FAMILY, alpha=.05, numeric_tolerance=ATOL,
        input_scope='persisted NPZ and per-update JSON; no new model inference or reference scoring',
        tables=comparisons, checks=CHECKS, selected_validation_winners=winners,
        primary_contrasts=[r for r in contrast_rows if r['primary']], trajectory_diagnostics=trajectory_rows,
        elapsed_seconds=time.monotonic()-began, script_sha256=file_sha(__file__))
    (OUT/'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False))
    (OUT/'SUMMARY_ZH.md').write_text(report_text(review, contrast_rows, trajectory_rows, task_rows))
    (OUT/'input_manifest.json').write_text(json.dumps(dict(files=[dict(path=p,sha256=h) for p,h in sorted(INPUT_FILES.items())]), indent=2))
    outputs = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('completion.json','run.log'))
    (OUT/'completion.json').write_text(json.dumps(dict(status='pass',script_sha256=file_sha(__file__),
        outputs=[dict(path=str(p),sha256=file_sha(p)) for p in outputs],
        elapsed_seconds=review['elapsed_seconds']), ensure_ascii=False, indent=2))
    print(json.dumps(dict(status='pass', seconds=review['elapsed_seconds'], tables=len(comparisons),
                         numeric_cells=sum(r['numeric_cells'] for r in comparisons)), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
