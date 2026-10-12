"""Scientific acceptance: descriptive statistics from saved experiment outputs.

No retraining, replay, file hashing, or engineering test suite is performed.
"""
import csv
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
OUT = HERE / 'acceptance'
GROUPS = ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster']

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def main():
    OUT.mkdir(exist_ok=True)
    p = read(HERE / 'protocol.json')
    rows, dynamics, missing, complete = [], [], [], []
    arrays = {}
    for family, arms in p['families'].items():
        for arm in arms:
            for seed in p['seeds']:
                run = f'{family}_{arm}_s{seed}'; d = ART / run
                required = ['complete.json', 'results.json', 'train.jsonl', 'validation.jsonl', 'test_best.npz', 'test_final.npz']
                absent = [f for f in required if not (d / f).exists()]
                if absent:
                    missing.append({'run': run, 'missing': absent}); continue
                result = read(d / 'results.json'); train = lines(d / 'train.jsonl'); val = lines(d / 'validation.jsonl')
                complete.append(run)
                selected = np.concatenate([t['selected_columns'] for t in train])
                dynamics.append({'run': run, 'family': family, 'arm': arm, 'seed': seed, 'updates': len(train),
                                 'center_fraction': float(np.mean(selected == 0)), 'center_count': int(np.sum(selected == 0)), 'selection_count': len(selected),
                                 'lambda_max': max(t['lambda'] for t in train), 'lambda_final': train[-1]['lambda'],
                                 'sampled_critic_nonzero_mean_updates': sum(t['critic_mean'] != 0 for t in train),
                                 'validation_critic_max_mean': max(g['critic']['mean'] for v in val for g in v['summary'].values()) if family == 'rest' else None,
                                 'training_seconds': sum(t['seconds'] for t in train), 'reference_seconds': sum(t['reference_seconds'] for t in train)})
                for checkpoint in ['best', 'final']:
                    with np.load(d / f'test_{checkpoint}.npz', allow_pickle=False) as loaded:
                        a = {k: loaded[k] for k in loaded.files}
                    arrays[(family, arm, seed, checkpoint)] = a
                    costs = np.concatenate([a[g + '__cost'] for g in GROUPS])
                    row = {'run': run, 'family': family, 'arm': arm, 'seed': seed, 'checkpoint': checkpoint,
                           'step': result['best']['step'] if checkpoint == 'best' else p['updates'][family], 'count': len(costs),
                           'mean': float(costs.mean()), 'p95_pooled_groups': float(np.quantile(costs, .95)),
                           'worst_ood_mean': max(float(a[g + '__cost'].mean()) for g in GROUPS[1:]),
                           'training_seconds': sum(t['seconds'] for t in train)}
                    for g in GROUPS: row[g] = float(a[g + '__cost'].mean())
                    keys = ['length', 'critic', 'predicted_reference_relative_error'] if family == 'rest' else ['wirelength', 'max_path'] if family == 'timing' else ['delay_scaled', 'length', 'buffers', 'skew_scaled']
                    for key in keys:
                        v = np.concatenate([a[g + '__' + key] for g in GROUPS]); row[key + '_mean'] = float(v.mean())
                        if key == 'critic': row.update(critic_zero_count=int(np.sum(v == 0)), critic_max=float(v.max()))
                    rows.append(row)
    aggregate = []
    contrasts = []
    for family, arms in p['families'].items():
        for checkpoint in ['best', 'final']:
            for arm in arms:
                subset = sorted([r for r in rows if (r['family'], r['arm'], r['checkpoint']) == (family, arm, checkpoint)], key=lambda r: r['seed'])
                if not subset: continue
                aggregate.append({'family': family, 'arm': arm, 'checkpoint': checkpoint, 'n_seeds': len(subset),
                                  'mean': float(np.mean([r['mean'] for r in subset])), 'seed_sd': float(np.std([r['mean'] for r in subset], ddof=1)) if len(subset) > 1 else None,
                                  'seed_values': [r['mean'] for r in subset], 'selected_steps': [r['step'] for r in subset],
                                  'worst_ood_mean': float(np.mean([r['worst_ood_mean'] for r in subset])),
                                  'training_seconds_mean': float(np.mean([r['training_seconds'] for r in subset]))})
            for arm in arms[1:]:
                base = {r['seed']: r for r in rows if (r['family'], r['arm'], r['checkpoint']) == (family, arms[0], checkpoint)}
                alt = {r['seed']: r for r in rows if (r['family'], r['arm'], r['checkpoint']) == (family, arm, checkpoint)}
                seeds = sorted(set(base) & set(alt)); delta = []
                for seed in seeds:
                    a = arrays[(family, arm, seed, checkpoint)]; b = arrays[(family, arms[0], seed, checkpoint)]
                    # Matched IDs are needed to define a paired scientific comparison.
                    if any(not np.array_equal(a[g + '__ids'], b[g + '__ids']) for g in GROUPS):
                        raise ValueError(f'Cannot form paired comparison for {family}/{arm}/{seed}')
                    delta.append(float(np.concatenate([a[g + '__cost'] - b[g + '__cost'] for g in GROUPS]).mean()))
                if not seeds: continue
                delta = np.asarray(delta); control_mean = float(np.mean([base[s]['mean'] for s in seeds])); alt_mean = float(np.mean([alt[s]['mean'] for s in seeds]))
                rng = np.random.default_rng(261011); bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
                contrasts.append({'family': family, 'arm': arm, 'control': arms[0], 'checkpoint': checkpoint, 'seeds': seeds,
                                  'control_mean': control_mean, 'alternative_mean': alt_mean, 'mean_delta': float(delta.mean()),
                                  'relative_change_percent': 100 * (alt_mean / control_mean - 1), 'delta_each': delta.tolist(),
                                  'wins': int(np.sum(delta < 0)), 'ties': int(np.sum(delta == 0)), 'losses': int(np.sum(delta > 0)),
                                  'descriptive_seed_bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist(),
                                  'training_time_ratio': float(np.mean([alt[s]['training_seconds'] for s in seeds]) / np.mean([base[s]['training_seconds'] for s in seeds]))})
    result = {'completed': len(complete), 'expected': 21, 'missing': missing, 'training_updates': sum(d['updates'] for d in dynamics),
              'training_examples': sum(d['updates'] * p['batch_size'] for d in dynamics), 'evaluation_rows': sum(r['count'] for r in rows),
              'aggregate': aggregate, 'contrasts': contrasts, 'dynamics': dynamics, 'per_run': rows}
    (OUT / 'statistics.json').write_text(json.dumps(result, indent=2) + '\n')
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (OUT / 'per_run.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    def agg(family, arm, checkpoint): return next(r for r in aggregate if (r['family'], r['arm'], r['checkpoint']) == (family, arm, checkpoint))
    def con(family, arm, checkpoint): return next(r for r in contrasts if (r['family'], r['arm'], r['checkpoint']) == (family, arm, checkpoint))
    report = ['# Critic 与 23/25ICCAD K3 实验验收', '',
              '本轮21/21项已完成，最终无失败或未完成项。degree10、batch128、每方法3个训练种子；基础WDRO每种子128步，两篇论文每种子256步。共4224次更新、540672个训练样本位置。', '',
              '每个checkpoint在四种分布各1024个net上评估，保存best与final共42份输出、172032条模型评估记录。这些包含同一net在多模型下的重复评估，独立训练重复数仍为3。best仅由真实validation最差OOD组均值选出，包含update0。', '',
              '**结论：本次Critic换算未展示有效预测能力；23ICCAD的K3基本持平；25ICCAD的K3出现约1%的成本改善，但耗时明显增加，且三种子的best结果不一致。**', '',
              '## 1. 真实测试结果', '',
              '表内为三个种子均值。REST指标为相对GeoSteiner误差（%）；Timing和Buffer为各自固定尺度物理成本，均越低越好，不能跨任务比较数值。', '',
              '| 任务 | 方法 | validation-selected best | final | best所选更新（101/102/103） | 单种子平均训练秒数 |', '|---|---|---:|---:|---|---:|']
    for family, arms in p['families'].items():
        for arm in arms:
            b, f = agg(family, arm, 'best'), agg(family, arm, 'final'); scale = 100 if family == 'rest' else 1; suffix = '%' if family == 'rest' else ''
            report.append(f"| {family} | {arm} | {b['mean']*scale:.6f}{suffix} | {f['mean']*scale:.6f}{suffix} | {b['selected_steps']} | {b['training_seconds_mean']:.2f} |")
    report += ['', '## 2. Critic换算：计算完成，但预测作用退化', '',
               '最终三个Critic分支的12288个测试预测全部为0。三个父模型在update0的四组validation上也全部输出0；该现象不是只能在本次训练结束后才观察到。所有保存的validation检查点同样如此。训练模式下部分批次出现过非零预测，不能说所有训练前向恒为0。', '',
               '因此，在已观测的最终测试集上，L_hat*=L_greedy/(1+critic)=L_greedy，所谓参考预测只是当前Actor自身线长。代码输出层为ReLU，这与输出被截到0相容；本轮未定位其为何成为0，不能把猜测写成根因。', '']
    for d in dynamics:
        if d['arm'] == 'critic_inverse': report.append(f"- 种子{d['seed']}：{d['selection_count']}次训练候选选择中，{d['center_count']}次选原中心（{100*d['center_fraction']:.4f}%）；lambda最大值{d['lambda_max']:.6g}。")
    c = con('rest', 'critic_inverse', 'final'); b = agg('rest', 'critic_inverse', 'best'); f = agg('rest', 'critic_inverse', 'final')
    report += ['', f"Critic分支best三个种子都选择update0，真实平均误差为{100*b['mean']:.6f}%；继续训练到final后为{100*f['mean']:.6f}%，比初始增加{100*(f['mean']-b['mean']):.6f}个百分点。",
               '', f"final相对Geo对照低{100*abs(c['mean_delta']):.6f}个百分点，三个种子都较低；但Geo对照也从初始变差，因此这是较少退化，并非训练优于初始。best与Geo完全相同，也不能作为等效替代证据，因为双方都退回未更新模型。",
               '', f"训练耗时相对Geo减少{100*(1-c['training_time_ratio']):.2f}%，但这一成本差对应几乎不进行对抗样本选择的退化机制。预测参考的平均相对偏差{100*f['mean']:.6f}%仅反映Actor原有误差，不能解释成Critic预测准确。",
               '', '基础K3在本轮best平均误差略差于Geo，final均值较低但仅2/3种子改善；并未建立稳定收益。', '',
               '## 3. 23ICCAD：K3基本持平', '']
    for cp in ['best', 'final']:
        c = con('timing', 'k3', cp)
        report.append(f"- {cp}：原方法{c['control_mean']:.8f}，K3为{c['alternative_mean']:.8f}，相对变化{c['relative_change_percent']:.4f}%；{c['wins']}胜/{c['ties']}平/{c['losses']}负。")
    c = con('timing', 'k3', 'best')
    report += ['', f"K3训练耗时为原方法的{c['training_time_ratio']:.2f}倍。种子102两种方法均选择update0，best/final真实输出成本相同；K3种子103也选择update0。现有数据不支持23ICCAD上checkpoint-bank带来稳定收益。", '',
               '## 4. 25ICCAD：有小幅成本收益，速度更慢', '']
    for cp in ['best', 'final']:
        c = con('buffer', 'k3', cp)
        report.append(f"- {cp}：原方法{c['control_mean']:.8f}，K3为{c['alternative_mean']:.8f}，相对变化{c['relative_change_percent']:.4f}%；{c['wins']}胜/{c['ties']}平/{c['losses']}负。")
    c = con('buffer', 'k3', 'best')
    report += ['', f"K3训练耗时为原方法的{c['training_time_ratio']:.2f}倍。按预设best口径，种子103反而变差，种子bootstrap区间包含0；final三个种子均改善，这是值得保留的探索性信号，不是稳定泛化优势的证明。", '',
               '物理分量（每格为三个种子、四组数据等权均值）：', '',
               '| checkpoint | 方法 | delay × 1e18 | 平均缓冲数 | 归一化线长 |', '|---|---|---:|---:|---:|']
    for cp in ['best', 'final']:
        for arm in ['original', 'k3']:
            subset = [r for r in rows if (r['family'], r['arm'], r['checkpoint']) == ('buffer', arm, cp)]
            values = [np.mean([r[k + '_mean'] for r in subset]) for k in ['delay_scaled', 'buffers', 'length']]
            report.append(f"| {cp} | {arm} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    report += ['', 'delay数值单位为1e-18秒，来自固定RC和归一化坐标的组件模型，不能外推为真实芯片延迟收益。综合成本降低不保证线长或缓冲数每一项都降低。', '',
               '## 5. 配对差值与不确定性', '',
               '差值=改动方法−对照，负值较好。REST用百分点，论文用成本原单位。区间为三个配对种子均值的10000次bootstrap 95%描述性区间；n=3很小，未作多重比较校正，也不将记录条数视为独立重复。', '',
               '| 任务/改动 | checkpoint | 三种子差值 | 平均差值 | 描述性区间 |', '|---|---|---|---:|---|']
    for c in contrasts:
        scale = 100 if c['family'] == 'rest' else 1
        report.append(f"| {c['family']}/{c['arm']} | {c['checkpoint']} | {', '.join(f'{v*scale:.7f}' for v in c['delta_each'])} | {c['mean_delta']*scale:.7f} | [{c['descriptive_seed_bootstrap95'][0]*scale:.7f}, {c['descriptive_seed_bootstrap95'][1]*scale:.7f}] |")
    report += ['', '## 6. 能支持的结论范围', '',
               '- 这轮支持把最直接的Critic代数换算判为未达到“有效预测替代”的目标；不能据此否定重新训练且校准良好的value predictor。',
               '- 两篇论文原方法均不使用GeoSteiner；K3增加三个教师的实际物理评分，还将J改成J/B_J−1，改变了目标归一化和lambda有效尺度，无法将变化单独归因于教师质量。',
               '- 25ICCAD沿用categorical-phase与local-greedy buffering组件，并非论文完整Shi–Li DP或工业流程复现。本轮仅为degree10短程继续训练。',
               '- 训练耗时计入样本生成、候选搜索、参考计算和优化，排除周期验证及最终测试；教师历史训练成本没有摊销。没有开展等总时间对照，不能宣称K3具有计算效率优势。', '',
               '数据入口：[统计JSON](statistics.json)、[各模型数据表](per_run.csv)、[原始产物](../artifacts/)。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'completed': len(complete), 'updates': result['training_updates'], 'examples': result['training_examples'], 'evaluation_rows': result['evaluation_rows'], 'aggregate': aggregate, 'contrasts': contrasts}, indent=2))

if __name__ == '__main__': main()
