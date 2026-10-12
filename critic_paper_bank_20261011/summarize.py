"""Produce scientific tables from completed jobs only, without selection on test."""
import csv, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'

def main():
    protocol = json.loads((HERE / 'protocol.json').read_text()); rows = []; details = {}
    for path in sorted(ART.glob('*/complete.json')):
        result = json.loads((path.parent / 'results.json').read_text())
        details[path.parent.name] = result
        for checkpoint in ['best', 'final']:
            groups = result['test_' + checkpoint]
            rows.append({'run': path.parent.name, 'family': result['family'], 'arm': result['arm'], 'seed': result['seed'], 'checkpoint': checkpoint,
                         'selected_step': result['best']['step'] if checkpoint == 'best' else protocol['updates'][result['family']],
                         'mean_cost': np.mean([v['cost']['mean'] for v in groups.values()]), 'worst_ood_mean': max(v['cost']['mean'] for k, v in groups.items() if k != 'uniform'),
                         'mean_group_p95': np.mean([v['cost']['p95'] for v in groups.values()]), 'training_seconds': result['training_seconds'], 'lambda_final': result['lambda_final']})
    if rows:
        tmp = ART / 'metrics.csv.partial'
        with tmp.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        os.replace(tmp, ART / 'metrics.csv')
    lines = ['# Critic 与论文 K3 小规模实验结果', '', '解释与结论见[实验验收报告](acceptance/REPORT.md)。该报告含Critic零输出与update0选择的限制，不能只凭下列代理误差判定替代成功。', '', f'已完成 {len(details)}/21 次训练。未完成项不计入均值。每组测试 1024 个 net，共四组；数值越低越好。', '',
             '基础 WDRO 报告真实 GeoSteiner 相对误差；Timing 报告固定尺度线长与最长路径加权成本；Buffer 报告固定 RC 延迟与缓冲数成本。不同任务之间不比较数值大小。', '',
             '| 任务 | 方法 | 种子 | checkpoint | 选择步数 | 四组平均成本 | OOD 最差组 | 训练秒数 |', '|---|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['family']} | {row['arm']} | {row['seed']} | {row['checkpoint']} | {row['selected_step']} | {row['mean_cost']:.8f} | {row['worst_ood_mean']:.8f} | {row['training_seconds']:.1f} |")
    lines += ['', '训练时间包含训练样本生成、搜索、参考计算和更新，排除周期验证与最终测试；K3 历史模型训练成本未摊销。', '', '## 配对差值', '', '下列差值为改动方法减对照，负值较好。种子数为 3 时的区间仅作描述，不支持广泛优越性结论。']
    contrasts = []
    for family, arms in protocol['families'].items():
        control = arms[0]
        for arm in arms[1:]:
            for checkpoint in ['best', 'final']:
                def selected(a): return {r['seed']: r for r in rows if r['family'] == family and r['arm'] == a and r['checkpoint'] == checkpoint}
                base, alt = selected(control), selected(arm); seeds = sorted(set(base) & set(alt))
                if not seeds: continue
                delta = np.array([alt[s]['mean_cost'] - base[s]['mean_cost'] for s in seeds]); ci = None
                if len(seeds) == 3:
                    rng = np.random.default_rng(261011); boot = delta[rng.integers(0, len(seeds), (10000, len(seeds)))].mean(1)
                    ci = np.quantile(boot, [.025, .975]).tolist()
                contrast = {'family': family, 'arm': arm, 'control': control, 'checkpoint': checkpoint, 'seeds': seeds, 'delta_each': delta.tolist(), 'mean_delta': float(delta.mean()), 'descriptive_seed_bootstrap95': ci}
                contrasts.append(contrast)
                lines.append(f'- {family} {arm} − {control}，{checkpoint}：{len(seeds)} 个配对种子，平均差 {delta.mean():.8f}；各种子 {delta.tolist()}；描述性区间 {ci}。')
    lines += ['', '## Critic 换算误差', '', 'Critic 原本拟合 sampled 相对误差；这里用 greedy 线长反推参考值。该转换不是最优值估计的一致性保证。']
    for name, result in details.items():
        if result['family'] != 'rest': continue
        groups = result['test_final']; bias = np.mean([v['predicted_reference_relative_error']['mean'] for v in groups.values()]); mae = np.mean([v['predicted_reference_relative_error']['mae'] for v in groups.values()])
        lines.append(f'- {name} final：预测参考值相对偏差 {100*bias:.4f}%，相对绝对误差均值 {100*mae:.4f}%。')
    lines += ['', '## 论文物理指标', '', '每项为四个测试分布等权平均；delay_scaled 的单位为 1e-18 秒，几何线长与路径使用归一化坐标单位。']
    for name, result in details.items():
        if result['family'] == 'rest': continue
        keys = ['wirelength', 'max_path'] if result['family'] == 'timing' else ['delay_scaled', 'length', 'buffers', 'skew_scaled']
        for checkpoint in ['best', 'final']:
            groups = result['test_' + checkpoint]
            values = {k: float(np.mean([g[k]['mean'] for g in groups.values()])) for k in keys}
            lines.append(f'- {name} {checkpoint}：{values}。')
    lines += ['', '## 结论边界', '', '论文 K3 将原固定尺度成本改为相对于教师库的成本，改变了样本权重与对偶价格尺度；不能把差异仅归因于 checkpoint 质量或称为替换 GeoSteiner 的加速。论文两个原分支本来都没有 GeoSteiner。', '',
              '25ICCAD 使用项目已有的 categorical-phase + local-greedy buffering 组件，不是论文完整 Shi–Li DP 或工业数据复现。此次 degree10、短程继续训练结论仅适用于固定协议。', '',
              '当前未完成、失败或种子不齐时，不给出完整实验胜负结论。']
    tmp = HERE / 'RESULTS.md.partial'; tmp.write_text('\n'.join(lines) + '\n'); os.replace(tmp, HERE / 'RESULTS.md')
    tmp = ART / 'contrasts.json.partial'; tmp.write_text(json.dumps(contrasts, indent=2)); os.replace(tmp, ART / 'contrasts.json')

if __name__ == '__main__': main()
