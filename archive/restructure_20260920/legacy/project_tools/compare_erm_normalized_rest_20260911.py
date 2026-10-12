"""Compare existing results only; no training, inference, or solver calls."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path('/remote-home/Guilin/DRO/Codex')
REV = ROOT / 'revisions/reference_precision_v2/Codex'
TABLES = REV / 'experiments/tables'
OLD = ROOT.parent / 'Test_001/results_scaling_experiment/full_20260818'
DEGREES = [10, 20, 30, 40, 50]
SEEDS = [101, 102, 103]
DISTS = ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster']
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path)
    digest = sha(path)
    SOURCES[str(path)] = dict(resolved_path=str(path.resolve()), sha256=digest,
                              bytes=path.stat().st_size)
    return digest


def read_json(path):
    bind(path)
    return json.loads(path.read_text())


def read_csv(path):
    bind(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def arrays(path):
    bind(path)
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'experiments/analysis' / ('erm_vs_normalized_rest_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    old_rows = {int(r['degree']): r for r in read_csv(OLD / 'summary.csv')}
    old_config = read_json(OLD / 'run_config.json')
    old_log_path = ROOT.parent / 'DRO/REST-main/results_ispd/ispd_runs.jsonl'
    bind(old_log_path)
    old_log = [json.loads(s) for s in old_log_path.read_text().splitlines()]
    latest_legacy = {int(r['degree']): r for r in old_log}
    frozen = read_json(REV / 'wdro_rest_min/manifests/stage6_models_frozen_v1.json')
    models = {(m['degree'], m['seed']): m for m in frozen['models'] if m['method'] == 'erm'}
    assert len(models) == 15
    new_rows = [r for r in read_csv(TABLES / 'stage7/per_degree.csv') if r['method'] == 'erm']
    new_main = next(r for r in read_csv(TABLES / 'stage7/main.csv') if r['method'] == 'erm')
    stage7_predictions = {}
    comparisons = []
    for degree in DEGREES:
        old = arrays(OLD / 'per_net' / f'degree_{degree}.npz')
        errors = old['scaled_errors_pct']
        assert len(errors) == int(old_rows[degree]['real_n'])
        assert np.isfinite(errors).all()
        close(errors, 100 * (old['scaled_rest_lengths'] / old['scaled_geosteiner_lengths'] - 1))
        close(errors.mean(), float(old_rows[degree]['real_scaled_error_pct']))
        means, p95s = [], []
        for seed in SEEDS:
            model = models[(degree, seed)]
            training = read_json(Path(model['result']['logical_path']))
            assert training['status'] == 'complete' and training['completed_updates'] == 2000
            assert training['initialization']['sha256'] == old_config['inputs'][str(degree)]['checkpoint_sha256']
            assert training['best_checkpoint_sha256'] == model['best']['sha256']
            pred_root = REV / 'experiments/logs/stage7/evaluation' / f'erm_d{degree}_s{seed}'
            result = read_json(pred_root / 'result.json')
            assert result['binding']['model']['best']['sha256'] == model['best']['sha256']
            assert result['binding']['mode'] == 'deterministic_greedy' and result['binding']['T'] == 1
            pred_path = pred_root / 'predictions.npz'
            pred = arrays(pred_path)
            assert SOURCES[str(pred_path)]['sha256'] == result['predictions']['sha256']
            regrets = pred['rest_lengths'] / pred['lstar'] - 1
            close(regrets.mean(), result['summary']['mean'])
            close(np.quantile(regrets, .95), result['summary']['p95'])
            row = next(r for r in new_rows if int(r['degree']) == degree and int(r['seed']) == seed)
            assert len(regrets) == int(row['count'])
            close(regrets.mean(), float(row['net_weighted_mean']))
            close(np.quantile(regrets, .95), float(row['p95']))
            stage7_predictions[(degree, seed)] = regrets
            means.append(100 * regrets.mean())
            p95s.append(100 * np.quantile(regrets, .95))
        current_mean = float(np.mean(means))
        comparisons.append(dict(degree=degree, old_nets=len(errors), current_nets_per_seed=len(regrets),
            current_training_seeds=3, old_mean_pct=float(errors.mean()),
            current_mean_pct=current_mean, mean_delta_pp=current_mean-float(errors.mean()),
            relative_mean_reduction_pct=100*(1-current_mean/float(errors.mean())),
            old_p95_pct=float(np.quantile(errors, .95)), current_p95_pct=float(np.mean(p95s)),
            original_march_log_mean_pct=latest_legacy[degree]['metrics']['rest_percentage_error'],
            comparison_type='historical_unpaired_different_sample_scope_and_reference'))
    all_new = [np.concatenate([stage7_predictions[(d,s)] for d in DEGREES]) for s in SEEDS]
    for metric, vals in [('net_weighted_mean', [x.mean() for x in all_new]),
                         ('p95', [np.quantile(x, .95) for x in all_new])]:
        close(np.mean(vals), float(new_main[metric]))

    curves = read_csv(TABLES / 'stage2_erm_validation_curves.csv')
    best = read_csv(TABLES / 'stage2_erm_best_validation_long.csv')
    initial = {(r['run_id'], r['distribution']): r for r in curves if int(r['update']) == 0}
    assert len(initial) == 60 and len(best) == 60
    validation_seed_rows = []
    for b in best:
        a = initial[(b['run_id'], b['distribution'])]
        d, seed = int(b['degree']), int(b['training_seed'])
        assert a['dataset_array_bundle_sha256'] == b['dataset_array_bundle_sha256']
        assert a['count'] == b['count'] == '2000'
        assert b['split'] == a['split'] == 'validation'
        assert b['best_checkpoint_sha256'] == models[(d, seed)]['best']['sha256']
        validation_seed_rows.append(dict(degree=d, seed=seed, distribution=b['distribution'],
            nets=2000, selected_update=int(b['update']),
            old_mean_pct=100*float(a['mean']), current_mean_pct=100*float(b['mean']),
            mean_delta_pp=100*(float(b['mean'])-float(a['mean'])),
            old_p95_pct=100*float(a['p95']), current_p95_pct=100*float(b['p95']),
            p95_delta_pp=100*(float(b['p95'])-float(a['p95'])),
            dataset_array_bundle_sha256=b['dataset_array_bundle_sha256'],
            current_best_checkpoint_sha256=b['best_checkpoint_sha256']))
    metrics = ['old_mean_pct','current_mean_pct','mean_delta_pp','old_p95_pct','current_p95_pct','p95_delta_pp']
    validation_cells = []
    for d in DEGREES:
        for dist in DISTS:
            rows = [r for r in validation_seed_rows if r['degree']==d and r['distribution']==dist]
            assert len(rows) == 3 and len({r['dataset_array_bundle_sha256'] for r in rows}) == 1
            close([r['old_mean_pct'] for r in rows], rows[0]['old_mean_pct'])
            validation_cells.append(dict(degree=d, distribution=dist, nets_per_seed=2000,
                seeds=3, **{k:float(np.mean([r[k] for r in rows])) for k in metrics}))
    validation_macro = []
    for dist in DISTS:
        rows = [r for r in validation_cells if r['distribution']==dist]
        validation_macro.append(dict(distribution=dist, degree_count=5, training_seeds=3,
            **{k:float(np.mean([r[k] for r in rows])) for k in metrics}))
    write_csv(out/'historical_ispd_comparison.csv', comparisons)
    write_csv(out/'same_validation_per_seed.csv', validation_seed_rows)
    write_csv(out/'same_validation_per_degree.csv', validation_cells)
    write_csv(out/'same_validation_macro.csv', validation_macro)
    test_rows = [r for r in read_csv(TABLES/'stage6_per_degree.csv') if r['method']=='erm']
    write_csv(out/'current_erm_synthetic_test.csv', test_rows)
    macro_old = float(np.mean([r['old_mean_pct'] for r in comparisons]))
    macro_new = float(np.mean([r['current_mean_pct'] for r in comparisons]))
    improved = sum(r['mean_delta_pp'] < 0 for r in validation_cells)
    report = [
        '# 当前 ERM 与原 REST 仅归一化结果对照', '',
        f'生成时间：{stamp}。仅回读、核对和汇总已有结果；新增训练、Actor 推理、GeoSteiner 求解均为 0。', '',
        '当前 ERM 指 precision_v2 主线：从同 degree 的 DAC21 Actor 出发，在 uniform 上进行 2,000 次 relative-regret Actor–Critic 更新，使用合成验证集选择 best checkpoint。它没有 WDRO 对抗选样。', '',
        '## 历史 ISPD 测试结果并列', '',
        '旧列采用 Test_001/full_20260818 的原 DAC21 模型仅归一化复现实验；新列采用 Stage 7 Protocol A 的 ERM 测试，三个训练 seed 的指标等权平均。两边均为 T=1 greedy。只比较共同的五个 degree。', '',
        '**这不是相同测试集的配对比较。** 旧数据来自 merged（未按当前 Protocol A 排除 metal5 重复版本）；新数据由 20 个基础 design 重建、排除六个重复版本。旧、新 GeoSteiner 参考精度和运行硬件也不同。旧输出的原始坐标先转 float32，再缩放；不能直接视为当前精度契约下的同坐标预测。', '',
        md_table(['degree','旧 net 数','新 net 数/seed','旧 mean %','新 mean %','新−旧 pp','旧 P95 %','新 P95 %'],
            [[r['degree'],r['old_nets'],r['current_nets_per_seed'],f"{r['old_mean_pct']:.4f}",f"{r['current_mean_pct']:.4f}",f"{r['mean_delta_pp']:+.4f}",f"{r['old_p95_pct']:.4f}",f"{r['current_p95_pct']:.4f}"] for r in comparisons]), '',
        f'五个共同 degree 等权 mean：旧 {macro_old:.6f}%，新 {macro_new:.6f}%，差 {macro_new-macro_old:+.6f} 个百分点，相对下降 {100*(1-macro_new/macro_old):.3f}%。这是两套历史协议的描述性差异，不能解释为控制变量下的训练收益。', '',
        '旧原始 2026-03-01 日志同样保留在 CSV 的 original_march_log_mean_pct 列，未与 2026-08-18 复现结果混合。当前未训练 degree 5/15/25/35/45 的对应 ERM，所以没有将旧 10-degree 总平均与新 5-degree 总平均直接比较。', '',
        '## 已有相同验证集上的初始化与微调后对照', '',
        '旧列为当前 precision_v2 ERM 日志中的 update=0：原 DAC21 Actor + 统一归一化，尚未微调。新列为同一 run 的最终选定 best；已核对 60 组数据 hash 相同，15 个 best checkpoint 身份与 Stage 7 测试一致，且初始 DAC21 checkpoint hash 与旧缩放实验一致。', '',
        '每个 degree×分布 2,000 个固定 validation net；先在各 cell/seed 计算 mean 和 P95，再等权平均。这里的 P95 是各 cell 的 P95 平均，不是混合全部样本后的 P95。validation 参与选模，因此不是独立测试收益。', '',
        md_table(['分布','旧 mean %','新 mean %','新−旧 pp','旧 P95 %','新 P95 %','新−旧 P95 pp'],
            [[r['distribution'],f"{r['old_mean_pct']:.4f}",f"{r['current_mean_pct']:.4f}",f"{r['mean_delta_pp']:+.4f}",f"{r['old_p95_pct']:.4f}",f"{r['current_p95_pct']:.4f}",f"{r['p95_delta_pp']:+.4f}"] for r in validation_macro]), '',
        f'20 个 degree×分布 cell 中，平均 regret 改善 {improved} 个、恶化 {20-improved} 个。全部 cell 均保留如下。', '',
        md_table(['degree','分布','旧 mean %','新 mean %','新−旧 pp'],
            [[r['degree'],r['distribution'],f"{r['old_mean_pct']:.4f}",f"{r['current_mean_pct']:.4f}",f"{r['mean_delta_pp']:+.4f}"] for r in validation_cells]), '',
        '## 可以得出的结论', '',
        '现有结果支持：当前 ERM 在若干指标上优于原 DAC21 初始化，但并非所有分布都改善。不能把差异单独归因于 Critic：同时改变了 Actor 的损失、追加了训练、重新初始化 Critic，并进行了验证集模型选择。', '',
        '要单独判定 regret Critic 是否优于线长 Critic，需要固定 Actor 的真实 regret 目标、训练输入、预算和选模规则，只改变 baseline 的回归目标及必要换算；当前表不包含这种消融。', '',
        '## 文件与核对', '',
        '- historical_ispd_comparison.csv：历史 ISPD 对照、样本数、P95、原始三月日志与相对变化。',
        '- same_validation_macro.csv：相同验证集上的四分布宏平均。',
        '- same_validation_per_degree.csv：完整 20 个 degree×分布对照。',
        '- same_validation_per_seed.csv：60 个逐 seed 比较与数据/模型 hash。',
        '- current_erm_synthetic_test.csv：当前 ERM 的完整 20 格合成独立测试结果，供查阅；未混入上述验证集对照。',
        '- evidence.json：源文件哈希及数值回读检查摘要。', '',
        '主要来源：', '',
        f'- {OLD / "summary.csv"}',
        f'- {OLD / "run_config.json"}',
        f'- {TABLES / "stage7/per_degree.csv"}',
        f'- {TABLES / "stage2_erm_validation_curves.csv"}',
        f'- {TABLES / "stage2_erm_best_validation_long.csv"}',
    ]
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    evidence = dict(at_utc=stamp, status='pass', scope='existing_artifact_comparison_only',
        added_training_updates=0, added_actor_calls=0, added_geosteiner_calls=0,
        historical_degrees=DEGREES, historical_old_nets=sum(r['old_nets'] for r in comparisons),
        current_ispd_unique_nets=sum(r['current_nets_per_seed'] for r in comparisons),
        old_prediction_rows_recomputed=sum(r['old_nets'] for r in comparisons),
        current_erm_prediction_rows_recomputed=sum(len(x) for x in stage7_predictions.values()),
        current_erm_model_bindings_checked=15, initial_checkpoint_identity_matches=15,
        matched_validation_rows=60, matched_validation_cells=20,
        validation_mean_improved_cells=improved, validation_mean_worse_cells=20-improved,
        historical_macro_old_mean_pct=macro_old, historical_macro_new_mean_pct=macro_new,
        historical_macro_delta_pp=macro_new-macro_old,
        source_files=SOURCES,
        limitations=['historical_ISPD_sample_scope_differs','historical_reference_precision_differs',
                     'historical_CPU_vs_current_GPU','historical_coordinate_precision_differs',
                     'validation_is_used_for_checkpoint_selection','not_a_critic_only_ablation'])
    evidence['outputs']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['source_files','outputs']},ensure_ascii=False,indent=2))
    print('OUTPUT',out)
    print('\n'.join(report[10:15]))
    print(json.dumps(validation_macro,indent=2))


if __name__ == '__main__':
    main()
