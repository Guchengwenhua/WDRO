# 当前 ERM 与原 REST 仅归一化结果对照

生成时间：20260911T155103Z。仅回读、核对和汇总已有结果；新增训练、Actor 推理、GeoSteiner 求解均为 0。

当前 ERM 指 precision_v2 主线：从同 degree 的 DAC21 Actor 出发，在 uniform 上进行 2,000 次 relative-regret Actor–Critic 更新，使用合成验证集选择 best checkpoint。它没有 WDRO 对抗选样。

## 历史 ISPD 测试结果并列

旧列采用 Test_001/full_20260818 的原 DAC21 模型仅归一化复现实验；新列采用 Stage 7 Protocol A 的 ERM 测试，三个训练 seed 的指标等权平均。两边均为 T=1 greedy。只比较共同的五个 degree。

**这不是相同测试集的配对比较。** 旧数据来自 merged（未按当前 Protocol A 排除 metal5 重复版本）；新数据由 20 个基础 design 重建、排除六个重复版本。旧、新 GeoSteiner 参考精度和运行硬件也不同。旧输出的原始坐标先转 float32，再缩放；不能直接视为当前精度契约下的同坐标预测。

| degree | 旧 net 数 | 新 net 数/seed | 旧 mean % | 新 mean % | 新−旧 pp | 旧 P95 % | 新 P95 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 23169 | 15079 | 0.2660 | 0.2406 | -0.0255 | 1.3836 | 1.2811 |
| 20 | 1805 | 1180 | 0.8850 | 0.8068 | -0.0782 | 2.9724 | 2.5484 |
| 30 | 2791 | 1798 | 1.4918 | 1.3634 | -0.1284 | 3.7403 | 3.5232 |
| 40 | 2249 | 1444 | 1.9087 | 1.7829 | -0.1258 | 4.4165 | 4.1176 |
| 50 | 4672 | 2997 | 2.6432 | 2.4787 | -0.1645 | 5.6547 | 5.8667 |

五个共同 degree 等权 mean：旧 1.438947%，新 1.334483%，差 -0.104464 个百分点，相对下降 7.260%。这是两套历史协议的描述性差异，不能解释为控制变量下的训练收益。

旧原始 2026-03-01 日志同样保留在 CSV 的 original_march_log_mean_pct 列，未与 2026-08-18 复现结果混合。当前未训练 degree 5/15/25/35/45 的对应 ERM，所以没有将旧 10-degree 总平均与新 5-degree 总平均直接比较。

## 已有相同验证集上的初始化与微调后对照

旧列为当前 precision_v2 ERM 日志中的 update=0：原 DAC21 Actor + 统一归一化，尚未微调。新列为同一 run 的最终选定 best；已核对 60 组数据 hash 相同，15 个 best checkpoint 身份与 Stage 7 测试一致，且初始 DAC21 checkpoint hash 与旧缩放实验一致。

每个 degree×分布 2,000 个固定 validation net；先在各 cell/seed 计算 mean 和 P95，再等权平均。这里的 P95 是各 cell 的 P95 平均，不是混合全部样本后的 P95。validation 参与选模，因此不是独立测试收益。

| 分布 | 旧 mean % | 新 mean % | 新−旧 pp | 旧 P95 % | 新 P95 % | 新−旧 P95 pp |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0.7358 | 0.7072 | -0.0285 | 2.0395 | 2.0136 | -0.0259 |
| mix_gauss | 1.2120 | 1.1472 | -0.0648 | 3.0681 | 2.9720 | -0.0961 |
| edge_bias | 0.7660 | 0.8316 | +0.0655 | 2.2116 | 2.3748 | +0.1632 |
| grid_cluster | 0.7589 | 0.7187 | -0.0402 | 2.1029 | 2.0298 | -0.0731 |

20 个 degree×分布 cell 中，平均 regret 改善 15 个、恶化 5 个。全部 cell 均保留如下。

| degree | 分布 | 旧 mean % | 新 mean % | 新−旧 pp |
| --- | --- | --- | --- | --- |
| 10 | uniform | 0.2336 | 0.2229 | -0.0107 |
| 10 | mix_gauss | 0.2452 | 0.2367 | -0.0085 |
| 10 | edge_bias | 0.2295 | 0.2235 | -0.0060 |
| 10 | grid_cluster | 0.2219 | 0.2342 | +0.0123 |
| 20 | uniform | 0.5665 | 0.5360 | -0.0305 |
| 20 | mix_gauss | 0.8421 | 0.8144 | -0.0277 |
| 20 | edge_bias | 0.5880 | 0.6049 | +0.0169 |
| 20 | grid_cluster | 0.6163 | 0.5947 | -0.0215 |
| 30 | uniform | 0.8058 | 0.7857 | -0.0202 |
| 30 | mix_gauss | 1.4613 | 1.3134 | -0.1479 |
| 30 | edge_bias | 0.8301 | 0.9075 | +0.0774 |
| 30 | grid_cluster | 0.8465 | 0.7785 | -0.0680 |
| 40 | uniform | 0.9420 | 0.9235 | -0.0185 |
| 40 | mix_gauss | 1.6055 | 1.5873 | -0.0182 |
| 40 | edge_bias | 0.9877 | 1.0814 | +0.0937 |
| 40 | grid_cluster | 0.9530 | 0.9181 | -0.0348 |
| 50 | uniform | 1.1309 | 1.0681 | -0.0628 |
| 50 | mix_gauss | 1.9059 | 1.7841 | -0.1218 |
| 50 | edge_bias | 1.1949 | 1.3405 | +0.1456 |
| 50 | grid_cluster | 1.1567 | 1.0679 | -0.0888 |

## 可以得出的结论

现有结果支持：当前 ERM 在若干指标上优于原 DAC21 初始化，但并非所有分布都改善。不能把差异单独归因于 Critic：同时改变了 Actor 的损失、追加了训练、重新初始化 Critic，并进行了验证集模型选择。

要单独判定 regret Critic 是否优于线长 Critic，需要固定 Actor 的真实 regret 目标、训练输入、预算和选模规则，只改变 baseline 的回归目标及必要换算；当前表不包含这种消融。

## 文件与核对

- historical_ispd_comparison.csv：历史 ISPD 对照、样本数、P95、原始三月日志与相对变化。
- same_validation_macro.csv：相同验证集上的四分布宏平均。
- same_validation_per_degree.csv：完整 20 个 degree×分布对照。
- same_validation_per_seed.csv：60 个逐 seed 比较与数据/模型 hash。
- current_erm_synthetic_test.csv：当前 ERM 的完整 20 格合成独立测试结果，供查阅；未混入上述验证集对照。
- evidence.json：源文件哈希及数值回读检查摘要。

主要来源：

- /remote-home/Guilin/DRO/Test_001/results_scaling_experiment/full_20260818/summary.csv
- /remote-home/Guilin/DRO/Test_001/results_scaling_experiment/full_20260818/run_config.json
- /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/tables/stage7/per_degree.csv
- /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/tables/stage2_erm_validation_curves.csv
- /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/tables/stage2_erm_best_validation_long.csv
