# WDRO 对比 ERM：PPT 图包

按截图中“少量基线、横轴 degree、每个数据集一张、再加总览”的展示思路，重绘已有真实实验结果。没有重新训练或调整原始结果。图中文字为英文，便于直接用于学术 PPT；PNG 为 3520×1980（16:9），另附 PDF 和可编辑 SVG。

## 文件

| 图 | 内容 |
|---|---|
| `00_overview.png` | 四个合成分布及 ISPD 的 mean/P95 相对 ERM 改善率 |
| `01_uniform.png` | Uniform，横轴 degree |
| `02_mix_gauss.png` | Gaussian mixture，横轴 degree |
| `03_edge_bias.png` | Edge bias，横轴 degree |
| `04_grid_cluster.png` | Grid cluster，横轴 degree |
| `05_ispd.png` | ISPD 外部测试，横轴 degree |
| `all_figures.pdf` | 上述六张图的多页 PDF |
| `plot_data_per_degree.csv` | 75 个 dataset×method×degree 绘图数值，包含绝对 regret 与改善率 |
| `plot_data_summary.csv` | 总览图数值，包含相对改善率和绝对百分点变化 |
| `make_figures.py` | 直接读取原始表的可复现绘图脚本 |
| `provenance.json` | 源表 SHA-256、存储路径、数值复核结果 |

## 可用于 PPT 的表述

> 在统一 REST 模型、预处理和推理设置下，WDRO 相对 ERM 在四个合成分布上的平均 relative regret 降低 1.43%–2.76%；在 ISPD 零样本外部测试上，平均 relative regret 降低 2.84%，P95 降低 2.72%。

这是一项幅度较小、可量化的改进，数据不支持“巨大提升”或“所有 degree 都显著改善”。四组合成测试的平均 regret 相对降幅依次为 Uniform 2.75%、Gaussian mixture 2.22%、Edge bias 1.43%、Grid cluster 2.76%。ISPD 平均 regret 从约 0.7571% 降至 0.7357%，绝对减少约 0.0215 个百分点，不能写成“线长降低 2.84%”。

每张分数据集图左侧展示原始平均 regret（从零起，同一纵轴范围），右侧直接展示相对改善率（保留零线、负值，所有图范围一致）。例如 Edge bias 的 degree=40，WDRO 相对 ERM 退化约 0.44%，图中保留此结果；degree=10 基本持平。总览图的改善率采用“聚合指标之比”，不是逐 degree 改善率的平均。

Mixed ERM 是已有代码中四个训练分布等比例混合的 ERM 对照，不是单纯的通用扰动增强。它在 Gaussian mixture 和 Edge bias 上优于 WDRO；在 ISPD mean 上也略优于 WDRO。图中完整保留这些比较。因此适合表述“相对标准 ERM 有改善”，不宜表述为“优于数据增强及所有基线”。图中没有画 Hard Mining 和 GroupDRO，只是按本次展示范围简化；其结果仍在原表中。

## 找到的早期实现

项目根目录：`/remote-home/Guilin/DRO/Codex`

- 最早阶段 4 的训练入口：`wdro_rest_min/scripts/train_wdro.py`。
- 有限候选选择及对偶更新：`wdro_rest_min/stage4/dual.py`，`select_wdro` 对每个中心选择 `regret − lambda × cost` 最大的候选，`update_lambda` 更新非负运输价格。
- 外层训练衔接：`wdro_rest_min/stage4/trainer.py`，候选选择与价格更新位于约 258–274 行。
- `wdro_rest_min/src/candidates.py` 与 `src/costs.py`：候选生成和几何运输成本。
- 早期阶段 4 是 degree=30、seed=101、500 次更新的实现诊断，不足以画出完整 degree 对比。因此绘图使用同一 Random/Sobol WDRO 路线的正式多 degree、多 seed 结果，而非后续 ES/Critic/RL 改进分支。

## 数据版本与指标口径

当前数值主线位于 `revisions/reference_precision_v2/Codex/`。旧 v1 的 GeoSteiner 参考长度曾有精度问题，已纠错重跑；本图包只读取 precision_v2，未混入旧 v1 数值。背景见项目 `docs/mainline/STAGE_6.md`。

源表均位于 `/remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/tables/`：

- 合成分布：`stage6_per_degree.csv`、`stage6_main.csv`；用 `stage6_per_seed.csv` 独立核对三 seed 平均。每个 degree×分布 10,000 个共享测试 nets，degree 为 10/20/30/40/50，训练 seeds 为 101/102/103。
- ISPD：`stage7/per_degree.csv` 和 `stage7/main.csv`。22,498 个唯一 nets、20 个设计。逐 degree 先取三 seed 平均；总览 mean 按 net 加权，不能拿五个 degree 的简单平均代替。
- 合成 P95 先在每个 degree/seed 内计算，再等权平均；ISPD P95 则每个 seed 在 pooled nets 上计算，再对 seed 平均。两者不可视为相同分层口径。
- 图为点估计，未添加显著性星号或误差线。原项目的配对区间见 `stage6_paired_ci.csv`、`stage8/ispd_paired_ci.csv`；结论与区间条件见 `阶段8_结果解读与交付清单.md`。不能从图中宣称每个点均有统计显著性。
- `REST / ERM` 指项目中采用统一预处理、继续训练的 ERM 基线，并非未经继续训练的 DAC21 原始 checkpoint；所有方法都以对应 degree 的 DAC21 Actor 为起点。正式训练设定为 2,000 次更新，推理为 T=1 greedy；不同训练方法的候选搜索额外开销不同，不能据此声称训练计算量完全相同。

截图提到的初稿 PDF 已定位到 `/remote-home/Guilin/DRO/output/pdf/WDRO-REST_ICCAD2026_anonymous_draft.pdf`，另一个路径为 `/remote-home/Guilin/DRO/paper_iccad_wdro_rest/build/main.pdf`。此次未改写论文内容。

## 复现

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/experiments/figures/wdro_erm_ppt_20260920/make_figures.py
```

脚本检验源表单元格完整性，重算合成数据的 seed 和 degree 聚合，并独立核对 ISPD 的 net 加权 mean。所有输出经 Codex 下的逻辑路径存储；实际附加盘路径记录于 provenance.json。
