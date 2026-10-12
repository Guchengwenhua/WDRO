# 阶段 0–5：协议、数据、基线与 WDRO 选择

本页合并早期各阶段的说明、验收与运行要点。**工程历史保留 v1，数值结果使用 precision_v2 纠错重放。**

## 固定比较设计

方法为 ERM、Mixed ERM、Hard Mining、GroupDRO、WDRO-REST；degree 为 10/20/30/40/50，训练 seed 为 101/102/103。同 degree 从同一 DAC21 初始化开始，正式训练 2,000 updates、batch256；模型选择只使用 validation。nominal 为 uniform，OOD 为 mix_gauss、edge_bias、grid_cluster。

每个正式 test 格 10,000 nets，评价为 T=1 greedy。nominal 最大允许退化是 relative regret 增加 0.002，即 0.2 个百分点。这个选模门槛不等同于训练中已经优化了一个严格的 nominal 约束。

## 每阶段交付与证据

| 阶段 | 核心交付 | 当前查证入口 |
|---|---|---|
| 0 | 协议冻结、环境与原模型兼容 | [protocol_v1.json](../../wdro_rest_min/configs/protocol_v1.json)、[历史报告](../../阶段0_协议冻结与环境报告.md) |
| 1 | 统一几何契约、数据、cost、评价和参考缓存 | [数据契约](../../wdro_rest_min/configs/data_contract_v1.json)、[纠错 reference 索引](../../revisions/reference_precision_v2/Codex/experiments/cache/lstar/protocol_v1/lstar_index.json) |
| 2 | 15 个统一 ERM，保存 optimizer/RNG/恢复状态 | [纠错完成索引](../../revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage2_completion_20260905.json) |
| 3 | 45 个 Mixed ERM / GroupDRO / Hard Mining 基线 | [纠错完成索引](../../revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage3_completion_v1.json) |
| 4 | Random-WDRO 与 cost/对偶更新/恢复审计 | [纠错完成索引](../../revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage4_completion_v1.json) |
| 5 | validation-only 预算和搜索选择、消融 | [纠错完成索引](../../revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage5_completion_v1.json)、[选定配置](../../revisions/reference_precision_v2/Codex/wdro_rest_min/configs/stage5_selected_wdro_v1.json) |

## 对阅读后续结果有影响的实现约定

- 数据逐 net 保留 ID、归一化与逆变换、边界、最小间距和 hash。validation/test 分离；真实设计只作冻结模型外部测试。
- 真实 regret 使用 GeoSteiner L★。阶段 6 发现旧尺度精度问题后，相关缓存和训练按原范围纠错重跑；旧 v1 的低层工程测试通过不能使其旧数值重新有效。
- WDRO 与 Hard Mining 共用候选 support 和真实评分入口。记录 Actor、REST、GeoSteiner 的实际调用；失败重试费用与有效训练轨迹分别计数。
- 当前选定主设置为 Random/Sobol K32、Hungarian-L1、learned λ、按 degree 的 q75 预算。早期旧 v1 文档中的 q50 不是当前选定主配置。
- checkpoint 记录模型、optimizer、scheduler、RNG 和调用账本；恢复等价性由原阶段审计验证，历史 smoke 不能代替正式完整验收。

## 历史资料如何使用

根目录的阶段 0–5 原报告通过兼容路径保持可读；包括启动命令、每次测试数量和中途资源快照。当前无需按历史“下一步”重启已结束的训练。定位原文与 hash 可查 [归档清单](../../archive/documentation_20260915/manifest.json)，运行入口见 [运行说明](../OPERATIONS.md)。
