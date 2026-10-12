# 阶段 6：参考精度纠错与合成主实验

本页合并阶段 6 的精度阻塞、纠错方案、多次续跑、最终交接及结果入口。

## 为什么产生 precision_v2

旧 v1 出现负 regret 反例，独立尺度与连通树复核发现部分 GeoSteiner L★ 偏高。修复采用独立版本 `wdro_precision_v2`，保留旧失败证据，并对受影响的缓存和阶段 1–5 按原设置复核或重跑。旧 v1 已产生的少量 test 结果没有混入新主表。

历史根源与修复细节可查 [精度阻塞报告](../history/阶段6_GeoSteiner参考精度阻塞与修复建议.md)；纠错全过程及完成项以 [重放完成凭证](../../../wdro_legacy_v1/wdro_rest_min/manifests/precision_v2_replay_completion.json) 为准。

## 已完成的正式范围

- 75 模型：5 方法 × 5 degree × 3 seed；全部在 test 前按 validation best 冻结。
- 20 个共享合成 test 文件，每文件 10,000 nets；共 300 评价格、3,000,000 条模型预测。
- 统一 T=1 greedy、batch256；保留模型、输入、reference hash、逐 net 预测与配对差值。
- 新增 WDRO 训练具有候选合法性、cost、λ 递推及真实 Hard Mining 预算配对审计；统计和交付经过完整回读。

完成来源：[75 模型索引](../../wdro_rest_min/manifests/stage6_models_frozen_v1.json)、[完成凭证](../../wdro_rest_min/manifests/stage6_completion_v1.json)、[交付验证](../../wdro_rest_min/manifests/stage6_delivery_verification_v1.json)。

## 数值解释

正式表同时保留 mean、P95、最差 OOD 组和配对 CI。mean 按 degree 与训练 seed 等权平均；P95 先在 degree/seed 内计算再平均，不能与混池 P95 混用。CI 条件于已训练的三个 seed；未作多重比较校正。

WDRO 与 ERM 的差异应在 [主表](../../experiments/tables/stage6_main.csv)、[逐 degree 表](../../experiments/tables/stage6_per_degree.csv)和[配对 CI](../../experiments/tables/stage6_paired_ci.csv) 中一起读。合成测试本身不能替代后来的外部泛化与强攻击结论。

## 恢复历史的统一解释

9 月 7–9 日的多份“剩余任务”“命令进度”“消融完成”“CEM 中断”文件是不同时间点的操作记录。最终训练、完整账目与交付均已完成；不应把中途的 DOING、旧 PID、旧 GPU 或未完成计数视为当前状态。完整原文已归档，旧文件路径仍可访问。

完整原始结果见 [正式验收报告](../阶段6_多种子合成主实验验收报告.md) 与 [最终交接复核](../history/阶段6_最终交接复核_20260909.md)。其“阶段 7 尚未启动”语句只对应出具报告当时；后续情况见 [阶段 7](STAGE_7.md) 与 [阶段 8](STAGE_8.md)。
