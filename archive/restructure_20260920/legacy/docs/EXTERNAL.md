# 真实芯片外部验证汇总

以下实验均为冻结模型的零样本验证。公开芯片 net 的二维线长指标不能直接等同于整芯片布线、拥塞或 PPA。

## 三组数据和实验的关系

| 实验 | 输入与模型 | 完成入口 |
|---|---|---|
| 主线阶段 7 | ISPD2018/2019 Protocol A；20 设计、22,498 nets，75 个按 degree 匹配模型，共 337,470 预测 | [阶段 7](mainline/STAGE_7.md) |
| 9 月 12 日公开设计扩展 | ISPD2008 的 16 个设计与 CircuitOps 13 个实现；28 个设计有目标 degree，共 134,550 合法 nets，75 模型、2,018,250 预测 | [原始完整报告](../external_benchmarks/20260912_public/REPORT.md) |
| v3.1 新模型全部真实数据评测 | 三类来源的 degree30；49 设计进入覆盖清单、43 有可用 net；每模型 11,032 nets，18 新模型 + 15 对照 = 364,056 预测 | [9 月 13 日独立验收报告](../external_benchmarks/20260912_v31_models/artifacts/review_20260913/REPORT.md) |

## 公开设计扩展

源数据、下载信息、合法性拒绝原因和原始 hash 保留在 [sources](../external_benchmarks/20260912_public/sources/) 与该实验目录。原始 14,392,871 条 nets 经冻结 degree 和几何契约筛选，形成实际评价集合；18 条目标 net 因重复 XY 被拒绝，GCD 保留零样本行。

75 个主线模型使用双尺度精确参考，269,100 次参考求解；6 张统计表和 160 行设计家族聚类配对 CI 已独立复算。总体 mean/P95 的小差异需要与具体分布、设计和对照一起解释，Mixed ERM/GroupDRO 的总体表现更低，不支持 WDRO 全面领先。

证据：[完成清单](../external_benchmarks/20260912_public/completion_manifest.json)、[独立交付](../external_benchmarks/20260912_public/delivery_verification.json)。

## v3.1 模型评测与重验

新增模型只有 degree30。A 中的 Critic/RL 攻击器不是新的 REST Actor，不计作这里的 18 个模型。所有模型使用合成 validation 选出的 best checkpoint，greedy 批形状固定 256 行。

9 月 13 日重验确认 1,645 项绑定，重新核验输入、模型、参考、逐网拓扑与线长和 1,452 个分片。逐种子、种子平均及配对 CI 表从原始逐网结果独立重算，输出 CSV 与原统计逐字节一致。本次重验没有新增推理或 GeoSteiner 调用。

已发布数据上，原 GroupDRO 总体 mean、P95 和设计宏平均最低；新增路线没有总体领先。新增方法内部 Random64 的 mean 最低，RL frozen4 的 P95 最低；ES64 与 Random64 的总体 mean 没有明确差异。Critic4 相对 RandomTop4 的总体 mean 更差，结论限定于对应聚合口径。

ISPD2008 对 net 加权总体指标贡献较大。CircuitOps eligible 子集较小，分域结果应作为探索性证据保留样本数，不推广到未覆盖工艺、设计或 degree。

证据：[收尾凭证](../external_benchmarks/20260912_v31_models/artifacts/review_20260913/closeout.json)、[验收详情](../external_benchmarks/20260912_v31_models/artifacts/review_20260913/acceptance.json)、[产物与统计目录](../external_benchmarks/20260912_v31_models/artifacts/)。
