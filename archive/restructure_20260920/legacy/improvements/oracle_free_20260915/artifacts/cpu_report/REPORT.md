# 无 GeoSteiner 学习：自动验收报告

**CPU/小样本工程 smoke，不能作为正式 GPU 科学结果。**

训练种子：[101]。全部任务 oracle_calls=0；源码、配置、产物哈希及锁均核验；使用保存的拓扑独立重算全部测试/攻击树长。
指标 q_lb 是 evaluator 相对最优值的保守上界，q_mst 是相对终端 MST 的可负指标；二者都不是实测真实 regret。
相同初始 Actor 和相同测试 net 已逐项核对。区间使用种子与 net 的配对分层 bootstrap；本报告 1 个训练种子，未做多重比较校正。

| 方法 | 测试最差组 mean q_lb（种子平均） | mean q_mst（四组等权） |
|---|---:|---:|
| erm_mst | 7.394213 | 2.541827 |
| erm_lb | 7.394213 | 2.541827 |
| wdro_random | 7.441407 | 2.591205 |
| wdro_es | 7.441407 | 2.591205 |
| wdro_critic | 7.441407 | 2.591205 |
| wdro_rl | 7.441407 | 2.591205 |

[逐方法/种子/分布 mean 和 P95](metrics.csv) · [训练前后及相对 ERM 配对差](paired_comparisons.csv) · [独立 Random32 压力测试](attacks.csv)

负配对差表示该指标降低。不能将有限候选攻击称为连续 Wasserstein 证书，也不能把未排除零的区间解释为等效。
运输 rho 是平均预算，训练的逐步 λ 更新不保证每批或全程精确满足；检查各任务 progress.jsonl 中的 mean_selected_cost。
全部结果保留，包括没有改善或退化的任务。此次范围为 degree30 从头训练可行性验证，不能据此比较历史预训练模型的优劣。
