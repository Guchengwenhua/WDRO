# 阶段 8：独立攻击、效率与最终统计

已完成 CEM-256 独立压力测试、完整候选审计、统计导出与交付收尾。原最终报告及图表保持原位和原始字节。

## 测试范围

75 个冻结模型，300 评价格，40,000 个独立中心，600,000 个模型—中心攻击。每次攻击包含原 X 的 256 次真实候选评分，总计 153,600,000；另有 600,000 次选中候选的双尺度参考核验。

CEM-256 是固定预算的局部搜索；观测到的最大 regret 不是连续支持上的全局最大风险。运输 cost 逐候选实际测量，没有以 ρ 充当逐 net 硬半径。

## 可以支持的比较

- WDRO 对 ERM：五 degree、三 seed 宏平均下，指定分布的 mean/P95 和 OOD worst-group 满足原稳定改善门槛；nominal 门槛通过。
- WDRO 对 Hard Mining：没有攻击宏指标满足稳定改善门槛。
- WDRO 对 Mixed ERM / GroupDRO：结果依赖分布；uniform/grid_cluster 较好，mix_gauss/edge_bias 较差。

统计条件和逐条件差值应随结论一起报告；多重比较未校正。共享服务器的 wall time、未完整覆盖的历史训练 segment 不支持受控加速比声明。

## 产物与恢复结论

| 目的 | 入口 |
|---|---|
| 完整结果与边界 | [结果解读与交付清单](../../revisions/reference_precision_v2/Codex/阶段8_结果解读与交付清单.md) |
| 权威完成收尾 | [stage8_closeout_v1.json](../../revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage8_closeout_v1.json) |
| 独立数值复核 | [stage8_delivery_review_v1.json](../../revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage8_delivery_review_v1.json) |
| 完整统计表 | [tables/stage8](../../revisions/reference_precision_v2/Codex/experiments/tables/stage8/) |
| 图 | [figures](../../revisions/reference_precision_v2/Codex/experiments/figures/) |

既有交付复核记录 640,587 项文件绑定。原流水线在攻击结束后发生 Python 包遮蔽错误，后续独立入口完成了统计和收尾；保留失败日志。最终任务退出、锁释放，阶段 9 未因本阶段完成而自动启动。

本页是资料整理摘要；上述大规模数值审计来自原验收，没有在本次文档整理中再次执行。
