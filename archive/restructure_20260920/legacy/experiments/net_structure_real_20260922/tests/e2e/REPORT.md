# net 结构先验真实芯片实验报告

更新：2026-09-22T06:04:59.626408+00:00；运行中，未完成正式验收。
完成最大训练预算 0/33；已有自然测试文件 19。

理论、数据隔离、方法边界见 PLAN.md。指标为真实 REST/L*−1；固定 K 候选不是连续 WDRO 证明。tile 保持仅适用于全局布线抽象，不证明详细布线或真实 cell 移动合法。

|方法|REST seed|生成器|预算|checkpoint|测试集|mean %|P95 %|设计macro %|选择步|
|---|---:|---:|---:|---|---|---:|---:|---:|---:|

完整各预算数据：tables/main.csv；逐设计：tables/per_design.csv；配对差与按设计bootstrap：tables/paired_vs_raw.csv。B仅3设计，CI不可支持通用显著性主张。

生成器调参/预训练/后训练曲线：artifacts/generators；几何诊断：artifacts/diagnostics；独立攻击：artifacts/attack；实际数据规模：artifacts/data_counts.csv。所有失败与负结果保留。

生成器使用扩展degree数据，历史VAE仅用d30；set_dae与graph_dae提供同数据预算对照。CPU/GPU耗时受并行负载影响，按实际调用数和训练暴露报告，不声称严格总算力公平。
