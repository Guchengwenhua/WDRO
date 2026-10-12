# 阶段 7：ISPD Protocol A 外部测试

该阶段只评价已经冻结的 precision_v2 模型；未使用 ISPD 训练、调参或选择 checkpoint。

## 输入、模型与计算

95 个存在的 design×degree 文件组成 20 个基础设计，共 22,498 合法 nets。5 个缺失的 degree 文件明确记录，6 个 `_metal5` 重复 design identity 被排除。75 个模型按 degree 适配输入，合计 337,470 条预测；不是每个模型都能评价所有 degree。

参考长度按 `2^27`/`2^28` 双尺度计算，并执行独立拓扑与长度检查。最终统计保留 net-weighted、design-macro、P95 和 worst-design 等不同口径；不得把样本权重不同的指标当作同一个结论。

## 完成证据与最新解释

| 用途 | 文件 |
|---|---|
| 输入、来源、原始阶段报告 | [阶段 7 验收](../history/阶段7_ISPD外部测试验收报告.md) |
| 完成凭证 | [stage7_completion_v1.json](../../wdro_rest_min/manifests/stage7_completion_v1.json) |
| 交付核验 | [stage7_delivery_verification_v1.json](../../wdro_rest_min/manifests/stage7_delivery_verification_v1.json) |
| 后续统一统计和解释边界 | [阶段 8 结果解读](../阶段8_结果解读与交付清单.md) |

后续统一 CI 支持 WDRO 相对 ERM 的指定 net-weighted mean、design-macro mean 和 P95 改善；不支持对所有基线和 worst-design 指标的全面优势。CI 条件于固定设计/degree 分层和已有模型，不等同于未见设计总体的置信保证。

9 月 12 日之后的 ISPD2008/CircuitOps 扩展及 v3.1 模型测试属于独立工作，见 [外部验证汇总](../../../summaries/EXTERNAL.md)。
