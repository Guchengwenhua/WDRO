# 当前状态

整理日期：2026-09-15。完成项依据既有完成凭证；本页的文档整理没有重新运行旧科学实验。

| 研究 | 状态与范围 | 权威证据入口 |
|---|---|---|
| precision_v2 主线阶段 0–8 | 已完成；75 模型、300 合成格/300 万预测，ISPD 及 CEM-256 攻击完成 | [阶段 8 收尾凭证](../revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage8_closeout_v1.json) |
| ES/Critic/RL v3.1 | 已完成；A30 固定父模型 / 1,080 格，B18 继续训练任务 | [最终状态](../revisions/adversarial_v3_1_20260912/Codex/FINAL_STATUS.md) |
| 公开芯片扩展 | 已完成；134,550 合法 nets、75 模型、2,018,250 预测 | [完整报告](../external_benchmarks/20260912_public/REPORT.md) |
| v3.1 新模型真实数据验证 | 已完成并在 9 月 13 日独立重验；33 模型、364,056 预测，degree=30 | [独立验收与统计](../external_benchmarks/20260912_v31_models/artifacts/review_20260913/REPORT.md) |
| tmp2 提出的 Critic 搜索七方向 | DONE；degree30、3 seeds、16项任务及独立复核完成 | [结果与验收](../improvements/critic_search_20260915/RESULTS.md) |
| 阶段 9 论文同步 | 既有主线记录尚未启动 | [历史论文同步清单](../论文同步清单_protocol_v1.md) |

## 需要保持的区分

当前主线数值来自 precision_v2。旧 v1 曾受 GeoSteiner 参考精度问题影响，旧表与旧参考值不能混入当前主表；早期 v3 又因 greedy 推理批形状的数值差异被 v3.1 替代。

历史进度原件 [01_WDRO-REST_项目进度.md](../01_WDRO-REST_项目进度.md) 保存截至 2026-09-13 的长篇日志，并被当日外部验证收尾文件绑定。它不承担 2026-09-15 新分支的动态状态；新研究在各自分支维护，完成后在本页增加证据链接。
