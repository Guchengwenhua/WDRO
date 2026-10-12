# 当前状态

整理日期：2026-09-15。完成项依据既有完成凭证；本页的文档整理没有重新运行旧科学实验。

最新分支更新：2026-09-19，仅新增三篇论文 × 基础 WDRO 验收记录；其余完成项未在本次重新验收。

| 研究 | 状态与范围 | 权威证据入口 |
|---|---|---|
| precision_v2 主线阶段 0–8 | 已完成；75 模型、300 合成格/300 万预测，ISPD 及 CEM-256 攻击完成 | [阶段 8 收尾凭证](../wdro_precision_v2/wdro_rest_min/manifests/stage8_closeout_v1.json) |
| ES/Critic/RL v3.1 | 已完成；A30 固定父模型 / 1,080 格，B18 继续训练任务 | [最终状态](../adversarial_v3_1_20260912/docs/FINAL_STATUS.md) |
| 公开芯片扩展 | 已完成；134,550 合法 nets、75 模型、2,018,250 预测 | [完整报告](../benchmarks/public_designs_20260912/docs/REPORT.md) |
| v3.1 新模型真实数据验证 | 已完成并在 9 月 13 日独立重验；33 模型、364,056 预测，degree=30 | [独立验收与统计](../benchmarks/v31_models_20260912/artifacts/review_20260913/REPORT.md) |
| tmp2 提出的 Critic 搜索七方向 | DONE；degree30、3 seeds、16项任务及独立复核完成 | [结果与验收](../critic_search_20260915/docs/RESULTS.md) |
| 三篇论文 × 基础 WDRO | DONE；degree10、24 次正式训练；9 月 19 日独立验收 PASS，尚未建立稳定 WDRO 优势 | [验收与统计](../papers_wdro_20260915/artifacts/review_20260919/REPORT.md) |
| 阶段 9 论文同步 | 既有主线记录尚未启动 | [历史论文同步清单](../wdro_legacy_v1/docs/history/论文同步清单_protocol_v1.md) |

## 需要保持的区分

当前主线数值来自 precision_v2。旧 v1 曾受 GeoSteiner 参考精度问题影响，旧表与旧参考值不能混入当前主表；早期 v3 又因 greedy 推理批形状的数值差异被 v3.1 替代。

历史进度原件 [01_WDRO-REST_项目进度.md](history/01_WDRO-REST_项目进度.md) 保存截至 2026-09-13 的长篇日志，并被当日外部验证收尾文件绑定。它不承担 2026-09-15 新分支的动态状态；新研究在各自分支维护，完成后在本页增加证据链接。
