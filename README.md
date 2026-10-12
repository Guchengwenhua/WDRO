# DRO / Codex 研究项目

2026-10-12 扩展实验完成：[原21项续到2048步及独立Critic修正对照](critic_paper_scale_20261011/README.md)，[30/30项验收报告与轨迹图](critic_paper_scale_20261011/acceptance_20261012/REPORT.md)，[逐项结果](critic_paper_scale_20261011/RESULTS.md)。

2026-10-11 实验：[Critic 参考值换算与 23/25 ICCAD checkpoint-bank 配对训练](critic_paper_bank_20261011/README.md)，[冻结协议](critic_paper_bank_20261011/PROTOCOL.md)，[21/21项完成后的验收报告](critic_paper_bank_20261011/acceptance/REPORT.md)。

正在进行：[全模型 × 全数据集扩展复测](benchmarks/all_models_all_data_20260928/README.md)，[唯一总表（随正式结果更新）](reports/全模型全数据集总表_20260928.html)。完成评估后自动进行目录分类。

这里按独立方法版本和研究协议平级组织。每个方法目录实际存放自己的代码、README、`docs/` 和产物入口；改进方法不再嵌套在父方法内部。同一协议下共用代码的算法变体仍在该方法内维护，避免再次复制代码和模型。

先看 [方法与测评总览](summaries/方法与测评总览_20260919.md)、[研究状态](summaries/STATUS.md) 和 [全目录索引](summaries/DIRECTORY_INDEX.md)。

强化学习攻击器的实现详解见 [强化学习对抗样本搜索方法与流程](/remote-home/Guilin/DRO/Codex/summaries/强化学习对抗样本搜索方法与流程_20260921.md)，包括各版本 REINFORCE、PPO、best-of-K、序贯、多智能体及历史对手训练。

| 方法 / 版本 | 实际目录 | 范围 |
|---|---|---|
| [WDRO 正式主线 precision_v2](wdro_precision_v2/) | `wdro_precision_v2/` | ERM、Mixed ERM、Hard Mining、GroupDRO 和 WDRO 的统一冻结实验；当前数值主线。 |
| [WDRO 历史 v1](wdro_legacy_v1/) | `wdro_legacy_v1/` | 保留早期实现和阶段记录；旧参考精度结果用于溯源。 |
| [对抗搜索 v3（历史）](adversarial_v3_20260912/) | `adversarial_v3_20260912/` | ES、Critic、RL 的早期实现；已由 v3.1 修订。 |
| [对抗搜索 v3.1](adversarial_v3_1_20260912/) | `adversarial_v3_1_20260912/` | 修正推理批形状后的 ES、Critic、RL 固定搜索及继续训练实验。 |
| [Critic 七方向研究](critic_search_20260915/) | `critic_search_20260915/` | D1–D7 的代理搜索与继续训练研究；共用同一研究协议。 |
| [Oracle-free 几何下界方法](oracle_free_20260915/) | `oracle_free_20260915/` | 使用几何下界构造目标，去除 GeoSteiner 依赖。 |
| [纯 RL 对抗训练](rl_only_20260915/) | `rl_only_20260915/` | 以 Actor 自身动作线长为目标，使用 indexed mean-L1 运输。 |
| [纯 RL 六方向研究](rl_research_20260915/) | `rl_research_20260915/` | REINFORCE/PPO、best-of-K、序贯、多智能体及种群方向。 |
| [WDRO 三论文迁移](papers_wdro_20260915/) | `papers_wdro_20260915/` | REST、Hub、Buffer/Timing 等论文任务的独立适配研究。 |
| [纯 RL 预训练后续方案](rl_only_pretraining_plan_20260919/) | `rl_only_pretraining_plan_20260919/` | 匹配预算的后续实验计划及既有结果分析；计划不等于已运行。 |

| 公共分类 | 内容 |
|---|---|
| [summaries/](summaries/) | 跨方法总览、状态、理论、运维、重构说明 |
| [benchmarks/](benchmarks/) | 公开芯片、v3.1 外部测试、RL-only 与主线比较 |
| [shared/](shared/) | 原始 DRO 依赖和附加盘存储的共享入口 |
| [literature/](literature/) | 文献调研和组会资料 |
| [tools/](tools/) | 项目工具、路径查询和重构完整性验证 |
| [archive/](archive/) | 历史原件、迁移快照、审计清单和旧缓存备份 |

每个方法的项目说明、理论、进展、结果和运行说明集中在其 `docs/` 中；代码包的测试与配置仍就近存放。大体积模型和数据没有再复制。

`revisions`、`improvements`、`wdro_rest_min` 等旧入口以及根目录少数旧 Markdown 名称是历史兼容链接。它们读取迁移前的源码/文档快照和同一份实验产物，以保留旧文件的 SHA-256；当前维护请从上表的新目录进入。完整列表、验证和限制见 [重构说明](summaries/RESTRUCTURE_20260920.md)。


GitHub 快照范围和超大报告下载方式见 [上传说明](UPLOAD_NOTES.md)。
