# WDRO-REST：主线、改进研究与外部验证

本项目研究怎样为 REST 生成有价值的困难引脚布局，并在平均几何运输预算下改善求解器的 relative regret。**当前数值主线是 precision_v2；新实验各用独立目录，结论按数据、预算和方法版本区分。**

## 从哪里开始

完整的方法说明、主要测评数字及“方法 × 测试方式”覆盖表见 [方法与测评总览（2026-09-19）](docs/方法与测评总览_20260919.md)。

1. [当前状态](docs/STATUS.md)：一页查看已经完成的研究和正在开展的方向。
2. [理论与方法](docs/THEORY.md)：真实评分、Critic、内外层目标和预算的关系。
3. [主线阶段汇总](docs/mainline/STAGES_0_5.md)：阶段 0–5；随后查看 [阶段 6](docs/mainline/STAGE_6.md)、[阶段 7](docs/mainline/STAGE_7.md)、[阶段 8](docs/mainline/STAGE_8.md)。
4. [改进研究](docs/improvements/ADVERSARIAL_V3_1.md)与[真实芯片外部验证](docs/EXTERNAL.md)。
5. [运行与恢复](docs/OPERATIONS.md)：环境、脚本入口、状态核实和资料维护。

## 按研究方向找文件

| 分类入口 | 内容 | 使用方式 |
|---|---|---|
| [mainline/](mainline/) | 当前 precision_v2 主 WDRO 与历史 v1 | 当前表、模型和配置用 precision_v2；v1 用于溯源 |
| [improvements/](improvements/) | 已验收 ES/Critic/RL v3.1 与 2026-09-15 Critic 七方向研究 | 每个独立版本保留自己的协议、代码、模型和结果 |
| [external/](external/) | 公开设计测试及 v3.1 degree=30 模型验证 | 全部属于冻结模型的零样本测试 |
| [docs/](docs/) | 合并后的阅读版本 | 当前阅读、解释及后续更新从这里进入 |
| [archive/](archive/) | 历史文档、失败分支、迁移备份和可恢复压缩包 | 查失败原因、旧阶段细节或原始报告时使用 |

分类入口大多为符号链接，进入的仍是已有真实目录。原实验源码、模型、缓存、表、日志及其绝对路径保持可用；不会因为分类入口产生另一套训练结果。

## 主线关键路径

当前版本的源代码和阶段实现位于 [precision_v2/wdro_rest_min](revisions/reference_precision_v2/Codex/wdro_rest_min/)。常用入口：

| 要找的内容 | 位置 |
|---|---|
| 冻结配置与方法参数 | [configs](revisions/reference_precision_v2/Codex/wdro_rest_min/configs/) |
| 完成和来源凭证 | [manifests](revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/) |
| 模型身份 | [75 个模型冻结索引](revisions/reference_precision_v2/Codex/wdro_rest_min/manifests/stage6_models_frozen_v1.json) |
| 模型权重与恢复状态 | [checkpoints](revisions/reference_precision_v2/Codex/experiments/checkpoints/) |
| 合成、外部和攻击统计 | [tables](revisions/reference_precision_v2/Codex/experiments/tables/) |
| 图与运行证据 | [figures](revisions/reference_precision_v2/Codex/experiments/figures/)、[logs](revisions/reference_precision_v2/Codex/experiments/logs/) |

`experiments/` 使用既有附加盘映射；所有新研究的逻辑路径仍放在 `DRO/Codex/` 内。各分类入口 README 说明原始目录与实际用途。

## 如何读结论

- 主线完成了 75 模型的合成测试、ISPD 和独立 CEM-256 压力测试；WDRO 相对 ERM 的指定指标有改善，不能概括为对所有基线、degree 或分布都更好。
- v3.1 的 ES64 更明确地改善固定模型上的有限预算搜索；继续训练实验没有建立相对既定对照的稳定优越性。
- 真实芯片测试是二维单 net 线长评价，degree、设计覆盖和统计权重需要一起报告。
- 代理输入梯度、有限候选最大值与局部压力测试均不构成连续 Wasserstein 风险证书。

## 本次资料整理

旧根目录的阶段、方案与恢复文档已经集中归档，并合并为上述专题阅读版本。旧文件名多数保留为兼容符号链接；根进度及已绑定的最终报告仍保留原件。这样旧链接和历史 SHA-256 可以继续核验。

旧报告中的 `DOING`、待启动、旧 GPU 编号和旧时间估算是历史快照；阅读当前情况请看 [docs/STATUS.md](docs/STATUS.md) 和新分支自身状态。完整整理范围、数量、原件清单和恢复方法见 [整理说明](docs/ORGANIZATION.md)。
