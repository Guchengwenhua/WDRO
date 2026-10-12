# WDRO 三论文迁移

REST、Hub、Buffer/Timing 等论文任务的独立适配研究。

- [本方法文档与进展](docs/README.md)
- [全项目方法与测评总览](../summaries/方法与测评总览_20260919.md)
- [返回项目入口](../README.md)

此目录是重构后的实际维护目录。历史源码及路径绑定另存于 [迁移归档](../archive/restructure_20260920/README.md)。

---

# 三篇布线论文 × 基础 WDRO

本目录研究 `23_ICCAD.pdf`、`25_DAC.pdf`、`25_ICCAD.pdf` 的构树/时序/缓冲机制怎样接入原 Random/Sobol WDRO。没有使用 ES、Critic-gradient 或 RL 对抗样本攻击器。

1. [详细理论方案](docs/THEORY.md)：适用性、改动量、数学目标与首批验证。
2. [2023 ICCAD 与 2025 DAC 审读](docs/PAPER_23_DAC25.md)。
3. [2025 ICCAD 缓冲论文审读](docs/PAPER_ICCAD25.md)。
4. [运行与恢复](docs/OPERATIONS.md)、[固定参数](protocol.json)。
5. [2026-09-19 独立验收与结论](artifacts/review_20260919/REPORT.md)、[验收凭证](artifacts/review_20260919/acceptance.json)：24 次正式训练完成，工程与数值验收 PASS；稳定 WDRO 优势尚未建立。
6. [原始自动报告](artifacts/report/REPORT.md)、[最新状态](docs/STATUS.md)。

**范围：degree10、24 次正式 ERM/WDRO 配对训练，属于任务迁移可行性研究。时序/skew/buffering 的改善不能冒充原纯线长 WDRO 的改善。** 基础纯线长 REST 作为独立匹配对照。各家族内比较、不同家族目标不排总榜。CPU 检查通过不等于 GPU 正式实验完成。
