# 三篇布线论文 × 基础 WDRO

本目录研究 `23_ICCAD.pdf`、`25_DAC.pdf`、`25_ICCAD.pdf` 的构树/时序/缓冲机制怎样接入原 Random/Sobol WDRO。没有使用 ES、Critic-gradient 或 RL 对抗样本攻击器。

1. [详细理论方案](THEORY.md)：适用性、改动量、数学目标与首批验证。
2. [2023 ICCAD 与 2025 DAC 审读](PAPER_23_DAC25.md)。
3. [2025 ICCAD 缓冲论文审读](PAPER_ICCAD25.md)。
4. [运行与恢复](OPERATIONS.md)、[固定参数](protocol.json)。
5. 最终报告将由队列生成至 `artifacts/report/REPORT.md`；是否完成应以 `artifacts/operations/exit.json` 和独立统计验收为准。

**范围：degree10、24 次正式 ERM/WDRO 配对训练，属于任务迁移可行性研究。时序/skew/buffering 的改善不能冒充原纯线长 WDRO 的改善。** 基础纯线长 REST 作为独立匹配对照。各家族内比较、不同家族目标不排总榜。CPU 检查通过不等于 GPU 正式实验完成。
