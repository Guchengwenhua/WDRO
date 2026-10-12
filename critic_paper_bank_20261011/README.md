# Critic 参考值换算与 ICCAD checkpoint-bank 实验

2026-10-11 新的探索性实验。入口：[冻结参数](protocol.json)、[理论与实验边界](PROTOCOL.md)、[结果](RESULTS.md)、[运行交接](HANDOFF.md)。历史目录只读导入。

[实验验收报告](acceptance/REPORT.md)：21/21项已完成。Critic最终测试预测全部为0，未实现有效预测替代；23ICCAD的K3基本持平，25ICCAD约1%成本改善但训练耗时为2.59倍，best结果仅2/3种子改善。数据表和统计见acceptance/。

- 基础 WDRO：GeoSteiner / K3 / online Critic 换算，3 种子 × 128 更新。
- 23ICCAD Timing-RST：原 fixed-scale WDRO / 同任务 K3-relative WDRO，3 种子 × 256 更新。
- 25ICCAD HarRL buffering 组件：原 fixed-scale WDRO / 同任务 K3-relative WDRO，3 种子 × 256 更新。

统一 degree10、batch128、32 个候选扰动加中心、几何预选4、原 Hungarian 运输；paper 使用 rooted transport。共21次继续训练。每8步原子存档，保留模型、Critic、Adam、随机数、lambda、步数、最佳权重；验证每32步。最佳与最后一步均用真实指标评估，不以代理误差判断改善。

运行 `bash launch.sh`。同一命令恢复；已有 complete 的任务跳过。任务失败保留日志，队列继续其他任务，结果只读取完成项。每次任务完成自动生成 RESULTS.md、artifacts/metrics.csv 和 contrasts.json。

大产物入口 `artifacts` 链接到 `/attached/remote-home1/Guilin/DRO/Codex/critic_paper_bank_20261011/artifacts`，用于日志、数据、检查点和缓存。没有开展工程测试、smoke、逐文件哈希或验收平台建设。
