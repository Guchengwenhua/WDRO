# RL-only 当前状态

2026-09-19T11:54:58Z：**18/18 个正式 GPU 模型、6/6 个 GPU smoke 已完成，整批验收及本轮复验通过。**

本轮重核原完成链、全部任务产物/检查点、444,096 条保存解的线长、选模及调用预算；重新生成的正式报告、四张正式 CSV 和四张独立统计 CSV 均与原结果逐字节一致。原实验及后台验收进程已退出，锁已释放，无需继续等待 GPU 实验。

[本轮验收报告](../artifacts/acceptance_20260919/ACCEPTANCE_REVIEW.md) · [验收凭证](../artifacts/acceptance_20260919/release.json) · [完整独立统计](../artifacts/acceptance_20260919/final_statistics/statistics_review.md)

纯 RL 学习降低自身线长的可行性得到本实验支持；ERM-rollout 平均线长最低，WDRO-RL 尚未显示相对 Random 的稳定优势。平均运输成本略超 ρ；有限窗口攻击不提供严格 ρ 球最坏风险或最优误差证书。

本轮没有新增训练、模型前向或参考求解器调用。原 2026-09-16 验收及实验结果保留在原目录。
