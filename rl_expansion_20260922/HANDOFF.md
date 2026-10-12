# 扩量实验交接：已完成并独立验收

2026-09-23 23:56:55 UTC 正式实验完成；2026-09-26 独立验收完成。

结论：核心工程、预算、数值和预注册统计检查通过；WD 权重衰减消融因 FP32 舍入退化为无效比较。验收状态为 `PASS_CORE_WITH_DEGENERATE_WEIGHT_DECAY_ABLATION`。

58 个训练任务、71 个测试任务全部完成；新增 712,704 次更新、50,331,648 条采样。独立检查 1,224 个文件哈希，复算 4,800,512 条保存线长，最大绝对误差为 0。队列进程已退出，锁已释放，无待运行任务，无需重新启动 tmux。

E50 训练由 12,288 延长到 49,152 步，五个种子的平均线长下降 2.428%，最差组下降 2.622%，达到预注册探索性门槛。训练步数和在线新样本数共同增加；结论不能单独归因于预训练、后训练或更新次数。其他调参和容量方案未通过晋级门槛；batch256 未显示同样本数优势。WD 不能用于判断权重衰减是否有效。

详细报告：[ACCEPTANCE_20260926.md](/remote-home/Guilin/DRO/Codex/rl_expansion_20260922/ACCEPTANCE_20260926.md)。

独立证据与交付清单：[artifacts/acceptance_20260926/complete.json](/remote-home/Guilin/DRO/Codex/rl_expansion_20260922/artifacts/acceptance_20260926/complete.json)。

原启动交接保存在 [HANDOFF_at_launch.md](/remote-home/Guilin/DRO/Codex/rl_expansion_20260922/artifacts/acceptance_20260926/HANDOFF_at_launch.md)，与原 `artifacts/operations/delivery.json` 的 handoff_sha256 完全一致。该旧 delivery 是 9 月 22 日的启动凭证，当前完成状态请看本文件、独立验收报告和 `artifacts/lifecycle.json`。原协议、源码、模型、数据与自动报告没有修改。
