# 给新 Xhigh 对话的执行指令

请读取 `/remote-home/Guilin/DRO/Codex/vae_wdro_real_20260921/PLAN.md`，按照该方案实现并运行真实数据 VAE 球壳 WDRO 实验。当前只有方案，没有新实验代码或训练结果。

我的目标：明天拿到可解释的实验结果，优先完成 degree30 的 RAW-WDRO 与 VAE-SHELL-WDRO 同初始化、同数据、同256-update预算对照；先seed101完整结果，再seeds102/103，随后补ERM-A与VAE-BALL消融。

不要做smoke、SHA/哈希校验、独立验收工程或长篇重新调研。正式运行内必要的数值/维度/坐标/数据划分检查要保留。不要改动冻结方法和历史结果。不需要再次询问是否开始，实现后直接运行授权范围内的实验，并持续推进到结果报告。

现成输入、参考解、父checkpoint和Python路径都已写入PLAN。真实数据共11032个degree30 nets，使用按设计family隔离的固定划分。VAE只学习A_train；A_cal校准，A_val选模；B/C不用于调参。候选池固定并预计算Lstar，以节省GeoSteiner成本，所有组用同一真实评分与运输成本。

工作目录：`/remote-home/Guilin/DRO/Codex/vae_wdro_real_20260921`。大产物放PLAN指定附加盘，通过本目录artifacts链接访问。启动前读一次实时GPU状态，只使用空闲资源。使用可恢复的tmux/nohup和checkpoint；不要停止其他作业。

从正式任务前32个updates估计吞吐，不另跑smoke。首个完整主对照完成就写INTERIM_REPORT.md，不等待所有消融。最终必须有中文REPORT.md、逐net自然测试结果、主对照表、训练时间/实际预算、last与val_best，负结果也如实交付。时间不足按PLAN删可选项，不悄悄改变主比较口径。
