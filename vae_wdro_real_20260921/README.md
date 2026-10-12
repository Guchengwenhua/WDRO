# VAE 真实数据先验与 WDRO 微调

12/12组正式实验与自然测试已完成，2026-09-22 已验收。完成时间为2026-09-21 17:26:34 UTC（北京时间9月22日01:26:34），实际运行78分35秒。

验收结论：实验执行、数据隔离、共同预算和结果复核通过；未验证VAE球壳能稳定改善REST。B_test三seed平均mean regret：父模型2.5142%，RAW 3.0120%，SHELL 2.9876%；所有组验证最优均为update0。

后台进程与tmux已正常退出，单实例锁已释放。大产物通过 `artifacts` 链接存于附加盘。

- [验收结论](ACCEPTANCE.md)与[检查记录](artifacts/acceptance_checks.json)。
- [最终状态](artifacts/status.json)：完成组、退出前的GPU快照及错误。
- [完成时间](artifacts/done/pipeline.json)。
- [最终中文报告](REPORT.md)：last@256、val_best、负结果、机制限制与实际开销。
- [首组阶段报告](INTERIM_REPORT.md)：seed101完整对照后4.0秒生成，保留当时内容。
- [实际执行配置](artifacts/config.executed.json)、[数据划分](artifacts/splits.csv)、[数据计数](artifacts/split_summary.json)。

已完成全部数据划分、正式VAE L16/L32拟合、三类固定候选池和共享参考，以及4方法×3seeds的256步训练。保存27组逐net自然测试NPZ/CSV，共160245行；额外的两次并发父模型测试计算已在报告中据实计入。

VAE 入场时 GPU0 空闲，已在那里完成拟合；之后全部 GPU 被其他任务占用。REST 改用空闲 CPU，每个 seed 的全部方法锁定同一设备类型，最多4个REST作业、每进程1个数值线程。预算、评分和输入序列不变。没有停止其他任务。设备记录见 `artifacts/devices/`。

执行顺序：seed101 RAW/SHELL → 完整自然测试与阶段报告 → seeds102/103主对照 → ERM-A和VAE-BALL三seeds消融 → 最终报告。主表始终为共同last@256，另报验证最佳（包括update0）。本轮不安排可选MIX/AUG/512扩展或独立攻击。

首组阶段报告实际于2026-09-21 16:37:45 UTC生成，全部12组最终报告于17:26:34 UTC前完成，早于原预计18:00 UTC。

查看留存日志：

```bash
tail -n 40 /remote-home/Guilin/DRO/Codex/vae_wdro_real_20260921/artifacts/launcher.log
```

任务已完成，不需恢复。以下是原恢复命令；完成标记存在时会直接退出：

```bash
bash run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python run_all.py --config config.json --resume
```

各REST组每32步保存Actor、Critic、optimizer、价格、RNG和下一步索引；VAE每epoch保存。参考缓存采用完整坐标BLOB相等键，跨方法/seed复用。日志、逐net结果、last和val_best均保留。

- [完整实施方案](PLAN.md)：数据、模型、候选生成、预算、对照、评价和限时降级。
- [具体配置](config.json)：可直接作为实现参数来源。
- [初始交接](HANDOFF.md)：保留原始计划阶段交接；实时状态以本README和artifacts为准。

优先问题：同一真实数据和训练预算下，VAE 球壳候选是否比原坐标候选改善 REST 的独立真实测试表现？

主对照 RAW-WDRO / VAE-SHELL-WDRO；degree30；256 updates；3 seeds，先完成 seed101。遵照用户要求，不安排 smoke、SHA 校验或独立验收工程。
