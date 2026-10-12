# 运行交接：GPU几何熵扩大训练

记录时间：2026-10-12 13:53:22 UTC+8。当前为正式训练运行中，不是完成验收。

24个独立运行：d10/d30 × original/random/entropy/geometry × 3种子。每个batch4096、625更新，先保存125更新阶段再自动续跑；总样本更新61,440,000。GPU2–7同时执行6个任务，未使用GPU0/1。

|任务|GPU|已完成更新/总更新|累计平均秒/更新|最近候选评分秒|
|---|---:|---:|---:|---:|
|d10_geometry_s101_u125|2|15/625|14.87|1.63|
|d10_entropy_s101_u125|3|15/625|14.53|1.29|
|d30_geometry_s101_u125|4|4/625|56.36|17.23|
|d30_entropy_s101_u125|5|5/625|43.59|6.56|
|d10_original_s101_u125|6|16/625|14.19|1.30|
|d10_random_s101_u125|7|17/625|12.96|0.00|

以上计时来自刚启动的正式训练，累计平均包含启动和周期验证开销。d10完整625更新约2.5–3小时/运行；d30几何目前约10小时/运行、普通熵约8小时/运行。假定6张卡持续可用，整队列暂估约1–2天；d30步数尚少，精确参考求解、阶段评估与资源等待会改变实际时间。此估计不是截止时间。

持久会话：`geometry-gpu-scale-20261012`。socket：`/remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/parallel.sock`。controller PID：`782874`。独立相关性监测在同会话的correlation窗口，使用保存的检查点，不反馈训练。

状态与日志：`artifacts/state.json`、`artifacts/controller.log`、`artifacts/logs/`、各运行`heartbeat.json`与`logs/events.jsonl`。完整启动快照：`LAUNCH_RECEIPT.json`。检查点每6更新及阶段末原子保存，含模型、优化器、调度器、随机状态和lambda；125/625阶段另保留latest/best快照及评估。

实际大文件位置：`/attached/remote-home1/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts`。项目内artifacts符号链接可定位全部数据。阶段结果位于`artifacts/train/d{degree}_{arm}_s{seed}/u125/`或`u625/`。所有阶段终结后自动汇总到`artifacts/REPORT.md`，失败与缺失会单列，不等同于24运行全部成功。

查看会话：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/parallel.sock attach -t geometry-gpu-scale-20261012
```

启动命令为在本目录执行`bash launch.sh`，当前已启动，不要重复启动。意外中断时，先确认原controller已退出；若tmux会话不存在可再次执行该命令。controller会按PID及启动令牌接管仍存活的所属任务，并按完成标记跳过完成项；训练从latest检查点恢复。若tmux会话还存在但controller已退出，可在该会话新建窗口运行下面命令；不要与活跃controller并行启动：

```bash
cd /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012
/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B controller.py >> artifacts/controller.log 2>&1
```

实现与比较边界见`PROTOCOL.md`：保留G数学定义但未做逐位等价检查；训练保留FP32、评分累加FP64。当前只是优化实现和扩大训练已经启动，尚无新一轮精度结论。
