# RL-only 与主线的交叉比较

[说明、进展与结果](docs/README.md) · [测评总目录](../README.md)

# RL-only 与主线方法的统一评测

本目录比较已冻结的 33 个 degree-30 Actor：主线 ERM、Mixed ERM、Hard Mining、GroupDRO、WDRO-REST 各三个种子，以及新 RL-only 六方法各三个种子。

- 两套完整既有测试集分别报告：RL-only 的 4,096 nets；precision_v2 主线的 40,000 nets。
- 新方案原 512 个攻击中心及每中心 32 个候选保持不变，为全部旧模型补充实际自身线长评分。
- 同一套数据内坐标、Actor 架构、greedy 模式、batch 64 与末批 padding 完全一致，不再做额外归一化。
- 复用新模型已经验收的 RL-only 测试/攻击结果，并为每模型每格执行 64 行兼容性重放；主线测试集对全部模型重新推理。
- 所有模型保留原 validation-best；不训练、不调参、不使用 GeoSteiner/最优标签。旧模型的历史训练仍依赖参考目标。

主线起点是 DAC21，微调预算为 2,000×256；新模型是随机初始化，预算为 2,048×64。最终表比较现有模型的效果，不是同起点、同计算量的算法消融。跨两批模型独立重采样训练种子，并对相同测试 nets 配对重采样。

## 文件

- `protocol.json`：已固定的评价范围、统计和限制。
- `manifest.json`：33 模型、12 数据格及源文件的哈希绑定。
- `artifacts/report/REPORT.md`：全部完成并通过独立数值核验后生成。
- `artifacts/operations/queue.json`：当前任务状态。

## 后台运行

调度器复用上一实验经过验证的实现。每 GPU 至多一个 worker、每 worker 3 个 CPU 线程；三次连续低利用率且显存余量满足门槛才接入，每进程显存分配上限 10%。不干预其他任务。输出按数据格原子提交，恢复时验证并复用已完成格。

```bash
tmux -S "$PWD/artifacts/operations/compare.sock" new-session -d -s rl_compare_20260919 "cd '$PWD' && exec bash run_env.sh scheduler.py --poll-seconds 5 --max-gpu-util 20 --min-free-mib 8192 --memory-margin-mib 2048 --max-workers 8 > artifacts/operations/scheduler.log 2>&1"
```

全部模型完成后自动运行 `report.py`，从保存的拓扑独立重算全部线长，核验哈希、调用账本和运行锁，再生成统计及完成凭证。全批结果预计小于 2 GiB；调度磁盘门槛为 20 GiB。
