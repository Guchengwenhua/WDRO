# v3.1 模型外部测试

[说明、进展与结果](docs/README.md) · [测评总目录](../README.md)

# v3.1 新训练模型：现有全部真实数据评测

新增 6 路线 × 3 种子共 18 个模型：RandomTop4、Critic4、RL frozen4、RL joint4、Random64、ES64。另对原 5 方法 × 3 种子共 15 个 degree=30 模型按完全相同的推理规则重测。

每个模型使用 11,032 条合法 degree=30 网络，来自现有 ISPD2018/2019 Protocol A、ISPD2008 global routing、CircuitOps Nangate45/ASAP7/SKY130HD。49 个设计全部进入覆盖清单，43 个含可用网络。没有截断引脚、抽样、重新训练或使用真实数据选模型。

新模型的适用 degree 只有 30；A 实验中的 Critic/RL 攻击器不是新 REST 求解模型。全部使用原合成验证集选择的 best checkpoint。固定 256 行 greedy 推理，尾批重复最后一行、输出丢弃补齐部分。

所有输出通过本目录 `artifacts/` 链接写入 `DRO/Codex/shared/experiments/real_models_v31_20260912`，使用项目既有 NVMe 存储映射。

- `artifacts/STATUS.json`：流水线实时状态（running / failed / complete）。
- `artifacts/logs/queue/state.json`：已完成数、活动模型、物理 GPU、已提交网络数。
- `artifacts/logs/queue/resources.jsonl`：持续 GPU 容量/利用率记录。
- `artifacts/models_frozen.json`、`input_manifest.json`、`runtime_frozen.json`：模型、数据和执行代码哈希。
- `artifacts/smoke/completion.json`：真实 GPU 尾批、原 v3.1 Oracle 一致性、跨卡续跑、幂等和独立审计测试。
- `artifacts/REPORT.md`、`summary.json`、`tables/`：完成后自动生成的中文报告与逐设计/数据集/种子统计。
- `artifacts/completion_manifest.json`、`delivery_verification.json`：仅在全部 33 个任务及最终验收通过后生成。

启动或恢复（已有 tmux 会话和匹配存活进程会保留，完成分片不会重复评分）：

```bash
bash /remote-home/Guilin/DRO/Codex/benchmarks/v31_models_20260912/run_tmux.sh
```

查看终端会话：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/shared/experiments/real_v31.sock attach -t real-v31-20260912
```

会话输出已持久化到 `artifacts/logs/pipeline.log`。运行完成后会话正常退出；断开 SSH 或结束 Codex 对话不会停止它。发生错误时保留 failed 状态、异常栈、完整日志和可恢复分片，不会把未完成工作标为 DONE。

队列动态检查全部 8 张 GPU，最多每卡 8 进程、全局 32 进程。按可用显存留出外部任务波动余量；已有外部任务接近饱和的卡待资源释放后再加入。显存分配以有效推理需要为准，主要用并发提高吞吐。
