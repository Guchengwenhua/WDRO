# 全模型 × 全数据集扩展复测

唯一结果表：[全模型全数据集总表](../../reports/全模型全数据集总表_20260928.html)。尚未评估的格子显示“待测”；只有完成标记为 COMPLETE 才表示本次队列和目录整理全部结束。

评估口径见 [PROTOCOL.md](PROTOCOL.md) 和 [protocol.json](protocol.json)。模型登记在 `artifacts/models.json`，数据列登记在 `artifacts/datasets.json`，排除的工程权重及恢复分片在 `artifacts/excluded_checkpoints.json`。这些 JSON 是机器可读登记，不是额外对外比较表。

后台状态：`artifacts/status.json`；总日志：`artifacts/logs/pipeline.log`；逐模型日志：`artifacts/logs/Mxxxx.log`；预测与断点：`artifacts/results/Mxxxx/d*/`；最终状态：`artifacts/completion.json`。

启动或恢复：

```bash
bash /remote-home/Guilin/DRO/Codex/benchmarks/all_models_all_data_20260928/launch.sh
```

进入 tmux：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/benchmarks/all_models_all_data_20260928/artifacts/evaluation.sock attach -t dro-all-data
```

恢复读取已提交的 8192-net 分片与游标，不重算完成的规模/模型。模型级锁限制并发写入。运行只读取已有模型权重；调度器根据实时 GPU 余量和利用率分配设备，GPU 暂不可用则等待。记录实际 UUID、batch、每秒处理数和条件性 ETA。

完成后自动将实际方法目录放入 `methods/`，总结、验收和文献归入 `reports/`，提供 `data/` 分类入口。保留必要的旧目录兼容链接，根目录普通文件只保留 README。迁移保留源文件内容，写入可恢复的移动记录；历史冻结任务若依赖解析后的绝对路径，可用 `tools/restore_layout_20260928.py` 恢复旧布局。

大文件实际存于 `/attached/remote-home1/Guilin/DRO/Codex/experiments/all_models_all_data_20260928`，由本目录 `artifacts` 链接访问。
