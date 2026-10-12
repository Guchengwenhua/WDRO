# net 结构先验真实芯片实验

详细中文理论与冻结协议见 [PLAN.md](PLAN.md)。本分支含可解释结构门控、图相关扰动、集合/图去噪预训练和全局布线 tile 保持方法。结果见持续更新的 [REPORT.md](REPORT.md)。完整科学结论须等待队列完成；工程启动不表示新方法优于旧方法。

- 9 个主方法，3 个 REST 种子，共 27 条；graph_tile 的另外 2 个生成器种子×3 REST 种子，再加 6 条，共 33 条。
- 6 次生成器调参（7500 updates），6 次正式训练（22500 updates），真实多degree预训练集 40630 net；d30 训练2810、验证1085、B测试3991、C测试1944。
- REST 每条保存256/1024/2048步的 last/val_best，自然测试与独立RAW-K32测试均自动执行。
- 所有大文件经 artifacts 链接存于附加盘；旧 VAE/CVAE 分支只读。

查看队列：
```bash
cat /remote-home/Guilin/DRO/Codex/net_structure_real_20260922/artifacts/heartbeat.json
cat /remote-home/Guilin/DRO/Codex/net_structure_real_20260922/artifacts/runtime_estimate.json
```

进入后台会话：
```bash
tmux attach -t net_structure_20260922
```
按 Ctrl-b，然后 d，可脱离会话而保留运行。

恢复入口（先检查现有会话；入口另有单实例锁，自动接续 checkpoint，不重复启动存活的 PID/starttime）：
```bash
bash /remote-home/Guilin/DRO/Codex/net_structure_real_20260922/run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python supervise.py
```

queue.json 记录任务、依赖、attempt、GPU UUID、PID及进程启动时间。每15秒心跳，生成器每250步、REST每32步持久化。失败最多重试两次，仍失败会显式阻止依赖任务并输出 supervisor_exit.json。重新启动不会擅自清除 failed；修正具体错误后才能将对应 failed/blocked 任务改回 pending。暂时没有GPU时等待，不降低正式预算或冒充完成。结束后 completion.json 才是工程完成标记。

科学边界：tile 保持固定的是粗粒度路由端点，不是完整电路单元移动的合法性；不保证 detailed routing、DRC、时序或可制造性。B/C为历史已见数据的设计隔离测试，不宣称首次盲测。
