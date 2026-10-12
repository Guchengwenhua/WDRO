# RL 预训练与后训练扩大实验

在8个相同seed上，基础预训练与后训练分别从2048扩大到8192步，另诊断攻击器预训练扩大；104项新增任务，继承预算单独记账。

* [详细理论和冻结实验矩阵](docs/THEORY_AND_PROTOCOL.md)
* [精确任务与参数](protocol.json)
* [当前状态](docs/STATUS.md)
* 实时队列和预计时间：`artifacts/operations/queue.json`、`eta.json`；准备阶段详见同目录`bootstrap.json`
* 最终报告：`artifacts/report/REPORT.md`；终态验收：`artifacts/operations/delivery.json`

旧轮 `rl_scaleup_20260921` 按原协议继续。新tmux先接续剩余工程试跑，核验CPU精确恢复、13项GPU路径和显存上限，全部通过后自动等待旧轮完成验收，然后接棒最多8张GPU；不需要保持对话在线或手工调度各阶段。任一工程门禁失败则记录NEEDS_REVIEW，不会绕过检查启动正式队列。

启动/恢复（拒绝重复队列、活跃旧worker和源码变化）：

```bash
bash /remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922/launch.sh
```

连接tmux：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922/artifacts/operations/prepost.sock attach -t rl-prepost
```

创建 `artifacts/operations/PAUSE` 可在训练边界保存并暂停；已开始的预定评价会先完成。删除此文件后由队列继续。资源暂停不消耗失败次数，普通失败最多重试3次。
断电/主机重启后用上述启动命令恢复；若旧worker仍活跃则先保留它们，不能重复启动。

代码和协议冻结后不得修改。GPU/CPU工程校准在 `artifacts/gpu_pilot`、`cpu_resume`，不会混入正式结果。
旧checkpoint由manifest和源码/父绑定校验后复制到 `lineage/`，该目录保持只读使用；所有新数据、临时文件、缓存、日志、模型和socket都在本实验目录。

工程COMPLETE表示104任务与审计完成，科学SUPPORTED另按预注册检验判断；训练更久可能没有改善，也可能退化，报告保留全部结果。
