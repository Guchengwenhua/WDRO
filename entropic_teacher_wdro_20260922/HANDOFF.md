# 后台交接（2026-09-22 UTC）

正式队列已启动，状态为 RUNNING，不是科学实验完成。8张GPU各运行1个本实验worker，PID身份、CUDA设备、内核锁、新鲜心跳、latest/best检查点已核验。详细启动快照见 artifacts/operations/handoff.json；之后以 queue.json 和各worker heartbeat.json 为准。

## 本轮研究

详见 docs/理论与实验方案.md。4个新方向：每批求解运输预算的熵正则分布、温度延续、冻结教师相对损失、mean/CVaR尾部风险。与ERM、hard-max和softmax递推价格对照。全程无GeoSteiner；线长和教师相对损失不冒充最优regret。

共116个训练任务（112主/跨degree + 4调参），再自动执行224个best/last模型评价。degree30八种子：每父4096步预训练，每子8192步后训练，batch128、K16。degree20/40各三种子：预训练2048、后训练4096。第一阶段queue显示18项，是调参和14个父模型；不是全实验只有18项。

## 时间与后台行为

按生产批量GPU试跑估算，训练约183个worker小时，最终评价约2个worker小时；8卡理想约23小时，实际整轮预计24–48小时，中心估计32小时（约2026-09-23下午UTC），CPU/GPU争用或排队可进一步延长。operations/whole_run_eta.json保存估算；eta.json只覆盖当前阶段，不能当成整轮剩余时间。

GPU试跑14个任务通过；14项数学/数据测试和4项调度测试通过；7方法CPU逐张量恢复一致，EXCESS/TAIL在生产batch/K下GPU恢复一致；best/last自然与Random/ES评价流程小规模跑通。这些是工程证据，不证明方法有效。首轮GPU恢复检查仅因比较器直接比较CPU与CUDA上的优化器计数张量而报错；统一读到CPU后，所有已保存数值逐张量相等，训练计算仍在GPU。

自动流程：独立温度pilot与共同预训练 → 只看验证集选温度 → 全部七分支后训练 → 冻结best/last哈希 → 自然及独立Random64/ES64评价 → 配对统计、Holm校正和报告。负结果同样保留。无需本对话持续监控。

## 查看、暂停、恢复

```bash
tmux -S /remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922/artifacts/operations/et.sock attach -t et-wdro
```

Ctrl-b d脱离。tmux内进程将运行到全部完成或需要处理的失败；失败见operations/delivery.json与对应logs。暂停可创建artifacts/operations/PAUSE，worker存断点后退出；移除它会继续排队。重启入口为本目录bash launch.sh，若已有本分支worker存活会拒绝重复启动。不要删除锁文件来绕过检查。

源码和方案在 /remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922；artifacts指向附加盘同名DRO/Codex目录。资源准入、定期检查点、心跳超时处理、有限重试和磁盘/显存余量检查已启用，仅管理本队列子进程。

最终产物：artifacts/report/REPORT.md、metrics.csv、audit.json；完成凭证：artifacts/operations/delivery.json。尚无正式效果结论。
