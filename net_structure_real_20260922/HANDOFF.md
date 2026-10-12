# 实验交接：net 结构先验与真实芯片验证

时间：2026-09-22T06:07:46.277626+00:00。状态：正式运行中，尚未完成全部训练与测试。
队列共166节点；完成27，运行8，待运行131；failed/blocked均为0。当前GPU：[0, 1, 2, 3, 4, 5, 6, 7]。

完整理论：[PLAN.md](PLAN.md)。当前结果：[REPORT.md](REPORT.md)。启动不表示方法改善；最终需查看逐设计、种子、预算、checkpoint和独立攻击结果。

## 实验规模

9种主方法×3 REST种子，加 graph_tile 2种额外生成器×3 REST种子，共33条训练。每条2048步，保存256/1024/2048预算。生成器6次调参、6次正式拟合；每正式生成器18000步多degree预训练+4500步d30后训练，batch128。预训练真实net40630，d30训练2810，B/C自然测试5935。

## 剩余时间

交接时保守预计还需约3–6小时，即2026-09-22约09:00–12:00 UTC。这是本机实测速度加验证/参考/队列余量的估计，不是承诺。外部GPU负载或显存变化会延迟。
当前动态估计：2.15–5.31小时；正式WDRO 0.763秒/更新。动态文件：artifacts/runtime_estimate.json。

## 查看与恢复

```bash
tmux attach -t net_structure_20260922
```
Ctrl-b，随后d可脱离，不终止任务。

状态：artifacts/heartbeat.json；逐项队列：artifacts/queue.json；日志：artifacts/logs；模型：artifacts/runs 与 artifacts/generators。每15秒心跳；每32 REST更新或250生成器更新checkpoint。

会话丢失时的恢复命令（监督器flock防重入，按PID/starttime接管存活worker）：
```bash
bash /remote-home/Guilin/DRO/Codex/net_structure_real_20260922/run_background.sh
```

后台包装器对监督器异常退出最多重试两次；工作任务也最多重试两次。若失败次数耗尽，supervisor_exit.json会列出failed/blocked。修复实际错误后才将相应状态改回pending；不要删除checkpoint或盲目重复启动。

## 已有验收证据

- tests/unit.json：集合/图等变、梯度、tile/bbox、optimizer/RNG精确恢复。
- tests/gpu_pilot.json：真实GPU、GeoSteiner和REST反向传播；535条真实net投影检查。
- tests/integration.json：隔离真实数据小规模34命令全链路通过，不作科学结论。
- tests/supervisor_adoption.json：监督器重启后8个worker原PID/starttime继续，无重复启动。
- tests/acceptance_guards.json：248个已保存候选存在pin至原tile的双射；未完成时禁止标完成。
- current_source_manifest.json、input_manifest.json、tests/additional_audit.json：源代码、数据、父模型、共享库绑定。

末阶段还会自动对所有10个候选池、4个tile候选池的每个候选匹配见证、33条训练及33项独立攻击执行独立产物检查，生成 artifacts/acceptance/final.json 与 completion.json。这里的独立指与生成实现不同的产物校验，不代表独立人类审稿。

## 资源与存储

实时余量调度8张卡；REST需至少2GB、生成器至少3GB空闲，allocator最多显存5%。GeoSteiner最多12个CPU进程，单GPUworker BLAS/torch为1线程。不干扰其他任务。
大产物实际存放于 /attached/remote-home1/Guilin/DRO/Codex/experiments/net_structure_real_20260922，经本目录 artifacts 访问。

最终解释边界：tile保持保证的是全局布线离散端点抽象不变；未包含完整cell移动、详细布线DRC和时序。历史已用设计做隔离测试，不称首次盲测；负结果同样保留。
