# 本轮扩展实验入口

启动时间：2026-09-22 04:17:52 UTC（北京时间12:17:52）。用户允许调参，其他要求沿用原实验；所有新代码及文档在本目录，历史目录只读。

## 实际执行规模与顺序

1. 共同微调：6种设置×ERM/RAW×2seeds，256updates；其中4项复用旧验证记录，20项新训练。只在A_val选一个设置，所有后续方法共用。
2. 局部条件VAE：L=[8,16,32]、beta=[1e-4,1e-3]共6项，用固定A_val条件先验生成指标选取；只用A_train拟合、A_cal校准。
3. RAW与LOCAL-SHELL先完成seed101、256updates的全部自然测试并自动生成INTERIM_REPORT.md，再补102/103。
4. RAW、LOCAL-SHELL、ERM-A、LEGACY-SHELL、MIX-LEGACY、LOCAL-BALL、MIX-LOCAL共7方法×3seeds，依次完成256/512/1024updates，并在每个预算报告last和验证最优。
5. 主实验完成若仍在启动后8小时内，追加2个独立生成器×3REST seeds的LOCAL-SHELL；11小时内则追加共享RAW-Random32攻击。是否入场只由时间决定，不由测试成绩决定。

最多47个新REST训练配置，外加6个CVAE调参拟合及2个独立生成器拟合；其中27个REST配置目标1024步，其余20个为256步调参诊断。准确规则见PLAN.md与config.json。config中`vae`及旧`methods_*`字段保留前轮来源；本轮实际方法由`primary_methods`、局部模型由`cvae`定义。

## 后台执行与恢复

训练队列tmux：`vae_wdro_extension_20260922`；守护进程tmux：`vae_wdro_extension_watch_20260922`。

进度：`artifacts/status.json`、`artifacts/progress/`；日志：`artifacts/launcher.log`及`artifacts/logs/`。守护进程只在本实验队列消失时恢复队列，队列会接管仍存活的本实验子进程，不停止其他任务。

每32步保存模型、优化器、RNG和lambda；各预算另存last/val_best。CVAE每epoch可恢复，候选参考使用持久缓存。异常连续失败两次会保留日志和部分报告，不伪称完成。

若服务器重启，进入本目录后在tmux运行：

```bash
bash run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python -u supervise.py
```

该入口有单实例锁；无需重做已经完成的训练、参考或测试。

## 资源与交付

GPU每次入场前检查其他计算进程、显存和利用率，只有完全空闲卡可用。目前8卡均有其他任务，使用4个单线程CPU计算任务；同REST seed比较固定设备类型。参考队列另最多8个worker。

大产物：`/attached/remote-home1/Guilin/DRO/Codex/experiments/vae_wdro_real_extension_20260922`，本目录`artifacts`为链接。未运行smoke、SHA或独立验收工程。

动态估时见`artifacts/runtime_estimate.json`，来自正式训练的实际速度。初始预计首对照2026-09-22 06:15 UTC，主实验12:30 UTC，含可选扩展15:00 UTC左右；并行负载或参考解耗时可使其推迟。

- `DIAGNOSTIC_REPORT.md`：调参及BN统计交叉诊断，不是测试成绩。
- `INTERIM_REPORT.md`：首组完整RAW/LOCAL-SHELL对照完成时自动生成。
- `REPORT.md`：中文动态报告，全部作业结束后自动标为最终报告。
- `tables/main_last_*.csv`、`main_val_best_*.csv`：共同预算主表。
- `tables/test_files.csv`：逐net自然测试CSV/NPZ索引。
- `tables/costs.csv`：各续训阶段新增实际请求、padding和耗时。
- `tables/independent_attack.csv`：共享攻击结果，若按预定时间门槛入场。

当前只是代码就绪和正式任务运行；训练完成、测试改善、鲁棒性改善必须分别根据产物判断，负结果完整保留。
