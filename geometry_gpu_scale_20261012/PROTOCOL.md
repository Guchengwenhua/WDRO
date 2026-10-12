# GPU几何熵优化及扩大训练方案

用户要求优化几何熵并扩充训练。本轮独立存放，保留旧结果，不覆盖旧CPU版本。几何训练沿用已授权的探索性范围，历史开发集门槛不再阻断本轮，但旧门槛结论不改写。

## 实现优化

仅计算当前使用的G或H，不再同时生成Q/C等辅助指标。解码器hook保留GPU张量，逐步骤累计分数，删除每步GPU到CPU的概率搬运；动作对只取上三角，利用对称性将G写成 `-sum_{a<b} p_a p_b D_ab (log p_a+log p_b)`，省略为零的对角线。路线描述仍是起点、路径中点、终点，正反向比较及包围盒宽高和归一化均保留。

GPU上构造与计算几何距离，不使用CPU NumPy生成大矩阵；概率、距离和分数累加采用FP64，路线构造采用原有FP32。梯度训练保留FP32，关闭TF32。新的求和/执行顺序可能改变接近并列时的排序，因此保留数学定义不等于已证明逐位数值等价。

候选排名推理microbatch由1024增至4096；四候选精评、validation及test仍使用1024。33候选及原样本+3个扰动的4参考预算不变。普通熵同步GPU化，原始/随机与G/H共享同一训练预算及4进程GeoSteiner参考求解。求解进程保持输入输出顺序、独立进程中的solver状态，所有臂同步调整。每更新记录shortlist、search、reference、materialize和optimization时间，速度从正式训练日志估计，不另跑调优基准。

## 训练规模与比较

d10/d30 × original/random/entropy/geometry × seed101/102/103，共24个新运行。batch4096，从同degree初始actor和每种子相同critic初始化，统一Adam学习率5e-5；新命名空间隔离旧轨迹。每个运行625更新=2,560,000样本更新，总61,440,000，较上一轮d10四臂125更新总预算扩大10倍。

在125与625更新保存权重并评估，125之后不按测试结果筛掉任何方法；继续训练不更换优化器、随机状态、lambda或候选设置。主结果固定latest625 OOD宏平均regret；G对三个对照、两个degree共6项主要比较，使用配对实例bootstrap的99.1667%区间，先对同实例跨种子平均，另列各训练种子效应。latest125、验证集best、最坏OOD、P95、成本为次级结果。记录真实求解预算和额外actor评分预算，不把等参考预算等同于等计算成本。

这是在历史测试数据已被查看后的探索性扩大实验，不是新盲测。新旧耗时差同时含GPU评分、microbatch、oracle并行和资源负载差异，不能只归因于某个kernel；G/H在当前队列中有共同对照。

## 持久运行

沿用物理GPU2–7，每卡最多一个本轮任务；d30训练需44,000MiB空闲，d10需35,000MiB，每进程最多55%显存并保留余量。六个worker配四个求解进程约使用30个CPU执行进程；不触碰他人任务。OOM最多重试一次并换卡。

所有大产物和临时缓存位于 `/attached/remote-home1/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts`，通过本目录artifacts链接访问。主盘8GiB、附加盘150GiB、总产物150GiB为边界。每6更新及阶段结束原子保存latest，保留best和125/625阶段快照；包含actor/critic/Adam/scheduler/RNG/lambda/参考计数。退出请求等待当前更新并保存，避免直接杀进程。

`bash launch.sh`启动tmux会话`geometry-gpu-scale-20261012`；状态`artifacts/state.json`、日志`artifacts/logs/`、阶段结果`artifacts/train/d{degree}_{arm}_s{seed}/u125或u625/`。独立CPU相关性监测不反馈训练，全部阶段结束自动写`artifacts/REPORT.md`。运行交接另存HANDOFF.md；协议本身不是训练完成证明。
