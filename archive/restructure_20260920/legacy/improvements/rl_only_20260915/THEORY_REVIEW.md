# 独立审查意见：严格无参考算法的 RL-only 方案

日期：2026-09-15。详细理论入口：[THEORY.md](THEORY.md)。本文件是设计审查，不代表 GPU 实验已经通过。

## 1. 对用户纠正的准确响应

此前使用 `B=max(HPWL,MST/2)` 的方案虽然没有 GeoSteiner，仍依赖非学习参考算法，不能满足新的明确限制。正确处理是暂停旧任务、保留历史证据，新建独立目录和 fresh initialization；不能只把 B 改名为“无监督尺度”后继续。

新方案的唯一主任务反馈是 `reward=−自身动作线长`。长度与合法性测量属于环境，不计算或近似参考解。若无法接受环境代价测量，问题本身就缺少可识别的“更好/更坏”信号；无需再训练一个没有外部锚点的 Critic 来掩盖这个缺口。

## 2. 推荐决定

- 主路线：sampled REINFORCE + learned state-value baseline，直接最小化自身实测线长。
- 必要内部替代：self-critical greedy baseline；它完全由自己的 Actor 产生，代价是额外前向。
- WDRO：优化绝对线长，使用有对应 pin 的 indexed L1 位移成本，不用 Hungarian。
- 最严格 learned adversary：坐标策略 RL；Random/ES/Critic 提案仅作明确命名的对照。
- 不混入 learned reference ratio、快照相对 regret、self-imitation 或 TD/GAE，以免第一轮同时改变过多机制。
- 固定物理窗口，不做逐 net bbox 归一化；原实验数据、ρ、λ、指标与结果不可直接当同一问题比较。

已选定六分支名为 `erm_critic`、`erm_rollout`、`wdro_random`、`wdro_es`、`wdro_critic`、`wdro_rl`。预算事先固定 `ρ=0.04`、`λ₀=1`、`ηλ=10`，不运行 q75 或其他几何分位数标定。数据统一仿射映射到 `[0.1,0.9]²`，扰动逐坐标限制 ±0.08 后裁剪到窗口。

## 3. 最容易写错的五个结论

1. **“Critic 学到了最优线长。”** 实际 state-value Critic 学的是当前策略平均成本 Vπ；除非证明已达到最优策略，否则不是 V★。
2. **“换成 baseline 后仍优化 regret。”** detached baseline不改变raw成本梯度；分母和状态分布改变则是另一个目标。
3. **“同步随机排列等于 Hungarian 集合距离。”** 只保持共同重排不变，仍是有标签/对应关系的成本。
4. **“没有求解器所以一定更慢/更快。”** 每次反馈较便宜，但样本效率、Actor 开销和对手训练共同决定总时间。
5. **“内部模型都同意就代表接近最优。”** 学习者可能共同犯错；自身改善和互相胜率都不能替代最优性证书。

## 4. 实现审查清单

### 4.1 禁止依赖

新目录源码/配置/产物中不应出现实际执行的 MST、HPWL/B 分母、GeoSteiner、Hungarian assignment、外部路由算法、旧最优缓存、旧 checkpoint 加载。禁止模式的 runtime guard、已加载模块/动态库检查和输入路径绑定应与静态检索互补。

同目录新实验自己的 resume checkpoint 可以加载，但必须绑定本次协议与初始化来源。禁止的是旧 oracle 或 B/MST 分支训练出来的权重；把旧权重复制到新目录再加载同样不符合要求。

禁止词可以出现在说明和 guard 中，因此不能把 `rg` 匹配个数机械地当违规结论。需要检查调用图和实际文件访问。为复用 pin 生成器而导入的旧模块若同时导入禁止算法，也应拆成干净的新模块，而非声称“没有调用就等于完全无依赖”。

### 4.2 Actor–Critic 数学

- Actor advantage 必须 detach。
- state baseline 在利用当前动作更新之前计算；不能先fit再重新预测同一动作的基线。
- SCST 的 greedy 输出不参与当前采样 log probability，也不反传。
- 部分树增量 reward 如使用，望远镜求和应精确等于最终环境记账成本；首轮推荐终局 reward。
- 动作 masks、拓扑连接合法性、索引范围和数值有限性必须验证。

### 4.3 对抗器数学

- 价格 λ 仅在本批选择之后按平均位移更新；使用旧 λ 评分，与同批奖励保持一致。
- indexed cost 在提案、选择、校准和验收中使用同一定义。
- RL 的 log probability 对应裁剪前采样动作；同状态多采样可以用 leave-one-out baseline。
- ES 若归一化方向，应明确具体平滑分布；其不是原始离散函数的精确梯度。
- 搜索 Critic 用 greedy C 标签，主 Critic 用 sampled C 标签；最终候选用实测长度复核。
- 最终选择包含原中心，若最高分低于原中心则属于实现错误。

### 4.4 公平性与恢复

- 相同 seed 的模型初始化 hash 相同，所有分支 fresh init。
- 输入流与动作 RNG 单独控制，攻击器的随机采样不能错开 Actor 的配对随机流。
- SCST 多出来的 Actor forward 单独记账，不把相同update预算说成相同算力。
- 新协议与旧目录的 resume 禁止混用；checkpoint绑定源码、数据/配置、优化器与随机状态。
- 固定 greedy批形状；保存 topology便于离线长度复核。
- 完整任务退出、锁释放、预算检查与报告生成后才标记正式完成。

## 5. 数据与评价解释

原始成本 WDRO可能偏向几何本来较大的实例。固定窗口和位移限制可以控制可达空间，不能让成本变成相对最优误差。不要为解决这个解释难题又偷偷加入 HPWL 或 learned L★ 蒸馏。

建议配对报告每组 mean/P95线长、before/after、learned methods相对Random-WDRO、运输成本与λ全程/尾段、每个训练seed。所有方法面对完全相同的held-out中心与预先固定攻击器；test不参与选模。

保存动作并重算线长，能验证环境记账和统计正确；没有最优参考时不能验证绝对 optimality gap。这是信息边界，不是换一个更复杂Critic就能绕开的误差。

## 6. 设计审查结论

此路线在理论与工程上都可执行，而且比“预测未知 L★ 再装作已知标签”更清楚。最短证据链应是：**无禁止依赖 → 随机初始化策略实际学短 → 相同输入上稳健改善 → learned adversary额外收益。** 各层分别检验，不能把一次smoke当作整条链都成立。
