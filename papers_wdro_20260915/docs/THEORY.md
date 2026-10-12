# 三篇布线论文与基础 WDRO：理论方案与验证协议

日期：2026-09-15。本文先于本分支正式 GPU 训练冻结。当前是独立的 degree-10 可行性研究，旧 precision_v2 的协议、模型与数值表保持原有身份。

## 1. 先回答问题：能用什么，改动有多大

**能够组合，但三篇论文并没有给出一种能直接提高原 WDRO 线长 regret 的通用技巧。它们主要改变内层布线求解器、其动作空间和优化目标。**

如果“现在的 WDRO 方法”指 Random/Sobol 候选 + Hungarian 运输代价 + 非负 λ 对偶训练，这个外层框架可以保留。如果要求连原 REST 网络、纯线长目标、无源点的输入语义都完全不变，则可直接迁移的内容很少；尤其缓冲器位置在原纯线长任务中没有对应的奖励，不能预期仅加谐波模块就提升线长。

| 本地论文 | 实际核心内容 | 保留纯线长任务可迁移部分 | 完整任务接入需要什么 | 改动判断 |
|---|---|---|---|---|
| 23_ICCAD | 动态图注意力构树；线长与最长源到接收端路径的权衡；大 degree 的几何分解合并 | 可单独研究图状态编码/构树网络；需重新训练并和 REST 做同任务消融，无法从时序结果推出线长改善 | 源点标签、最长路径评价、目标权重；大 net 还需分解合并与全局代价审计 | 只接目标与已有解码器是小到中等；换原论文网络为中等；完整大 net 框架较大 |
| 25_DAC | Selector 选 hub，Conductor 决定继续/停止，再由 MSPD 构树，优化线长-skew | hub 结构可作为另一种求解器，但 MSPD 本身已有较强算法成分；须独立核算其费用，不能把收益归于 WDRO | hub 动作、停止概率、源点、MSPD 评分器、线长-skew 权重 | 中等到较大，不能直接加载 REST 权重 |
| 25_ICCAD | 联合构树与谐波缓冲候选，结合缓冲插入优化 Elmore 延迟与缓冲数量 | 原纯线长目标中缓冲动作缺少用途；没有直接迁移收益保证 | RC 参数、源点/负载、拓扑+相位动作、几何合并、缓冲插入、实际延迟评价 | CPU/GPU 组件试验为中等；工业级全流程与论文完整算法复现较大 |

这里“中等”表示需要新的 adapter、目标评价和配对训练；“较大”表示还要增加全新物理/几何算法与验收数据。不能把本次小 degree 原型的开发时间当作工业实现工期。

论文与源码分别核对自[作者发表页](https://hu-ding.github.io/publications.html)、[Timing-RST](https://github.com/LiynnYang/Timing-RST)、[Hub Node Selection](https://github.com/USTC-DIAG-LAB/An-Adaptive-Learning-Approach-for-Hub-Node-Selection)、[RLBufferTree](https://github.com/USTC-DIAG-LAB/RLBufferTree)。`25_ICCAD.pdf` 是缓冲谐波论文，**不是**同一作者 2025 ICCAD 的 CoTL。PDF 页码、公式、固定源码 commit 和实现位置详见 [两篇时序与 hub 论文审读](PAPER_23_DAC25.md)及[缓冲论文审读](PAPER_ICCAD25.md)。

## 2. 当前基础 WDRO 到底做了什么

以 `wdro_precision_v2/wdro_rest_min` 为依据：

1. 单个 net 是归一化到 `[0.1,0.9]`、保持纵横比的二维点集 X。
2. REST Actor 生成离散拓扑。greedy 输出的线长为 Lgθ(X)，真实搜索损失是 `rgθ(X)=Lgθ(X)/L*(X)-1`；L* 使用精确 binary scale `2^27` 的 GeoSteiner。
3. `src/candidates.py:generate_support` 生成 32 个 Random/Sobol 扰动，加上原中心，共 33 个候选。扰动类型是 jitter/sparse/correlated/axis-biased，幅度固定为 0.01/0.02/0.04/0.08。
4. `shortlist_support` 用不训练的 MST/HPWL 几何代理选出“原中心 + 3 个扰动”；只有这 **4 个**需要当前 Actor 的真实评分。K32 不代表 32 次真实评分。
5. `stage4/dual.py` 选择 `Z_i=argmax[r_g(Z)-λ c(X_i,Z)]`，`λ←max(0,λ+ηλ(mean c−ρ))`。
6. 在选中 Z 上重新采样 Actor 动作，使用真实 sampled regret 的 REINFORCE 更新；Critic 是只依赖输入的方差降低 baseline，不是 greedy loss 的替代函数。

旧主线每 degree 的 ρ 来自训练候选成本 q75；新任务若改变 ground cost，就不能无说明继承旧 ρ。原主线正式模型的 2000 updates/batch256 与本分支 2048/batch64 的用途不同，本分支不冒充旧主线重放。

## 3. 共用理论：替换布线任务后，WDRO 外层仍成立

设输入包含几何和必要属性 `x=(V,s,C,RC)`，策略 `πθ(a|x)` 输出合法动作轨迹，真实任务代价为 `J(x,a)`。固定一个预先确定的正常数 sJ，定义

\[
\ell_\theta(x)=\mathbb E_{a\sim\pi_\theta(\cdot|x)}[J(x,a)/s_J].
\]

理想目标为

\[
\min_\theta\sup_{Q:W_c(Q,P)\le\rho}\mathbb E_Q\ell_\theta(X).
\]

若问题的可积性、紧性/增长控制与对偶成立条件满足，可写为

\[
\min_{\theta,\lambda\ge0}\left[\lambda\rho+
\mathbb E_{X\sim P}\sup_Z\{\ell_\theta(Z)-\lambda c(X,Z)\}\right].
\]

这说明 WDRO 不要求目标一定是线长、Actor 一定是 REST，也不要求对坐标有可用导数。真实代价能计算、动作概率能求导即可构造 REINFORCE。**这不是对任意神经网络/任意启发式内层都提供全局收敛或泛化证书。**离散 greedy 解码的连续性需单独论证；本次不假定它有已知 Lipschitz 常数。

实际仍用原有限候选 C(X) 与几何预筛 S(X)（含中心，大小 4），以 greedy 代价近似内层：

\[
\hat z_i=\arg\max_{z\in S(x_i)}[J(z,a^g_\theta(z))/s_J-\lambda c(x_i,z)].
\]

对选中输入，Actor 梯度估计为

\[
\hat g_\theta=\frac1B\sum_i
\left(J(\hat z_i,a_i)/s_J-b_\phi(\hat z_i)\right)
\nabla_\theta\log\pi_\theta(a_i|\hat z_i),
\]

其中 `a_i~πθ`，baseline 与动作独立，优势中的 b 要 detach。Critic 最小化对相同真实 sampled cost 的 MSE；不能把动作相关的额外奖励混作 baseline。由于内层 greedy 与外层 sampled 本来不同，这只是当前训练机制的任务迁移，并非精确求解上式理想 min-max。

λ 的更新符号来自对偶目标梯度 `ρ−mean c`；对 λ 做梯度下降得到 `λ+η(mean c−ρ)`。单个 z 的成本可以大于 ρ，ρ约束的是整体平均运输成本。一次更新的平均成本超标不自动证明发散；需要报告 λ、预算缺口、时间平均与有限支持风险。

## 4. 必须处理的语义与标度

### 4.1 源点不能被当成普通接收端匹配

无标签全点 Hungarian 有一个直接反例：把同一坐标集合的源点与 sink 身份交换，原 c=0，而最长路径、skew 与 RC delay 可能变。对有源点的三篇论文，本次使用

\[
c_{root}(X,Z)=\frac{\|x_0-z_0\|_1+
\min_{\pi\in S_{n-1}}\sum_{i=1}^{n-1}\|x_i-z_{\pi(i)}\|_1}{n}.
\]

这保留 Hungarian-L1 主体，只限定标签匹配。原点仍位于每个候选的 index0；候选生成的坐标排序用于生成噪声，回写原索引后不改变源点身份。原纯线长 REST 对照仍用全点无标签 Hungarian。

缓冲分支固定全部 sink 电容及 RC 参数，只研究坐标分布偏移。若未来扰动电容、驱动强度或源点身份，需扩展输入和 ground cost，并重新校准，不能把本次结论外推过去。

### 4.2 原始 L* 不能变成时序任务的最优值

GeoSteiner 只给纯线长参照，不能拿它作为时序、skew 或缓冲目标的最优解。因此三种新任务报告固定标度 cost 及物理分量，不称“相对全局最优 regret”。

固定标度规则如下：

| 家族 | 本次真实训练/搜索 cost | 原始分量 |
|---|---|---|
| REST 对照 | L/L*−1 | L、L*、relative regret |
| Timing-RST | `[(1−α)L+αR]/[2(n−1)]`，α=0.5 | 线长 L、最长源到 sink 路径 R |
| Hub | `[βL+(1−β)skew]/[2(n−1)]`，β=0.5 | 线长、skew、hub 数量；每次评分内部有 11 次 PD 参数扫描 |
| Buffer | `(delay_seconds×1e18+0.25×buffers)/10` | 最大 Elmore delay、skew、线长、buffer 数量 |

不同家族的 cost 不能直接排名。sJ 改变也会改变合适的 λ 标度与有限步轨迹，所以所有单位、权重、学习率均先冻结，不观察测试结果后调整。

## 5. 逐论文接入方案

### 5.1 2023 ICCAD：动态图状态与时序权衡

输入增加 source 标记，Actor 使用原 TripleGAT 动态拓扑状态，从合法连接中逐步构树。沿用公开代码的图到 rectilinear tree 语义，计算并去重线段后的 L，以及 rooted tree 的最长路径 R。当前已用官方 `libeval.so` 独立对照 NumPy 线长，避免把索引对错误理解为普通 pin-pin 边。

最小新增接口是 `adapter_timing.py`；WDRO 的 Random/Sobol 生成、预筛、对偶更新保持共用。α 是线长-路径权重，λ 是运输价格，二者不能混淆。

若只想提高原纯线长问题，更合适的后续实验是 α=0 的动态图编码替换消融，使用与 REST 相同真实线长目标、初始训练信息、预算和评测。但当前原论文异构网络无法直接加载 DAC21 REST 权重，从头训练和预训练权重比较会混入初始化差异。这个问题不能由本次时序 ERM/WDRO 配对结果回答。

论文的大 degree polar quadtree divide-and-merge 在本轮 d10 不实现。把小块分别做到 robust，不保证合并后的全局时序或运输预算 robust；需对完整 net 定义候选、完整构树评分，或额外证明组合误差界。

### 5.2 2025 DAC：可学习的 hub 选择与停止

Actor 含 Selector 和 Conductor：选择 sink 作为 hub，并决定何时停止；之后调用原始 MSPD C++ 算法得到树。策略 log-prob 必须包含有效选择与停止动作，停止后的无效动作不再计入。

本轮 degree10 允许至多 5 个 hub，固定 β=0.5。保留原 C++ 构树及其 11 点 PD 参数扫描；固定坐标整数化倍率 1000。原脚本中的 hub 数量额外奖励在本轮系数固定为 0，使内层选点与外层真实目标一致。**保留了选择/停止机制，不声称复现原论文整个训练配方。**

整数化是这个公开求解器的一部分；几何运输在原归一化坐标上度量，真实损失是经过该确定性量化/构树流水线的值。可能出现小移动不改变输出的区间；不能据此宣称坐标损失平滑。跨求解器比较时还必须统一量化契约，本轮只在该家族内部比较 ERM/WDRO。

成本核算不能只数神经网络 forward。WDRO 每中心 4 次真评等于最多 44 次内部 PD 扫描，测试全 33 候选对应最多 363 次。所有真实评分计数保留，以便换算；不得把 MSPD 的计算收益归于 Actor 或 WDRO。

### 5.3 2025 ICCAD：联合拓扑与谐波缓冲候选

论文以固定空间周期 `T=sqrt(2RbCb/(RwCw))` 的正弦波峰生成候选；Actor 对每条边选择相位。对 L 形边拐点继续累计弧长，相同物理段的不同方向需要统一坐标后合成波形，再调用缓冲选择器。

本分支保留公开网络与真实 RC 代价，采用 16 个离散相位并把相位分类 log-prob 加入 REINFORCE。原因是公开代码将确定性相位 detach 到 NumPy，不能直接让末边相位获得真实反馈梯度。离散化只保证相位可训练；一个峰的几何定位误差至多约 T/(2×16) 的结论不能直接转成非连续插入/构树代价的误差保证。

已记录的重要源码差异与修正：公开评分的多目标权重不同于论文式(1)；公开插入器是局部 greedy，并非论文描述的完整 Shi–Li 动态规划；修正固定源点的伪随机 log-prob、相位/重叠方向、sink 有子树时的电容累计。详细列表见 `vendor_harrl/PATCHES.md`。

固定上游 RC、所有 sink 相同负载、统一的归一化物理尺度。这里是可重复合成 RC 任务，不是 signoff 时序，也没有测量真实功耗。CPU 独立两节点 Elmore 手算与代码相符，仍需 GPU 训练和独立测试才能判断实证收益。

## 6. 首批实验：能够回答什么

本轮实施的是**基础 WDRO 对这些真实求解任务的可组合性与有限预算效果**，外加原纯线长 REST 的匹配对照。它不把时序任务的改善冒充原纯线长 WDRO 的改善。

固定 `protocol.json`：

- degree=10；4 个家族 × ERM/WDRO × seeds101/102/103，共 24 次正式训练。
- 每次 2048 updates、batch64、Adam1e-4；每家族 ERM/WDRO 从完全相同 Actor/Critic 初始化出发。原 REST 使用可信 DAC21 Actor；三篇异构 Actor 从头初始化。
- nominal 仅 uniform。候选、shortlist、训练样本按 seed/update 确定，CPU 预生成并 hash 固定，不从测试数据训练。三种论文共享 rooted ground cost；REST 使用 unrooted cost。
- ρ 在独立 training-only 256 中心上对全部 32 个扰动成本重新取 q75，数据与源码绑定。λ0=0，学习率1。
- 每128更新，在每分布256个固定 validation nets 上评估。只按 OOD 三组最大平均 cost 选最早最佳 checkpoint，允许选择初始化；同时报告 nominal。这个新任务 pilot 没有沿用旧线长0.002门槛，因为新目标单位不同。
- 完成选模后才在每分布1024个固定测试 nets 上计算 cost 与各分量的 mean/P95。训练3seed，保留每net结果。
- 每分布另取128个 heldout 中心，重新生成原 Random/Sobol 32+中心。这次全部33个真实评分，用下节 LP 统一执行平均ρ预算。没有 ES、Critic-gradient、RL attacker，也没有 CEM。

运行性与数值有效性的门禁：候选合法且中心保留；源点匹配反例；真实代价/概率 finite；梯度 finite；恢复后有效轨迹相同；最终 checkpoint/数据/模型/源码 hash 一致；LP 可行；任务正常退出、锁释放、完整配对统计生成。**不以“WDRO必须获胜”作为工程验收门槛。**

结论门槛：报告每任务、分布、seed的差值；区分均值、P95、有限支持风险。3个训练seed规模有限，单net bootstrap 条件于这些模型，不能等价于大量独立训练重复。若区间跨0或seed符号不一致，只能说没有建立稳定改善。即使某任务有改善，也不能推出另两个任务、未测试degree、大net或真实芯片改善。

## 7. 用共享平均预算验证有限候选风险

对 m 个测试中心、每个 k=33 个候选，记真实 greedy loss `l_ij` 与运输成本 `c_ij`。求

\[
\max_{p_{ij}\ge0}\frac1m\sum_{ij}p_{ij}l_{ij},\qquad
\sum_jp_{ij}=1,\quad\frac1m\sum_{ij}p_{ij}c_{ij}\le\rho.
\]

记录 primal mass、行和残差、平均成本、原点平均 loss、无约束逐点最大值。原点成本为0，因此 LP 必有可行解；解出的有限支持最坏均值不低于原点均值。这比用一个固定λ后逐点取最大却不检查总预算，更直接对应“平均运输预算”的原意。

此 LP 对每中心指定的候选边精确，不包含任意中心向全部候选并集的所有运输边，不能认证连续空间的全局最坏风险。MST/HPWL 预筛没有对时序或 buffer 目标的覆盖定理；训练4次、压力测试33次的真实评分预算也必须同时报告。ERM/WDRO使用相同更新数，WDRO每步多4次greedy评分，因此不是同计算预算；必须同时报告实际调用与耗时。

## 8. 执行、资源与可恢复交付

所有新文件位于本目录。公开代码只读快照保留 commit/SHA；本地修正隔离并逐项说明。CPU数据预生成、编译、单元/集成检查先完成；GPU以最近多次显存与算力采样准入，每张可用卡最多一个本分支 worker，尽量并行家族/seed。只处理自己启动的进程，不抢占或终止他人任务。

tmux 内依次执行数据准备、协议冻结校验、GPU队列、统计与验收。每个训练保存 Actor/Critic、Adam、λ、RNG、更新游标和训练链 hash；容量退出保留 checkpoint 后自动等待。进程身份核对 PID、启动时钟与 boot_id，并用 flock 防重复。预期产物为 `artifacts/runs/*/{latest.pt,best.pt,complete.json,test_best.npz,attack.npz}` 与 `artifacts/report/`。

ETA使用本轮 GPU smoke 的时间和实际训练进度估计，计算时间与等待资源时间分开。不能从空闲显存推断GPU算力空闲，不能承诺不可控排队时长。
