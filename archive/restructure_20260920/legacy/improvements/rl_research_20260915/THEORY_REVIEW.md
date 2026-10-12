# WDRO 内部强化学习攻击器：六种优化路线与对抗共同训练

日期：2026-09-15。本文先于本研究的新训练运行撰写，属于理论与实验设计，不宣称这些方案已经有效。实际规模、随机种子、运行结果与完成状态以本目录最终冻结协议和产物为准。

## 0. 范围、反馈与已有项目的关系

本研究遵守项目当前更严格的纯内部强化学习反馈约束：**不用 GeoSteiner、MST、HPWL、最优性下界、Hungarian 匹配、参考最短树、参考值缓存、旧预训练模型或这些工具生成的教师标签。** 可计算的信息是输入坐标、当前系统自产动作及其实际线长、动作合法性、坐标位移和策略概率。主 REST Actor 与攻击器均在本次研究中初始化、训练；若使用共同起点，必须来自本次从零训练并明确记账的 warm-up。

核对的当前项目文件包括：

- `docs/THEORY.md`：旧 relative-regret WDRO 与主 Critic / 搜索代理的角色。
- `revisions/adversarial_v3_1_20260912/Codex/adv_v3/rl.py`：旧一步攻击器已经有共享参数、置换等变 pin 编码、全局 pooling、Gaussian 位移和 λ 条件输入；它并非逐 pin 多智能体训练。
- `revisions/adversarial_v3_1_20260912/Codex/adv_v3/common.py`：旧合法化包含逐 net 归一化、Hungarian 成本和参考求解；这些部分不能直接搬到本研究。
- `improvements/rl_only_20260915/THEORY.md`：本研究承接其中纯内部反馈、固定窗口和 indexed 位移距离的定义，不混用旧 regret / 最优性结果。

本研究的改进问题是：**相同实际评分成本下，怎样学到更有效的坐标攻击策略；这种攻击策略与 REST 共同训练，能否降低独立测试中的实际线长和有限攻击下的成本？**

这里的“更有效攻击”指原始目标 F 增大，不是 learned Critic 分数变大。“更鲁棒”必须另用冻结后的独立攻击验证，不能由训练对手变弱或攻击器训练 loss 下降推断。

## 1. 所有路线共用的数学问题

### 1.1 主任务与动作测量

固定 degree `n=30`，输入为有对应关系的 pin 元组 `X∈[0.1,0.9]^(n×2)`。REST 策略 `πθ(a|Z)` 采样离散树动作 a；环境只测量这组动作的实际线长 `C(Z,a)`。沿用 pure-RL 分支明确规定的 REST span 记账语义，不把它未经检查就称为完整物理线段并集长度。

令

\[
\bar C_\theta(Z)=\mathbb E_{a\sim\pi_\theta(\cdot\mid Z)}C(Z,a),\qquad
C^g_\theta(Z)=C(Z,a_g(Z;\theta)).
\]

主模型最小化实际 sampled 成本；攻击搜索通常用可复现的 greedy 成本。这种 sampled / greedy 区别需要在实验中公开。

### 1.2 固定几何与位移成本

所有坐标攻击共用环境变换：

\[
Z=T_X(u)=\operatorname{clip}_{[0.1,0.9]}
\{X+\operatorname{clip}_{[-\epsilon,\epsilon]}u\},\qquad\epsilon=0.08.
\]

这里不做逐 net 平移缩放；最终有效位移 `Z−X` 逐坐标不超过 ε。近重复 pin 的判定、拒绝或回退规则应冻结并记录；不能某一方法额外获得修复搜索。

\[
c_{idx}(X,Z)=\frac1n\sum_{i=1}^n\|Z_i-X_i\|_1.
\]

同步重排 X、Z 不改变此成本，但独立重排 Z 会改变对应关系。因此这是带 pin 对应关系的空间，不能称为无序集合的最优匹配 Wasserstein 距离。`ε=.08` 是局部可达集限制，`ρ=.04` 是平均运输成本预算；两者含义不同，不能把二者都叫“攻击半径”。本设定每样本成本上界不超过 `.16`，单样本成本可以超过 `.04`。

### 1.3 鲁棒目标与价格更新

理论目标可写为带可达集限制的运输鲁棒优化：

\[
\min_\theta\sup_{Q\,\text{可由合法运输到达},\;W_{c_{idx}}(P,Q)\le\rho}
\mathbb E_Q\bar C_\theta(Z).
\]

相应拉格朗日形式为

\[
\min_{\theta,\lambda\ge0}\left\{\lambda\rho+
\mathbb E_X\sup_{Z\in\mathcal Z(X)}[\bar C_\theta(Z)-\lambda c_{idx}(X,Z)]\right\}.
\]

Wasserstein 对偶依赖状态空间、可测性、增长与适用的对偶条件；神经网络有限候选实现不自动继承强对偶求解能力或全空间鲁棒证书。[Gao 与 Kleywegt 原论文](https://arxiv.org/abs/1604.02199)

本研究的真实攻击分数固定为

\[
F_{\theta,\lambda}(X,Z)=C^g_\theta(Z)-\lambda c_{idx}(X,Z),\qquad
F(X,X)=C^g_\theta(X).
\]

所有最终候选按实测 F 排序，保留原 X 为 no-move 候选。训练中 λ 用选中候选的成本更新：

\[
\lambda\leftarrow[\lambda+\eta_\lambda(\overline c_{idx}-\rho)]_+.
\]

这是对 λ 的最小化方向；成本超预算时价格上升。固定攻击器比较先冻结 λ，避免方法之间通过不同价格改变题目。共同训练阶段所有方法使用相同初始 λ、ρ 与更新规则，日志保留成本偏差。不能让每个候选独立满足 `c≤ρ` 后仍声称做的是同一个平均预算问题。

### 1.4 纯内部反馈的解释边界

原始 regret `L/L★−1` 消除了部分几何难度差异。现在 raw C 同时包含“几何本来就更长”和“策略在此处解得更差”两部分；攻击器可能主要学到把 pin 向窗口两端移动。这样仍可优化声明的绝对成本鲁棒任务，但不能解释为发现了最大的最优性误差。

报告应包含实际 C、ΔF、位移成本、边界裁剪比例、攻击多样性、原输入 nominal 成本及独立攻击成本；禁止输出虚构 optimality gap。研究不以重新引入 MST、几何下界、预训练教师或最优参考作为“归一化修复”。

## 2. 路线一：校准的一步 REINFORCE 基线

### 2.1 假设和结构

原一步方案最大的问题未必是算法名字，而是高维联合动作方差、弱 baseline、训练价格与测试价格不匹配，以及单样本训练与多候选择优不匹配。先建立实现正确且预算清楚的控制组。

攻击策略输出未裁剪 Gaussian 动作 `u∼pω(u|X,λ)`，再由 T 产生 Z。共享参数 per-pin 编码与全局集合 pooling 构成置换等变均值，所有 pin 共用二维或逐 pin 噪声模型。初始均值接近零，标准差可从 `.02` 起；这些是冻结起点，不是已证明最优参数。

\[
J_1(\omega)=\mathbb E_u F(X,T_X(u)),\quad
\nabla_\omega J_1=\mathbb E[(F-b)\nabla_\omega\log p_\omega(u\mid X,\lambda)].
\]

基线可选实测 `b=F(X,X)`，或独立训练的攻击价值网络 `Vν(X,λ)`。V 与主 REST Critic 是两个网络：前者预测攻击回报，后者预测当前 REST sampled 线长。用原 X 基线会增加实测请求；所有路线都计入这笔开销。

### 2.2 必须验证的细节

- log probability 对实际采到的未裁剪 u 计算。将裁剪后的位移代回 Gaussian 密度会错误处理边界点的概率质量。
- 策略梯度中的 advantage detach；本批基线先取值，随后才用本批回报训练价值网络。
- 不把批内最高分候选当作其他独立动作的无偏 baseline；包含自身动作的 baseline 需要专门推导。
- 若使用批均值/标准差缩放 advantage，须注明这是优化尺度处理。按 net 的难度尺度归一化会改变样本权重，不能悄悄加入。
- 同时报告 entropy / std、clip 比例、优势方差和梯度范数，识别“策略不动”与“饱和到边界”。

### 2.3 最小实验

同一初始主 Actor、相同 λ、相同中心流、相同实际调用数下，比较未训练 Gaussian、训练后的 REINFORCE，以及实测原 X。固定 K 测量 `max(F(X),F_1,...,F_K)−F(X)`。这一组区分“网络真的学会提案”和“只因多次采样自然取到较大值”。

## 3. 路线二：一步 PPO，重复利用同一批真实反馈

### 3.1 理论方案

仍然是 contextual bandit：观察 `(X,λ)`，一次输出整组坐标位移，环境给一个终局 F。它是 H=1 强化学习，没必要虚构长轨迹、Bellman bootstrap 或 GAE。

保存旧策略 log probability `log p_old(u|s)` 和动作无关旧价值 `V_old(s)`，优势 `A=F−V_old(s)`。最大化 clipped surrogate：

\[
r(\omega)=\exp[\log p_\omega(u\mid s)-\log p_{old}(u\mid s)],
\quad
L^{clip}=\mathbb E\min\{rA,\operatorname{clip}(r,1-\delta,1+\delta)A\}.
\]

价值网络回归 F；同一批实测样本可做数个更新 epoch。PPO 原论文提出这一 clipped 代理与多 epoch 更新；它不是本非凸双层项目的单调改进或收敛定理。[PPO 原论文](https://arxiv.org/abs/1707.06347)

### 3.2 为什么可能有用

一次真实评分往往比小攻击网络反向贵，重复利用标签有机会改善单位评分的学习。但 degree30 有 60 个连续动作分量，联合 log-ratio 为分量之和；轻微均值变化也可能让整个联合 ratio 大幅偏离 1。大 clip fraction、低有效样本量或高 KL 都可能使更多 epoch 适得其反。

候选初始配置为 `δ=.1`、2–4 epochs、小学习率、梯度裁剪、同一 rollout 内冻结主 Actor 和 λ。应记录联合 KL/clip fraction，而不是只看每 pin KL。不能随意把 joint log probability 除以 n 后称为相同概率比；那会改变 surrogate。

### 3.3 辨别性实验

REINFORCE 与 PPO 使用同一每批候选数、同一新鲜反馈数、同一网络表达能力。报告两条横轴：实际评分数与时间。PPO 可能在第一条更好、第二条更差；这种结果仍有解释价值。额外 epoch 是额外计算，不算免费。

## 4. 路线三：直接优化 best-of-K 的 PPO

### 4.1 训练目标与部署目标对齐

部署通常不是随机抽一个 Z，而是取 K 个候选中 F 最大者。优化 `E[F]` 不一定最有效地优化 `E[max F]`：高均值窄分布可能不如保留少量高值模式的分布。

令 K 个动作独立条件采样，`F_0=F(X,X)`，

\[
M=\max\{F_0,F_1,\ldots,F_K\},\qquad J_K=\mathbb E M.
\]

对第 j 个候选定义排除自身的 baseline：

\[
b_j=\max\big(\{F_0\}\cup\{F_l:l\ne j\}\big),\qquad A_j=M-b_j.
\]

条件于 X 和其他动作，b_j 与 u_j 独立，因此精确 score-function 梯度为

\[
\nabla J_K=\mathbb E\sum_{j=1}^K
(M-b_j)\nabla\log p_\omega(u_j\mid X,\lambda).
\]

这一推导是本研究对现有设定的直接构造，不依赖最优参考。只有超过原 X 和其他候选的样本获得正优势；连续分数通常只有一个 winner 非零。并列第一时，排除自身后仍有同值候选，对应优势可为零。

PPO 可以对候选级概率比和冻结的 `A_j` 作 clipping。这是局部 surrogate；多个 epoch 后不再等于新的策略分布下重新计算的无偏 `∇J_K`，应以实测 test best-of-K 判断。

### 4.2 优点、风险与实现选择

优点是目标直接匹配候选择优，排除其他候选不增加任何评分。风险是优势稀疏、初始 X 过强时全组信号为零、策略早熟和协同多样性不足。不要把 `winner−runner-up` 偷换成 rank 1 奖励后仍宣称优化同一 raw F 目标。

若采用额外 entropy、混合 mean-F 辅助目标或 group reward 标准化，都要作为另一个清楚标注的变体。首轮可只跑纯目标以识别机制；进一步可用事先固定的 `J_K+βJ_1`，但它已经改变学习目标。

### 4.3 最小实验

与普通 PPO 共用 K、中心数、实测总数。至少报告 K=1 与部署 K 的曲线：best-of-K 方法可能牺牲单次均值以提高尾部命中率。记录每组非零优势比例、候选重复率、候选间位移差与边界比例，避免把“模式多样”仅凭 std 大来判断。

## 5. 路线四：短时域顺序 PPO，学会观察反馈再移动

### 5.1 真正的序列决策

一步动作无法利用“这次移动提高还是降低了真实 F”。引入 H=2–4 的短搜索轨迹：状态包含原 X、当前 Z_t、历史最佳候选 Z*_t、它们的实际 F、λ 与 t/H。行动给下一步位移。

所有累计位移相对原 X 裁剪，不能每步仅限制相对 Z_t 的 `.08`，导致 H 步的可达范围变大。得到 `Z_{t+1}` 后做一次实际评分；令

\[
M_t=\max\{F(X,X),F(X,Z_1),\ldots,F(X,Z_t)\},
\quad r_t=M_{t+1}-M_t.
\]

选择 `γ=1` 时，累计回报严格为 `M_H−F(X,X)`；因此 dense feedback 没有奖励无效来回振荡，也没有改变最终 best-of-trajectory 目标。γ<1 会偏好较早改善，成为不同问题。主 Actor 与 λ 在整段 rollout 和对应 PPO 更新内冻结。

轨迹价值可拟合剩余改善 `Vν(s_t)=E[M_H−M_t|s_t]`。首轮使用完整 Monte Carlo 剩余回报，避免短轨迹里增加没有必要的 bootstrap 混淆；以后再把 GAE 作为方差消融。

### 5.2 同预算比较

一个 H 步轨迹消耗 H 次新候选评分。若总共允许 K 个新候选，则应运行 `K/H` 条轨迹，而不是给序列方法 K×H 次评分。原 X 基线可在同中心共享；缓存命中、重复请求和实际 Actor 前向行数分别记账。

序列方案有望更省候选，但存在阶段奖励稀疏、每轨迹吞吐低、响应迟滞和训练不稳定。它是否优于 K 个独立一步候选，必须在相同真实反馈成本下测量。

### 5.3 最小实验

固定 total K，比较 H=1 与预先选择的 H=2；若正式 K 不能整除 H，剩余预算的使用规则须预先冻结。输出每一步真实最佳 F 和接受移动率。不能只展示最后一步而抹掉曾找到的更好候选。

## 6. 路线五：合作 pin-agent MAPPO

### 6.1 什么才是本任务中的多智能体

让每个 pin 对应一个同构攻击 agent，agent i 控制二维 `u_i`，所有攻击 agent 共享同一个任务回报 F。所有 pin 移动共同生成一个 Z 并只需一次主 Actor 评分。主 REST Actor 仍是对抗的解题方，**不是这些合作攻击 agent 中的一个**。

每个 agent 执行共享策略 `pω(u_i|o_i)`；`o_i` 可以包含自己的原位置/当前位移、λ、时间和允许广播的置换不变全局摘要。训练时 centralized value `Vν(s)` 看到完整 pin 集合与历史，执行时不需要 value。若所有 actor 也看完整集合，应如实称为共享参数、全观测的 per-pin MAPPO 实现；若宣称部分可观测 CTDE，则 actor 不得读超出其声明 observation 的训练信息。

MAPPO 原论文研究合作共享奖励，centralized value、参数共享和训练超参数是关键组成；其 benchmark 结果不构成在连续坐标扰动或零和对抗 REST 中同样有效的保证。[MAPPO 原论文](https://arxiv.org/abs/2103.01955)

### 6.2 更新与置换结构

\[
p_\omega(u\mid s)=\prod_i p_\omega(u_i\mid o_i),\qquad
r_i=\frac{p_\omega(u_i\mid o_i)}{p_{old}(u_i\mid o_i)}.
\]

共享团队优势 `A=G−Vν(s)`，更新

\[
L_{MAPPO}=\mathbb E\frac1n\sum_i
\min\{r_iA,\operatorname{clip}(r_i,1-\delta,1+\delta)A\}.
\]

这是 per-agent PPO surrogate，不是把 joint ratio 算错。未 clipping 的策略梯度在旧策略处与共享奖励 joint score-function 梯度相同到正的 n 倍尺度；多 epoch 和 clipping 后两者有不同偏差，正是应测试的算法差别。

actor 使用共享 per-pin 网络，critic 用集合池化/attention 的 invariant 输出。不能按输入 pin 编号加入可学习 ID 而仍声称严格置换等变。固定采样种子的 pathwise 置换检查还需要同步重排 noise；仅分布上等变不必在不同索引 RNG 下逐点输出相同。

### 6.3 核心风险：团队信用分配

一个 pin 的坏移动可能被其他 pin 的好移动掩盖。单一全局 baseline 降低状态方差，但不会自动解决每 pin 的贡献归因。首轮先测试团队优势，避免昂贵的逐 pin counterfactual 评分。以后可训练 action-dependent centralized Q，或抽少量 pin 计算保持其他动作不变的差分奖励；这些额外评分必须计入预算。

同一个 net 的 30 个 pin-agent 不是 30 个独立环境样本。bootstrap 以 net 与训练 seed 为单位，不能把 30 倍 pin 数当作样本量扩大。

### 6.4 最小实验与命名验收

与整组动作 PPO 共用网络容量、团队回报、候选数与训练数据；明确差异是 per-pin ratio/clipping 和 centralized value，而非单纯把同一个 `Policy` 改类名。检查每 pin log probability、centralized critic 输入、共享团队 reward 和同步 action 联合环境步。

## 7. 路线六：合作 pin-agent MADDPG bandit

### 7.1 可实施的限定版

每个 pin agent 的确定性策略输出 `μω(o_i)`，探索时加噪声。centralized critic `Qν(s,u_1,...,u_n)` 用完整 joint action 预测真实 F。一步任务终止，因此

\[
y=F,\qquad \mathcal L_Q=\mathbb E_{\mathcal D}(Q_\nu(s,u)-F)^2.
\]

此处不应写没有后继状态的 `r+γQ_target`。在主 Actor、λ 固定时，历史真实反馈可以 replay，降低新环境反馈需求；Q 是学得的动作回报回归，不是最优线长参考。

标准 pin i deterministic policy gradient 为

\[
\nabla_{\omega_i}J_i\approx
\mathbb E[\nabla_{u_i}Q_\nu(s,u)\vert_{u_i=\mu_i(o_i)}
\nabla_{\omega_i}\mu_i(o_i)].
\]

参数共享时求各 agent 更新之和；实现应清楚说明更新 agent i 时其他动作来自 replay 还是当前策略、是否 detach。若直接对整个 joint deterministic policy 回传所有动作梯度，那是 centralized shared-team deterministic policy gradient 变体，不能隐去该差别。

MADDPG 的来源是利用训练时的其他智能体信息构建 centralized critic，并允许合作或竞争奖励。[MADDPG 原论文](https://arxiv.org/abs/1706.02275) 本研究的 H=1 合作攻击队伍是其简化使用场景，不等于重现原文全部算法或基准。

### 7.2 为什么这是高风险备选

- 真实 REST greedy 线长对坐标有离散拓扑切换；Q 的小回归误差不保证其动作梯度方向正确。攻击器可能利用 Q 外推误差。
- 确定性策略容易缺乏多样性；若测试只给 μ，不公平地少用 K，若给探索噪声则噪声分布要预先冻结。
- 边界裁剪可能造成平坦区。可用 bounded tanh 均值或对最终合法位移建模，但全组比较必须明确其合法化和可达集仍一致。
- 主 Actor 或 λ 改变时，旧 reward 已过时。replay 必须记录对手版本、λ 与真实 C；价格改变可由保存的 C 和 cost 重算 F，主 Actor 改变则必须重新实测 C 或丢弃旧数据。
- Q 的非光滑目标回归、replay 和 actor 互相追逐使调参成本可能显著高于 PPO，不能因为 off-policy 就预设更高样本效率。

### 7.3 最小实验

冻结主 Actor 与 λ，给与 PPO 相同数量的新 F 标签；MADDPG 允许预先规定的固定 replay 更新次数，额外反向耗时单列。必须用独立候选的实际 F 验证 Q gradient 提案，并同时记录 Q 预测值与真实值偏差。GPU pilot 中失败、退化或 Q exploitation 是有效负结果。

### 7.4 为什么不直接把整个 REST 换成 MADDPG

攻击坐标是连续动作，REST 构树动作是离散 categorical/指针选择。原始 deterministic continuous-action gradient 不直接适用于后者。把离散动作换成 Gumbel/soft relaxation 会改变训练环境与梯度估计，还要验证硬解码一致性、合法树和温度偏差。

因此本轮可做“REST REINFORCE 解题者 + MADDPG pin 攻击队伍”的混合对抗系统；把整个项目称为纯 MADDPG 需要真正实现并验证离散动作扩展。本研究不靠名字回避这个技术代价。

## 8. 第七项系统方向：主模型与攻击器共同进化

六种攻击路线可以作为同一个二方系统的不同内层 learner。为分清“攻击本身更好”和“共同训练对主任务有用”，先做固定主模型，再做共同训练；这两个结论不能互相代替。

### 8.1 二方零和目标与当前工程近似

若内外层都基于 sampled 主策略成本，固定 λ 时可写成

\[
\min_\theta\max_\omega
\mathbb E_{X,u\sim p_\omega,a\sim\pi_\theta}
[C(T_X(u),a)-\lambda c_{idx}(X,T_X(u))].
\]

攻击者最大化，主 Actor 最小化。价格 λ 与主 Actor 同为最小化侧，附加 λρ 项。对抗扰动与主策略联合训练在 RARL 中有先例，但其控制环境实验不证明本组合优化系统必然改善。[RARL 原论文](https://proceedings.mlr.press/v70/pinto17a.html)

本研究的实用版本若使用 greedy C 训练攻击器、sampled C 训练主 Actor，就不是对同一个精确 payoff 的标准零和梯度；应标注为近似交替训练。为了研究一致性，可在后续单独比较攻击 reward 用 sampled C 平均若干次，但额外主采样、测量和方差应记账。

### 8.2 先做分块交替，不在同一 rollout 中更新对手

每轮执行：

1. 固定 `(θ,λ)`，采集攻击 rollout，并完成攻击器 PPO / REINFORCE / MADDPG 更新。
2. 固定攻击器，用原 X 和候选实际 F 选出 Z*。
3. 主 Actor 在 Z* 上采样合法树，以 `C−V_rest` 作 REINFORCE 成本梯度；训练主 baseline。
4. 用选中候选的实际位移成本更新 λ。
5. 到固定间隔保存双方、所有 optimizer/RNG、λ、计数和对手版本。

攻击更新次数与主更新次数的比例是事前固定参数，例如 1:1。不能看当前测试谁领先就给它更多训练。过强攻击可能导致课程失配，过弱攻击近似普通训练，双方同时快速更新可能震荡；都以日志和独立评测判断。

### 8.3 快照池与 population self-play

仅对最新对手训练容易循环：攻击器找到 A 漏洞，主策略修 A 但忘记 B，攻击器又回 B。保存有限历史快照可在相同总更新预算下改善覆盖。

轻量版本：保留初始、中期、最新的少量主 Actor / attacker 快照；按事前冻结权重选择对手，最终每个主模型接受所有攻击快照的交叉评估。每个周期增量训练仍从自己的当前权重继续，历史对手只用于反馈，不是外部参考教师。

更完整版本建立有限 payoff matrix：

\[
A_{ij}=\mathbb E_X[\text{攻击器 }j\text{ 对主策略 }i\text{ 的实测 }F],
\]

在该矩阵上算混合策略，并为混合对手训练新的近似 best response。PSRO 使用 policy population、对手混合和经验博弈分析提供了该思路。[PSRO 原论文](https://arxiv.org/abs/1711.00832)

但简单均匀随机选历史快照只能称为 population self-play / PSRO-inspired，不能叫完成了 PSRO；后者至少应有 payoff matrix、明确 meta-solver 与新 response 迭代。有限矩阵的 Nash 或受限 exploitability 也不等于连续全策略空间的 exploitability。

### 8.4 公平的共同训练验证

至少包含无攻击 ERM、随机候选择优、学习攻击器交替训练和带快照池交替训练。共用从零起点/本次 warm-up、主 Actor 更新数、输入流、选择规则、最终测试集合。快照池若训练时同时评分多个主模型，这些额外 forward 必须算入。

最后冻结主模型，对每个模型使用：固定随机攻击、全部学习攻击器跨模型迁移、相同预算的攻击器短适应（若协议允许）。只用自己的训练攻击器评测，可能把双方共同失灵误认为鲁棒。

## 9. 可辨别、可执行的实验分层

### 9.1 不用一次大重构掩盖机制

| 层次 | 回答的问题 | 最小对照 | 有效完成证据 |
|---|---|---|---|
| A：环境和数学实现 | 反馈、log probability、预算与命名是否正确 | 手算线长、clip 边界、策略 ratio、best-of-K baseline、per-pin 分解 | 测试输出与源码绑定 |
| B：冻结主模型攻击研究 | 新攻击 learner 是否改善实际 F | 原 X、未训练随机、REINFORCE、五种替代 | 每中心候选、动作、C/c/F、固定测试与多 seed |
| C：共同训练 | 攻击改善是否转化为主模型改善 | ERM / Random / learned / population | 相同起点、外层预算、独立输入与交叉攻击 |
| D：推广确认 | 能否跨 degree、分布与训练预算复现 | 预注册扩展 | 新协议及完整重复，不能用 pilot 代替 |

这一设计不要求根据 B 的正结果才保留 C；如本轮预先授权每方向都验证，可以给所有路线同等有限 C 预算。负结果完整保留，不根据 test 自行延长表现较好的路线。

### 9.2 推荐的资源受限首轮

建议固定 degree30、至少三个独立攻击训练 seed、多个独立中心分布。先以同一从零训练主策略快照回答 B，各路线给相同新鲜 candidate score 预算；再以完全相同初始主 Actor 回答 C。pilot 的数量与 epoch 在运行前写入机器可读协议，不能只写在 prose 里。

若预算很小，只能称为探索性 GPU 验证；它可以排除实现故障、明显无效方法和大幅退化，不能把“跑完”写成算法优越性。只有一个训练 seed 时，net bootstrap 只反映输入不确定性，不估计训练稳定性。

### 9.3 统一记账

- 原中心 baseline 评分、新候选评分、重复/缓存请求、主 Actor greedy forward 行数、padding 行数。
- 主 Actor sampled 训练数，攻击器/价值/Q 更新数，PPO 重用 epoch、MADDPG replay 次数。
- 固定与序列候选的真实总数；所有中途实测 reward 均计数。
- 训练/validation/test 分开；test 不参与重新选超参、冻结步数或奖励定义。
- GPU 型号、进程、峰值显存、CPU 工作线程、端到端时间、有效吞吐与失败重试。
- 原始坐标、树动作、C、位移成本、F、种子、checkpoint hash 与协议 hash；保存最终选中候选及所有被评分候选。

### 9.4 统计与实际验收

固定主模型攻击报告配对 `ΔF`、`ΔC`、成本和胜率；共同训练报告 nominal / 各分布 / 独立攻击的 mean、P95 与 worst-group 实际 C。报告每 seed 数字，再用 seed 与 net 分层的配对 bootstrap；不要将各超参格、pin-agent 或同轨迹步骤当成独立训练重复。

多方法比较先指定主比较（例如 PPO 对 REINFORCE、population 对对应 single-opponent），其余探索性比较公开多重比较限制。CI 含零应写“本预算未验证差异”，不能写“方法等价”。

## 10. 优先级与明确的理论判断

1. **PPO 一步与 best-of-K PPO：** 最容易隔离真实问题，也最直接复用当前网络；重点看昂贵反馈的利用效率和训练/部署目标错位。
2. **顺序 PPO：** 如果一步提案主要靠随机命中，短反馈轨迹可能有价值；必须守住总评分预算。
3. **合作 pin-agent MAPPO：** 有清楚的多智能体定义，可能减缓高维 joint ratio 问题，但团队信用分配仍难，不能因为 pin 多就预设优势。
4. **MADDPG bandit：** 合理的连续攻击备选，较适合冻结主 Actor 的 replay 实验；对 Q 方向可靠性的要求更高。
5. **population 对抗共同训练：** 处理对手过拟合和遗忘，可能比更换攻击算法名字更影响最终主任务；必须跨对手评测。

本研究允许从小方案进到较大重构，但先把角色与目标说清楚：PPO 是优化方法，MAPPO 的多智能体可以是合作移动 pins，REST 与攻击器则是竞争双方，population 是训练对手分布。这几个维度可以组合，却不互相等同。

## 11. 本轮冻结的探索性 GPU 方案

以下规模在实验启动前确定；如机器可读协议对每步具体评分行数进一步细化，以两者核对后的冻结文件为准，不根据最终测试改变方案。

### 11.1 共同起点与数据

| 项目 | 本轮决定 |
|---|---|
| degree | 30 |
| 三个独立训练 seed | 201、202、203 |
| 主 Actor 初始化 | 每 seed 从零；同 seed 下所有方法共用相同初始化 |
| 共同 ERM warm-up | 256 更新，batch32 |
| 位移 / 窗口 / 距离 | ε=.08；[.1,.9]²；indexed mean L1 |
| 正式 checkpoint | 固定最后一步；没有根据 test 选 checkpoint |
| 输入分布 | 四种事前冻结分布；具体参数和数据 namespace 写入协议 |

warm-up 不是旧预训练，属于本轮内部训练计算。每个方法如果从同一个 warm-up 分叉，报告共同 warm-up 成本与分支增量成本，并另给独立部署时需要承担的总成本。

### 11.2 A：固定主模型，比较六种学习攻击器

固定 warm-up 后主 Actor；六条路线各训练 256 更新：`reinforce`、`ppo`、`bestof_ppo`、`sequential_ppo`、`mappo`、`maddpg_bandit`。训练 λ=5；所有学习攻击器只用相同预先规定的新鲜候选反馈预算，不因 PPO epochs 或 replay 增加而偷偷增加真实标签。

最终测试采用 λ∈{1,5,10}、四分布、每分布128中心、统一 K=8 实际候选评分预算；原 X 的纳入方式及序列方法如何使用预算在实现协议中统一。配套 `random` 和 `es` 对照不作外部教师。所有候选与原输入按实测 raw F 比较。

λ=1/10 的结果是训练价格之外的泛化与价格敏感性，不能解释为这些价格已分别充分训练。若策略把 λ 作为输入，仅在 λ=5 训练仍不足以保证其跨价格条件响应正确。

### 11.3 B：共同训练，十个配对分支

从同一 seed 的 warm-up 主 Actor 出发，分别继续 256 更新：

`erm`、`random`、`es`、`reinforce`、`ppo`、`bestof_ppo`、`sequential_ppo`、`mappo`、`maddpg_bandit`、`population_ppo`。ERM 在同一中心输入流继续正常 sampled 训练，不调用攻击器。

learned 分支接 A 已训练256更新的攻击器；population 使用相应 PPO 攻击器起点。`λ₀=5`、`ηλ=10`、`ρ=.04`，各分支采用相同规则更新。固定学习攻击器分支与未训练 Random/ES 的外层更新数相同，不等于总计算相同；A 的攻击预训练成本单列并给摊销解释。这样可以测量“已有攻击器后继续主任务”的效果，但不能把学习路线所有收益都归于 B 的256步。

Random/ES 是纯内部反馈对照，却不是 RL 算法。共同 warm-up checkpoint 是无继续训练对照；新增的配对 ERM 继续训练256步，用于排除额外主训练本身的贡献。总任务数为3个warm-up + 6×3个固定攻击任务 + 10×3个共同训练任务 = 51。

### 11.4 population 的实际形式与理论边界

每64次更新归档主 Actor，最多保留三个历史版本，再加当前版本；评分对手从该池均匀采样。每批只使用选到的一个对手做同样数量的候选评分，选中候选由实时主 Actor 做 sampled 更新。历史版本全部来自本次训练，不是旧教师。

该方法的优点是用有限同等评分数增加对手覆盖；但对旧版本的困难输入未必对实时主 Actor 困难。并且

\[
\mathbb E_j\max_Z F_{\theta_j}(X,Z)
\ne\max_Z\mathbb E_j F_{\theta_j}(X,Z)
\]

一般不相等。先抽 scorer 再选它的最难候选，对应左侧形式的有限搜索课程；它既不等于对混合 payoff 求内层最大值，也不保证实时主 Actor 的 WDRO saddle 更新。此分支应命名为 `population_ppo` / 历史对手课程，不称完整 PSRO、Nash 求解或已证明 minimax 收敛。

### 11.5 最终独立测试

- 每个分支最后 checkpoint 在四分布各1024个独立输入上测量实际 greedy C。
- 每分布另取128中心，分别做独立 Random32 与 ES32 压力测试；相同原中心、相同预算、相同 λ 规则配对。
- 此压力测试固定 `λ=0`，最大化允许位移范围中的 raw C；它不要求每个样本 `c≤ρ`，也不是对训练中每个分支最后价格的攻击。对应的实际成本单列。
- 评测存储 topology 和线长，可直接重算环境 C；不借助最优参考确认质量。
- 表格包括 mean、P95、worst-group、nominal 变化、运输成本、各 seed、真实调用量与时间。
- A 和 B 都只有本次小预算探索性范围；即使全部运行完成也不能宣称 degree10–50 普遍有效、达到收敛或获得全空间鲁棒保证。

### 11.6 实现细化：与前文候选理论配置的区别

最终实现采用 `sigma=.04`、PPO `clip=.2`、4个更新epoch。这些是冻结的探索起点；前文 `.02`、`.1`、H=2–4 属于可选设计示例，不是本轮实际运行值，也不表示根据 test 调参。

`K=8` **包含原 X**，因此各方法每中心总计8次实际评分。一步方法有7个独立提案；`sequential_ppo` 与 `mappo` 都有一条7步轨迹，每步 latent action 先限幅 `.04`，累计对原中心位移仍限制 `.08`。两条序列路线共用状态、奖励和评价数，直接比较主要隔离 joint PPO ratio 与 per-pin ratio 的差异；与一步 PPO 比较同时改变了序列交互，不能单独归因于多智能体结构。

REINFORCE 实际使用其他独立提案 F 的 leave-one-out 均值作 baseline；仅一个提案时用原 X。它仍是动作无关的合法条件基线。PPO value / terminal Q 回归 `F(X,Z)−F(X,X)`，其减去的是同一冻结 scorer 的原中心实测成本，不是几何下界；它不改变固定中心的攻击目标。bestof_ppo 使用原始 `M−b_j` 优势，不作 advantage 标准化。

顺序策略观察原坐标、当前位移、价格、已使用步数比例、当前 F 相对原中心的差和历史最佳 F 差；没有把历史最佳候选的完整坐标放入策略状态。完整理论状态可以扩展到最佳坐标，但当前环境转移和 best-so-far 奖励只需当前位置与最佳标量，因而这一压缩状态足够定义本轮受限控制问题。

`maddpg_bandit` 实际对共享 actor 的整个 joint action 回传 centralized Q 梯度，属于合作队伍的 centralized deterministic policy-gradient 简化变体；不是逐个冻结其他 actor 的原始 MADDPG 更新。target Actor/Q 虽被保存及 Polyak 更新，终止 bandit 不使用它们构造 bootstrap target。固定 A 阶段 replay 可以复用；移动主 Actor 的 B 阶段在每次搜索前清空 replay，保留网络与优化器学习，避免旧对手标签污染。它因此在 B 中不享有跨步 replay 的优势。

### 11.7 数学与预算独立复核

独立审查检查了 latent Gaussian 密度、旧策略 likelihood、旧价值 baseline 的 detach、PPO 更新期内固定优势、best-of-K 条件基线、顺序奖励望远镜求和、MAPPO per-pin ratio 和 terminal Q 目标。未修改 `policies.py`。新增 `tests/test_objective_identity.py` 枚举共享 categorical 策略的全部联合动作，比较真实期望 max 的 autograd 梯度与 `Σ_j(M−b_j)∇log p(u_j)`：包含 center 占优、奖励并列、1/2/3/5个提案。另用 Bernoulli 闭式导数 `Kp(1−p)^K` 验证。两个独立数学测试通过，误差阈值 `1e−12`；连同已有测试共11项通过。

候选评分数量相同不等于 GPU 总计算相同。固定 microbatch64、训练 batch32、K8 时：

| 方法结构 | 每步有效候选评分 | 每步含 padding 的 greedy forward 行数 |
|---|---:|---:|
| Random 一次联合评分 | 256 | 256 |
| ES 探测加最终提案 | 256 | 320 |
| 一步学习攻击器：原 X 单独评分 | 256 | 320 |
| Sequential PPO / MAPPO 七步反馈 | 256 | 512 |

所以序列路线付出真实更高的 forward/padding 与同步开销。这是本实现限制，最终报告同时比较有效评分、实际 forward 行数与端到端时间，不能笼统写成“相同计算预算”。

各训练 seed 使用相同测试中心，seed 与 center 是交叉维度。总体不确定性应保留这种结构：每次 bootstrap 抽训练 seed，并对每个分布抽一组中心索引，再将同一组中心索引用于所有抽到的 seed。不能在每个 seed 内独立抽 center 后把共同输入误当作嵌套的独立样本。单 seed 区间仅反映该训练实例条件下的输入不确定性。

## 12. 一手文献入口

- PPO：Schulman 等，2017，<https://arxiv.org/abs/1707.06347>。
- MAPPO：Yu 等，2022，<https://arxiv.org/abs/2103.01955>；正式论文 <https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html>。
- MADDPG：Lowe 等，2017，<https://arxiv.org/abs/1706.02275>。
- Robust Adversarial RL：Pinto 等，2017，<https://proceedings.mlr.press/v70/pinto17a.html>。
- PSRO：Lanctot 等，2017，<https://arxiv.org/abs/1711.00832>。
- Wasserstein DRO：Gao 与 Kleywegt，<https://arxiv.org/abs/1604.02199>。

文献支持方法的来源和结构；本项目中的 best-of-K baseline 推导、固定坐标限制、反馈记账和具体组合是本文方案，不声称原论文已经验证了本问题。
