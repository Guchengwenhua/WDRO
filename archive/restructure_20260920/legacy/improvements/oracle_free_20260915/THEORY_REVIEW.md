# 独立理论审查：无需 GeoSteiner 的 REST / WDRO 学习

日期：2026-09-15。用途：为本次独立实验提供理论检查；本文件本身不宣称训练或验证已完成。实验参数、实现范围和结果以同目录冻结协议、代码、日志为准。

**当前实施入口为 [THEORY.md](THEORY.md)。** 本审查的“可选/建议”段落涵盖较广设计，不代表全部实施。已冻结首轮是 degree30、3 seed、6 分支、2,048 更新、batch64、K8，采用保守 `B=max(H,M/2)`；更强 Hwang 分母和其他候选变体未测试。

**长度语义补充。** 重新检查 `eval.c` 后，文中 L 严格定义为历史 REST 的 per-terminal span 记账长度 `L_eval`。其满足 `L_eval≥L_union≥L★`；同 x/y 的不同 pins 可能重复累计重合 spans。因此文中双端 regret 区间针对 `L_eval/L★−1`；物理并集 regret 只直接继承 `[0,L_eval/B−1]`，其下端不能使用 `max(0,L_eval/M−1)`。NumPy 实现保留该历史语义，具体证明见 THEORY.md 第 3–4 节。

## 1. 结论与当前实现的依赖位置

**可以完全不使用 GeoSteiner 来训练 REST，也可以让 WDRO、ES、Critic 坐标搜索和一步 RL 攻击器完全不使用 GeoSteiner。** 必须同时替换实际学习损失、攻击评分和选模指标，不能只修改 Critic 的回归标签。

另外，当次读取本地原始 `/remote-home/Guilin/DRO/DRO/WDRO-REST/train.py` 第 80–92 行确认：原始 REST 已直接以 sampled 树长及其 baseline 进行 Actor–Critic 更新，没有使用最优标签。因而本次不是发现“RL 可以不求最优解”的新原理，研究增量是让当前 regret-based WDRO 和三条搜索路线、选模/测试一并脱离最优参考，并给出计算便宜的可审计几何界和运行时依赖隔离。

已读当前 `docs/THEORY.md`、`docs/improvements/ADVERSARIAL_V3_1.md`，并检查 v3.1 的 `adv_v3/common.py`、`adv_v3/joint.py`、`adv_v3/critic.py` 和 `wdro_rest_min/stage6/trainer.py`：

- 现有 `Oracle.evaluate` 用 Actor 的实际 greedy 树长除以 GeoSteiner 参考长度。
- `sampled_update` 用实际 sampled 树长除以复用的参考长度，Actor 的 REINFORCE 与训练 Critic 都依赖这个量。
- ES 的有限差分和一步 RL 的奖励均经上述实际评分间接依赖 GeoSteiner；它们并不都具有一个必须预测最优长度的 Critic 层。
- 训练 Critic 是状态基线；搜索 Critic 是另一个拟合 greedy 评分的代理。两者不可混同。

最短的替代链条是：

```text
无标签 pin 坐标 → Actor 采样合法树 → 直接计算该树线长 L
             → 计算坐标的便宜确定性尺度 D
             → q_D = L / D - 1 → REINFORCE / 训练 Critic

候选坐标 Z → Actor greedy 树长 L_g → q_D^g(Z) - λ c(X,Z)
             → Random / ES / Critic 提案 / RL 奖励与最终复核
```

这里的 `L` 只是计算一棵已经给定的树有多长，不是在全部树中寻找最优树，因此计算 `L` 无须调用 GeoSteiner。应检查实际树长动态库及导入链，避免把“算法上不需要”误当成“运行时未加载”。

## 2. 两种简单的无求解器尺度

令非退化二维 pin 集合为 `X={(x_i,y_i)}_{i=1}^n`，允许在平面上布置无障碍水平/垂直线段。定义：

\[
H(X)=(\max_i x_i-\min_i x_i)+(\max_i y_i-\min_i y_i),
\]

\[
M(X)=\min_{T\text{为terminal spanning tree}}
\sum_{(i,j)\in T}(|x_i-x_j|+|y_i-y_j|).
\]

`H` 是包围盒半周长（HPWL），`M` 是只以原始 pins 为顶点、边权为 Manhattan 距离的最小生成树长度。`M` 可用稠密 Prim 算法在 `O(n²)` 时间内精确计算；这不是 Steiner 树问题。degree 10–50 时计算规模很小。

对最优 rectilinear Steiner 树长度 `L★`，存在不需要参考标签的关系：

\[
\boxed{B_2(X):=\max\{H(X),M(X)/2\}\le L^\star(X)\le M(X).}
\]

**证明。**

1. 任意连通 rectilinear 网络的水平线段总长至少覆盖 pin 的横向跨度，垂直总长至少覆盖纵向跨度，所以 `H≤L★`。
2. 将最优 Steiner 树每条边重复两次，得到长度 `2L★` 的闭合遍历。按该遍历首次出现的 terminal 顺序取 shortcut，Manhattan 三角不等式保证 terminal 闭合游走总长不增加。删去一条边得到 terminal spanning path，其长度不大于 `2L★`。最小生成树不长于这条路径，因此 `M≤2L★`。
3. 将 Manhattan MST 的每条边实现为一条水平加垂直路径，即得到连接全部 pins 的可行网络；合并重合线段及去环只会缩短，所以 `L★≤M`。

还有更强、仅适用于本节平面无障碍 rectilinear 设置的经典界：

\[
\boxed{B_{3/2}(X):=\max\{H(X),2M(X)/3\}\le L^\star(X)\le M(X).}
\]

`2M/3≤L★` 是 Hwang 的 rectilinear Steiner ratio 定理。本次实施若已冻结使用 `M/2`，保持该实现即可，它只是较保守；不要在观察结果后切换尺度。更强界可在预先声明后统一用作附加报告，或用于下一次独立协议。[Hwang, 1976, 原论文摘要与出版信息](https://epubs.siam.org/doi/pdf/10.1137/0130013?download=true)

**适用边界。** 带障碍物、金属层切换代价、拥塞罚项、非 Manhattan 边权或不同目标不能直接套用这些不等式。`H=M=0` 的重合点实例应单独定义零长度答案并排除除法；正常归一化若最长跨度等于 1，则 `H≥1`，无须用任意 epsilon 改写理论分母。

## 3. 可学习目标不是唯一的

设合法动作序列 `a` 对应树长 `L(X,a)`，策略为 `π_θ(a|X)`。

### 3.1 MST 相对长度：推荐为便宜直接的主学习目标

\[
q_M(X,a)=L(X,a)/M(X)-1.
\]

优势：尺度不随坐标整体放大改变，完全无标签，直接鼓励同一 net 上更短的树。**`q_M` 可以为负，负值表示比 terminal MST 短，不表示比 Steiner 最优树短。** 禁止沿用旧 `relative_regret` 中“负值即报错”的断言，也不应对 `q_M` 做 `max(q_M,0)`：截断会让优于 MST 后的继续改进失去目标信号。训练 Critic 和 greedy 代理建议使用线性输出头，以覆盖负值。

### 3.2 可证明保守上界：适合作为另一种学习目标或报告指标

\[
q_B(X,a)=L(X,a)/B(X)-1,
\quad B=B_2\text{ 或预先冻结的 }B_{3/2}.
\]

对任意可行树，真实相对 regret 满足：

\[
\boxed{\max\{0,q_M(X,a)\}\le
r^\star(X,a):=L(X,a)/L^\star(X)-1
\le q_B(X,a).}
\]

这是**无需 GeoSteiner、针对每一个输出的 evaluator 记账成本**的 regret 区间，不是预测或经验近似；若记账成本包含重合 span，物理并集 regret 只保证位于 `[0,q_B]`。浮点计算应用容差检查可行性和上下界；不能通过截断掩盖明显负的上界。

设 `κ=2`（保守界）或 `κ=3/2`（Hwang 界），则：

\[
0\le r^\star\le q_B\le\kappa(r^\star+1)-1.
\]

这个上界可能很松：即使 Actor 输出已经最优，`q_B=L★/B−1` 仍可显著大于零。因此 `q_B=0.20` 的含义是“该树的真实 regret 至多 20%”，而非“观测到 20% regret”；`q_B` 下降也是上界下降，不能单独声称真实 regret 按相同数值下降。

### 3.3 与旧目标何时相同、何时不同

对**固定 X**，`M(X)`、`B(X)` 和 `L★(X)` 都与动作无关且为正，因此最小化它们对应的期望归一化长度，都以最短可表示动作作为逐实例最优解。

对共享参数 θ 的有限容量模型，跨实例平均时三种目标对不同 X 给予不同权重：

\[
J_D(\theta)=\mathbb E_{X\sim P,a\sim\pi_\theta}[L(X,a)/D(X)-1],
\quad D\in\{M,B,L^\star\}.
\]

它们并非相同优化问题，局部极值、有限训练结果、跨分布排序均可能不同。WDRO 内层也会根据不同分母选择不同 Z。特别是 `q_B` 的高值可能来自下界松弛，而非 Actor 特别差；必须同时报告 `M/B`，观察攻击是否主要寻找松弛更大的几何形状。

仅使用 `L` 或 `L/H` 也可学习，前者更强调尺度较大的 net，后者经常有更松的最优性关系。将既有数据中的 `L★` 蒸馏成预测器只能减少在线求解，不能叫完全无 GeoSteiner 的训练。

## 4. 不需要最优标签的 Actor–Critic

固定状态分布和 `D(X)>0` 时，score-function 恒等式给出：

\[
\nabla_\theta J_D
=\mathbb E[(q_D(X,a)-b_\phi(X))\nabla_\theta\log\pi_\theta(a|X)].
\]

因为 `b_φ(X)` 与当前采样动作无关：

\[
\mathbb E_{a|X}[b_\phi(X)\nabla_\theta\log\pi_\theta(a|X)]
=b_\phi(X)\nabla_\theta\sum_a\pi_\theta(a|X)=0.
\]

不需要对离散树结构或 C++ 长度 evaluator 求导，只需要动作的 log probability。实际最小化：

\[
\mathcal L_{actor}=\frac1m\sum_i
\operatorname{stopgrad}(q_{D,i}-b_\phi(X_i))\log\pi_\theta(a_i|X_i),
\]

\[
\mathcal L_{critic}=\frac1m\sum_i
(b_\phi(X_i)-\operatorname{stopgrad}(q_{D,i}))^2.
\]

学习 Critic 拟合当前 sampled 策略的期望代价 `E[q_D|X]`，它不拟合 `L★`。MSE 条件均值是合理基线，但并非一般情况下严格方差最小的基线：后者还含 score gradient 范数权重。经典 REINFORCE 的梯度恒等式不要求 Critic 完美，只要求基线不依赖当前动作并在 Actor 更新中停止梯度。[Williams, 1992](https://link.springer.com/article/10.1007/BF00992696)

实现时在本次动作反馈更新 Critic **之前**计算本次 baseline，然后对 advantage detach。不要先用当前动作训练基线，再重新计算基线放回同一动作的 Actor loss；这会破坏动作独立性。批量中心化若包括自身样本也会带入有限批量偏差，首轮无需额外加入。

### 可选：完全去掉训练 Critic

可以使用冻结 Actor 副本的 greedy 代价作为基线 `b(X)=q_D(X,a_greedy)`，或者同一 X 的独立多次采样 leave-one-out 基线。前者增加 greedy 前向和长度计算，后者增加采样数；都无需最优参考。组合优化中的 greedy rollout baseline 已有直接文献先例，但新 REST 上是否更省时间、是否更稳定仍需实验。[Kool、van Hoof、Welling, 2019](https://arxiv.org/abs/1803.08475)

## 5. GeoSteiner-free WDRO

在无序等 degree pin 集合上保持原运输成本：

\[
c(X,Z)=\min_\sigma\frac1n\sum_i\|X_i-Z_{\sigma(i)}\|_1.
\]

令真正外层训练代价 `\bar q_D(θ,Z)=E_{a∼π_θ}[q_D(Z,a)]`。理想问题为：

\[
\min_\theta\sup_{Q:W_c(P,Q)\le\rho}\mathbb E_Q[\bar q_D(\theta,Z)].
\]

在相应可测性、增长/紧性和对偶成立条件下，对偶形式为：

\[
\min_{\theta,\lambda\ge0}\lambda\rho+
\mathbb E_{X\sim P}\left[\sup_{Z\in\mathcal Z}
\{\bar q_D(\theta,Z)-\lambda c(X,Z)\}\right].
\]

相关对偶理论应根据实际状态空间和损失条件引用，不能仅写出公式就声称现有有限候选程序有连续全局证书。[Gao 与 Kleywegt, 原始研究](https://arxiv.org/abs/1604.02199)

与现有实现兼容的工程方案将内层改成 greedy 实际评分：

\[
F_D(X,Z;\theta,\lambda)=q_D^g(\theta,Z)-\lambda c(X,Z),
\quad q_D^g=L^g_\theta(Z)/D(Z)-1,
\]

保留原 X，选实际评过候选中 `F_D` 最大者，再在该 Z 上用 sampled q_D 做外层训练。内层 greedy / 外层 sampled 的差异仍然存在；严格说这是混合启发式更新，不是所写 sampled 双层目标的精确随机梯度。

价格更新方向保持：

\[
\lambda\leftarrow\max\{0,\lambda+\eta_\lambda(\overline c-\rho)\}.
\]

这是对偶最小化的投影梯度形式；不能因为叫“对抗”就把 λ 更新方向反转。ρ 是平均成本约束，不是每个样本位移上界。改变 q_D 尺度后不能照搬旧 λ 数值并声称相同强度；本轮应从预先固定 λ 初值与更新率开始，并报告全部训练和尾段运输成本。

由于逐点 `r★≤q_B`，对同一不确定集合有 `sup_Q E[r★]≤sup_Q E[q_B]`。但是**有限候选得到的最大 q_B 是连续内层最大值的下界，不是其上界**，因此训练中的有限候选分数不能直接用于认证整个 Wasserstein 球。

## 6. 三种改良方法如何去掉 GeoSteiner

### 6.1 ES：直接在便宜实际代价上搜索

设合法化映射为 `Z=T_X(u)`，定义 `h_X(u)=F_D(X,T_X(u))`。对 iid 高斯方向 `ε_j`，正反评分构成：

\[
\widehat g(u)=\frac1{2k\sigma}\sum_{j=1}^k
[h_X(u+\sigma\epsilon_j)-h_X(u-\sigma\epsilon_j)]\epsilon_j.
\]

在适当可积条件下，其期望是**含合法化映射的函数 h 的高斯平滑**梯度。它不是原始 greedy 离散树代价的普通精确梯度。归一化、截断、rank shaping 若加入，会进一步改变更新规则；不可继续把整条算法称为无偏原函数梯度。[Salimans 等, 2017](https://arxiv.org/abs/1703.03864)

每次评分只计算 greedy 树长、M/B 和运输成本。保留全部已评合法点中的最佳实际 `F_D`，包括 X；不能仅返回最后搜索中心。

### 6.2 Critic 搜索：用自生成标签拟合 greedy 代价

另设搜索代理 `s_ψ(Z)`，训练标签为当前 Actor 的 `q_D^g(Z)`；不复用 sampled Critic 的语义。坐标上升优化：

\[
s_\psi(Z)-\lambda c(X,Z).
\]

实现需保留输入 autograd 图；最终候选按实际 `q_D^g−λc` 复核。Actor 不断更新时搜索标签会陈旧，应预先固定代理刷新节奏，并记录每次标签所属 Actor 的更新号/权重 hash。便宜标签使在线刷新可行，但不保证代理方向有效。

Hungarian 匹配可在 autograd 外求当前 assignment，再在 Torch 中求该分支 L1：唯一 assignment 且符号固定处对应局部分支导数；ties 与绝对值零点只有选定次梯度约定。训练代理 q_D 时不需要对 M 的 Prim 决策求导，代理学习整体 q_D；若显式计算一个 `预测L / M` 的微分，则必须明确如何处理 MST 分支变化。

至少报告保留数据上的 MSE、Spearman、零梯度比例，以及相同起点/位移/实际评分预算下的无移动、随机方向、代理方向比较。非零梯度不能证明方向提高实际代价。

### 6.3 一步 RL：奖励只需当前解的线长

攻击器 `p_ω(u|X,λ)` 产生合法化位移 `Z=T_X(u)`，最大化奖励 `R=F_D(X,Z)`。其 loss：

\[
\mathcal L_{adv}=-\mathbb E[
\operatorname{stopgrad}(R-b(X,\lambda))\log p_\omega(u|X,\lambda)].
\]

baseline 不得依赖同次动作。若 u 经过 clip/合法化后评分，log probability 必须对应实际采样的原始随机变量 u；不要把 clipped Z 误当为原始高斯样本来求 log probability。攻击器与主 Actor 的训练方向相反：主 Actor 最小化代价、攻击器最大化罚运输后的代价。

使用时多候选择优与训练单样本平均奖励仍不同，必须报告。搜索策略的训练/验证标签开销、重放缓存和刷新开销都计入总账，不能只比较在线 K 次评分。

## 7. 严格的“完全不依赖 GeoSteiner”应检查什么

本次建议采用最强且容易审计的定义：训练、搜索、选模、评测都不调用、不加载 GeoSteiner，不读取最优长度列，也不使用曾经通过最优标签训练得到的模型初始化。

1. Actor、训练 Critic、greedy 代理、攻击器全部 fresh initialization；随机种子匹配。
2. pin 数据独立生成并落盘为只含坐标、ID、分布标签的 bundle；training、validation、test ID/坐标去重核查。
3. 不加载旧 `best.pt` 或带旧 Critic 权重的 checkpoint；复制架构源文件不等于复制权重。
4. 不复用旧最优参考缓存；不调用会隐式打开 reference sqlite/GeoSteiner 动态库的旧 Oracle 类。
5. 避免为读取归一化元数据而打开混含 `L★` 的旧数据文件；本轮新建 label-free 数据最清楚。
6. runtime guard 对 GeoSteiner 导入、可执行文件调用和对应动态库加载 fail-closed；另记录已加载模块/动态库及输入文件 hash。静态搜索和一个人为填写的 `geosteiner_calls=0` 字段单独不足以证明无依赖。
7. validation 和最优 checkpoint 选择使用预先冻结的 q_D 指标。test 只在定型后评测，不用于换训练损失、λ、攻击步数或训练时长。

若另开只做最终分析的 GeoSteiner 审计，可以称“训练和选模无 GeoSteiner，最终审计使用 GeoSteiner”。但不能称“端到端零 GeoSteiner”。本次若用户目标为完全摆脱，首轮无需这一步。

## 8. 不计算 L★，依然可以验证什么

### 8.1 首要质量指标

- 同一 test net 的实际 greedy 树长 `L`、`L/M−1`、`L/H`，均值/P95/最差分布组。
- 逐实例真实 regret 区间 `[max(0,L/M−1), L/B−1]`，均值与分位数的保守界。
- `M/B` 分布，检查保守上界松弛。
- 相同 net 上两模型 `L_A/L_B−1` 或长度差；负值说明 A 的实际解更短，无需最优参考即可确认。
- MST win rate、输出合法率、重复前向一致性。

MST 比较可以验证是否优于一个可构造的便宜基线，不能代替最优解比较。没有 L★ 时不能给出精确 mean regret、精确最优率或“与 GeoSteiner 相差 x%”的结论。

### 8.2 可选部署保底，不替代训练验证

若真正构造并输出 `L` 和 Manhattan MST 两者中较短的可行网络，则 `L_out≤M≤κL★`，得到保守 2-approximation 或本设置下 1.5-approximation。只在报告中写 `min(L,M)` 而没有实现对应 fallback 输出结构，不能宣称系统已实现该保证。fallback 也可能掩盖未训练 Actor 的退化，因此主报告仍保留原始 Actor 结果。

### 8.3 实验比较应控制的量

首轮至少包括无标签 ERM，以及同一损失的 Random-WDRO、ES-WDRO、Critic-WDRO、RL-WDRO；若算力足够，用至少三个预先固定 seed。相同 seed 共享初始化、中心数据、外层 update 数、optimizer/选模规则与评测集。若并行任务需要分配不同物理 GPU，应记录型号和实际时间，避免硬件差异被误报为算法加速。

需要同时报告：外层 update 数、实际训练样本数、greedy/采样 Actor 前向、padding 行、树长/MST/transport 计算次数、代理/策略额外数据量、训练秒数与峰值显存。相同更新数不等于相同计算预算；相同在线 K 不等于相同总训练开销。

对不同模型在同一 test net 的差值作 paired bootstrap；同时列出每个训练 seed，不能把数千 test nets 误当成数千独立训练重复。三 seed 的一致方向只能提供初步稳定性证据。初始 checkpoint 也评测，区分“实现跑通”“确实学会缩短树”“WDRO 改善偏移分布”“三种搜索器进一步改善”四个层次。

### 8.4 最低验收门槛

1. 数学单元验证：H、M、退化实例、排列/平移/缩放不变性；已知小例与手算 MST 相符；强/弱 B 关系一致。
2. policy 梯度实际非零、权重改变、训练损失/基线数值有限；q_M 的负值合法且可拟合。
3. greedy 批尺寸固定、尾批行为预先冻结，避免旧 v3 曾出现的批形状改变树结构问题。
4. 所有训练/选模/评测输入仅含 label-free 数据，运行审计未发现 GeoSteiner 使用。
5. finite-search 保留原 X；best 实际分数不低于原 X；transport 与 score 可离线重算。
6. 在独立 test 上报告质量、区间和方差；没有提升也完整保留，并明确结果不支持提升。
7. checkpoints、配置/代码/data hashes、事件日志、job exit 状态、锁释放和统计结果齐备后才标记完成。

## 9. 推荐的理论主张

完成无泄漏实验后，可以分别检验以下主张：

1. **可学习性**：不给最优标签，REST 仅靠自己的合法树长可学习更短的解。
2. **可迁移的鲁棒训练机制**：WDRO 的候选选择和三个搜索器不本质依赖最优求解器，可以在新的可计算代价上运行。
3. **可审计的质量区间**：每棵输出树都可获得基于 H/M 的保守最优性区间。
4. **经验鲁棒性**：在指定分布、degree、seed、训练预算和 finite-search 评测下，某方法是否改善 mean/P95/最差组实际长度。

不能从这些主张推出：原相对 regret 目标被无损替代、某新方法普遍优于旧最优标签训练、Actor 达到最优、或者有限候选训练获得连续 Wasserstein 鲁棒性证书。

## 10. 本审查依据

本仓库证据为 2026-09-15 当次实际阅读的理论入口与 v3.1 代码。独立外部理论引用已在相应推导段落给出。记忆文件仅用于定位训练 Critic / 搜索代理的历史解释和 v3.1 批尺寸约束，相关代码与当前文档已重新核实；未根据旧摘要直接宣称本次实验结果。
