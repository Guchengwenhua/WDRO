# 无 GeoSteiner 的 REST 与 WDRO：理论方案与本次验证边界

日期：2026-09-15。本次独立研究目录：`oracle_free_20260915/`。

**核心方案：保留 REST 构树，把真实最优长度归一化的 regret 换成只需坐标和当前输出树即可计算的目标。** Actor、训练 Critic、对抗搜索、选模与测试均从这条无最优标签的数据流出发。可以得到严格的逐实例保守质量上界；不能据此声称原目标被完全等价替换，或获得连续 Wasserstein 鲁棒性证书。

本文给出理论及 2026-09-15 冻结的第一轮 degree30 概念验证。实际参数绑定以 [../protocol.json](../protocol.json) 为准；测试和训练是否完成以产物为准。更详细的独立推导与审查见 [THEORY_REVIEW.md](THEORY_REVIEW.md)。

## 1. 为什么现有方法需要 GeoSteiner，哪里可以去掉

当前主线用

\[
\ell^\star(X,a)=\frac{L(X,a)}{L^\star(X)}-1
\]

作为主 Actor 的 sampled 代价，同时用 greedy 动作的同类代价进行 WDRO 候选评分。`L★` 来自 GeoSteiner；`L(X,a)` 只是对已选动作对应的线段求长度。

**现有代码其实已经包含不需要最优标签的学习先例。** 本地原始 REST 的 `/remote-home/Guilin/DRO/DRO/WDRO-REST/train.py` 第 80–92 行直接计算 sampled 输出的 `lengths`，用 `length−prediction` 更新 Actor、用 `MSE(prediction,length)` 更新 Critic；第 107–113 行也直接按 greedy 长度验证，没有在这些训练损失中除以 L★。这说明“RL 用自身构造结果的代价学习”不是本次新提出的原理。本次新增研究是把当前 **regret-based WDRO 的候选选择、三种攻击器、选模和评价**整体改为可计算无最优标签目标，增加 H/M 的可审计界，并移除旧长度动态库潜在的 GeoSteiner 链接依赖。对原始 REST 的全部运行时依赖不在这里作额外无调用结论。

训练 Critic 拟合 sampled 代价的条件均值，属于 REINFORCE 的减方差基线。Critic 坐标搜索中的代理则拟合 greedy 代价。ES 和一步 RL 的实际奖励同样经 `L/L★−1` 间接依赖 GeoSteiner。因此仅更换 Critic 标签，不能让整个学习系统摆脱最优求解器。

本次同时更换四个位置：

1. 主 Actor 的真实 sampled 学习代价。
2. 训练 Critic 的目标值。
3. 对抗候选的实际 greedy 评分，以及 ES / 搜索代理 / RL 的反馈。
4. validation 选模和独立测试的指标。

对树结构的解码和实际长度计算保留现有 REST 语义，所有网络权重重新随机初始化。

## 2. 无求解器的几何上下界

### 2.1 定义与证明

设 X 是 n 个二维 pin，目标是无障碍平面上水平、垂直线段的总长度。定义包围盒半周长：

\[
H(X)=\max_i x_i-\min_i x_i+\max_i y_i-\min_i y_i.
\]

定义 terminal Manhattan MST：

\[
M(X)=\min_{T\text{为terminal spanning tree}}
\sum_{(i,j)\in T}\left(|x_i-x_j|+|y_i-y_j|\right).
\]

M 只在原始 pins 之间选择最小生成树，可以用稠密 Prim 在 `O(n²)` 时间内精确计算。它不搜索 Steiner 点。本次使用：

\[
\boxed{B(X)=\max\{H(X),M(X)/2\}\le L^\star(X)\le M(X).}
\]

证明分三步：

- **H 是下界。** 连接全部 pins 的网络必须覆盖横向和纵向跨度，水平总长与垂直总长分别至少等于这两个跨度。
- **M/2 是下界。** 把最优 Steiner 树每条边加倍，可得到长 `2L★` 的遍历。按该遍历到达 terminals 的顺序用 Manhattan shortcut，三角不等式保证不增长度；删去闭合路径的一条边得到 terminal spanning tree。M 最小，所以 `M≤2L★`。
- **M 是上界。** 把 MST 每条边实现成一个水平加垂直的 L 形路径，会得到连接全部 pins 的可行网络，总长度至多 M；重合合并、去环只会缩短。

理论上无障碍二维 rectilinear 情况还满足更强界 `2M/3≤L★`，可取 `max(H,2M/3)`。这是 Hwang 的经典定理；**本轮没有采用这个更强分母，不把它作为已测试变体**。[Hwang 原论文](https://epubs.siam.org/doi/pdf/10.1137/0130013?download=true)

带障碍物、拥塞代价、金属层切换、非 Manhattan 度量均超出上述保证。归一化保留纵横比，把最长跨度映射到 `[0.1,0.9]` 的跨度 0.8，因此正常实例 `H≥0.8`，分母不接近零。退化重合输入必须在除法前拒绝或单独处理。

### 2.2 无 GeoSteiner 的精确小例

对任意三个 pins，令 `s=(median(x_i),median(y_i))`。将三点用 L 形路径连到 s，可构造长度恰为 H 的网络：每一坐标三点到中位数距离之和就是该坐标跨度。因此：

\[
\boxed{n=3\ \Longrightarrow\ L^\star(X)=H(X).}
\]

这给出不调用任何精确求解器的单元测试：随机生成三点，检查 `sum_i||X_i−s||₁=H` 和 `B≤H≤M`。degree30 的 M 则可与独立 SciPy MST 实现逐实例对照，不需要最优 Steiner 参考。

## 3. 长度 evaluator 的精确定义与保证

本次用纯 NumPy 重写 `algorithms/eval.c` 中的 `eval`，避免加载同一旧动态库中包含的 `gst_rsmt` 包装函数。对动作对 `(u,v)`，铺设：

- 位于 `x_u`、从 `y_u` 到 `y_v` 的竖直线段。
- 位于 `y_v`、从 `x_v` 到 `x_u` 的水平线段。

每个 pin 的所有竖直段合成一个 span，所有水平段合成一个 span，再把各 pin 的 span 长度求和，得到 `L_eval`。动作对组成连通的 n−1 边 terminal tree，是铺设网络连通的充分条件；实现检查索引范围与并查集连通/无环性质。

这里必须区分两个长度：

\[
L^\star\le L_{union}\le L_{eval}.
\]

`L_union` 是物理线段并集长度。不同 pins 若共享横坐标或纵坐标，它们的 spans 可能重合，旧 evaluator 仍按不同 pin 求和，故可能重复记账。横段和竖段相交只贡献零测度的点，本身不会重复累计长度。交叉可能形成几何环，但剪环仍可得到不更长的可行树。

因此 `L_eval` 与历史 evaluator 完全相同，且始终是一个可行连通网络长度的保守记账值。**本次报告中的 L 默认指 `L_eval`，不把它自动称为去重后的物理长度。** 已知同 x/y 的重合场景应保留为回归测试。无重合 spans 时两者相等。

## 4. 两个学习目标，以及它们到底保证什么

本次比较两种归一化：

\[
q_M(X,a)=L_{eval}(X,a)/M(X)-1,
\]

\[
q_B(X,a)=L_{eval}(X,a)/B(X)-1.
\]

`erm_mst` 使用 q_M；`erm_lb` 与四种 WDRO 分支使用 q_B。

### 4.1 MST 目标

q_M 可以为负，表示 evaluator 成本小于 terminal MST 成本，并不表示超过最优解。禁止截断负值，否则 Actor 比 MST 更好后将失去继续优化的信号。训练基线和搜索代理均使用线性输出头。

### 4.2 保守下界分母目标

定义与历史记账语义一致的相对代价：

\[
r^\star_{eval}=L_{eval}/L^\star-1.
\]

有逐样本严格区间：

\[
\boxed{\max\{0,q_M\}\le r^\star_{eval}\le q_B.}
\]

并且 `q_B≤2(r★_eval+1)−1`。对实际物理并集 regret，则有：

\[
\boxed{0\le r^\star_{union}=L_{union}/L^\star-1\le q_B.}
\]

**有重复记账时，`max(0,q_M)` 不能声称是物理并集 regret 的下界。** 如果未来实现物理线段去重，可把同一公式中的 L 换为其精确并集长度，本次不混用两种定义。

q_B 不是观测的精确 regret。例如 q_B 为 20%，只能说明实际可行解的真实 regret 不超过 20%。即使某个动作已经最优，`L★/B−1` 仍可能显著大于零。下界松弛会影响攻击目标，所以报告 `M/B`，辅助判断攻击是否主要寻找几何下界更松的输入。

### 4.3 固定实例等价，跨实例不等价

对固定 X，M、B、L★ 都是正的、与动作无关的常数；这些目标保持所有动作的长度排序。对共享参数网络，平均目标对不同 X 分配不同权重，因此有限容量最优参数和训练过程可以不同。

在 WDRO 内层，Z 本身也会改变分母。用 q_B 替换 regret 会改变对抗实例排序及问题定义，不能把新方法称为旧 regret-WDRO 的严格等价加速实现。

## 5. Actor 与训练 Critic：只有自身线长，也足够学习

令 `π_θ(a|X)` 为 REST 的动作分布，固定正尺度 D 为 M 或 B。目标：

\[
J_D(\theta)=\mathbb E_{X,a}[q_D(X,a)].
\]

动作独立基线 `b_φ(X)` 给出：

\[
\nabla_\theta J_D=
\mathbb E[(q_D(X,a)-b_\phi(X))\nabla_\theta\log\pi_\theta(a|X)].
\]

原因是对每个 X，`E_a[∇θ log πθ(a|X)]=0`，乘上只依赖 X 的基线后仍为零。无须对离散动作和树长计算求导。[REINFORCE 原始论文](https://link.springer.com/article/10.1007/BF00992696)

实现采用：

\[
\mathcal L_{actor}=\operatorname{mean}
[\operatorname{detach}(q_D-b_\phi)\log\pi_\theta(a|X)],
\]

\[
\mathcal L_{baseline}=\operatorname{mean}[(b_\phi-\operatorname{detach}(q_D))^2].
\]

训练基线在更新前预测，再同时使用本批反馈更新；不能先利用当前动作训练基线再重算同次 advantage。基线近似当前 sampled 代价的条件均值，不是 L★ 预测器，也不是 greedy 搜索代理。

本轮基线是小型集合网络，不复用旧 Critic 权重。也可以去掉训练 Critic，改用冻结 Actor 的 greedy rollout 基线，或同状态独立采样的 leave-one-out 基线；组合优化已有这种直接用路径代价训练的先例。[Kool 等原论文](https://arxiv.org/abs/1803.08475) 这些可选基线不是本轮新增消融。

## 6. WDRO 的新目标和实际近似

沿用 pin 排列不变的运输成本：

\[
c(X,Z)=\min_\sigma\frac1n\sum_i\|X_i-Z_{\sigma(i)}\|_1.
\]

若使用 sampled 期望代价 `\bar q_B(θ,Z)=E_a[q_B(Z,a)]`，理想鲁棒目标为：

\[
\min_\theta\sup_{Q:W_c(P,Q)\le\rho}\mathbb E_Q[\bar q_B(\theta,Z)].
\]

满足对偶成立的技术条件时，对应：

\[
\min_{\theta,\lambda\ge0}\left\{\lambda\rho+
\mathbb E_X\sup_Z[\bar q_B(\theta,Z)-\lambda c(X,Z)]\right\}.
\]

对偶的使用需要对应可测性、增长/紧性等条件；公式本身不意味着有限候选程序获得了连续最优值。[Gao 与 Kleywegt 原始研究](https://arxiv.org/abs/1604.02199)

本次与现有 REST 工程结构相同，在内层用 greedy 动作真实记账长度评分：

\[
F_B(X,Z)=q_B^g(Z)-\lambda c(X,Z).
\]

在有限候选中选最大 F_B，再在所选 Z 上做 sampled Actor 更新。内层 greedy 与外层 sampled 不完全一致，故这是实用混合近似，不宣称是上述理想目标的精确梯度算法。

每批价格更新：

\[
\lambda\leftarrow\max\{0,\lambda+\eta_\lambda(\operatorname{mean}c-\rho)\}.
\]

本轮 λ 从 1 开始、更新率为 1。ρ 不读取旧最优标签或旧模型价格，而由 2,048 个独立训练分布校准中心、每中心 7 个随机候选的运输成本 q75 冻结，数值约为 0.062502；校准扰动与实际随机搜索都先裁剪到 ±0.08 再合法化。ρ 是平均预算，候选自身的位移限制是另一个机制；末期平均靠近预算不等于全程严格满足预算。

对每个候选，q_B 是最优性保守上界；但有限候选 max 仍是连续 `sup_Z q_B` 的下界。因此不会把有限攻击结果当作全 Wasserstein 球的证书。

## 7. 四种无 GeoSteiner 的候选生成器

所有 WDRO 分支每个中心实际评分 8 个候选，包含原中心 X，最终都用当前 Actor 的实际 `F_B` 选优，保证不劣于原中心分数。位移经同一合法化，最大预合法化坐标位移 0.08；合法化包括裁剪、统一归一化和非法输入回退，会改变最终位移。

### 7.1 Random

产生 7 个标准差 0.04 的高斯坐标扰动，加原中心形成 K=8 对照。它是区分“额外候选评分收益”和“学习/方向收益”的最低对照。

### 7.2 ES

产生 3 个随机方向及其反向，实际评分原中心与 6 个正反探测点，再根据正反分数差移动一次并评分，合计 8 次。

本轮方向由 Gaussian 向量经 RMS 归一化而得，等价于 d=2n 维、半径 `√d` 的均匀球面方向。令 `h_X(u)=F_B(X,T_X(u))`，T 包含实际裁剪/合法化。对 σ=0.04、k=3，未再次归一化的原始估计量为：

\[
\widehat g=\frac1{2k\sigma}\sum_{j=1}^k
[h_X(\sigma\epsilon_j)-h_X(-\sigma\epsilon_j)]\epsilon_j.
\]

在相应可积和可微意义下，其期望对应 h 与半径 `σ√d` 均匀球体平滑后的梯度；球面公式的因子 `d/(√d)^2=1` 给出上式尺度。它**不是未归一化 Gaussian smoothing 的梯度**。实际更新还对 `ĝ` 再次 RMS 归一化，因此实际方向本身不再保留无偏梯度解释，也没有有限步全局最优保证。经典 ES 提供零阶平滑优化的动机，本轮的方向分布、合法化和保留最好点共同定义实际算法。[ES 原论文](https://arxiv.org/abs/1703.03864)

### 7.3 Critic 坐标搜索

另一个小型集合回归器 `s_ψ(Z)` 使用当前 Actor 的实际 greedy q_B 标签在线训练。7 个随机起点各做 4 次、每次步长 0.01 的 RMS 归一化坐标上升，再加原中心作实际评分。

为减少提案阶段 CPU 匹配开销，本轮代理上升用固定 pin 索引罚项：

\[
c_{id}(X,Z)=\frac1n\sum_i\|X_i-Z_i\|_1\ge c(X,Z).
\]

**提案目标是 `s_ψ−λc_id`，最终选择目标仍是 `q_B^g−λc`。** 两者并不相同；这只是低成本提案器。代理梯度不等于离散构树代价的精确输入梯度，也不保证提高实际目标。本轮不将代理性能直接解释为真实方向质量；更细的匹配方向消融可作为后续固定协议实验。

### 7.4 一步 RL

小型置换等变策略读取 X，产生 7 个高斯位移，经过同样合法化并用 F_B 奖励。**本轮策略输入没有 λ；当前 λ 参与奖励，策略在线适应。** 这是简化 contextual-bandit 攻击器，不是旧 v3.1 策略的完全复现。

令同中心第 j 个独立采样奖励为 R_j，基线取其余 6 个奖励的均值：

\[
b_{-j}=\frac16\sum_{k\ne j}R_k,\qquad
\mathcal L_{RL}=-\operatorname{mean}_j[(R_j-b_{-j})\log p_\omega(u_j|X)].
\]

其余独立动作提供 leave-one-out baseline，避免同一动作反馈进入自己的基线。log probability 对应裁剪/合法化前实际采样的 u，不把合法化后的坐标错当原始高斯样本。该目标最大化平均单次奖励，推理时多候选择优，两者仍有差异。

## 8. 本轮冻结验证设计

| 项目 | 本轮设置 |
|---|---|
| 范围 | degree30；无标签从零训练的概念验证 |
| 网络起点 | 每 seed 相同随机 Actor 初始权重；无旧 checkpoint |
| seed | 101、102、103 |
| 分支 | ERM-MST、ERM-LB、WDRO-Random、WDRO-ES、WDRO-Critic、WDRO-RL，共 18 模型 |
| 外层预算 | 每模型 2,048 更新，每批 64 |
| 搜索预算 | 每中心 8 次实际 greedy 评分，含原中心 |
| 数据 | 训练中心为 uniform；validation/test 为 uniform、mix_gauss、edge_bias、grid_cluster；全部新生成且无参考标签 |
| validation | 每分布 256；每 128 更新；最小 worst-group mean q_B，平局保留更早 checkpoint |
| 常规 test | 每分布 1,024，独立于训练/validation |
| 独立有限攻击 | 每分布 128 中心，统一 32 候选随机攻击 |
| greedy 前向 | 本次固定 64 行，尾批复制填充并丢弃冗余行 |
| checkpoint | 每 32 更新；保存更新号、优化器、随机状态与配置绑定 |
| 核心统计 | before/after、同 seed 配对比较；2,000 次 paired bootstrap |

本轮 greedy 批尺寸 64 是新实验选择；旧 v3.1 使用 256，不能将本次称为其完全复现。共同固定批形状是为了避免已知的批形状引起 greedy 输出变化风险。

总工作量为 18×2,048=36,864 次外层更新，对应 2,359,296 个 sampled Actor 训练样本；12 个 WDRO 任务每任务实际候选评分 1,048,576 次，合计 12,582,912 次。validation、test、独立攻击、padding 行和中断重放另计。

四个 WDRO 分支都更新同一规模的 greedy 代价代理，使这部分小网络计算一致；仅 Critic 分支使用它生成候选，其他三分支的这一步对提案没有作用。共同外层更新数和在线 K 控制主要比较，但不同方法仍有不同的方向梯度/策略计算开销。报告实际前向行数、长度/MST/transport 次数、额外梯度、wall time 与峰值显存，不能由“K 相同”推断总计算量相同。

## 9. 评价与验收结论的层次

### 9.1 不使用最优标签也可直接回答

1. 训练后同一测试 net 的 evaluator 长度是否比初始模型更短。
2. 不同训练分支在相同 net 上哪个记账代价更低，mean/P95/最差组是否改善。
3. 有多少输出优于 terminal MST；其保守 regret 上界 q_B 有多大。
4. 新 WDRO 方法在指定随机攻击评测下是否降低记账长度与上界。
5. 训练、搜索、选模、测试是否完全无 GeoSteiner、无 L★ 标签、无旧预训练权重。

报告每个训练 seed，不能把大量测试 net 当作同等数量的独立训练重复。三 seed 提供初步稳定性证据；CI 跨零时不宣布稳定提升。

### 9.2 本轮无法直接回答

- 精确的平均 `L/L★−1`、精确最优率、与 GeoSteiner 相差多少个百分点。
- 新无标签方法相对旧最优标签方法在同训练预算下的收益；旧模型不是本轮配对训练对照。
- 更广 degree、真实芯片分布或任意未来分布的普遍优势。
- 无限候选内层最优，或连续 Wasserstein 球的全局证书。

### 9.3 无依赖与数学验收

纯几何检查包括：M 与独立 MST 算法一致、三点精确最优 H、小例和随机 tree 的 `L_eval≥B`、排列/平移/缩放关系、连接结构合法性。NumPy 长度实现与**独立构造所有物理线段并按所在直线合并区间的实现**对照：横纵坐标均不重复的随机输入，物理并集与历史 span 记账必须相等。本服务器缺少可用的 C 编译器，因此未执行独立 C 动态库对照；不能把源码逐行复核和区间并集对照写成“C 二进制测试通过”。

最终输出保留 pin 坐标、动作 topology、线长和几何界，验收可直接从 topology 重新计算 evaluator 长度，再复核 q_B/q_M 与候选分数；不需要再次执行 Actor 或求解器。

运行时 fail-closed guard 阻止已知 GeoSteiner 导入/动态库/外部程序、旧参考缓存和本目录之外的 checkpoint。新训练/选模/测试数据只存坐标和标识。训练日志与 audit 同时保留输入、源码、配置和 checkpoint hashes，不能仅凭手工写一个零调用字段证明无依赖。

最后区分四种结论：**工程跑通、从零学习有效、WDRO 改善偏移表现、三种搜索器额外改善**。它们需要逐项证据。训练完成不保证后三项成立；负结果也应完整汇总。

## 10. 下一步可能的扩展（本轮不执行）

更强 Hwang 下界、物理线段并集 evaluator、无 Critic 的 rollout baseline、代理匹配运输梯度、RL 输入显式加入 λ、其他 degree、与最优标签训练的公平对照，均可成为后续独立协议；不能观察本轮 test 结果后悄悄加入或替换。

若将来部署时真正输出 REST 与构造好的 Manhattan MST 两者中较短的可行网络，可取得保守 2-approximation，或本设置下借 Hwang 定理的 1.5-approximation。只在表格中把数值改成 `min(L,M)` 而没有实际输出 fallback 网络，不构成已实现的保证。
