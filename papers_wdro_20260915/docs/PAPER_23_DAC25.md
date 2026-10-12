# 23_ICCAD 与 25_DAC：接入基础 WDRO 的理论、实现和验证边界

日期：2026-09-15 UTC。本文对应服务器原文件 `DRO/23_ICCAD.pdf`（9 页）与 `DRO/25_DAC.pdf`（7 页）。范围：基础 Random/Sobol WDRO，不采用 ES、Critic 输入梯度或 RL 对抗样本生成器。

## 1. 结论

两篇都能与目前 WDRO 的**训练分布鲁棒化层**结合，因为 WDRO 内层只需要给定输入点集后得到真实模型损失，并不要求 routing decoder 或损失可对坐标求导。

但要区分两个问题：

1. **让现在只优化线长的 REST-WDRO 更好**：23_ICCAD 的动态图表示与大网分块更直接；25_DAC 的 hub 机制主要服务于 cost–skew，未提供它会改善纯线长 WDRO 的理论理由。后者设纯线长权重后可以运行，但不能因此预期比 REST 更好。
2. **让 WDRO 服务于 timing/skew 路由目标**：两文均高度适用。维持原 Random/Sobol+真实重评分+Hungarian+learned dual price 外层，把 Actor 与真实成本替换为论文机制即可。这是新的应用分支，需改变有根数据语义、目标及部分评估契约。

从代码量看，可集中在两个 adapter；从研究问题看，23 的完整 timing 接入是中等改动，25 的 hub+MSPD 接入是中到大改动。已有 REST checkpoint 不能直接当作两种异构 Actor 的预训练权重。短实验能验证结合可训练且在固定预算下是否有改善信号，不能直接复现论文 Pareto 图或声明工业 timing 收益。

## 2. 论文与官方源码身份

作者 [Hu Ding 的 publications 页](https://hu-ding.github.io/publications.html) 的 ICCAD 2023、DAC 2025 对应入口已现场核对。下载固定 commit 的 ZIP 并安全解包，未执行原仓库训练入口。当前环境缺少 git 二进制，因此保留 API commit、ZIP SHA256 与下载 URL 作为替代溯源。

| 文件 | 标题/作者 | 官方仓库与本次 commit |
|---|---|---|
| `23_ICCAD.pdf` | **Towards Timing-Driven Routing: An Efficient Learning Based Geometric Approach**；Liying Yang、Guowei Sun、Hu Ding | [Timing-RST](https://github.com/LiynnYang/Timing-RST)，`354e25747c741a643c9cc48d9a4370e9e8d9fca7`，commit 日期 2026-08-22 |
| `25_DAC.pdf` | **To Tackle Cost-Skew Tradeoff: An Adaptive Learning Approach for Hub Node Selection**；Guowei Sun、Lin Chen、Qiming Huang、Hu Ding | [Adaptive Hub Selection](https://github.com/USTC-DIAG-LAB/An-Adaptive-Learning-Approach-for-Hub-Node-Selection)，`b42fc1fe7a2099772f950f01d97773de09d45f44`，commit 日期 2026-08-13 |

源码与其 `*_PROVENANCE.json` 均在本目录 `sources/`。这些是论文作者入口现在提供的发布版本，commit 晚于论文发表；不能假称已恢复论文发表时的实验环境或原始 checkpoint。25_DAC PDF 的字体字符映射有损，`sources/25_DAC_recovered.txt` 对部分拉丁字符做了映射恢复；数学含义以原 PDF 与源码互相核对，不能逐字信任恢复文本中的希腊字母或不等号。

## 3. 共同的数学接口：保留基础 WDRO，明确变化的对象

### 3.1 当前主线与分支的关系

主线损失是线长相对最优值的 regret：

\[
\ell_\theta(X)=\frac{L(T_\theta(X))-L^*(X)}{L^*(X)}.
\]

其有限候选训练近似为

\[
\min_{\theta,\lambda\ge0}\lambda\rho+
\frac1B\sum_i\max_{Z\in\mathcal S_4(X_i)}
\{\ell_\theta(Z)-\lambda c(X_i,Z)\}.
\]

本分支保留候选产生和预筛：原点加 32 个 Random/Sobol 扰动，廉价预筛后保留原点与 3 个扰动做真实成本评估。不能将其写成“训练精确搜索了全部 33 个候选”。新的内层损失改为论文实际物理目标的固定尺度值，见后文；这不是 `L*` regret，也不调用 GeoSteiner 来假装获得多目标最优参考。

### 3.2 Root 不能与 sink 任意匹配

两文均指定第 0 个节点为源点/root。原无标签点云的全点 Hungarian 距离不识别 root；一个代价很小的匹配可能把 root 交换成 sink，从而把不同的路由任务视为相同输入。采用有标签的距离：

\[
c_{root}(X,Z)=\frac1n\left(\|x_0-z_0\|_1+
\min_{\pi\in S_{n-1}}\sum_{i=1}^{n-1}\|x_i-z_{\pi(i)}\|_1\right).
\]

仍是 Hungarian-L1；区别只是匹配仅允许同类端点。随机扰动照常作用在 root 与 sinks 的坐标上，root 的**身份**不变。若未来固定 root 坐标，必须作为另一个预先规定的扰动空间，不能在实验后因结果不佳切换。

新 `rho` 仍事前用训练校准集候选成本的 q75，但 rooted 与 unrooted 必须分别校准。原数值不可无条件复制。以下 `alpha`、`beta` 是物理目标权重，`lambda` 是 WDRO 的非负运输价格；两者数学角色完全不同。

### 3.3 Policy gradient 不要求离散路由可微

对选中的输入 `Z`，Actor 采样轨迹 `a`，真实 evaluator 返回 `J(Z,a)`。固定尺度 `M_n=2(n-1)` 下，训练 score 为 `s=J/M_n`：

\[
\widehat{\nabla_\theta R}
=\frac1B\sum_i(s(Z_i,a_i)-b_\phi(Z_i))
\nabla_\theta\log p_\theta(a_i\mid Z_i).
\]

`s` 和 advantage 中的 baseline 必须停止梯度；baseline 自身通过 MSE 学习。路由转换、MSPD、最大路径和 skew 都可以是离散或不可微运算。没有必要为了使用这些论文而引入三种改良 attacker。

本轮 Actor 采用论文发布网络，Critic/baseline 统一为本分支 action-independent MLP，以使 ERM/WDRO 配对使用同一个控制变量实现。此项是明确的 pilot 简化，不是对作者原三路网络结构的完整复现。

固定 `M_n` 只做正比例缩放，保持同 degree 内原物理目标的排序。它不改变某个输入上树的最优解，但使合适的 dual price 和优化步长的数值尺度变化。因此新分支重新学习 `lambda`，不直接继承原 regret 模型的 lambda。不同论文目标的 score 数值不可跨行当作同一质量指标比较。

## 4. 23_ICCAD：哪些机制能接入

### 4.1 原文关键内容与源文件位置

| 机制 | 原 PDF | 本次官方源码 |
|---|---|---|
| timing–wirelength 目标 `(1-alpha)L+alpha R` | p.2，式 (1) | `Timing-RST/train_lambda.py`，训练成本组合处 |
| 有向 pin 边表达 L 形 rectilinear segment，逐步生成树 | p.2–3，图 1–3，式 (2)–(3) | `models/model_sorted.py:Actor.forward` |
| 位置+source indicator | p.4，Initialization | `models/model_sorted.py` 输入额外第 3 维，`pad[:,0,0]=1` |
| 一阶、有公共父/子节点的二阶三通道 GAT | p.4–5，式 (4)–(6) | `models/gat.py:TripleGATLayer`；`models/model_sorted.py:TripleGAT` |
| 动态已连/未连图表示、合法边概率与 mask | p.5，式 (7) | `models/model_sorted.py:GraphEmbed3,Actor.forward`；`models/utils.py:Pointer` |
| 相邻 alpha 模型暖启动 | p.5，II-C | `train_lambda.py` |
| polar quadtree、bounded bidirectional search、局部源点及拼接 | p.5–7，III-A/B、图 5–9 | `utils/divide_merge.py`、`algorithms/polar_quadtree/`、`utils/nn_solver.py` |

`R(T)=max_sink d_T(root,sink)` 是线性路径长度近似，并非完整 Elmore RC delay。其原实验的归一化参考分别使用 FLUTE 线长与 CL 最大路径（p.7），不是 GeoSteiner 多目标最优值。

### 4.2 路线 A：只借动态图表示，保持纯线长任务

把原 REST Actor 换成 TripleGAT Actor，令 `alpha=0`，仍用原线长 regret 与精确 RSMT 参考训练。可以保留不带 root 语义的原任务，但发布 Actor 本身显式从 0 号点开始扩树；若按任意 root 处理，需事前定义 root 采样或输入排列规则并保证比较公平。更稳妥的独立实验是把 root 也作为明确条件，专门比较 topology-aware Actor 在纯线长目标上的收益。

改动：中等；改变模型架构和 checkpoint，不必改变 attacker。**本轮不运行这个额外分支**，避免把同一论文的多种机制全部混进一个无法归因的实验。

### 4.3 路线 B：完整 moderate-degree timing Actor + 基础 WDRO（本轮实现）

定义

\[
J_{23}(X,a)=(1-\alpha)L(X,a)+\alpha R(X,a),\qquad\alpha=0.5,
\]

采用官方 `Actor` 全部参数及三通道动态更新。`adapter_timing.py` 只在内存改写导入 namespace；原模型源文件未改。真实线长使用 `algorithms/eval.c` 的逐端点区间长度算法的 NumPy 等价转写；最大源汇路径直接加载 `utils/distance.py:get_distance`，与官方训练 `utils/myutil.py:eval_distance` 相同。

这里的“真实”指**官方训练成本实现**，不是独立工业 STA 或重新实现 sweep-line 后经证明的物理网络延迟。论文论述了 graph-to-tree sweep-line 转换，发布训练脚本实际调用的是上述 `eval.c` 与 `distance.py`；本轮忠实复用其评分口径，并记录这一实现边界。合法性检查验证 pin-level 输出有 `n-1` 条无环边且连通。

### 4.4 路线 C：polar quadtree 分块 + 已有 WDRO 局部求解器

若目标是从 degree≤30 扩大到 100、1000，III 节的框架可作为推理时组合层：先按 root 周围的 data-dependent rings/sectors 分块，容量约 30；确定块的连接次序和局部 root；把现有局部 solver 包装为 `solver(points)->tree`；再拼接。

这条路线能重用已有局部模型，推理代码改动小到中等，但如果训练目标是**全网** WDRO 风险，则必须在每个扰动全网后重新分块、重新选局部 root、重新拼接并评估全局损失。对每个块独立 robustify 不等价于全网 Wasserstein ball，更不保证全局 max path。当前 mainline 只输出边序列，需要输出实际节点/rectilinear 边的转换接口。

本轮 degree10 不涉及分块；因此任何 pilot 结果均不验证论文大规模加速或拼接质量。未来最小验证可固定已有模型，degree100，每种方法 100 个同源 net，比较整网合法性、wirelength/max path 和时间，再决定是否增加全网 WDRO 训练。

### 4.5 无法直接转移的结论

- p.7–8 的经验 Pareto 优势不是一般 WDRO 理论，也不保证本数据、本初始化、本训练预算下更好。
- 原文 large-degree 比较本身有边界：degree1000 时该方法不是在所有质量指标上超过 SALT；速度结论来自指定硬件和样本规模。
- 单个权重 `alpha=.5` 的实验只能评价该标量目标附近的点，不足以报告 Pareto frontier。
- `alpha` 暖启动策略可省训练，但与 learned WDRO lambda 无关；顺序微调收益需要另一个受控实验，不能从论文论述直接认定成立。

## 5. 25_DAC：Selector–Conductor 为什么可与 WDRO 组合

### 5.1 原文机制与源文件位置

| 机制 | 原 PDF | 本次官方源码 |
|---|---|---|
| 总线长及 sink 间 skew | p.2，式 (1)–(2) | `cpp/mspd.cpp:calcWL,calcSkew` |
| `beta*cost+(1-beta)*skew` | p.2，式 (3) | `models/evaluate.py:objective_weights,evaluate_batch` |
| root-aware embedding、root 向其他节点拼接 | p.4，式 (4)–(5) | `models/selector_conductor.py:Actor.forward`、`models/layers.py:Embedder` |
| Selector pointer、禁止重复选 hub、上下文更新 | p.4–5，式 (6)–(8) | `models/selector_conductor.py:Actor` |
| Conductor adaptive continue/stop | p.5，式 (9)–(10) | 同文件 `Bernoulli`/`over_flag`/`done_probs` |
| Critic、trajectory policy gradient、附加中间奖励 | p.5，式 (11)–(13) | `train.py`、`models/selector_conductor.py` |
| 已选 hubs 交 MSPD/STT 构造 | p.2–3，II-B、图 3 | `cpp/mspd.cpp:solve_single,PrimDijktra` |

设源汇路径为 `d_i=d_T(v0,vi)`，则源码实际用

\[
S(T)=\max_{1\le i<n}d_i-\min_{1\le i<n}d_i.
\]

root 自身不参与 min；否则 min 总为 0，skew 会退化成最大路径而变成另一问题。原 PDF 式 (2) 集合符号易被理解得过宽，应结合正文“n−1 sinks”和源码 `for(i=1;i<orig_num_nets;i++)` 确认。

### 5.2 本轮最小完整机制

使用原 Selector 与 Conductor 网络：每一步先作 continue/stop 决策并选择尚未使用的 sink 为 hub，轨迹 log probability 必须同时包括 selector categorical probability 和 conductor Bernoulli probability。只对“选中了哪些 hub”求 log probability、忽略停止动作，会丢失对 hub 数量的学习。

给定最终 hubs，调用原 C++ MSPD。发布 evaluator 对每一组 hubs 进行 **11 个 Prim–Dijkstra alpha 值**的构造，再选择 `beta L+(1-beta)S` 最小者。这里内部 alpha sweep 属于原路由求解器，和 WDRO 的 32 个对抗输入候选完全不同。应计入路由成本：一次“真实对抗样本评分”约含 11 次 MSPD 构造；不能与 REST 的一次 decoder/evaluator 调用按相同 wall-time 假比较。

本轮固定 `beta=.5`、最大 5 个 hubs、保留自适应停止行为。`adapter_hub.py` 接入原 `Actor` 和原 `evaluate_batch`，评分为

\[
s_{25}(X,a)=\frac{0.5L(MSPD(X,H(a)))+0.5S(MSPD(X,H(a)))}{2(n-1)}.
\]

MSPD 按官方发布接口把 `[0,1]` 浮点坐标乘 1000 再截断为整数，结果除 1000。因此小于 .001 的几何变化可能不改变 MSPD 成本；这属于本轮明确冻结的数值口径，不能将其当作连续坐标下精确的 WDRO 最优值。Hungarian 成本仍在原归一化浮点坐标计算，记录该轻微量化差异。

### 5.3 原文中间奖励的处理

官方 Actor 返回 `reward=sum_t 0.1*gamma^t*continue_t`，默认 `gamma=.5`。原 `train.py` 的 advantage 是 `physical_cost - critic - intermediate_reward`。奖励取决于选择/停止动作，**不是**合法的 action-independent baseline；保留它会令外层目标包含偏好更多 hubs 的 shaping 项。

为了使本轮 WDRO 内层真实评分与外层最小化的是同一物理 `cost–skew` 目标，协议事前固定 `policy_bonus_weight=0.0`。网络仍按原源码产生 raw bonus，adapter 将其记录为 `raw_policy_reward`，但 `policy_reward=0`。因此本轮验证的是**原 Selector–Conductor+MSPD 主机制的 cost–skew WDRO 适配**，未完整复现作者包含额外 hub 数量奖励的训练配方。

若无 shaping 难以训练，结果应报告这一限制；不能看结果后打开奖励并仍沿用同一冻结实验标签。当前 CPU 验证已确认关闭 bonus 时采样与 finite nonzero gradient 均可运行，但这不是其长期优化质量的证据。

### 5.4 适用性与改动大小

其自适应停止减少需要学习/枚举的 hub 集合，适合“用少量关键节点控制可扩展的传统构造器”的问题。它是**决策网络**，不是对抗样本攻击器，因此把它用于 route construction 不违反本轮排除三种 learned attacker 的范围。

与当前 pure-wirelength REST 的接口差异较大：动作从 `n−1` 条有向边变为 variable-length hub 集合，真实环境从 REST 的线长 evaluator 变为 MSPD，目标增加 skew，输入新增 root 语义，训练、评估和 checkpoint 都应隔离。WDRO 外层本身的改动仍小。

不能直接把“Conductor 自动停止”改成“自动增加/停止对抗候选数”并声称复现论文；那会改变本轮已冻结 attacker/budget，且需重新处理可变真实评分预算。

### 5.5 不可转移的实验结论

原文 p.6 的 degree10 模型使用 batch2048、约100k iterations，训练/比较规模、PD/MSPD/BST-DME 基线与本轮短 pilot 不同。大网 degree1000 的外推也不由 degree10 本轮验证。原文 normalized wirelength/skew 分别除以 Prim/Dijkstra 的参考值，这些不是 cost–skew 全局最优值。单个 `beta=.5` 上的 WDRO 配对优势不能升级为 Pareto frontier 优势。

## 6. 本轮可执行实验与验收

以 `protocol.json` 为正式冻结配置。本轮每个 paper family 做 **degree10、3 个 seed、ERM/WDRO 配对**，2048 updates、batch64；每对共享相同网络初始化、优化器设置、输入序列、baseline 架构及 checkpoint 选择规则。WDRO 使用原 32 候选产生方式与原点+3扰动真实重评分；候选机制不学习。

评估采用相同 held-out nets：nominal 与三个分布偏移，分别报告物理 wirelength、max path/skew、归一化标量成本。最终固定 Random/Sobol 的全 33 候选 true score 压力测试与运输预算有限支持 LP 用于检查候选集合内的鲁棒损失；依然不是连续 Wasserstein certificate。随机种子是独立训练重复，测试 net 内 bootstrap 不能代替训练重复。

可判定三个层次：

1. **工程可用**：导入/结构合法、成本有限、greedy 可重复、Actor 梯度 finite 且非零、checkpoint 可恢复。
2. **机制信号**：在相同 family 和预算内，WDRO vs ERM 的平均成本/尾部成本、OOD worst-group、相同 rho 的有限候选压力指标有方向一致的差异。
3. **研究结论**：只有当配对统计与训练重复支持时，才说“此 degree、此数据、此预算下有改善”；否则为不确定或未观察到收益。2048 updates 远低于论文原始训练量，应把欠训练与方法无效区分开。

与现有 WDRO 的比较需分层：REST-WDRO 行作为基础方法复核，各 paper 的 ERM/WDRO 行检验该机制能否接受分布鲁棒化。不同任务成本之间不能直接排名“哪个论文最好”。

## 7. 已实现产物、编译与 CPU 证据

- `adapter_timing.py`：官方 TripleGAT Actor；官方路径 evaluator；与 `eval.c` 等价的线长计算；namespace 隔离；合法树检查。
- `adapter_hub.py`：官方 Selector–Conductor；原始 C++ MSPD；adaptive stop 的 trajectory probability；成本/bonus 分离；参数注册。
- `verify_adapters_cpu.py`：和主线 `src` 共存、合法性、greedy 重复、目标等式、stop 后空 hub、训练梯度等检查。
- `artifacts/build/mspd_build.json`：使用分支 `.deps/ziglang` 与 `.deps/pybind11` 编译原 `cpp/mspd.cpp` 的命令、源文件 SHA256 和产物 SHA256，未改算法源码。
- 编译命令由记录可精确重放；输出 `.so` 在 `artifacts/build/`，运行时不要求系统安装 compiler 或 pybind11。C++ 实现使用全局工作数组，一个 Python 进程内不应并行多线程调用；不同进程互相隔离。

本次实测仅 CPU：Timing B64/d10 单次 greedy 约0.964s、64/64 pin 树合法，NumPy线长与仓库 `libeval.so` 比较 max absolute difference 为0；Hub B64/d10 greedy 约0.222s、64/64 hub 输出合法，两次 greedy 完全相同，train+backward约0.317s且梯度有限非零。证据见 `artifacts/timing_cpu_benchmark.json`、`artifacts/hub_cpu_benchmark.json`。这些时间只用于判断可运行度；GPU 全程 ETA 必须来自后续实际 GPU profiler，不能以论文表格或 CPU 时间直接宣称。

## 8. 优先级建议

若目的仍是**当前 wirelength WDRO 改善**，优先保留主任务，研究 23 的 topology-aware Actor 或大网 divide–merge；25 的 hub 方法先作为新 cost–skew 分支，不宜期待它直接提升线长 regret。

若目的是**让 WDRO 面向更接近时序约束的路由任务**，先完成本轮配对 pilot：23 的 moderate-degree timing 接入成本更低、无需 native 编译；25 的工程接口稍多，但已用原 MSPD 源码打通。只有在冻结 pilot 有稳定收益或清晰诊断之后，才扩训练量、degree 和多个 alpha/beta 点。无需先升级对抗样本搜索器。
