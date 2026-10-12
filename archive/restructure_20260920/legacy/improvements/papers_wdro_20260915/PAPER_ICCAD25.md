# 25_ICCAD：HarRL 与基础 WDRO 的适用性、理论方案和组件验证

## 一、结论

**能结合，但主要用途是把当前 WDRO 从“线长稳健性”扩展到“带缓冲的时序与资源稳健性”。这不是替换对抗样本寻找器，也不是直接提高原纯线长指标的小补丁。**

如果实验目标严格保持当前 relative wirelength regret、原 Actor 输出和纯坐标输入，论文最核心的谐波缓冲候选机制没有对应的物理优化对象；直接接上通常不会带来有理论依据的线长提升。若接受增加缓冲位置、驱动源、电容与 Elmore 时序目标，基础 Random/Sobol + Hungarian + λ 对偶更新框架可保留，需改 Actor 动作、真实评估器、输入语义和损失标度。

工程上，本分支已实现可运行的 degree-10 组件适配器，可进入小规模 ERM/WDRO 对照；它不是原论文完整复现。完整实现论文宣称的 Shi–Li 最优缓冲插入算法、工业 RC 标定及全 degree 训练，需要明显更大的工作量。

## 二、论文与源码对应

- 本地论文：`/remote-home/Guilin/DRO/25_ICCAD.pdf`，9 页。
- 标题：*Achieving Simultaneous Buffering and Steiner Tree Synthesis via Harmonic Based Reinforcement Learning*，Lin Chen、Yuxuan Li、Hu Ding，ICCAD 2025。
- [作者发表页](https://hu-ding.github.io/publications.html)对应[官方 RLBufferTree 仓库](https://github.com/USTC-DIAG-LAB/RLBufferTree)。同年 ICCAD 还有 CoTL，但它不是用户提供的这份 PDF。
- 固定 commit：`c4f921a45a46eaec07b1f4afe1b5a73ec5a38e31`。
- 完整源码文件快照放在 `sources/RLBufferTree/`，`SOURCE_MANIFEST.json` 记录每文件 SHA256，`upstream_tree.json` 保留仓库完整树。未下载其约 600 MB 的训练日志、权重和数据，未运行上游训练入口。
- 上游与本项目都由 REST Actor–Critic 框架衍生，复用接口的基础很好。

## 三、论文到底贡献了什么

### 3.1 问题不是对抗坐标搜索

输入为有源节点的 net：`V={(x_i,y_i,C_i)}`，节点 0 是 source，其余是 sinks；还需要 wire/buffer 的 R、C 与 buffer intrinsic delay。输出为树拓扑及插入缓冲器后的物理 RC 树。论文式 (1) 目标为

\[
J(V,B)=\max_{i\ge1}\tau_i(V,B)+\beta_B |B|,
\]

其中 `β_B` 是缓冲器数量惩罚，**不是 WDRO 的 transport multiplier λ**。缓冲数是面积/功耗的代理，不能直接等同最终布局面积、总功耗或 signoff 时序。

### 3.2 谐波把变数量的位置决策压缩为相位决策

论文式 (7) 给出简单两端线段的理想均匀缓冲密度。式 (8) 由此固定空间频率：

\[
T=\sqrt{\frac{2R_bC_b}{R_wC_w}},\qquad
f_t(s)=\sin(2\pi s/T+\phi_t).
\]

边上的候选点取峰值，间距 T，由每条边一个相位 `φ_t` 决定起始偏移。跨 L 形拐点继续按路径长度累计相位；多个边段重叠时，应在同一空间坐标方向上把正弦波相加，再寻找合成峰值。Actor 将此前拓扑和缓冲候选信息都放入状态，后续连边可响应此前缓冲选择。

其收益是大幅降低缓冲候选位置策略的参数维数。其约束是同一段上峰间距固定，多缓冲位置不能任意独立移动。二端均匀间距的解析结论不等于整个 branching tree 上的最优位置定理。

## 四、当前基础 WDRO 与 HarRL 的数学连接

### 4.1 当前实现边界

`revisions/reference_precision_v2/Codex/wdro_rest_min/src/candidates.py` 先生成每个中心原点 + 32 个 Random/Sobol 扰动，经固定 MST/HPWL 代理预筛成原点 + 3 个扰动，再用 Actor greedy 输出和 GeoSteiner 长度计算真实 relative regret。`stage4/dual.py` 按

\[
r_\theta(z)-\lambda c(x,z)
\]

选择扰动，更新 `λ ← max(0, λ+η(mean c−ρ))`。`stage6/trainer.py` 外层重新采样 Actor 动作，真实 sampled regret 是 REINFORCE 反馈，Critic 是与动作无关的 baseline。候选 greedy 评估与外层 sampled 训练目标不严格相同，这个近似原本就存在。

### 4.2 替换任务损失后，对偶结构不变

固定任务 loss 的单位和归一化常数 `s_J>0`，定义

\[
\ell_\theta(v)=\mathbb E_{a\sim\pi_\theta(\cdot|v)}[J(v,a)/s_J].
\]

考虑带 source 属性的 nominal distribution P，理想 WDRO 为

\[
\min_\theta\sup_{Q:W_c(Q,P)\le\rho}\mathbb E_Q[\ell_\theta(V)].
\]

在满足相应对偶成立条件时，写成

\[
\min_{\theta,\lambda\ge0}\lambda\rho+
\mathbb E_{X\sim P}\sup_Z\{\ell_\theta(Z)-\lambda c(X,Z)\}.
\]

实际 pilot 仍用现有有限候选、预筛和 greedy task loss 近似内部 supremum。外层按采样的树拓扑及相位真实 J 训练，Critic 拟合同一 sampled task cost。`β_B` 固定，λ 由 transport budget 更新，两个系数职责不同。

不能继续把 `L*` 当时序/缓冲目标的最优值。GeoSteiner 只为纯线长给精确参照。若没有带缓冲时序任务的最优解，就使用固定标度的 cost，分别报告 max delay、skew、wirelength 和 buffer count；不要称 relative optimality regret。

### 4.3 source、sink 属性决定 ground cost

无 source 的无标号 net 可以在所有 pin 上 Hungarian 匹配。时序 net 的 source 与 sinks 不可互换；否则交换 source 与 sink 后可能 ground cost 为零、时序损失却改变。固定 source 的合理扩展为

\[
c_{root}(X,Z)=\frac{\|x_0-z_0\|_1+
\min_{\pi\in S_{n-1}}\sum_{i=1}^{n-1}\|x_i-z_{\pi(i)}\|_1}{n}.
\]

它保留 Hungarian-L1 核心机制，只在 source 语义上加约束。所有扰动生成需保留 index 0 的身份。此 pilot 固定所有 sink 电容，因此无需电容 transport；后续若扰动负载，需要把负载差异加入 ground cost，或明确条件 WDRO 只对坐标分布偏移稳健。改变 cost 后应重新校准 rho，不能不经校准继承旧数值。

### 4.4 纯 loss 缩放也影响 λ 学习

若 J 缩小 s 倍，在等价优化问题中最优 λ 也缩小 s 倍。有限步学习时固定原 λ 学习率不会自动产生等价轨迹。因此必须预先固定时序单位、目标归一化和 λ 学习率，不能观察结果后随意修改尺度再把差异归于 robust 训练。本 pilot 用 `(delay_seconds*1e18+0.25*B)/10`。

## 五、不能略过的源码与理论差异

源码是机制证据，不应在未经检查时等同论文描述。

1. **奖励不一致。** 论文为 max delay + β_B buffer count，源码 `utils2/rsmt_utils5.py:eval_batch` 返回 `0.4 delay+0.4 skew+0.2 length+0.2 count`。此分支明确选择论文启发的两项目标，四项指标都保留。
2. **buffer solver 不一致。** 论文描述使用 Shi–Li 最优插入方法，源码 `optimal_buffer_insertion` 是从深处开始、某段 delay 局部改善则插入的 greedy 规则；没有论文算法所需的 Pareto 状态集 DP，也没有对 max delay+count 的全局最优保证。此分支保留该真实 greedy 物理插入过程，明确不称 Shi–Li 复现。
3. **相位梯度缺项。** 原 Actor 的相位是确定性网络输出，真实评估前 `detach().numpy()`；policy log probability 只包含离散连边动作。相位会通过后续网络状态间接影响之后拓扑 logits，但末条边没有后续拓扑，因而没有与该相位实际缓冲收益对应的直接训练信号。不能把“相位函数可微”表述成“真实缓冲目标已端到端可微”。
4. **相位、负载和固定 source 的实现问题。** 原 `atan(sin/cos)` 丢失象限；分段后相位起点重置；内部 sink 的下游电容被提前 return 忽略；source 被固定为 0 却仍计入一个未采样类别的 log probability。这些均在本分支修正，详见 `vendor_harrl/PATCHES.md`。
5. **几何循环处理。** 上游会对几何图用 BFS 投影成 rooted RC tree；本分支保留此行为。交叉或重叠产生的拓扑，最终以该显式树的实际 RC 指标评估。它不是所有几何 RC 回路的电路仿真。

## 六、当前已实现的最小可验证方案

### 6.1 适配器接口

`adapter_buffer.py`：

```python
model = Adapter(degree=10, device='cuda:0', params={})
cost, log_prob, metrics = model.evaluate(pins, greedy=False)
```

Actor 仍为上游三层 attention/REST pointer 架构，真实输出为各 L 形连边及各边相位。相位改为固定 16 类 `2πk/16`；每条边的相位类别也计入 log probability，因此常规 REINFORCE 即可为末边相位提供梯度，不需要对插入 greedy、buffer count 或 Elmore 后处理求导。这是明确的适配，不是论文原相位模型的精确复现。

离散相位近似：16 个相位中最近者的角度差最多 π/16，固定单边上相应峰位置偏差最多 T/32。该位置界不意味着目标误差界：峰进出边界、插入集合改变、BFS 拓扑改变都能产生不连续，除非额外证明稳定区间内的 loss Lipschitz 和边界余量。

### 6.2 预先固定的合成物理参数

- 所有 source/sink 坐标仍来自本项目归一化几何域；source 为 index 0。
- sink C 固定 `0.5e-15`，来源是上游 `[0.1,0.9]*1e-15` 的中点。固定它用来单独验证几何分布偏移，不声称代表某个工艺角。
- degree 10：`Rw=0.003,Cw=2e-15,Rb=0.0002,Cb=5e-16,Kb=3e-21`，T≈0.182574；均来自固定源码。
- delay 乘 `1e18` 作为合成缩放单位，β_B=0.25，整体再除 10。不能把该数字直接写为工业 ps。
- 指标返回 `delay_raw_seconds, delay_scaled, skew_scaled, length, buffers, objective, phase_mean`。

### 6.3 先做什么对照才能证伪

首先固定同一 HarRL 适配器、参数初始化、训练样本流、更新数、学习率、Critic 和 held-out 验证数据，对比 ERM 与基础 WDRO。degree10、3 个配对训练 seed 的 pilot 足以检查训练是否可行、是否方向一致及运行成本，无法替代完整论文规模验收。

主指标是固定两项目标的 held-out 原分布与既定几何偏移分布 cost；分解指标报告 delay/skew/length/count，防止通过增加缓冲器或拉长线长换取 delay 改善却没有说明。内部最坏候选分数需报告实际真实评分次数，保持两组评估预算相同。按 net 配对 bootstrap 的区间仅反映该已训练 seed 下样本不确定性；3 seed 的一致性另行报告，不能将大量 net 当作大量独立训练实验。

若 WDRO 相对 ERM 的 cost CI 跨零，或三个 seed 方向不一致，则结论是本设置未建立稳健收益。若 WDRO improves cost 但原 wirelength regret 更差，应报告真实的目标权衡。即使所有 cell 改善，也不能据此推断原 precision_v2 纯线长 WDRO 被改进，或者已得到连续 Wasserstein ball 上的 certificate。

后续最小机制消融可在同一冻结协议下另开：固定 uniform 相位 vs learned16相位；actual局部buffer插入 vs no buffer；repair之前/之后的实现对照仅用于源码审计，不用有已知误差的 evaluator 当最终参考。它们不是本轮提前承诺的主 pilot 成果。

## 七、误差与保证边界

- 候选集合是合法连续域的子集，候选最大值至多等于连续内部 supremum。在同一 loss、同一 λ 下扩大候选不降低内部最大值；这只能比较固定有限模型的搜索下界，不能提供原连续 robust risk 上界。
- MST/HPWL 预筛是为线长任务设计，与 delay/缓冲数不一定一致。保留它是满足“基础寻找器不变”的控制条件，但必须写出适配后筛选误差可能更大。主结论限定在预筛后的有限候选目标。
- 如果能独立证明整个任务 loss 对合法几何扰动 K-Lipschitz，并且最终真正评分的集合是连续域的 ε-cover，则固定 λ 的内部 supremum 误差可界为 `(K+λ)ε`。当前没有 ε-cover，且 greedy tree/峰数量变化可不连续；不能直接套用此界。
- 强对偶、经验风险泛化、非凸训练收敛、有限真实评分搜索质量是不同问题。构造合法 dual 更新不自动解决其余问题。
- 本分支只验证固定 RC 与固定负载下的坐标偏移。真实 source 位置、load、wire RC、工艺角或 cell library 偏移未进入模糊集合，不能称覆盖这些偏移。

## 八、工程量判断

| 路线 | 改动 | 判断 |
|---|---|---|
| 保持纯线长 WDRO 不变，只阅读借鉴 Actor–Critic | 基本无新机制 | 改动小，理论新增收益也小 |
| 后处理缓冲评估，Actor/目标不变 | 新增 RC evaluator 与实际 buffer insertion | 中等，可评估迁移性，不能训练时序最优策略 |
| 本分支组件 pilot | 原 Actor+相位动作+RC真实cost+基础WDRO外层 | 中等，已完成 CPU 接口/机制验收，可GPU验证 |
| 论文完整接入 | 正确最优buffer DP、连续相位学习、source/C语义、工业单位、规模和基线全复现 | 较大，属于独立研究子项目 |

## 九、当前证据状态

- `buffer_cpu_smoke.json`：degree10、batch8，3 次评估及一次反传约 0.244 秒（该服务器当时 CPU，小样本计时，不是 GPU ETA），所有 metric 有限、greedy 输出完全一致、一般 phase gradient 非零。
- `buffer_cpu_acceptance.json`：9 项 PASS，包括真正末条边的相位 logprob 单独反传（梯度范数 14.9647）、固定 source 无伪 policy gradient、batch1、greedy 确定性；独立两节点 Elmore 手算与实现均为 `4.77e-20`；内部 sink 下游电容、分段相位连续和完全相消波均通过。
- `test_adapter_buffer.py` 是可重跑的验收程序。
- 上述证据只说明适配器可计算、训练信号有定义、若干重要物理/几何案例正确。GPU ERM/WDRO 训练收益和本分支总体验收，以主实验目录最终 manifest、日志与统计结果为准；此处未标记其完成。
