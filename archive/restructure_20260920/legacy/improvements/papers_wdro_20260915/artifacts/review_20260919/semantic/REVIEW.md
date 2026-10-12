# 2026-09-19 独立语义验收

## 结论

**语义与目标接线审查通过，附明确科学边界。** 本审查未发现需要推翻本轮执行完成状态的目标接线错误；它不代替根代理对生命周期、hash、全部统计的最终验收。当前产物支持“基础有限候选 WDRO 能接入这三类求解任务并完成训练”，不支持“已稳定优于 ERM”“三篇论文改善了现有纯线长 WDRO”或“获得连续 Wasserstein 鲁棒证书”。

最需要补充到最终解释中的事实是：

1. **REST 的三个 WDRO 最佳 checkpoint 全为 update 0。** 它们逐 tensor 等于初始化，测试预测也等于初始化；目前没有选出一个由 WDRO 训练带来的线长改进模型。表中 WDRO 对 ERM 的细小均值优势来自 seed103 的 ERM 选择了 update256 后在测试上略变差，而不是 WDRO 学出了优于初始化的模型。
2. **Hub 的大多数最佳策略选择 0 个 hub。** 三个 ERM seed 和 WDRO seeds102/103 在全部 4×1024 测试样本上都立即停止；只有 WDRO seed101 的平均 hub 数为 0.0488–0.1104/net。这些结果很大部分对应原始无 hub 的 MSPD 路径，不能表述为 Selector 稳定找到了更好的 hub 结构。
3. **Buffer 不是论文完整复现。** 本分支学习 16 类相位，真实评分使用固定 RC 的 Elmore 模型；插入器是源码局部 delay-gain greedy，不是 Shi–Li 全局最优 DP。最优插入保证、工业时序结论和单独谐波模块的贡献都未建立。
4. **预算活跃与严格可行应分开。** 三个新任务的 λ 实际活跃，REST 的 λ 多数步骤为 0；有限步投影对偶训练并不要求每个 minibatch 预算可行。Buffer 时间平均运输成本仍略高于 rho，应作为未完全收敛的诊断；不能因此判工程失败，也不能写为训练严格满足预算。最终测试 LP 独立施加共享平均 rho，是另一个准确限定的问题。

## 审查范围与新证据

只读取 protocol、runner、adapter、物理 evaluator、THEORY/REPORT 和 24 个正式运行的日志/checkpoint/NPZ。新文件仅写入本目录；未启动 GPU、训练或改动旧产物。CPU 测试隐藏全部 GPU，禁止 Python 写旧目录 bytecode。

- 独立重算全部 24 个运行在 17 个 validation 点上的最早最小 OOD worst-group 选择，全部与选中 update 一致。初始化合法参与选择，测试集不参与该选择。
- 对 12 个 family/seed 配对，直接读取 `initial.pt` 比较 ERM/WDRO Actor state_dict，全部逐 tensor 相同。共同 Critic 的初始化一致性由相同构造次序与 seed 代码保证；此处没有保存的 initial Critic 可单独逐 tensor 核验。
- 从 `test_best.npz` 和 `test_initial.npz` 的物理分量独立重构 **196608 条** cost。所有 cost 与保存的真实 raw objective 经过相同 float32 转换后精确一致。Hub 的 L、skew、objective 分别以 float32 保存，因此由已舍入 L/skew 重新相加会产生最多 2.9802322388e-08 的 cost 绝对差、最大相对差 1.3364271e-07；与官方 evaluator 的分量舍入一致，不是目标不一致。验收容差为 `4*float32_eps` 的相对误差，未修改任何原值。
- 12 个 WDRO run 各有 2048 条有效顺序 progress；逐步按上一 λ、记录的 mean transport 和 rho 重算投影更新，最大绝对误差为 0。
- 重新执行现有 9 项缓冲 CPU 组件探针，输出重定向到本目录，全部通过。末条边 phase logprob 单独反向传播的梯度范数为 14.9646835；两节点无缓冲 Elmore 独立手算与实现均为 4.77e-20。还验证固定 source 无伪 policy gradient、batch1、greedy 确定性、内部 sink 下游电容、跨分段相位和相消波。

机器记录见 [summary.json](summary.json)，可重跑审查逻辑见 [audit.py](audit.py)，物理组件复核见 [buffer_cpu_acceptance.json](buffer_cpu_acceptance.json)。完整生命周期和统计复核由其他独立审查结果负责；本审查未重跑全部神经网络预测。

## 1. 真实目标是否一致

| family | 实际 score | 语义 |
|---|---|---|
| REST | L/L*−1 | GeoSteiner 纯线长相对 regret |
| Timing | (0.5L+0.5R)/18 | 线长与最长源汇几何路径，不是 RC delay |
| Hub | (0.5L+0.5skew)/18 | 原始 MSPD 几何代价；固定 scale1000 整数量化、每评分内部11点PD扫描 |
| Buffer | (delay_seconds×1e18+0.25B)/10 | 固定合成RC、固定负载下的最大 Elmore delay 与缓冲数量 |

所有 family 的 ERM/WDRO、训练/候选评分/测试都使用同一 adapter 的该公式。Hub 的额外 action-dependent bonus 系数事前固定为 0，runner 不把它纳入 cost；这正是冻结协议，不能暗示复现了原文包含 hub bonus 的训练配方。Buffer 的局部插入规则只检查 delay 局部收益、没有在内部优化 0.25B；但是最终返回与训练使用的 cost 包含 0.25B。因此这是求解器启发式的最优性限制，不是 cost 漏项。

根代理 runner 中 `actor_loss=((true.detach()-base.detach())*logp).mean()` 是最小化 cost 的 REINFORCE 符号。Critic 输入只含几何/source 标记，不含被采样动作，拟合同一次真实 sampled cost，属于合法 action-independent baseline；Actor 与 Critic 梯度通过 detach 分开。

**同一物理公式不等于精确同一个 min-max 目标。** 内层用 deterministic greedy action 的 cost 选离散候选，外层在该候选上重新采样动作做 REINFORCE；再加上 MST/HPWL 预筛只保留中心+3扰动，这仍是当前基础 WDRO 的工程近似。不能把它表述为精确求解 stochastic-policy risk 的连续内部最大化。

代码位置：`runner.py:151`（选模）、`:168`（greedy候选）、`:174`（REINFORCE）；`adapter_buffer.py:137`（phase logprob）、`:152`（物理目标）；`vendor_harrl/evaluator.py:105`（局部插入）、`:380`（BFS构树）。

## 2. 初始化与最佳点的实际含义

下表各单元依次是 seeds101、102、103 的最佳 update：

| family | ERM | WDRO |
|---|---|---|
| rest | 0, 0, 256 | 0, 0, 0 |
| timing | 1664, 128, 2048 | 1024, 128, 1920 |
| hub | 128, 0, 128 | 640, 0, 128 |
| buffer | 1664, 1920, 1280 | 2048, 2048, 1536 |

REST WDRO 三个最终选中权重均与 DAC21 Actor 初始化完全相同。ERM seeds101/102 也选择初始化；ERM seed103 在四分布等权测试均值上比初始化高约 0.00041696。于是其与初始化的差别占据 REST 汇总表全部非零 WDRO−ERM 改善，不能称为 WDRO 训练贡献。

Timing 与 Buffer 的两种方法均有训练后选中 checkpoint，保存测试 cost 相对其各自随机初始化总体下降，说明接口可训练。此事实不区分 WDRO 与 ERM 的贡献，也不是单独模型模块的消融证据。Buffer WDRO seeds101/102 最佳在预算终点2048，现有记录不足以宣布优化收敛。

Hub ERM 的全部最佳预测无 hub，WDRO seeds102/103 也如此；两方法在后两个 seed 的预测 cost 一致。关闭 hub bonus 使真实物理目标一致，但没有保留论文对选择更多 hubs 的额外激励。本轮能说明该简化目标下的训练行为，不能概括论文原完整方法的强弱。

## 3. λ 与运输预算：实际诊断

`λ零比例`指本步更新后的值；第一步选择前 λ 固定为0。新任务仅第一次候选选择价格为0，之后 λ 全部正值。REST 的选择前/更新后零比例在此三个完整轨迹上相同。

| family/seed | λ零比例 | λ最终值 | 全程平均c/rho | 末256步平均c/rho | 单批c>rho比例 |
|---|---:|---:|---:|---:|---:|
| rest/101 | 64.79% | 0.000000 | 0.920924 | 0.933195 | 24.37% |
| rest/102 | 71.63% | 0.000000 | 0.904860 | 0.925946 | 20.56% |
| rest/103 | 67.24% | 0.000000 | 0.912216 | 0.937755 | 23.88% |
| timing/101 | 0.00% | 0.088803 | 1.001469 | 0.999734 | 48.78% |
| timing/102 | 0.00% | 0.093552 | 1.001547 | 0.999996 | 49.46% |
| timing/103 | 0.00% | 0.092189 | 1.001525 | 1.000810 | 49.12% |
| hub/101 | 0.00% | 0.091186 | 1.001508 | 0.999825 | 49.37% |
| hub/102 | 0.00% | 0.112227 | 1.001856 | 1.001191 | 50.34% |
| hub/103 | 0.00% | 0.112178 | 1.001855 | 1.000044 | 48.97% |
| buffer/101 | 0.00% | 0.545898 | 1.009028 | 1.002315 | 51.03% |
| buffer/102 | 0.00% | 0.562974 | 1.009310 | 1.003789 | 50.68% |
| buffer/103 | 0.00% | 0.584985 | 1.009674 | 1.004973 | 51.27% |

REST 在 λ=0 的步骤退化为当前预筛集合上的 cost hard mining；不能说每一步都由运输价格抑制扰动。三个新任务的正 λ 与接近预算的 mean transport 表明对偶价格发挥作用，但这不是连续最坏分布被精确求出的证明。

Buffer 全程平均预算超出约0.90%–0.97%，末256步约0.23%–0.50%。对从0开始、全程未触及投影边界的 λ 轨迹，`λ_final = η∑(c_t−rho)`，正的终值本来就对应有限时间的正平均残差。必须结合固定步数和学习轨迹解释，不能将 batch 超标约一半误判为求解错误，也不能从 λ>0 宣称预算已严格满足或鞍点已收敛。

## 4. 缓冲相位、几何构树与最优性边界

末相位 logprob 已加入总策略概率，因此即使真实 evaluator 在 NumPy 中执行、对 buffer count 或相位位置不可微，仍有合法离散 score-function 梯度。新 CPU 探针只证明梯度路径存在；不证明方向有利，不证明 learned phase 优于 fixed/random phase，更不证明收益来自谐波而不是联合拓扑学习。

最终 evaluator 将交叉/重叠段拆开、合成谐波后插入候选，并用上游 BFS 规则投影为 rooted RC tree。局部插入器不是论文所称的完整最优 DP，也不是对任意真实电路网表的精确模拟。相位离散化、固定 RC、固定相同 sink C、BFS 投影均是明确的实现契约。新成果只能称“论文机制启发并修正后的组件 pilot”。

## 5. 能对现有纯线长 WDRO 说什么

- 可以说：Random/Sobol候选、Hungarian运输价格、非负λ更新、真实cost REINFORCE外层可与三类真实求解器组件组合；工程可运行性与有限设置下训练行为已有证据。
- 可以说：在该degree10、2048×64预算、三seed、验证选模规则下，没有建立相对于ERM的稳定WDRO优势。最终统计结论应以另一个独立统计审查为准，而不是仅看某个net条件CI。
- 不可以说：三篇论文已经改善当前 precision_v2 纯线长主线。三个新任务改变了目标、source语义、动作空间和初始化；纯线长REST对照最终又全部保留初始化。当前没有把论文网络接到原纯线长任务、在匹配初始化/预算下隔离测试模型结构改进。
- 不可以把跨family的数值cost排榜、把Timing/Hub几何路径称为Elmore/signoff delay，或将Buffer合成delay直接解释为工业ps/真实功耗降低。
- 测试LP只在每中心指定33候选运输边上精确施加共享平均rho；它没有全部中心到候选并集的跨行边，更没有覆盖连续合法坐标域。MST/HPWL预筛对新目标没有覆盖误差界。
- 24个训练run完成不等于有24个独立WDRO训练seed：每family只有3个ERM/WDRO配对seed。net条件bootstrap不能消除训练方差；相同updates也不代表相同compute，WDRO每步多4次真实greedy评分。

## 6. 对已有报告的评价与建议

`artifacts/report/REPORT.md` 已明确family内部比较、不同预算、三seed与net条件区间、初始化可选、有限LP及非工业/连续证书的边界；没有发现与上述边界直接矛盾的假保证。建议补上REST全部保留初始化、Hub多数无hub策略、λ活跃诊断三项，使读者不会把表中的细小负差或工程PASS读成方法提升。执行验收PASS与“稳定优势尚未建立”可以同时成立。
