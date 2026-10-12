# 独立方法与依赖验收：RL-only degree30

审查快照：2026-09-16 04:05:09 UTC。机器可读证据：[method_review.json](method_review.json)。本审查只读取源码、协议和已有产物，并在 CPU 上重建坐标用于查重；没有构造 Actor、执行推理/训练、调用求解器，或修改冻结文件。

## 1. 结论

**已检查的学习目标、基线、搜索反馈、依赖链和协议绑定符合本次 RL-only 定义；未发现需要重跑已完成任务的科学实现错误。** 这里的 RL-only 指任务反馈只来自自身动作长度，不使用外部参考解/最优值/几何界。Random、ES、梯度提案仍然只是明确命名的对照，不能把所有六分支都称为双策略强化学习。

**整批验收在本快照仍不成立：18 项中只有 17 项具有完成产物。** `wdro_es_s103` 只有 1,530 条更新记录，目标为 2,048，尚无 `complete.json`。它的已有源码/协议/初始化绑定通过，但不能用其他两个 ES seed 补齐，也不能把现有统计说成预定三 seed 全集。后续完成状态由运行验收更新，本快照不替代该确认。

## 2. 本次独立核验得到的证据

| 检查 | 结果 | 范围 |
|---|---|---|
| 18 项 run 绑定到当前冻结协议 | PASS | 协议 SHA256、每个 effective protocol 字段 |
| 训练/环境/搜索源码与模型依赖哈希 | PASS | 18 项保存绑定均与当前文件一致 |
| 相同 seed 初始 Actor 一致 | PASS | 每个 seed 的六方法只有一个初始权重 hash |
| 已完成模型的 checkpoint 选择 | PASS | 17 项均等于 validation worst-group mean 最低的最早更新 |
| λ 更新递推 | PASS | 18 项已有更新全重算，最大数值差为 0 |
| nominal 训练中心与 held-out 重合 | PASS | 393,216 个训练中心与 5,632 个验证/测试/攻击中心无重合 |
| 同一数据侧重复 | PASS | 训练内部及 held-out 内部均无重复 pin 集合 |
| 训练 RNG seed32 碰撞 | PASS | 6,144 个中心批次的初始派生 seed 无碰撞 |

坐标查重不是只比较 namespace 字符串：按冻结生成器重建全部三个 seed、全部 2,048 更新的 nominal 中心；每个 net 按 x、y 字典序排列 pins，再对精确 float32 坐标做 SHA256。因此也能发现仅 pin 顺序不同的同一集合。held-out 包括 validation 1,024、test 4,096、attack 中心 512。

本次协议 SHA256 为：`808b800d15718e9fed7e37292ea0978ab6e2a941674dbd8eebd5ae6e03971d97`。

## 3. “不依赖参考算法”的证据及其边界

### 3.1 调用链检查通过

- `geometry.py:57` 的生成器只生成固定窗口坐标；没有旧数据文件或参考标签读取，没有逐 net bbox 归一化。
- `geometry.py:95` 的成本是直接对应 pin 的平均 L1，没有 assignment、Hungarian 或 MST。
- `geometry.py:141` 测量已给定 topology 的 span 长度，并检查其连接结构；它不构造较优树，也不提供最优值尺度。
- `geometry.py:172` 只导入旧 REST 的 **Actor 架构源码**；检查其三份 `models/*.py`，依赖为 Torch/NumPy/标准库，未发现求解器或权重读取。
- `train.py:127–137` 设置随机种子后直接构造新 Actor、baseline、surrogate、proposal；未加载旧分支教师。
- 训练代码只有两个 `torch.load` 调用：当前 run 的 `latest.pt` 恢复，以及当前 run 的 `best.pt` 最终评价。恢复要求 method、seed、源码、依赖和协议 binding 完全一致。
- `train.py:533` 在模型构造前安装 guard；17 项完成产物的 forbidden module 检查为空。

这些证据支持“本次经过检查和绑定的流程没有使用禁止的参考算法/标签/旧权重”。合法性检查、坐标裁剪、长度测量和神经网络训练自身的数值计算是必要环境/学习操作，不是额外参考求解器。

### 3.2 不能把零计数字段当独立证据

`train.py:160–163` 把 `oracle_calls` 初始化为 0，但代码没有通过该字段拦截并计数未知外部调用。因此该字段本身只是声明，不是独立的审计仪表。guard 也主要阻止已知模块、库名和路径，最终模块列表不是全过程文件访问日志，且未保存操作系统层的完整动态库/系统调用轨迹。

**验收依据应是绑定源码的实际数据流、全部受控 imports/loads、guard 与完成产物共同支持的结论，不能宣传为任意隐藏依赖都不可绕过的安全证明。** 本次没有发现隐藏依赖迹象；上述是审计证据的强度边界。

## 4. 学习数学与冻结协议一致

### 4.1 Actor 与训练基线

`train.py:349–369` 的 target 直接来自 sampled topology 的实测长度。Actor loss 使用 `(own_length−prediction).detach()×log_prob`；学习基线的 MSE target 是同一自身长度，且本次 prediction 在利用该动作更新基线之前取得。未把 Critic 预测替代实际 reward，也未拟合 L★。

`erm_rollout` 在 `train.py:337–342` 先用当前 Actor 的 eval/greedy 解码取得停止梯度的成本，再采样 Actor 动作；该 baseline 不依赖本次 sampled action。它不训练 value baseline，并单独累计 rollout 前向量。这一实现符合 self-critical 控制变量的要求。

Actor 的随机动作 seed 每步独立设置，攻击器的采样不会改变下游 Actor 的随机流。共同随机数不意味着不同状态/参数下动作必然相同，它提供配对控制。

### 4.2 对抗搜索

- 搜索代理拟合本批全部候选的 **greedy 实测长度**；训练 baseline 拟合 sampled 长度，两者没有混同。
- 四个 WDRO 都更新相同规模的 surrogate；只有 Critic 分支用它提案。其他三分支因此带有相同的额外小网络开销。
- `search.py:139–155` 的 RL 对手用裁剪前高斯样本计算 log probability；reward 为实际 `length−λ×indexed_cost`；leave-one-out baseline 排除自身样本，停止梯度正确。
- 价格使用本批旧 λ 评分，之后按 `max(0,λ+10×(mean_cost−0.04))` 更新；已有日志全部递推一致。
- 每种训练搜索实际 K=8，包含原中心；最终统一按实测成本选择。ES 使用中心、三对探测和一次移动；Critic 使用七个终点与中心。

没有发现 reward 符号、Actor/对手更新方向、基线动作依赖或 score/transport 不一致的问题。

## 5. 必须保留的科学限制

### 5.1 实际优化的不是旧 regret 问题

内层 greedy 线长与外层 sampled 线长不同；既有 BatchNorm/Dropout 在 train/eval 模式也不同。因此不能把这套更新称为连续 sampled WDRO 目标的精确梯度算法。raw 线长还会将输入本身的几何难度计入攻击收益。

indexed L1 是有对应 pin 元组上的距离，不是无序集合最小匹配距离。当前 λ/ρ 和性能不能与旧 Hungarian/regret 方案视为同一个问题进行因果比较。

### 5.2 压力测试没有实施 ρ 约束

`train.py:403` 的 held-out 压力测试使用 Random32、λ=0，选择窗口内自身线长最大的候选；它不筛选 `cost≤0.04`，也没有求满足均值预算的混合分布。这里验证的是预先规定位移窗口内的有限随机压力测试，不能宣称测得 ρ=0.04 Wasserstein 球内最坏风险。

它也没有单独比较冻结父模型下 Random/ES/Critic/RL 的攻击强弱，不能将主 Actor 的训练结果直接解释成某个搜索器具有更好的方向。

### 5.3 训练选择过程不能完全离线重放

训练只保存每步平均 sampled 长度、baseline MSE、surrogate MSE、选中/原中心平均 greedy 长度、平均位移和 λ；没有持久保存全部训练候选坐标、逐候选分数及选中 topology。

因此最终测试/攻击可以从保存的 topology 完整重算，λ 递推和训练预算也可重算；但不能仅凭当前产物逐步重放完整计划的 12,582,912 次候选评分及相应选优。模型相关的训练扰动也无法全部做事后坐标查重。已核验的来源隔离与 nominal 中心无重合不是对这些未保存数组的逐字节证明。

本轮还未保存 advantage 方差、独立代理排名相关或匹配随机方向消融。baseline MSE 不等于已证明降方差，surrogate MSE 不等于已证明方向有效。方案文字中的这些机制诊断不能在本次验收中标记已完成。

### 5.4 测试数据没有驱动代码内选模，但并非从未访问

`train.py:488` 在任何训练更新前先评测初始随机 Actor 的 test，用于 before/after 对照；源码只用 validation 选择 best，并未使用 test 指标更新参数或选择 checkpoint。本次可称 **test 未用于代码内选模**，不应称 test 在最后评价前从未被读取/打开。

`mix_gauss` 的六个 cluster centers 在每个 split 的单次 `make_data` 调用中只随机生成一次，所以该 split 的所有 nets 共享一个固定布局。对 net/seed 的 bootstrap 是此布局条件下的不确定性，不涵盖跨 cluster-layout 的变化。三种训练 seed 也不是很多独立重复；未校正的多个 CI 不能作为普遍胜出的证明。

### 5.5 长度是历史 evaluator 记账值

同 x/y pins 的不同 spans 可能重叠而重复记账。验收确认的是这个冻结 evaluator 的值和合法连接结构，不是每个结果都已去重、去环后的物理线段并集长度。未计算最优参考，不能从本次产物给出精确 regret、最优率、近似比或连续鲁棒性证书。

## 6. 如何汇总本次验收

可以接受的方法层结论是：**在冻结 raw 线长、有对应 pin 的传输成本和从零训练设置下，已检查的实现符合无外部参考的 RL 反馈设计。** 是否有效学习、是否改善偏移表现、是否优于 Random-WDRO，必须使用独立统计结果分别判断，不由本方法审查代替。

本快照仍需 `wdro_es_s103` 完成后，核验其剩余更新、最终数据/拓扑、完整三个 seed 统计、退出与锁释放，才能形成整批 18 项验收结论。若用户本轮只需已有结果的阶段验收，应明确标注 17/18 与缺失单元，不能偷偷减少既定样本数。
