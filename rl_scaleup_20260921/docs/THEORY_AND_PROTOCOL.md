# RL 对抗搜索扩大训练：理论与冻结实验方案

日期：2026-09-21。状态：启动前方案；效果尚待完整实验与审计。
本次全部新增文件位于 `/remote-home/Guilin/DRO/Codex/rl_scaleup_20260921`。
机器可执行参数以同目录上级的 `protocol.json` 为准；正式启动时冻结 SHA-256。

## 1. 已有证据与选择理由

已阅读 `summaries/强化学习对抗样本搜索方法与流程_20260921.md`，核对 `rl_research_20260915` 的协议、策略、训练、调度及独立验收表。
旧实验为 degree 30、3 seeds、batch 32，warmup / 固定攻击器 / 共同训练各 256 updates，K=8。

| 共同训练方法 | 四分布常规平均线长 | Random32 | ES32 |
|---|---:|---:|---:|
| Random | 4.089438 | 4.817139 | 4.857655 |
| Sequential PPO | 4.029407 | 4.834345 | 4.874315 |
| Best-of-K PPO | 4.092096 | 4.806512 | 4.848241 |

来源：旧分支 `artifacts/review_20260919/coev_summary.csv`。常规四分布等权，不是仅 uniform。
Sequential PPO 常规线长较 Random 低 1.468%；独立攻击下旧均值反而略高，未建立稳定优势。
Best-of-K PPO 的两项独立攻击均值略好，但旧样本不足以认定优胜。
固定攻击中 λ=5 的 gain：Best-of-K 1.048442、Sequential 1.030966、REINFORCE 1.025723；相关旧区间跨零。
因此选择这两种候选继续研究，保留 REINFORCE 为主动对照，不把旧探索性赢家当作已确立结论。

## 2. 目标、几何与 WDRO 联系

中心 X=(x_1,...,x_n) 属于 [0.1,0.9]^(n×2)，n=30。保持 pin 身份，运输成本

    c(X,Z) = (1/n) Σ_i ||x_i-z_i||_1.

每坐标相对原中心最大位移 ε=0.08；窗口裁剪后检查最小 L1 pin 间距 1e-6，非法提案回退到原中心并照常计入评分预算。
不做 Hungarian，不重新按单个 net 跨度归一化；这是一种有标号坐标成本。
qθ(Z)=C(Z,greedy(πθ,Z)) 是 REST 在该输入上自己生成拓扑的线长；使用原 evaluator 的 terminal-span 口径。
它不是最优性误差：输入自身变长与求解质量变差可能共同导致 q 增大。

固定 θ 和 λ 时，攻击器最大化

    Fθ,λ(X,Z) = qθ(Z) − λ c(X,Z),
    G_K(X) = max_{j=0,...,K−1} Fθ,λ(X,Z_j) − qθ(X),  Z_0=X.

因此 G_K≥0；K=8 含原中心，每张 net 恰好 8 个真实评分请求，重复/回退候选也计数。
λρ 对同一中心的择优是常数，故无需加进 G。

Wasserstein 对偶的形式为 inf_{λ≥0}{λρ+E_X sup_Z[qθ(Z)−λc(X,Z)]}。
本实现只在局部合法集合内搜索有限候选，得到内层 sup 的下界；对这个下界再做 λ 优化不会自动成为真实鲁棒风险的认证上界。
此外攻击用 greedy 树、REST 更新用 sampled 树，不是精确同一光滑鞍点问题。
对偶背景见 Gao–Kleywegt 原文：https://arxiv.org/abs/1604.02199 。本方案不声称已满足其理论定理的全部假设。

## 3. Best-of-K PPO

给定 X、λ，策略独立采样 7 个高斯潜在动作，再确定性裁剪和合法化得到候选。策略概率始终计算在采样的潜在动作上，不能把裁剪后的坐标当作原高斯动作。
设 F_j 为真实分数，M=max(F_0,...,F_7)，M_{−j}=max_{i≠j}F_i。第 j 个提案使用

    A_j^replace = M − M_{−j}.

在条件独立采样、固定 θ/λ、原始 score-function 估计条件下，M_{−j} 不依赖动作 j，可作为不改变该期望梯度的 baseline。
这把学习信号集中到改变整组最大值的提案。若多个提案并列最高，单提案替换优势可能为零，需要监测非零优势比例，不能只看 loss 下降。
PPO 采用 joint action 概率比 r=exp(logπ_new(a|X)−logπ_old(a|X))，最大化

    E[min(r A, clip(r,0.8,1.2) A)].

一批采样复用 4 epochs，lr=3e-4，width=64，初始 σ=.04；继承已核查实现，不更换网络或添加 surrogate scorer。
PPO clipping 是更新近似，上述 score-function 恒等式不等于 clipped PPO 的无偏性或单调提升保证。
PPO 原文：https://arxiv.org/abs/1707.06347 。

## 4. Sequential PPO

状态包含原中心、当前相对位移、λ、步进度、前一步 F−F_0 和历史最优 F−F_0。
从当前点继续提出增量动作，相对原中心执行总位移约束；不强制每步回到历史最佳点。
令 M_t=max(F_0,...,F_t)，奖励 r_t=M_t−M_{t−1}，t=1,...,7，则

    Σ_t r_t = M_7 − F_0 = G_8.

γ=1，按后续增量累计回报 G_t=Σ_{u=t}^7 r_u 更新 PPO，并回归状态 value。整段轨迹完成后才更新，最后按所有候选真实 F 择优。
这个 telescoping 恒等式保证轨迹回报与最终搜索目标一致，不保证有限训练 PPO 找到最佳策略。
主要风险：高维 joint ratio clipping 饱和、前期探索不足、序贯 CPU/GPU 同步成本、攻击器对特定 θ 的过拟合。
保留 clip fraction、KL 近似、策略方差、增量回报、telescoping error、真实调用和 padding 行数诊断。

## 5. 共同训练及预算跟踪

同一 seed 的所有方法复用本次新生成的 ERM warmup Actor、value、Adam 状态。
每轮中心数据和 REST 动作随机种子在方法间配对；攻击器固定阶段学习后继承到共同训练。

1. 冻结本轮 θ、λ，采集 K=8 候选及真实评分。
2. 学习型攻击器用本轮采样更新；所有方法从已评分集合择优 Z*。
3. REST 在 Z* 上重新采样拓扑 Y，最小化 E[stopgrad(C(Z*,Y)−V(Z*)) logπθ(Y|Z*)]，value 回归自身 sampled 线长。
4. λ←max(0,λ+10(mean c−.04))；初始 λ=5。
5. 写进度、调用日志、心跳，按固定频率原子保存模型、优化器和 RNG。

ERM 继续在 nominal 上训练；Random/ES 不训练攻击器。因此匹配的是 REST 更新数与攻击阶段每轮真实候选数，不是相同总训练 FLOPs。
REINFORCE 与两种 PPO 具有相同固定预训练真实评分预算，但 PPO 优化器 epoch 和计算时间仍另报。
旧实验 24/27 个对抗共同训练 run 的全程平均 cost 超过 .04；本次必须报告全程、末 64、末 256 updates 的 cost−ρ、超额比例和 λ，不以公式正确代替预算可行。

## 6. 冻结矩阵与规模

| 项目 | 旧配置 | 本次 |
|---|---|---|
| degree | 30 | 30 |
| seeds | 201–203 | 301–308，8 个新 seed |
| batch | 32 | 64 |
| warmup / fixed / coev updates | 各 256 | 各 2,048 |
| 每阶段中心数/seed | 8,192 | 131,072（16 倍） |
| 搜索 K | 8 | 8 |
| fixed 测试中心/分布 | 128 | 512 |
| 常规测试/分布 | 1,024 | 4,096 |
| 独立攻击中心/分布 | 128 | 512 |
| 独立攻击 | Random32、ES32 | 相同算法与预算，λ=0 |

固定攻击器：REINFORCE、Best-of-K PPO、Sequential PPO；共同训练：ERM、Random、ES、REINFORCE、Best-of-K PPO、Sequential PPO。
总任务 8×(1+3+6)=80：8 warmup、24 fixed、48 coev。每个 seed 是一次独立训练重复，不能把测试 net 数充当训练 seed 数。
每 fixed 或非 ERM coev run：2,048×64×8=1,048,576 个训练 greedy 评分；warmup/coev 每 run 有 131,072 个 sampled REST 评分。
总训练 greedy 评分 67,108,864；sampled REST 评分 7,340,032；最终测试 greedy 评分 10,616,832，不含 smoke 或重试额外开销。
batch=64 与 greedy microbatch=64 对齐，使训练中 Random、ES、单步和序贯方法都无需 padding，分别真实计时仍然必要。
训练输入采用新 namespace `rl-scaleup-20260921-v1`，与旧研究分离；fixed/coev 测试各自独立，所有方法和 seed 的测试中心与顺序相同。
只用最后 update 的模型；不看 test 选 checkpoint、调 λ、改奖励或追加有利 seed。

## 7. 假设、统计与判定

预注册 8 个有方向主比较，合在同一 Holm family，familywise α=.05：

* 两个 fixed 比较：每种 PPO 在 λ=5 的四分布等权 gain 减 REINFORCE gain。
* 六个 coev 比较：每种 PPO 相对 Random，在常规、Random32、ES32 平均线长上的下降。

每个比较先在相同 seed、相同中心配对，对四分布等权平均，得到 8 个 seed 层差值 d_s；统一正值表示改善。
预先指定单侧改进假设，枚举 2^8=256 个 seed 符号翻转，p 为随机符号均值≥观测均值的比例；其推断依赖零假设下差值符号可交换/对称的假设。
最小 p=1/256；八个训练 seed 仍有限，不承诺检验功效。Holm 校正后 p≤.05，至少 6/8 seed 改善，并满足最小效应，才标记该项 `SUPPORTED`。
fixed 最小绝对 gain 差 .01；coev 最小相对线长改善 .5%，分母为同一新协议 Random 均值。
“独立攻击下稳定改善”要求 Random32 和 ES32 两项均达标；常规达标而独立攻击未达标，只报告常规收益。
另提供 2,000 次 crossed seed/common-center bootstrap 的双侧 95% 描述区间；所有 seed 使用同组抽样中心，不把 seed×net 当独立训练重复。
其他 λ=1/10、ES/ERM/REINFORCE 对照、分布分项、P95、worst-group 和训练曲线完整保留，标为次要/探索性。不使用未校正区间作主结论。
对预算超额只作测量，不因某项显著就声称满足连续 Wasserstein 约束。

## 8. 执行、资源与失败处理

优先使用实测未饱和的 8 张 GPU，每卡最多本实验 1 worker，每 worker 1 BLAS/PyTorch CPU 线程，nice=10。CPU 用于数据、几何评分、统计和审计，不为占满核数而启动无用进程。
每次启动记录最新显存、GPU 进程、利用率、主机负载、内存、磁盘、物理 GPU index/UUID、命令及 PID/start_ticks/boot ID。
初始准入至少 8 GiB 空闲显存，GPU 利用率滚动平均≤80%，进程显存分配上限 12%（约 5.76 GiB），另留 CUDA context/安全余量；持续饱和时等资源。
不终止、降优先级或驱逐其他用户进程。训练中资源收紧时本实验自己保存状态并退让，队列等待准入恢复。
初始剩余磁盘约 92 GB；保留至少 25 GiB，预计结果与模型 <15 GiB。每 run 只保留 latest、日志和最终结果，不逐步保留大 checkpoint。
每 64 updates 原子 checkpoint；中断最多重做未提交的 63 updates，额外调用通过原始 ledger 单独计入。每任务最多 3 次失败尝试；耗尽则明确失败，不无限占资源。
tmux 启动调度器、ETA、终态审计；无需对话持续监控。可创建 `artifacts/operations/PAUSE` 请求本实验在步边界保存并退出 worker，删除后由队列自动恢复。
ETA 由实测训练耗时与完整规模 pilot 的每阶段速率更新；包含剩余任务、依赖和评估余量，不能保证未来共享资源等待时间。

## 9. 交付与验收

启动门：策略/目标恒等式/调度单元检查通过，连续运行与中断恢复一致，生产 batch 的 GPU pilot 实测通过，再冻结源码协议与启动清单。
终态门：80 个 GPU 任务均完成且 worker 退出；manifest/hash/父 checkpoint 绑定正确；正式预算一致；原始调用账本解释重试；训练/测试无重叠；合法性、线长、gain、cost 和 λ 递推独立重算；八项主比较与完整次要表输出；锁和进程生命周期验收通过。
自动交付 `artifacts/report/REPORT.md`、`metrics.csv`、`comparisons.csv`、`primary_comparisons.csv`、`training_budget.csv`、`runtime_and_calls.csv`、`confirmatory.json`，以及 `artifacts/operations/delivery.json`。
`COMPLETE` 表示协议工程与数值验收完成，`SUPPORTED` 表示单个预注册改善达到阈值；没有改善也可以完成有效实验。smoke/pilot 不是正式结论。
本次不证明 degree 泛化、真实芯片设计收益、相对最优解 regret、连续最坏分布认证或相同总算力下优势。

## 10. 原始依据

* 本地：`../rl_research_20260915/artifacts/review_20260919/REPORT.md` 与 fixed/coev 汇总、`runner.py`、`policies.py`。
* Schulman et al., Proximal Policy Optimization Algorithms：https://arxiv.org/abs/1707.06347 。
* Gao & Kleywegt, Distributionally Robust Stochastic Optimization with Wasserstein Distance：https://arxiv.org/abs/1604.02199 。
* 本文件的 Best-of-K / telescoping 推导针对本仓库具体实现，不是对上述论文实验结论的转述。
