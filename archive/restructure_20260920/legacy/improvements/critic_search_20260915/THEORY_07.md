# 候选方向 7：搜索代理刷新与运输乘子更新节奏

## 1. 实验定位与问题

本实验属于 `tmp2` 建议的独立机制研究，degree=30，使用已冻结 ERM 父模型的三个训练种子 101、102、103。每个种子进行四个分支、每分支继续训练 128 次外层更新。实验不会修改、替代已有 v3.1 或 reference_precision_v2 主线，也不会根据本次结果追加训练预算或改变评价范围。

搜索代理预测当前 Actor 的确定性 greedy 相对 regret，而外层 Actor 通过 sampled REINFORCE 优化。Actor 每次更新后，代理拟合的目标函数也发生变化。因此，刷新频率影响代理的时效性；内层攻击选择的平均运输成本影响双变量 λ 的更新，成本噪声可能引起乘子波动。本实验分别操纵这两个因素，检验它们是否改善代理误差、预算动态及最终模型。更低代理误差或更强有限搜索均不能预先推出更好外层模型。

## 2. 同一预算问题及有限内层近似

令归一化 pin 集为 X，合法候选为 Z，Actor greedy 解长为 Lθ(Z)，精确 GeoSteiner 解长为 L*(Z)。真实评分定义为

\[
r_\theta(Z)=L_\theta(Z)/L^*(Z)-1,\qquad
c(X,Z)=\operatorname{Hungarian\!\!\!-L1mean}(X,Z).
\]

对 λ≥0，内层比较目标为 `rθ(Z) − λ c(X,Z)`。每个训练中心固定真实评分预算 4：原点 X 和 3 条 Critic 梯度轨迹终点。轨迹保持旧 calibrated proposer 的高斯初始标准差 0.02、8 步、RMS 归一化梯度步长 0.01、相对中心预归一化位移裁剪 ±0.08，以及既有合法化流程。选择真实惩罚分数最大的候选，平分时保留最早候选。原点参加选择，所以真实惩罚分数不会比原点差。

ρ 和 λ₀ 来自同 seed 的 WDRO 父模型。ρ 保持父模型原已选择 q75 预算，所有分支相同，且每步不变。这个设计改变算法更新节奏，没有改变允许的运输预算问题。4 次评分只是有限候选近似，不提供连续 Wasserstein 最坏情况证书。

## 3. 2×2 因子对照

| 分支 | 代理校准间隔 | λ 更新的成本信号 |
|---|---:|---|
| refresh32_raw | 每 32 步 | 当步实际平均成本 |
| refresh8_raw | 每 8 步 | 当步实际平均成本 |
| refresh32_ema | 每 32 步 | EMA 平滑成本 |
| refresh8_ema | 每 8 步 | EMA 平滑成本 |

相同种子的四分支起始 Actor、Actor baseline Critic、搜索代理、Adam 初始化、学习率、训练中心和每步随机种子全部相同。不同分支因状态分化产生不同选中 net 是算法作用的一部分。新的训练/validation/test 命名空间不使用已有研究测试数据。

主模型与 baseline Critic 使用 fresh Adam，学习率 1e-5，batch=16，梯度裁剪=1。搜索代理从旧 v3.1、同种子的 `experiments/fixed/erm_d30_s{seed}/critic/best.pt` 加载；代理 Adam 也重新初始化，学习率 1e-4。真实 greedy 推理始终固定 padding 到 256 行，因为既有项目发现改变 batch shape 可能改变浮点路径和离散 greedy 输出。小训练 batch 仅限制有梯度的外层激活规模。

## 4. 校准时序与标签时效性

外层第 t 步严格依次执行：

1. 使用 θₜ₋₁、ψₜ₋₁ 和 λₜ₋₁ 搜索 4 个候选并真实评分。
2. 在这 64 个本步已评分候选上，测量当前代理 MSE、MAE、偏差和 Spearman。
3. 若 t 是刷新间隔的倍数，用本步 64 个标签做一次代理 MSE 更新。Actor 此时尚未更新，所以标签与当前 Actor 完全对应。无额外真实评分，不收集历史 regret 缓冲区。
4. 用真实分数选出的 16 个候选进行一次 sampled Actor–Critic 更新。L* 从本步已付费参考结果复用；sampled Actor/REST 调用单独计账。
5. 更新 λ；按固定日程 validation；原子提交完整 checkpoint。

代理校准损失为 `mean((Cψ(Z)-rθₜ₋₁(Z))²)`，保持 regret 的数值尺度。校准使用 eval mode 固定 dropout 和 running statistics，仍保留参数梯度。刷新8比刷新32执行更多代理优化，所以同时报告校准步数与 wall time，不将新增计算隐藏。只复用几何参考 L*，不将旧 Actor 的 regret 当成新 Actor 标签。下一步 Actor 已变化导致的一步陈旧无法由此次刷新消除，正是在线跟踪误差的一部分。

## 5. λ 更新及成本平滑

令当步选中候选的真实平均运输成本为 \(\bar c_t\)。基线使用

\[
\lambda_t=\max(0,\lambda_{t-1}+\eta_\lambda(\bar c_t-\rho)),\qquad\eta_\lambda=1.
\]

平滑分支使用 β=0.9：

\[
m_1=\bar c_1,\quad m_t=0.9m_{t-1}+0.1\bar c_t,\quad
\lambda_t=\max(0,\lambda_{t-1}+\eta_\lambda(m_t-\rho)).
\]

以首个实际成本初始化 m，避免从零启动人为压低早期成本信号。EMA 可能降低高频噪声，也可能滞后于真实预算违约；因此必须同时记录原始成本、EMA 信号、ρ、λ。不能仅因为 λ 曲线更平滑就宣布预算处理更好。报告平均成本减ρ、平均绝对偏差、超过ρ比例、成本标准差、总变差，以及末32步统计。这里ρ是期望运输约束目标，不是每个 net 的硬半径。

## 6. 选模、评价与预先固定分析

每个分布均使用独立 validation 64 个 net，四个分布是 uniform、mix_gauss、edge_bias、grid_cluster。在更新0、32、64、96、128评价；所有种子和分支共享同一组 validation 几何。选择 OOD 三组中最坏组 mean regret 最低的 checkpoint，但要求 uniform mean 不超过初始模型 +0.002。初始模型也参加选模；相同值保留更早模型。

训练和选模完成后，使用另一命名空间下每分布256个全新 test net，对初始、validation选中和第128步三个 checkpoint 进行相同 greedy 评价。第128步结果只作训练动态诊断，不参加选模或改变最终模型。报告每组 mean、P95、maximum，以及 nominal mean 和 OOD worst-group mean。主比较是同种子的分支差值；三个种子只适合机制探索，不能把同一 seed 内数千 net 当成数千独立训练重复。若作置信区间，应明确条件于这些模型、重采样的单位与未覆盖的训练随机性。无本次有限实验可支持普适优势、等价性或全局鲁棒保证。

预先固定判断：刷新频率是否降低当前搜索候选上的代理误差；成本平滑是否改善原始成本预算动态而非只改善乘子曲线；两因素及交互是否改善独立 test。三个种子方向不一致、收益很小或选中更新0，均如实报告，不追加种子/步数挽救效果。

## 7. 预算、资源与产物

每个正式任务：8192 次 greedy 搜索评分（128×16×4），2048 次 sampled Actor/REST 训练调用，1280 次 validation 评分（5×4×64），3072 次初始/选中/第128步 test 评分（3×4×256）。合计12544次 greedy调用，不含中断后可能重做的实际请求；参考 GeoSteiner 可命中缓存，实际 miss 次数独立记录。最差情况下完整重做一次未提交步骤，真实调用计账不会回滚。

每任务 torch线程1，GeoSteiner workers=3，Torch分配器显存比例上限0.10。采用 CUDA_VISIBLE_DEVICES 显式绑定单卡，实际并发和可用显存由主调度器决定。不会终止、暂停或修改其他用户进程。

运行示例：

```bash
CUDA_VISIBLE_DEVICES=3 ADV_V3_REFERENCE_WORKERS=3 \
  /remote-home/Guilin/miniconda3/envs/py311/bin/python \
  /remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915/training.py \
  --seed 101 --variant refresh32_raw --physical-gpu-index 3
```

可选 `--outdir` 必须位于 DRO/Codex；默认输出 `artifacts/training/{variant}_d30_s{seed}`。`--smoke` 独立输出路径，4更新/batch2/validation2/test4，仅验证程序。`--stop-after N` 在已提交步骤后暂停，用同一命令删除此参数恢复。

每步 `latest.pt` 保存 Actor、baseline Critic、搜索代理、全部 Adam、λ、EMA、Python/NumPy/Torch/CUDA RNG、输入身份、历史和最佳状态。每32步另存 checkpoint。搜索标签产物绑定真实 Actor 内容hash；输入/validation/test 分家保存并做精确 net hash 不相交检查。`identity.json` 绑定源码、理论、父模型、旧代理和设置；恢复时不匹配即拒绝混合运行。`task.lock` 防止重复writer；`launch.json` 保存 PID起始ticks；heartbeat 支持恢复检查。

最终 `summary.json`、所有 checkpoint、逐步log、npz结果由 `complete.json` hash绑定。只有完整128步、评价和数据审计结束才能写正式 pass。SIGTERM/SIGINT 在安全边界暂停，实际 Oracle 调用单调保留；完成与资源释放分别由 complete.json、exit.json 和进程/锁检查证明。
