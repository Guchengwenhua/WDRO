# WDRO-REST 项目说明

> 新聊天固定阅读顺序：本文件 → [../../../summaries/history/01_WDRO-REST_项目进度.md](../../../summaries/history/01_WDRO-REST_项目进度.md) → [WDRO-REST_最简实验执行计划.md](WDRO-REST_最简实验执行计划.md) → [../../../summaries/history/02_新对话执行提示词.md](../../../summaries/history/02_新对话执行提示词.md)。

> 文件导航：[项目文件总索引](../../../summaries/history/03_项目文件总索引.md)。原引用的 `summaries/history/02_新对话执行提示词.md` 本次在DRO范围未找到；运行约束可查本文件第9节与执行计划第7节。

## 1. 一句话目标

在统一归一化、同一 REST 模型和同一训练/推理预算下，验证带平均几何运输预算的 WDRO 训练，能否比 ERM、Mixed ERM、Hard Mining 和 GroupDRO 更稳定地降低指定合成分布及 ISPD design-derived nets 上的尾部 relative regret，同时不明显损害 nominal 性能。

项目不承诺对任意未知分布都鲁棒，也不把归一化、新 REST 网络或学习型候选生成器作为贡献。

## 2. 为什么要做

REST 原始训练数据是在单位正方形中独立均匀采样的 pins，但布局产生的真实 nets 可能具有聚集、边界集中和标准单元行结构。普通 ERM 在 uniform 上表现好，不代表在这些几何分布偏移上仍有较低的 P95 或 worst-group regret。

本项目不只是把若干目标分布混进训练集，而是定义一个以 nominal 分布为中心、由几何运输成本限制的 Wasserstein 邻域，并训练 REST 降低该邻域中的最坏期望 regret。

## 3. 方法究竟做什么

### 3.1 统一预处理

每个 net 独立处理：

1. 平移至 bounding-box 原点；
2. 按最长边等比例缩放到 `[0.1, 0.9]`；
3. 保持纵横比并记录逆变换；
4. 用固定规则处理重复 pin、零跨度、最小间距和无效 net；
5. 所有方法调用同一个实现。

归一化是标准预处理，不单独宣称贡献。

### 3.2 优化的损失

对每个 net 使用 GeoSteiner 参考长度：

```text
relative_regret = (L_REST - L_star) / L_star
```

`L_star` 对固定数据集按 hash 缓存。Critic 只做 REINFORCE 的方差降低基线，不能替代 `L_star`。

### 3.3 几何运输成本

对相同 degree 的两个无序 pin 集合，使用 Hungarian matching L1：

```text
c(X,Z) = (1/n) min_sigma sum_i ||x_i - z_sigma(i)||_1
```

它应对 pin 输入顺序不变，同时保留 rectilinear L1 几何。最简消融使用 symmetric Chamfer-L1；两种 cost 各自按扰动 cost quantile 标定 rho。

### 3.4 WDRO 内层搜索

对每个 nominal 中心 net `X_i` 生成 `K` 个合法候选并加入原样本：

```text
Z_i0 = X_i
score_ik = estimated_relative_regret(Z_ik) - lambda * cost(Z_ik, X_i)
Z_i_star = argmax_k score_ik
```

必须为每个中心 net 独立选一个候选，不能为整个 batch 只选一个。候选生成器只是搜索支持点，不是最坏分布 `Q_star`，其采样概率也不是 `Q_star` 的概率质量。

主版本使用 Random/Sobol 的 jitter、sparse、correlated 和 axis-biased 扰动，`K=32`。候选先用便宜 evaluator/surrogate 初筛，top-4 再用真实 Actor、真实 evaluator 和精确/缓存的 `L_star` 打分。

### 3.5 对偶训练

Wasserstein primal 中对最坏分布的优化，通过对偶变成一个非负运输价格 lambda 和逐 net 内层最大化。Actor/Critic 在选中的 `Z_i_star` 上更新，lambda 按平均已选 cost 更新：

```text
lambda <- [lambda + eta_lambda * (mean_selected_cost - rho)]_+
```

若预算活跃，训练后期 `mean_selected_cost` 应接近 `rho`。lambda 恒零、发散或 budget gap 长期失控时，先按实现错误排查，不能解释为鲁棒性证据。

## 4. 最简实验范围

### 4.1 主方法

| 方法 | 作用 |
|---|---|
| ERM/REST | 统一预处理后的 nominal 标准基线 |
| Mixed ERM | 排除收益仅来自看过更多目标分布的解释 |
| Hard Mining | 排除 WDRO 只是无约束困难样本挖掘的解释 |
| GroupDRO | 比较已知组重加权是否已经足够 |
| WDRO-REST | Random/Sobol、Hungarian-L1、learned lambda 的主方法 |

KL-DRO 和 CVaR 不做；论文中相应比较和优越性表述必须删除。

### 4.2 数据与预算

- degree：`{10, 20, 30, 40, 50}`；
- nominal：uniform；
- 三个目标分布：mix_gauss、edge_bias、grid_cluster；
- 每个 `degree × distribution` 测试格固定至少 10,000 nets；
- 训练 seeds：`{101, 102, 103}`；validation seed：`202`；test seed：`303`；
- 所有方法从相同 degree 对应的 DAC21 Actor checkpoint 开始；
- 统一 2,000 policy updates、batch size 256；
- 推理统一 `T=1`、deterministic greedy、相同测试 batch 和硬件。

### 4.3 ISPD 协议

采用协议 A：所有模型只用合成数据训练，所有超参数只用合成 validation 选择，ISPD 仅在配置冻结后作模型的零样本外部测试。

现有仓库已经生成过全量 merged REST 结果，因此可以说“模型零样本迁移”，不能说研究者完全未见的 blind test。ISPD 数据必须保留 design/net ID，排除 `_metal5` 重复版本，并报告：

- net-weighted mean；
- design-macro mean；
- P95；
- worst-design mean；
- 按 degree 分层的结果；
- 每个 design/degree 的样本数和无效 net 数。

## 5. 必须完成的验证

1. 数据、seed 和 test/validation 不泄漏；
2. normalization/inverse 和 Hungarian cost 单元测试；
3. 五种主方法、五个 degrees、三个训练 seeds 全部完成；
4. rho=`0/selected/larger`、fixed/learned lambda、`K={1,8,32}`；
5. Random-64 与 CEM-64 等 evaluator-call budget 比较；
6. Hungarian/Chamfer 分别按 cost quantile 标定 rho；
7. 固定小候选集上的 primal-dual LP gap 审计；
8. 所有方法接受同一个独立强攻击，例如 CEM-256；
9. 记录 lambda、mean cost、rho、budget gap、合法率、calls、训练和推理时间；
10. 结果不好也不得删除不利 degree、分布或 seed。

## 6. 论文可以怎样下结论

只有同时满足以下条件，才能支持 WDRO 的主要性能主张：

- 相对 ERM、Mixed ERM 和 Hard Mining，目标分布 worst-group 或 P95 稳定改善；
- 多个训练 seed 改善方向一致，置信区间支持结论；
- nominal mean regret 恶化不超过预先固定的 0.2 个百分点；
- 独立攻击下优势仍存在；
- lambda 与平均运输成本符合预算解释；
- ISPD 结果不与合成结果严重矛盾，或明确限制结论范围。

若只改善尾部而不改善 mean，只声称尾部鲁棒性。若 Mixed ERM/GroupDRO 相当，只强调显式几何预算和诊断能力。若无稳定改善，应如实报告，不能通过 test 调参或删结果修正。

## 7. PDF 草稿与最简执行版的关系

`output/pdf/WDRO-REST_ICCAD2026_anonymous_draft.pdf` 描述了完整研究设想，但当前实验执行以最简流程为准。PDF 仍包含 canonicalized ERM 单独基线、KL-DRO/CVaR、五类 shift、强搜索主模型、GW/FGW、多档 K、log-sum-exp/raw-loss 消融和 design-level blind split；这些超出最简范围，填结果前必须从论文中同步删除或改写。

该 PDF 是 ICCAD 2026 模板。若目标是 ICLR 9 月 25 日截稿，除实验外还必须完成模板迁移和论文结构调整。

## 8. 四个核心文件各自负责什么

- 本文件：稳定的项目目标、方法、范围和判定标准；除研究协议变化外不频繁修改。
- `summaries/history/01_WDRO-REST_项目进度.md`：唯一的动态状态源；每次实质工作后更新完成项、证据、阻塞项、下一步和日志。
- `wdro_legacy_v1/docs/history/WDRO-REST_最简实验执行计划.md`：阶段、排期、工期估算和止损规则。
- `summaries/history/02_新对话执行提示词.md`：新对话的可复用启动提示词，统一约束产物路径、动态 GPU/磁盘预检、多卡利用、持久任务和分步恢复。

## 9. 运行与存储约束

- 所有新产生的项目文件和数据必须通过 `/remote-home/Guilin/DRO/Codex` 下的逻辑路径存储和访问。大文件允许使用 Codex 内的符号链接落到附加盘，但必须记录 resolved path。
- 每次启动、恢复或扩展批量任务前，重新检查 GPU 进程/显存/利用率和目标文件系统空间。GPU 上已有进程不构成自动排除；只要剩余显存能容纳实测任务峰值和安全余量、计算资源未持续饱和，就应尽量纳入调度。
- 长任务必须可恢复、可追踪，并优先在服务器端 `tmux` 会话中脱离本地 VS Code/网络连接运行。详细规则以新对话提示词文件为准。
