# WDRO-REST 最简实验执行计划（面向 9 月 25 日截稿）

> 新聊天固定阅读顺序：[00_WDRO-REST_项目说明.md](00_WDRO-REST_项目说明.md) → [01_WDRO-REST_项目进度.md](01_WDRO-REST_项目进度.md) → 本文件 → [02_新对话执行提示词.md](02_新对话执行提示词.md)。

> 2026-09-10整理说明：阶段0–8已完成，当前数值与模型使用reference_precision_v2；阶段9未启动。本文件排期、初始代码盘点与硬件数字保留为2026-09-04计划快照，不是当前待办。当前状态见项目进度，文件入口见[总索引](03_项目文件总索引.md)。梯度搜索见[独立研究评估](04_梯度对抗样本_可行性与数学流程.md)，尚未纳入原冻结协议。

## 1. 结论与边界

本计划已交叉阅读 `WDRO-REST_最简实验流程.md` 和 `output/pdf/WDRO-REST_ICCAD2026_anonymous_draft.pdf`，实验取舍以最简流程为最高优先级。目标是在不删减硬性要求的前提下，完成一套可复现、可写入论文主表的最小实验。这里的“最低要求”指实验协议完整，不代表保证论文录用。

目标 PDF 与 `paper_iccad_wdro_rest/build/main.pdf` 的 SHA-256 完全一致（`c724df...d97da`），并由仓库中的 `paper_iccad_wdro_rest/main.tex` 生成，因此已通过对应源码逐项核对。论文草稿目前是 ICCAD 2026 格式；若最终目标确为 ICLR 9 月 25 日截稿，模板迁移和篇幅重组是实验之外的额外任务，最晚应在 9 月 21 日开始。

2026-09-04计划制定时的代码盘点：当时尚未实现 WDRO；`DRO/WDRO-REST/train.py` 仍是原版 REST 训练代码。现有资产可以复用的是 REST Actor/Critic、degree 5–50 的 DAC21 checkpoint、GeoSteiner、三个目标分布生成器及 ISPD 解析代码。WDRO、Mixed ERM、Hard Mining、GroupDRO、matching cost、候选搜索和统计汇总都需要补齐。

为赶截稿，正式范围固定如下：

- 方法：ERM/REST、Mixed ERM、Hard Mining、GroupDRO、WDRO-REST，共 5 个；
- degree：`{10, 20, 30, 40, 50}`；
- 测试分布：uniform、mix_gauss、edge_bias、grid_cluster；
- 每个 `degree × distribution` 固定 10,000 个测试 net；
- 3 个独立训练 seed，不做 5 seeds；
- 所有方法从相同 DAC21 Actor checkpoint 开始，统一微调 2,000 policy updates、batch size 256；
- 推理统一 `T=1`、deterministic greedy、相同 batch、相同预处理；
- 主 WDRO 固定为 Random/Sobol、`K=32`、Hungarian-L1、learned lambda；
- 不做 KL-DRO、CVaR、GW/FGW、额外两个合成分布、T=8 和 FLUTE 主对比。

PDF 草稿必须与上述缩减同步，否则会出现“正文承诺了但实验没做”的问题：

- 删除 abstract 和正文中“优于 canonicalization、CVaR/KL-DRO”的未验证表述；归一化只作为五种方法共享的预处理；
- controlled shifts 从五类改成三类，删除 anisotropic 和 void 的必做表述；
- 主表从当前 8 行缩成上述 5 种方法，CEM 只放等预算消融/独立攻击；
- 消融从当前的 K 五档、GW/FGW、max/log-sum-exp、raw length 等缩成最简流程要求的 rho、lambda、`K={1,8,32}`、Random/CEM、Hungarian/Chamfer；
- 将当前设计级 train/validation/blind 协议 B 改成协议 A：所有模型和超参数只用合成数据，ISPD 只作外部测试；由于旧 merged 结果已存在，不使用 “researcher-blind” 一词；
- 删除或改写所有超出最终统计证据的 TBD 句子，结果不显著时明确写不显著。

## 2. 服务器盘点与时间估算依据

盘点时间：2026-09-04（UTC）。

| 资源          | 当前状态                                                                | 对实验的影响                                                            |
| ------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| GPU           | 8×NVIDIA RTX 6000 Ada，每卡 49,140 MiB                                 | GPU 0–3 空闲；GPU 4–7 正被其他任务占用。按 4 卡并行排期               |
| CPU           | 2×Xeon Gold 6326，32 物理核/64 线程；容器可见 63 线程                  | 足够并行做 GeoSteiner、数据生成和统计                                   |
| 内存          | 1.0 TiB，总可用约 832 GiB                                               | 不是瓶颈                                                                |
| 磁盘          | 截至 2026-09-04 07:23 UTC 仅余约 24 GiB，使用率接近 100%                | 当前首要风险；正式运行前至少腾出 100 GiB；动态值以进度文件为准           |
| 环境          | `py311`：Python 3.11、PyTorch 2.6.0+cu124、CUDA 可见 8 卡、SciPy 可用 | 可直接开发；系统默认 Python 没有 PyTorch，运行脚本必须显式使用`py311` |
| 调度/版本工具 | 未发现 Slurm、`git` 命令                                              | 采用单机 GPU 锁文件/任务清单；配置和代码快照用 manifest/hash 留档       |

实测/历史测速：

- degree=30，1,000 个 mix_gauss net，`T=1`：REST 推理 0.61 秒，GeoSteiner 5.70 秒；
- degree=30，ERM 100 updates、batch 256：约 21.8 秒（含启动和一次小验证）；据此 2,000 updates 的普通方法约 7–10 分钟；
- 历史日志中，每 10,000 个 net 的 GeoSteiner 平均约为：degree 10/20/30/40/50 = 4.4/17.6/42.9/78.7/127.5 秒；
- WDRO/Hard Mining 还要搜索候选，按普通训练的 4–10 倍并预留 2 倍工程缓冲估算；
- 下述“总工期”包含编码、测试、返工和机器运行，且假设 GPU 0–3 持续可用。

预计核心实验在 9 月 20 日前结束，9 月 21–24 日只做补跑、统计和论文同步。若只能使用 1 张 GPU，正式训练阶段预计额外增加 1–2 天，但开发关键路径基本不变。

## 3. 固定实验协议

### 3.1 数据与参数

| 数据            | 最简固定设置                                |
| --------------- | ------------------------------------------- |
| nominal         | uniform                                     |
| mix_gauss       | `nonuniform=0.8, clusters=6, sigma=0.04`  |
| edge_bias       | `nonuniform=0.8, clusters=4, sigma=0.06`  |
| grid_cluster    | `nonuniform=0.8, clusters=8, sigma=0.03`  |
| 训练 seed       | `{101, 102, 103}`，同时对应 3 个模型 seed |
| validation seed | `202`，每格 2,000 nets                    |
| test seed       | `303`，每格 10,000 nets                   |

所有数据按 net 做同一套 bbox 原点平移和最长边等比例缩放到 `[0.1, 0.9]`，保留纵横比和逆变换参数。重复 pin、零跨度、最小间距和无效 net 的规则写进一个共享模块，训练与测试不得各写一份。

训练使用固定、带 ID 的数据池和确定性候选 seed；这样候选与 `L_star` 才能缓存，且 `R=10` 的刷新含义明确。测试文件一经生成只读，五种方法共享同一文件。

### 3.2 训练、公平性与选择规则

- 每个 degree 使用对应的同一 DAC21 Actor checkpoint；五种方法各自重新初始化/恢复相同 Critic 状态，并使用相同 2,000 updates；
- ERM 只用 nominal；Mixed ERM 均匀混合 nominal 和三个目标分布；GroupDRO 将这四类数据作为四个组；
- 所有方法只用合成数据训练和选择超参数；ISPD 只作冻结配置后的外部测试（协议 A）；
- Hard Mining 和 WDRO 使用同一批 nominal 中心点、同一合法候选生成器、同一 top-4 精确评价和 evaluator-call budget；区别仅是 Hard Mining 没有运输惩罚/预算；
- 主 WDRO：`K=32`，原样本必在候选中，候选每 10 updates 按中心 ID 刷新；全部候选先初筛，top-4 用真实 Actor、真实 evaluator 和缓存的 GeoSteiner `L_star` 打分；
- `rho` 用各 degree 的扰动 cost quantile 表示，避免跨 degree 和跨 cost 直接复用数值；lambda 投影到非负区间；
- checkpoint 只根据 validation worst-group relative regret 选择；约束为 nominal mean regret 相比 ERM 恶化不超过 0.2 个百分点；nominal test 不参与选模；
- 每个 run 只保留 latest、best 两个 checkpoint和一份结构化日志，控制磁盘占用。

### 3.3 指标与最终产物

主损失为逐 net：

```text
relative_regret = (L_REST - L_star) / L_star
```

`L_star` 只由 GeoSteiner 计算并按数据 hash 缓存。最终自动输出：

- 主表：5 methods × nominal/3 OOD/ISPD，报告 mean、P95、worst-group 和 95% CI；
- paired per-net difference 及 bootstrap 95% CI；
- 按 distribution、degree、seed 的完整长表，不只保留均值；
- ISPD 的 net-weighted mean、design-macro mean、P95、worst-design mean、逐 degree 样本数及无效 net 数；
- lambda、mean selected cost、rho、budget gap 同图；
- evaluator/GeoSteiner calls、训练 wall-clock、推理时间和峰值显存。

## 4. 分阶段执行过程与工期

### 阶段 0：冻结协议和运行环境（9 月 4 日，0.5 天）

要做：

1. 把本节的 degree、分布参数、seeds、2,000 updates、指标和 nominal 退化上限写成只读配置；
2. 把 PDF 中要删除/改写的额外 baseline、消融、分布和 blind-test 表述列成论文同步清单；
3. 为每次运行保存配置 hash、代码 hash、环境版本和硬件快照；
4. 动态确认当次安全可用的 GPU 集合，并建立失败重跑清单；
5. 安全清理重复数据、旧缓存或移到其他盘，使剩余空间达到至少 100 GiB；不擅自删除原始 benchmark。

通过条件：配置冻结、`py311` 环境 smoke test 通过、磁盘空间达标。若磁盘仍低于 100 GiB，不启动批量训练。

### 阶段 1：统一数据、评价与审计基础设施（9 月 5–6 日，1.5–2 天；机器约 1–3 小时）

要做：

1. 实现共享 normalization/inverse、合法性投影、数据 ID 与 seed 隔离；
2. 生成并冻结 validation/test 文件，缓存所有固定测试集的 GeoSteiner `L_star`；
3. 实现逐 net regret、P95、worst-group、paired bootstrap 和调用次数统计；
4. 实现 Hungarian-L1 和 symmetric Chamfer-L1；测试 `c(X,X)=0`、置换不变、小位移、幅度单调、不同结构不坍缩、跨 degree 尺度、重复 pin/NaN；
5. 修复 ISPD 数据整理：现有 `compress.py` 会遍历 `_metal5` 变体且可能把 `merged` 再合并，必须按 design identity 去重并保留来源 ID；
6. 冻结 20 个基础 ISPD design（排除 `_metal5` 重复版本）为协议 A 的外部测试集，新实验在最终配置冻结前不读取其指标。

ISPD 采用协议 A：所有模型只用合成数据训练，所有超参数只用合成 validation 选择。现有仓库曾产生全量 merged REST 结果，因此可以称“模型对 ISPD 零样本迁移”，但不能称研究者完全未见的 blind test。若必须声称严格 blind，需要另给从未评测过的新 designs。

通过条件：数据不泄漏；预处理和 cost 测试全过；固定测试集 hash 与 `L_star` cache 可复用。

### 阶段 2：复现 ERM 并锁定统一训练器（9 月 7 日，0.5–1 天；机器约 2–4 小时）

要做：

1. 先在 degree=30、200 updates、seed=101 上完成训练/恢复/评价闭环；
2. 验证无 NaN、逆变换后的长度等价、日志和 checkpoint 可恢复；
3. 扩到五个 degree、3 seeds、2,000 updates，形成 ERM 基准；
4. 固定五种方法共用的 optimizer、batch、policy updates、验证频率和 checkpoint schema。

通过条件：ERM nominal regret 与现有 DAC21 checkpoint 在合理范围内，3 seeds 训练稳定。

### 阶段 3：完成三个基线（9 月 8–9 日，1.5–2 天；4 GPU 并行约 6–12 小时）

要做：

1. Mixed ERM：四组数据等比例采样，普通 ERM loss；
2. GroupDRO：四组 loss 的指数权重更新，优化最差组；
3. Hard Mining：对与 WDRO 相同的候选取最高真实 regret，不使用 cost 或 lambda；
4. 用统一 schema 跑五 degrees × 三 seeds，并自动比较 policy updates、测试文件和推理配置是否一致。

通过条件：三种方法可恢复、日志完整；Hard Mining 与 WDRO 的搜索预算可逐 run 对账。

### 阶段 4：打通 Random-WDRO 与数学审计（9 月 10–12 日，2.5–3 天；机器约 8–16 小时）

要做：

1. 实现 jitter、sparse、correlated、axis-biased 四类 Random/Sobol 候选和合法性去重；始终加入原样本；
2. 实现逐中心 `argmax(regret - lambda × cost)`，禁止把整个 batch 只选一个候选；
3. 实现 lambda 非负投影更新，记录 mean cost、rho 和 budget gap；
4. 检查 `rho=0` 接近 ERM、扰动变强时 cost 增大、lambda 更新方向正确；
5. 在固定小中心/候选集合上用线性规划比较 primal value、dual value 和 gap；
6. degree=30 单 seed 连续跑 500 updates，确认预算长期移动平均不失控后才进入调参。

通过条件：cost/候选/对偶测试全过；lambda 不恒零、不发散；合法候选率高且无 pin 顺序漏洞。

### 阶段 5：最小 validation 搜索与消融（9 月 13–14 日，1.5–2 天；4 GPU 并行约 8–16 小时）

只用 degree=30、seed=101 和 validation 数据完成以下最小集合，其他 degree 不重复网格搜索：

1. rho：`0 / q50 / q75`，其中 q50、q75 来自扰动 cost 分布；
2. lambda：selected rho 下 fixed / learned；
3. K：`1 / 8 / 32`；
4. searcher：Random-64 / CEM-64（4 轮×16、每轮 elite 4），严格同为 64 calls；
5. ground cost：Hungarian / Chamfer，各自重新按 quantile 标定 rho；
6. 选择 nominal 退化不超过 0.2 个百分点时 validation worst-group 最低的配置；主方法仍使用规范要求的 Random/Sobol K=32、Hungarian、learned lambda，其他结果作为消融。

通过条件：配置选择只引用 validation；生成一张消融表和一张预算诊断图。9 月 14 日结束后冻结配置，禁止看 test 后回调。

### 阶段 6：完整多种子训练与合成主测试（9 月 15–17 日，1.5–2.5 天；预计 25–50 GPU-hours）

要做：

1. 按冻结配置补齐 `5 methods × 5 degrees × 3 seeds = 75` 个模型；
2. 4 张 GPU 按 degree/seed 分队列运行，失败只按原配置重跑；
3. 在共享的 20 个 `degree × distribution` 测试文件上逐一评价，每格 10,000 nets；
4. 做完整性审计：75 个 checkpoint、所有测试格、所有 seeds、调用预算和日志不得缺失；
5. 生成主表、逐 degree 附表和 paired CI。

通过条件：五种方法结果齐全且没有 test-time 特权。结果不好也保留，不删除不利 degree、分布或 seed。

### 阶段 7：ISPD 最终测试（9 月 18 日，0.5–1 天；机器约 2–6 小时）

要做：

1. 冻结配置后才在新实验流水线中解锁 ISPD design IDs；
2. 用相同 normalization、`T=1` 和 evaluator 评价全部方法；
3. 输出 net-weighted、design-macro、P95、worst-design、逐 degree 样本数和无效数；
4. 不因 ISPD 结果返回阶段 5 调参，只允许修复经单元测试证明的实现错误并整套重跑。

通过条件：每个结果能追溯到 design/net ID、模型 hash 和 `L_star` cache。

### 阶段 8：独立攻击、效率与最终统计（9 月 19–20 日，1–2 天；机器约 12–24 小时）

要做：

1. 对每个 degree/distribution 固定抽取 2,000 个独立 attack nets；
2. 冻结所有模型，用训练时未使用的 CEM-256 同等攻击五种方法；
3. 报告攻击后的 regret、攻击 cost、合法率、calls、wall-clock；
4. 汇总训练时长、推理时间、峰值显存及 GeoSteiner/evaluator calls；
5. 自动生成论文所需主表、消融表、lambda/cost/rho 图和 CSV/JSON 原始表。

通过条件：WDRO 在独立攻击下的结论与正文一致；若只改善 P95/worst-group，就只写尾部鲁棒性，不扩大为 mean 全面领先。

### 阶段 9：同步论文草稿（9 月 21 日，0.5–1 天）

要做：

1. 按本计划第 1 节的清单，删除 PDF 草稿中已经取消的 baseline、分布、消融和 blind-test 主张；
2. 用冻结配置填写 configuration table，用自动生成的数值填写主表、消融表、摘要和结论中的所有 TBD；
3. 核对正文、表格、图、CSV 的数值和置信区间完全一致，不手工复制未追溯的数字；
4. 若投 ICLR，将内容迁移到 ICLR 模板；若仍投 ICCAD，则重新编译并检查页数与匿名性。

通过条件：正文不再承诺未执行的实验，所有结论都能追溯到冻结的结果文件。

## 5. 截稿倒排与止损规则

| 日期           | 必须达到的里程碑                                            |
| -------------- | ----------------------------------------------------------- |
| 9 月 6 日晚    | 数据、预处理、指标、cost tests 完成                         |
| 9 月 9 日晚    | ERM/Mixed/GroupDRO/Hard Mining 全部可运行                   |
| 9 月 12 日晚   | Random-WDRO 与 primal-dual 审计通过                         |
| 9 月 14 日晚   | validation/消融结束并冻结最终配置                           |
| 9 月 17 日晚   | 75 个正式模型和合成主测试齐全                               |
| 9 月 18 日晚   | ISPD 测试结束                                               |
| 9 月 20 日晚   | 独立攻击、效率和全部统计结束                                |
| 9 月 21 日晚   | PDF 草稿实验范围与最终结果同步；如投 ICLR，完成模板迁移初版 |
| 9 月 22–24 日 | 只用于失败重跑、核表、论文润色和备份                        |

止损规则：

- 不删减五个主方法、五个 degrees、三个训练 seeds、三个目标分布或每格 10,000 测试 nets；
- 若时间不足，先取消可选可视化和额外参数点，只保留规定的最小消融；
- CEM 只做 degree=30 的等预算消融和独立攻击，不作为所有 degree 的主训练器；
- 若 WDRO 到 9 月 14 日仍无稳定改善，继续完成冻结协议下的主实验，但收缩论文结论；不得使用 test 继续调 rho；
- 任一阶段发现数据泄漏、cost/对偶方向错误或 test 被用于选模，立即停止后续批跑，修复后从受影响的最早阶段重跑。

## 6. Codex 可自主交付的目录结构

后续实现和实验产物尽量集中到 `Codex/`，不直接覆盖原 REST：

```text
Codex/
├── WDRO-REST_最简实验执行计划.md
├── wdro_rest_min/
│   ├── configs/          # 冻结协议与各 run 配置
│   ├── src/              # 数据、方法、训练、评价、统计代码
│   ├── tests/            # normalization/cost/对偶/泄漏测试
│   ├── manifests/        # 数据、代码、环境和模型 hash
│   └── scripts/          # 单卡任务与 4-GPU 队列入口
└── experiments/
    ├── cache/            # L_star；按 hash 去重
    ├── checkpoints/      # 每 run 仅 latest/best
    ├── logs/             # JSONL/CSV
    ├── tables/
    └── figures/
```

我可以自主完成代码实现、测试、批量运行、失败重跑、结果审计、统计和表图生成。需要保持的外部条件只有两项：至少约 100 GiB 可写空间，以及尽量连续可用的 GPU 资源。GPU 编号和每卡并发数必须在每次启动时动态确认；已有进程不构成自动排除，只要剩余显存能容纳实测峰值和安全余量、计算未持续饱和，就应尽量利用。

## 7. 动态资源与持久任务补充规则

- 本文件中的 GPU 编号和磁盘容量只是带时间戳的历史估算，不是永久分配。每次启动、恢复或扩展批量任务前，必须重新查询 GPU 进程/显存/利用率和目标文件系统可用空间，并生成当次 host-runtime/preflight 记录。
- 在不造成显存超配/OOM、计算持续饱和且不改变冻结实验语义的前提下，尽可能使用所有仍有容量的 GPU，包括已有进程但尚未满载的 GPU。按实测任务峰值安排每卡并发数，优先并行独立 run，不固定假设任何历史 GPU 编号始终可用。
- 所有新产物的逻辑路径必须位于 `/remote-home/Guilin/DRO/Codex`。附加盘只通过 Codex 内的稳定符号链接使用，并在 manifest 中记录 resolved path。
- 超过单次可靠交互时间的阶段拆成可独立验收和恢复的子步骤。训练和大批量评价优先在服务器 `tmux` 持久会话中运行，并将日志、PID/锁、心跳、任务清单、checkpoint 和恢复命令全部存入 Codex。
- 可复用的完整启动文案见 `02_新对话执行提示词.md`。
