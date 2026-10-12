# 真实 net 的 VAE 球壳先验 × WDRO：次日交付实验方案

状态：PLAN_ONLY。2026-09-21 编写；已只读核对数据与代码，尚未训练或产生实验结果。

## 0. 执行约定与交付优先级

用户要求：明天看到结果；不要 smoke 和 SHA 校验。执行者直接实现并进行有实际训练预算的实验，不新建 smoke、哈希清单、源码绑定、独立验收流水线或通用测试工程。不要因历史流程要求这些步骤而反复请求确认。

保留真实运行所必需的有限数值、维度、数据划分、坐标合法性检查；保存日志、checkpoint、逐 net 结果和统计表。这些检查在正常数据处理/训练中执行，不另跑预实验。出现 NaN、参考解失败、数据错位时不能继续生成虚假结果。

新文件只放本目录。冻结方法目录只读；不修改历史 benchmark 的数据、checkpoint、报告、模型实现。此次是新的真实数据微调实验，不替换历史主线。

时间目标按执行开始 T0 计算，不承诺未经计时的运行时：

- T0+4～6 小时：优先输出 seed 101 的 RAW-WDRO 与 VAE-SHELL-WDRO 完整 256-update 对照及自然测试结果。
- T0+8～12 小时：争取主对照 3 seeds 完成，同时补齐 ERM 和 VAE-BALL 消融。
- 预留最后 60～90 分钟用于测试、汇总和中文结论，不把全部时间耗在训练。
- 若启动时已很晚，以用户实际截止时间为准；没有更具体时间则以上述相对时间执行。每完成一个完整方法对即可更新阶段报告，不等所有可选实验结束。

最低交付：RAW-WDRO 与 VAE-SHELL-WDRO 在 seed 101、同一父模型、同一数据、相同 256 updates 下的对照。优先完成这两组的 seeds 102/103，再追求其他消融。一个完整主对照比多个没训练完的方向更重要。

不从零训练 REST，不做所有 degree，不接 Critic 梯度、PGA、PPO，不移植整套 AGRO，不改真实 regret 定义。方案采用固定候选池，必须称为“有限候选 WDRO 微调”；不是历史在线 Random/Sobol 主线的逐项复现。

## 1. 研究问题与成功/失败含义

问题：在同样真实训练数据 A 上，用 VAE 学得的几何先验和潜空间球壳限制产生候选，是否比原坐标扰动的 WDRO 更能改善独立真实数据上的 REST regret？

主假设是最终模型的泛化改善，而不是 VAE 找到更低的攻击分数。生成器可能去掉不必要的极端样本，也可能遗漏有效的困难样本。二者必须通过共同测试区别。

VAE 的编码均值半径限制只是一项经验规则，不是物理真实性、概率覆盖或连续 Wasserstein 鲁棒性的保证。真实测试集在旧研究中已经用过，此次按设计隔离训练/调参，但不宣称“从未看过的全新盲测数据”。

## 2. 当前已核实的资源与代码

### 2.1 现成真实数据

输入：`/remote-home/Guilin/DRO/Codex/benchmarks/v31_models_20260912/artifacts/inputs/degree_30.npz`

参考值：`/remote-home/Guilin/DRO/Codex/benchmarks/v31_models_20260912/artifacts/inputs/reference_30.npz`

读取 `pins, net_ids, design_ids, dataset_ids, family_ids`；输入形状为 `(11032,30,2)`、float32。参考文件有 `net_ids, lstar, verification_lstar`，此次只复用 `lstar`，不重跑双尺度检查。按 net_id 对齐；不能假定另存后的行序仍一致。已有 content_hashes 字段不参与此次计算或校验。

已核实来源：ISPD2008 9088 nets；ISPD2018 468；ISPD2019 1330；CircuitOps 146（65+79+2）。无需重新下载、解析 LEF/DEF 或重跑旧 prepare.py。

### 2.2 REST 父模型、环境和可复用模块

- Python：`/remote-home/Guilin/miniconda3/envs/py311/bin/python`
- 基础包：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min`
- REST 原模型：`/remote-home/Guilin/DRO/DRO/WDRO-REST/models/actor_critic.py`
- 动态库：`/remote-home/Guilin/DRO/DRO/WDRO-REST/algorithms/libeval.so`
- 父 checkpoint：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/checkpoints/stage2/formal_erm_d30_s{101,102,103}_u2000/best.pt`
- 模型与真实树长：基础包 `src/rest.py` 的 `build_actor, build_critic, RestTreeEvaluator`。
- 参考最优值：基础包 `src/geosteiner.py`，使用现有 `2**27` 修正版，显式传上述 libeval.so；不使用旧 v1 参考实现。
- 几何与运输：基础包 `src/geometry.py, src/datasets.py, src/costs.py`。
- 训练损失可参考 `/remote-home/Guilin/DRO/Codex/adversarial_v3_1_20260912/adv_v3/joint.py` 的 `sampled_update`（约 349 行）。
- 有限 WDRO 选择/价格更新可参考基础包 `stage4/dual.py`，不要照搬旧工程的整套运行审批或验收逻辑。

直接 `torch.load` 本地可信 checkpoint，加载 Actor 和训练基线 Critic 的 state_dict，执行 strict shape/key loading。不要调用会强制 SHA 检查的 `load_parent`、`load_trusted_legacy_actor` 或旧冻结清单校验入口；用新分支内的简单加载器，不改原文件。读取现有索引仅用于找路径。

### 2.3 存储与 GPU

计划核查时：GPU 0/1/2/3/6/7 显存为 0；4/5 有占用。状态会变化，执行前再读一次 nvidia-smi，优先独占空闲卡，不结束其他任务。建议最多同时 4 个训练进程，每卡一个。

根盘当时仅余约 115 GB；附加盘 `/attached/remote-home1` 约余 2.2 TB。代码/文档留在本目录；执行者把大产物保存到 `/attached/remote-home1/Guilin/DRO/Codex/experiments/vae_wdro_real_20260921`，在本目录建立 `artifacts` 链接。不要复制原始 benchmark 或历史模型全集。

新建自己的 run_env.sh，将 TMPDIR、缓存、MPLCONFIGDIR 指向新 artifacts，设置 OMP/MKL/OPENBLAS_NUM_THREADS=1、CUBLAS_WORKSPACE_CONFIG=:4096:8。无需改全局环境。

## 3. 数据划分：直接使用下表，不看旧性能来挑设计

主实验数据源为 ISPD2008；family_ids 如下。同一 family 的全部 net 留在同一 split，不随机拆 net。

| split | family_ids | 原始行数 |
|---|---|---:|
| A_train | adaptec1, adaptec2, adaptec3, bigblue1, bigblue2, bigblue3, newblue1, newblue2, newblue4 | 2810 |
| A_cal | bigblue4, newblue3 | 1202 |
| A_val | adaptec4, newblue6 | 1085 |
| B_test | adaptec5, newblue5, newblue7 | 3991 |
| C_external | 全部 ISPD2018、ISPD2019、CircuitOps | 1944 |

具体计数：A_cal=959+243；A_val=234+851；B_test=320+1320+2351。A_train+A_cal+A_val+B_test=9088。

使用已有 family_ids 合并规则；若发现已有 family 跨表中 split，先按 family 合并，不能拆开。在本地 `splits.csv` 记录 net_id/design_id/family_id/dataset_id/split。

不跑 SHA。可以直接将 float32 坐标按 (x,y) 排序后做字节相等比较，检测跨 split 的完全相同几何；保留优先级为测试 > 验证 > 校准 > 训练，从低优先 split 去掉重复行并记录计数，不把近似几何当作相同。测试集本身的合法原始重复行保留并在报告注明。以上行数是处理前的实测值。

用途隔离：

- A_train：VAE 拟合、REST 微调、候选池生成、初始参考缓存。
- A_cal：球/球壳边界、共同 rho 的几何校准；不更新模型权重。
- A_val：VAE checkpoint/一次预设回退选择、REST checkpoint 选择。
- B/C：只用于输出测试结果，不调超参、不挑方法、不决定是否换 VAE。

所有 REST 组看到同一 A_train、同一 seed 的逐步原始 net 索引序列。训练采用 net-uniform 有放回采样，对应同一经验分布；同时报告 design-macro 结果，避免只看大设计。

## 4. 实验矩阵与优先顺序

| 方法名 | 每个中心的候选 | 作用 | 优先级 |
|---|---|---|---|
| RAW-WDRO | 原样本+7 个原坐标扰动 | 主对照 | 必须 |
| VAE-SHELL-WDRO | 原样本+7 个球壳解码候选 | 用户提出的主方法 | 必须 |
| ERM-A | 原样本 | 真实数据微调对照 | 第二优先 |
| VAE-BALL-WDRO | 原样本+7 个只有外半径的解码候选 | 内半径消融 | 第二优先 |
| MIX-WDRO | 原样本+4 RAW+3 SHELL | 生成先验是否适合作补充 | 可选 |
| VAE-AUG | 由 SHELL 池均匀采样，不按 regret 选 | 区分普通增强与困难筛选 | 最后可选 |

主 seeds=101,102,103。每个 seed 的所有 REST 组从同一个对应 ERM 父 checkpoint 出发，优化器重新初始化。VAE 和候选池先固定一份，跨这三个 seeds 共享；所以三 seed 只检验 REST 微调随机性，不等于三次独立生成器实验。

先做 seed101 的两项主对照，再优先保证主对照其他 seeds；空闲卡可同时做 ERM-A/VAE-BALL。绝不先做大量消融再发现主对照没有结果。

## 5. VAE：简单集合编码器，直接重建坐标

不替换 REST Actor。VAE 完全独立，预训练后固定。

### 5.1 确定的初始结构

- degree=30，潜变量维度 L=16。
- Encoder：共享逐点 MLP `2→64→128`，SiLU；mean pooling 与 max pooling 拼接为 256 维；再 MLP `256→128`，两个线性 head 输出 mu/logvar，各 L 维。
- Decoder：MLP `L→128→256→60`，SiLU；输出 reshape(30,2)，用 sigmoid 映射到 [0.1,0.9]。
- 在重建损失前使用可微版项目坐标归一化：逐轴减最小值，以两个坐标轴中较大的跨度共同缩放到 0.8，再加 0.1。span 加数值下限。不要对两个坐标轴分别缩放从而改变长宽比。
- 无数据增强、无 test 数据输入；训练随机打乱点序，集合 encoder 不依赖顺序。

### 5.2 损失、预算、选模

重建使用双射 Hungarian matching 下的平均平方 L2 距离：在 detach 的距离矩阵上求最优匹配，再对选中配对用 Torch 求重建损失梯度。不要使用逐行 MSE；不使用只靠 Chamfer 的多点塌缩方案。

`loss = mean_matched_squared_L2 + beta * mean_latent_KL`，KL 对 batch 和潜维都取平均，beta 从 0 在前 30 epochs 线性升至 0.001，之后保持。logvar 可裁剪 [-10,5]。

- Adam lr=0.001，batch=128，最多 300 epochs；每 5 epochs 在 A_val 上用 mu（不采样）重建。
- 以 A_val 平均 Hungarian-L1 重建误差选 checkpoint；连续 8 次验证不改善就停止。设至少 50 epochs。
- 单模型软时间预算 75 分钟。保存 best/last 和曲线。
- 将全量 A_val 的中位重建 Hungarian-L1 与共同 rho 比较。若中位误差 > rho，只允许一次预设回退：L=32，其余不变，最多再 60 分钟；按同一验证重建指标选择两者中较好者。不能根据 B/C 选择。
- 不因重建差而无限调模型。若回退后仍差，照实记录并完成主对照；若 >95% 选中 nominal，说明此实现未产生足够可用的局部困难样本，不能把它解释为真实性提升。

这里没有条件生成器或 residual decoder：第一版保持“解码出完整坐标”的明确含义。不能临时改成 X+[G(z)-G(mu(X))] 却仍声称候选限制在同一个解码流形上。

## 6. 球/球壳边界与候选池：不做潜空间梯度

### 6.1 半径校准

用固定 VAE 的编码均值定义 e(X)=mu(X)，对 A_cal 计算 r=||e(X)||2。

- r_max=经验 97.5% 分位数。
- r_min=经验 2.5% 分位数。
- BALL：0≤||z||≤r_max；SHELL：r_min≤||z||≤r_max。
- 两种方法共用 r_max；报告实际 inner/outer clipping 比例。若几乎没有触及内界，就不能用无差异结果证明内界一定无用。
- 这不是有限样本 95% 置信覆盖承诺，也不是 VAE posterior sample 的半径分位数。

### 6.2 VAE 候选

每个 X 的中心 z0=mu(X)。各潜维尺度 s_j=max(std_A_train(mu_j),1e-3)。共享固定候选种子 20260921。

产生 7 个原始潜变量 `z_raw=z0+tau*s*epsilon`，epsilon 标准高斯；tau 固定 `[0.25,0.5,1.0,0.25,0.5,1.0,0.5]`。BALL 和 SHELL 使用相同 z_raw，仅投影规则不同。

径向投影：外界超过则缩到 r_max；SHELL 内界不足则沿原方向放到 r_min，零向量用固定随机单位方向。decode 后进行同一坐标规范化和最小间距修复。记录修复率，不能把大幅修复后的结果当成仍严格属于解码集合。

原样本 X 始终保留为 slot0，即使 X 无法被 VAE 精确重建。因此“VAE 限制”指非 nominal 候选的来源，不能声称整个支持都严格等于 G(Z)。

先统一按坐标规范排序候选，避免 VAE 输出槽位顺序引入额外 REST 效应；所有方法的候选和原始输入采用同一排序约定。运输仍用无序 Hungarian-L1。

同中心重复候选最多重采样 16 次，仍不足则保留重复并记 unique_K，不用 RAW 偷偷补满 VAE-only 池。重复不带来额外有效搜索能力；报告实际候选多样性。

### 6.3 RAW 候选与共同 rho

RAW 使用既有四类坐标扰动思想（jitter、sparse、correlated、axis-biased）。固定 7 个 `(family,amplitude)`：`(jitter,.01),(jitter,.02),(jitter,.04),(jitter,.08),(sparse,.04),(correlated,.04),(axis-biased,.04)`。合法化/归一化/排序与其他方法相同。不得比较旧32候选预筛选组与新8真评分组。

在 A_cal 上，用独立固定 RNG=20260922 生成同样 RAW 几何候选；不调用 Actor。所有非 nominal 候选运输成本的 q75 定义共同 rho。所有 WDRO 组共用这个 rho，不按方法分别重新校准。rho、半径、VAE都不在 REST 微调过程中变动。

RAW/BALL/SHELL 对 A_train 分别一次生成完整 K=8 候选池，保存 pins、center_index、slot、cost、来源。对每个 distinct 候选只算一次 Lstar，跨全部 seeds 复用。

如果全量训练池太慢，允许按预定计划同时对所有方法把 A_train 截为每 family 用固定 RNG 均匀抽取、总计最多 2048 nets 的子集；此决定仅依据预计算吞吐和剩余时间，发生在各组 REST 正式训练前，写入配置。VAE仍可用全 A_train，但所有方法拥有同等数据访问；报告实际训练中心数量。不能根据模型效果选择子集。

## 7. 控制主要时间成本：参考解一次计算，候选分数实时计算

全量预处理最多 3×2810×7=59010 个非 nominal 参考请求（重复可复用），而不是每步再次调用 GeoSteiner。BALL/SHELL 相同结果可以直接复用。已有 X 的 Lstar 从原参考文件读取。

不要使用强制文件/实现 SHA 身份绑定的旧 ReferenceCache。新建轻量缓存：规范化 float32 坐标的完整 bytes 作为相等键（可用内存 dict 或 SQLite BLOB primary key），值为 Lstar。不要 SHA、MD5 或全盘内容扫描；只处理本实验输入。

参考解使用现有修正版数值过程，成功状态/有限值检查保留。不跑双尺度复验。单个参考解超时 30 秒可终止本实验对应 worker，把该候选标记无效并跳过其选择；不能填造参考值。记录失败率；若任一池无效率 >1%，停止该池的效果宣称，先定位错误。

CPU reference workers 总数最多 8～16，依据 CPU 占用调整；参考预处理与 VAE/RAW 训练可流水执行。不能每个 GPU 作业各自启动 16 个 oracle worker。

Greedy Actor 分数必须随 Actor 参数更新重新计算，不能缓存旧 regret。候选几何、运输成本、Lstar 可以缓存。

真实评分统一 `actor.eval(); no_grad(); actor(pins, True)`；batch=256，末块复制末行补齐，输出再截取。相同输入排列、相同 padding 规则用于所有模型。sampled 训练仍 batch=64、actor.train()、actor(pins,False)。

## 8. REST 微调：固定可完成预算

- 首要完整预算 256 updates，batch=64，seed 101/102/103。
- Adam lr=1e-5，Actor/Critic 联合优化，clip norm=1.0；新 optimizer，不恢复父 optimizer。
- 所有组复用同 seed 的 center_index[256,64]；sampled动作 RNG 与候选 RNG分开。候选池在训练前固定，不消耗后续动作 RNG。
- WDRO 每中心把 8 个候选全部真实 greedy 评分，选择 `regret-lambda*cost` 最大者；同分优先 nominal，再取较小 slot。
- `regret=L_REST/Lstar-1`。不换成原始线长、HPWL/MST 代理，不使用 Critic 预测取代最终真实评分。
- 选择后调用 sampled Actor；重新计算 sampled tree length，沿用这个候选的 Lstar。Actor loss=`((sampled_regret-baseline).detach()*log_probs).mean()`；Critic loss=`(baseline-sampled_regret)^2.mean()`。
- lambda 初始 0，所有 WDRO 组相同；每步用旧 lambda 选样本，再更新 `lambda=max(0,lambda+1.0*(mean_selected_cost-rho))`。
- 不导入历史 synthetic 最优 lambda。lambda 可在各组自适应分化，但每组日志都记录数值、cost与budget gap；有限训练并不保证平均运输严格≤rho。
- 每32 updates保存 latest（Actor、Critic、optimizer、lambda、RNG、下一步索引）；每64 updates在完整 A_val 验证。验证不进行训练。
- update0、64、128、192、256 为共同观测点。每组保存 last.pt 和 val_best.pt；best 按验证 design-macro mean regret 选，包含 update0；同分保留较早 checkpoint。
- 主结果是所有方法共同的 last@256，val_best 为次结果，二者均报告。不能只报告某组有利的 checkpoint。

单个 WDRO run 请求 256×64×8=131072 个 greedy candidate 分数和 16384 个 sampled 训练分数。三 seeds 的主对照共 786432 greedy requests。实际 GPU 前向 padding 行数与真实候选数分开计数。上述计数不包含自然验证/测试，另行记录。

若 core 全部完成且仍有至少 3 小时，可将参与比较的方法统一续训到512；256结果保留且仍为共同首要预算。不允许拿512步新方法与256步baseline作主比较。

## 9. 评价与结论规则

### 9.1 必须完成的自然测试

对每个 seed 的父模型、last@256、val_best 在完整 B_test 和 C_external 上 greedy评估，全部用已有 Lstar。保存逐 net 长度、regret、design/family/source、method、seed、checkpoint_role。

报告：

1. B_test net-weighted mean regret、pooled P95、design-macro mean、每设计 mean/P95。
2. C_external 分开列 ISPD2018、ISPD2019、CircuitOps；CircuitOps只有146 nets且小设计多，明确探索性。
3. 逐 seed 配对差 `SHELL-RAW`（负数更好）、跨seed均值/标准差；百分比点差和相对regret下降率都写分母。不能称为相同比例的线长缩短。
4. lambda、选中cost、nominal选择率、修复率、unique_K、VAE内/外界投影率；VAE重建误差与潜变量方差/KL。
5. 训练时间、VAE与候选/Lstar一次性成本、每组训练时间、实际更新数。

若生成器只学出规则平均点集、nominal几乎全选，报告这一机制失败。若 val_best=update0，直接说明未发现验证收益，不能称为“VAE训练改善”。

### 9.2 有时间则做独立攻击（不阻塞自然测试交付）

使用与 VAE 无关的原坐标 Random32：固定取 B_test 3个设计各32 nets（总96），同一候选池、slot0原样本+31扰动，原始loss最大化（lambda=0）。全模型共同输入与预算，参考值预计算共享。不要用各模型自己的VAE候选评价自己的鲁棒性。

资源充裕再扩展192中心/CEM32；不要优先跑大规模CEM256。若未做独立攻击，结论限于自然测试，不宣称全面鲁棒性提高。

### 9.3 不夸大统计证据

主要报告3 seeds方向、效果大小、逐设计分解。B只有3个held-out设计，不能用几千个net把设计级不确定性伪装得很小。

可附2000次按设计整体重采样、再按seed重采样的配对区间，必须注明仅3个测试设计、区间不稳定。单seed结果只做初步描述。不把未显著等同于无效，也不以最好seed作为主结果。

分类：

- “值得继续”：SHELL较RAW在B主指标上3 seeds同向改善，逐设计/外部来源无明显集中的退化；若有独立攻击同时报告是否保持。
- “条件性收益”：均值改善但种子/设计不一致，或自然数据改善但独立攻击退化。
- “未见收益”：共同预算下无一致提升，或与update0相比无改善。
- “机制未跑通”：VAE重建/覆盖差、解码修复严重、有效候选不足；与充分验证方法无效不同。

BALL比SHELL好时如实报告下界可能有害；SHELL与BALL几乎相同且内界触发很少时，只能说本配置未充分检验内界。

## 10. 快速执行顺序与降级

1. 读取现成输入和参考，保存split与配置；检查可用GPU/目录。直接启动正式VAE拟合，同时构建RAW池与共同rho。
2. RAW参考完成后先启动RAW seed101；VAE完成后构建BALL/SHELL池，优先SHELL参考与seed101。此时ERM可占另一空闲卡。
3. 主对照seed101到256立即做完整自然测试并写 INTERIM_REPORT.md；保持另外两seed排队，不因单seed效果决定取消或改参。
4. 补主对照seeds102/103；有容量补BALL、ERM的同预算seeds。
5. 写共同256步主表、val_best表和图；余时再做独立攻击、MIX、AUG，顺序如上。

吞吐估计直接使用正式训练前32个updates和真实候选参考队列的完成计数，不额外跑smoke或丢弃这些训练步。记录预计完成时间并每小时更新。

若超时：先取消可选MIX/AUG和512续训，再缩减独立攻击，再只保留主对照3seeds；最后才退为主对照1seed。不要取消自然测试。可以报告共同128步的阶段性结果，但它不替代256预算完成，且不同预算分表。

若VAE完全失败且无法在预算内修复，仍交付RAW/ERM正式结果、VAE真实失败日志与诊断，不能换成普通噪声并保留VAE方法名。

用tmux/nohup与简单本实验单实例锁，保存pid、命令、日志、开始时间和最新update，避免SSH断开丢任务。不引入复杂跨项目调度器。发现同名本实验作业已运行时先接续，不重复启动。

## 11. 实现文件与交付物

建议新代码：`run_env.sh, data.py, vae.py, candidates.py, oracle.py, train.py, evaluate.py, report.py, run_all.py`。命名避免把common/evaluate等旧目录塞入路径后意外import错模块。只复用必要的src工具，不import旧主程序触发整套流程。

本目录已提供 `config.json`，作为具体参数起点。建议命令接口（由执行者实现，不是当前已经存在的可执行命令）：

```bash
bash run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python run_all.py --config config.json --resume
```

交付至少包括：

- `artifacts/splits.csv`、实际执行的config与资源/进度摘要（无SHA）。
- VAE best/last、训练曲线、校准半径、重建/生成图（每组固定同一批验证输入）。
- 候选池、参考值与几何/失败计数，供复用，不需要每步dump海量坐标。
- 每组 latest/last/val_best checkpoints和训练JSONL日志。
- `tables/main_last256.csv, main_valbest.csv, per_design.csv, costs.csv`。
- 逐net测试NPZ/CSV、两张图：主对照mean/P95、逐设计配对差。
- `INTERIM_REPORT.md`（首个完整对照后）、`REPORT.md`（中文自包含：问题、实际做了什么、结果、失败/局限、下一步）。

执行结果只按实际文件和真实日志描述：代码完成、训练完成、性能改善是三种不同状态。不安排SHA交付验收，也不把“没有抛异常”当成方法有效。
