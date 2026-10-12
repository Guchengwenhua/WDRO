# 真实 net 的结构先验与全局布线单元约束：理论及冻结实验方案

制定日期：2026-09-22。新分支独立于已完成的 VAE 与正在运行的 CVAE 扩展。本文先于本分支训练与测试制定。代码实现、启动、工程完成、科学改善是四个不同状态。

## 1. 要解决什么问题

一个 net 是若干需要电气连通的 pin。REST 的输入是无序二维坐标，输出直角 Steiner 树的近似构造。真实芯片还涉及 pin 所属单元、层号、障碍物、金属宽度/间距、routing tile 容量、驱动/负载与时序。仅靠坐标 VAE 不可能恢复这些丢失的条件。

旧方案：degree30 集合 VAE 以 Hungarian 匹配重建 + KL 学习；对编码后扰动施加潜向量半径球壳，再解码并以真实 regret 选困难候选。半径是潜空间统计，不是设计规则。旧 REPORT 显示 last@256 普遍劣化，验证最优仍为 update0；不能据此宣称流形过滤有效。并行 CVAE 分支刚由 A_val 选择 freeze-BN/lr1e-5；本方案采用此已知训练修正，全方法一致，记录对历史验证的复用。

研究问题分开检验：
1. 新先验能否保留真实测试 net 的局部结构、减少预定义极端几何？
2. 保留先验后仍能否找到有实际 regret 的困难样本，还是只退化为原样本？
3. 相同训练预算、真实参考、父模型下，REST 微调能否改善隔离设计上的 mean/P95？

## 2. 五种新方法与对照

### M1：可解释布线统计门控（feature_gate）

对 net X 定义 phi(X)：bbox 宽高/长宽比、HPWL、L1 最小生成树长度/HPWL、最近邻距离分位数、轴向近对齐比例、四象限 pin 占用。HPWL=w+h 是最短连接的下界，MST 描述连接代价代理；MST 不是真实已布树。对训练 phi 用中位数和 MAD 标准化，再计算最大绝对偏差作为异常分数。在 A_cal 真实 net 上以 99% 分位数固定门限。RAW 的同一候选若超过门限则沿中心插值回退（1,1/2,1/4,1/8,0）；若真实中心本身超限则保留中心并记 fallback，不偷偷删除罕见 net。

这是可解释经验 support gate，不是工业 DRC，也不在当前跨设计依赖下宣称 conformal coverage。A_cal 只用于几何校准，不看 Actor/test。

### M2：几何邻接图的平滑扰动（laplacian）

每个真实 net 构造对称 4-NN 几何图，A_ij=exp(-d1(X_i,X_j)/s)，L=D-A。采样 E~N(0,I)，Delta=(I+tau L)^(-1)E，再调整为与 RAW 相同的初始 RMS 幅度。tau=2 固定。其高频图谱分量被衰减，相邻 pin 倾向共同移动，减少把局部 pin 簇打散。图是 net 内几何邻接，并非额外获得的门级 netlist/真实路由拓扑。

### M3：多尺度普通集合去噪先验（set_dae）

在真实 net 上以 Y=X+sigma E 自监督，sigma∈{.005,.01,.02,.04,.08}。DeepSets 逐 pin 特征 + mean/max 全局池化，输出每个 pin 的残差。输入保留 pin 对应，不使用固定排序输出，也不把复杂度混同于 VAE。训练损失为 mean ||D(Y,sigma)-X||²/sigma²。网络看不到干净 X；噪声配对仅来自 train。

局部候选：对真实中心加入与 RAW 同一高斯噪声，D 去噪，再取 X + 0.5(Y-X)+0.5(D(Y)-X)。混合保留探索，避免所有候选收缩成原样本。这是单步多尺度去噪生成，不冒称完整 DDPM 反向采样。

### M4：布线结构图去噪先验（graph_dae）

用带相对 dx,dy,|dx|,|dy| 的 4-NN 消息传播替代纯集合模型。训练时只在含噪 Y 上构图，避免干净目标信息泄漏到图输入。附加损失为：干净目标 kNN 边的 L1 长度恢复 + bbox 跨度恢复 + 轴向对齐软统计恢复。公式：

L = L_denoise + eta [L_edge + L_bbox + .1 L_axis]，eta∈{.1,1}。

干净图只定义监督目标中的边；推理时不用干净图来指导网络消息传播。该损失与直角连接结构有关，但不包含层/容量/时序信息。set_dae 是同宽度、同数据、同训练预算的必要消融。

### M5：图去噪 + routing tile/层号保持（graph_tile）

从 ISPD08 .gr 文件恢复 grid origin、tile size、pin layer、pin 物理坐标；严格对 net ID、归一化坐标进行反向校验。给每个 pin 固定其原 routing tile 与 layer。对 M4 的有对应候选裁剪回该 tile 的内部，并保留原 bbox 极值 pin 的对应轴，避免后续归一化改变物理映射；若出现极近 pin，则整条位移按比例回退而非单独推离，仍失败则回到原样本。

若 pin 的 tile/layer 集合和所有环境容量不变，原全局 routing 问题的离散端点与容量输入不变；因而在相同 grid 抽象下不会产生新的端点可行性问题。此性质并不证明 tile 内 detailed route 可行，也不证明移动 pin 对应一个可实现的单元布局。真实 pin 必须随 cell 移动，宏障碍、pin shape、LEF 与完整网表约束尚未纳入。此分支称为全局布线抽象一致的几何反事实。

为了辨别“只是扰动变小”，增加 tile_raw：同一 RAW 噪声仅作 tile 投影，与 graph_tile 使用同样硬约束、生成次数和尺度。若 graph_tile 仅优于大幅 RAW 而不优于 tile_raw，不能把收益归因于学习先验。

### 对照与完整矩阵

主方法共 9 个：erm_a、raw_wdro、vae_shell_wdro（原训练池原样复用）、feature_gate、laplacian、set_dae、graph_dae、tile_raw、graph_tile。除 ERM 外均 K=8（含中心 slot0），共享中心序列和真实评分；去重后的有效 K、fallback 和位移单独报告，不能以低有效 K 的低风险冒充优秀搜索。

后续未执行的完整物理扩展：基于 LEF/DEF+cell/netlist 的合法 cell ECO 扰动→所有连接 pin 同步更新→OpenROAD 增量全局/详细布线→DRC、overflow、timing 检查。这需要独立的全芯片实验协议，不能用本分支 net 坐标结果代替。检索确认工具所需信息后再做，不把装个 router 当做已实现约束。

## 3. 数据协议

沿用旧设计隔离，保证可比：A_train=adaptec1/2/3,bigblue1/2/3,newblue1/2/4；A_cal=bigblue4,newblue3；A_val=adaptec4,newblue6；B_test=adaptec5,newblue5,newblue7；C_external=ISPD18/19 与 CircuitOps。degree30 的 train=2810，cal=1202，val=1085，B=3991，C=1944（以准备产物再次校验）。

扩展预训练：仅 A_train 设计的全部合法 degree10/20/30/40/50 net，变长网络，不截断高 degree 为 d30。按 degree 均匀、再按设计均匀、最后按 net 均匀采样。degree30 后训练仍只用 A_train。按归一化 pin 字节跨划分去重，优先保留测试；不声称近重复/同系列设计完全无相关。逐 design、degree、split 输出规模；这些测试设计在历史项目中已使用，不称全新盲测。

主 REST 微调固定旧 2810 个中心，扩展数据只用于新生成先验。VAE 历史基线仅用 d30，额外加入 set_dae 公平比较同预训练数据；不把更大数据的好处统归于图结构。

## 4. 训练与调参预算（固定配置见 config.json）

生成器预热调参共 6 次：set_dae width96/160 × eta0；graph_dae width96/160 × eta .1/1。种子301，每次6000多degree预训练updates +1500 d30后训练updates，batch128（约96万训练 net 暴露）。AdamW lr3e-4，梯度裁剪1，去噪噪声范围固定；每250更新验证，最后阶段验证最优选 checkpoint。以 A_val 的固定噪声 d30 坐标去噪误差（design-macro、sigma归一）选同一方法超参，不用测试。

正式生成器：选定 set/graph 超参各 seeds301/302/303 从头训练18000多degree +4500 d30 updates，batch128，即每run288万样本暴露；等效epoch按实际 train 数量报告，d30 后训练约205遍。每250步保存完整 optimizer/RNG/阶段，报告 pre/post 各阶段指标。训练预算固定，不以测试好坏扩展。

REST：9方法×3父种子101/102/103，共27条主训练；最大2048 updates×batch64，共同保存 last/val_best@256/1024/2048。每128更新验证，每32恢复checkpoint。Adam lr1e-5，冻结 BN running stats 但训练 affine，与已知稳定配方一致；全部从对应同一父 ERM checkpoint 起步、同一全新 optimizer。主生成器301固定，另 graph_tile generators302/303 ×3 REST seeds，共6条交叉扩展，总33条REST训练。

WDRO 目标 max_k [L_theta(Z_k)/L*(Z_k)-1-lambda c(X,Z_k)]，c=归一化 Hungarian-L1 mean。每步 lambda←max(0,lambda+mean(c_selected)-rho)，共同rho沿用 A_cal RAW q75。有限候选、有限步对偶更新不构成连续 Wasserstein 球最优保证；报告实际平均预算违反。物理对应约束作用在 pin index 上，Hungarian只是无序 net 的评分成本，不代替 pin 身份。

## 5. 验证与验收

A. 生成器：固定 held-out corruption，误差相对 noisy identity 基线，按sigma/design汇报；图结构 loss 消融；模型置换等变测试。
B. 候选：真实 held-out pin 几何统计、HPWL/MST代理变化、cost分位数、有效 K、gate通过率、tile crossing、层号保持、fallback率；注入 collapse/outlier/axis distortion 仅作几何压力诊断，不能标为真实非法电路标签。tile约束必须在物理映射中100%满足，否则整项失败。
C. 固定父 REST 上的候选压力：同K、同lambda、真实 GeoSteiner L*；记录max/selected regret 和cost，区分过滤降低难度与模型训练收益。
D. 自然测试：每个预算的 last 与验证最优均评 B/C 全 net，列 net-weighted mean/P95、design-macro、worst-design；逐net NPZ。按seed配对、design聚类bootstrap 2000次；B仅3设计，不宣称高置信通用改进。报告正负结果。
E. 独立攻击：最终验证最优在B每设计固定32中心，以新的RAW K32、lambda0构造共同攻击池，真实L*、不使用训练候选。按方法/seed报告压力下regret，计入额外预算。
F. 工程：数据ID对齐、跨split无精确重复、置换等变、tile保持、参考非负regret、oracle错误/timeout、checkpoint恢复、单实例锁、完成数量与hash manifest。小规模端到端通过只是启动门槛，全部项目终态才可标工程完成。无改善也应完成报告，不无限调参。

## 6. 资源、存储和交接

所有入口在 DRO/Codex/net_structure_real_20260922。artifacts 链接到附加盘 /attached/remote-home1/Guilin/DRO/Codex/experiments/net_structure_real_20260922（本机主盘仅约79GB余量，附加盘约2.2TB）。历史分支只读，复用都记录源hash。

使用8张GPU的实际余量，最多每卡一个本分支worker；初始入场为6GB。2026-09-22 06:03 UTC依据实测约438MB allocator峰值，调度细化为REST至少2GB、生成器至少3GB空闲，每worker allocator上限显存5%（约2.4GB）；此调整只影响资源入场，不改模型、预算或评价。保留实际余量。无需空占显存；增加有效任务比抢占/耗尽显存更有价值。不停止、杀死、重新配置他人任务。CPU按当前进程可用31核，准备最多8并行，GeoSteiner最多12并行，单GPUworker torch/BLAS 1线程；CPU与GPU任务流水线。资源变满时队列等候，OOM最多重试两次并恢复，反复错误停止依赖分支并写状态，不能静默跳过。

独立tmux会话 net_structure_20260922；队列、各worker日志、checkpoint、60秒内心跳、阶段ETA写入 artifacts；监督器单实例flock、防重复worker PID/starttime检查。时间估计必须用本机实测速度动态校正，不以理论FLOPS估计。任务成功退出时生成 REPORT.md 与完成manifest；失败生成可恢复错误清单。运行稳定、有恢复证据和ETA后可结束本次对话。

## 7. 理论来源与边界

- ISPD08 官方数据与输入语义：https://www.ispd.cc/contests/08/ispd08rc.html
- RUDY 原论文（矩形需求估计，只有全网需求叠加并结合容量才涉及拥塞）：https://past.date-conference.com/proceedings-archive/2007/DATE07/PDFFILES/08.7_1.PDF
- OpenROAD global placement / RUDY：https://openroad.readthedocs.io/en/latest/main/src/gpl/README.html
- 全局布线层/容量/拥塞：https://openroad.readthedocs.io/en/latest/main/src/grt/README.html
- 详细布线 pin access、track、search/repair：https://openroad.readthedocs.io/en/latest/main/src/drt/README.html
- 多尺度去噪动机：https://arxiv.org/abs/2006.11239 。本方案实现的是局部去噪先验，不是该文完整DDPM。

本次不使用“单net HPWL/area”冒充全芯片RUDY拥塞，也不以MST、VAE分数或有限候选测试宣称DRC/时序/可制造性保证。
