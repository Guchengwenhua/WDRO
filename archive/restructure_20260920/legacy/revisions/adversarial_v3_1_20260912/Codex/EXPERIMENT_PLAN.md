# 对抗生成框架 v3：三路线验证协议

日期：2026-09-12。用户已授权：保留全部既有代码，在DRO/Codex内建立版本副本，先写实验文档后直接执行三方向验证，使用所有有容量的GPU并发。此版本为独立研究扩展；不覆盖precision_v2，不改写旧实验结论，不推进阶段9论文任务。

## 版本、数据与产物

- 版本根：DRO/Codex/revisions/adversarial_v3_1_20260912。
- 新执行目录：该版本内Codex；新实现置于adv_v3，父实现副本置于wdro_rest_min。
- 原REST的models/algorithms/utils已复制到版本内DRO/WDRO-REST，避免修改共享legacy依赖。
- 801个父文件已逐文件复制并校验，清单为manifests/parent_snapshot.json。父模型只读加载，更新只写新checkpoint。结束时再次检查父文件hash。
- 新实验产物仅写Codex/experiments，链接到DRO/Codex/experiments/adversarial_v3_1_20260912；附加盘resolved path记入manifest。
- 合成数据另设版本化seed命名空间；train、validation、evaluation、joint evaluation不复用旧test/ISPD指标。四种分布沿用原参数和合法性定义。

## 问题与预先规定的比较

固定X、主模型θ和价格λ，最大化f(Z)=Lθ(Z)/Lstar(Z)−1−λc(X,Z)。参考值使用precision_v2 GeoSteiner；成本使用Hungarian mean L1；保持degree、[0.1,0.9]等比bbox归一化、最小间距及float32合法性。

固定模型攻击使用eval/greedy。外层主策略继续用sampled REINFORCE，因此联合训练属于现有评价目标的近似优化，不宣称精确零和均衡或连续Wasserstein鲁棒性证书。ρ为平均成本预算；不得擅自改为每个样本c≤ρ。

三个研究问题：

1. 一步RL是否可跨X学习直接生成高分Z？与主模型交替更新是否优于固定攻击器？
2. 高斯平滑的成对零阶梯度能否在相同真实评分预算下优于Random/CEM？
3. 已有Critic梯度是否可用，经过greedy标签校准后能否以较少真实评分有效攻击？

## A：固定模型验证（完整网格）

- 主模型：ERM与WDRO，各degree={10,20,30,40,50}、seed={101,102,103}，共30个父best checkpoint。
- 价格：使用同degree/seed父WDRO best checkpoint的λ，对该对ERM/WDRO共用；记录ρ和λ，模型间不混淆目标。
- 评价中心：每degree/四分布256个新net，同degree所有模型共用同一集合；每模型1024中心，总30720模型-中心。
- 学习用数据：只在新uniform训练来源学习攻击器/代理，OOD作为迁移评价。训练、验证、评价hash交集必须为零。
- 新攻击器训练种子由版本、degree、父模型seed与方法名确定；不根据评价结果重选。

| 方法 | 每中心真实搜索评分 | 作用 |
|---|---:|---|
| random_top4 | 4 | 原32扰动+X，33几何初筛，X+前三真实评分 |
| critic_raw4 | 4 | 原Critic输入适配；保留死区现象 |
| critic_calibrated4 | 4 | greedy校准Critic引导上升 |
| rl4 | 4 | 一步RL直接输出3个候选，加X |
| random64 | 64 | 全真实随机/Sobol比较 |
| cem64 | 64 | 原同目标CEM四轮，每轮16 |
| es64 | 64 | X+7轮×(4对正反+新中心)=64 |
| critic_calibrated64 | 64 | 同真实预算的代理搜索 |
| rl64 | 64 | 同真实预算的策略采样 |

原X始终占一次真实评分，并在原f下选已评分历史最佳。每个候选请求都计入预算，即使reference缓存命中；同时另计真正GeoSteiner cache miss。预算为搜索候选数，不把一个batch调用记作一次。

Gaussian-ES：μ=0.02，4对方向、7步，位移上限0.08；方向按每net RMS归一化后步长0.01。该归一化属于搜索更新，不改变成对估计公式。固定确定性合法映射，重复点评分缓存但不因seen历史改点；失败固定回退X并记数。

Critic：复制父Critic，修通输入图，raw版本保持原输出头；calibrated版本更换为允许有效梯度的输出头并用真实greedy regret拟合。训练8192个合法uniform来源点，独立validation1024点；Adam lr=1e-4，batch128，16 epochs，按validation MSE选best。多起点坐标上升每起点8步，步长0.01，位移上限0.08；逐步Hungarian分支L1梯度，合法性处理后真实复核。raw不使用评价结果挑checkpoint。记录MSE、排序、原始/校准零梯度比例。

RL：共享点编码+全局池化+逐点高斯位移头，输入X与λ；REINFORCE更新，不经过真实评分器反传。256 updates、batch64、Adam lr=3e-4、初始std0.02、最大位移0.08；用原X真实分数作动作无关baseline。真实奖励f，固定主模型。每32步独立uniform validation128点检查，按真实平均f选best；不读取evaluation指标。全部奖励/validation真实调用另计。策略不输出pin重排。

固定超参数用于第一轮验证，不承诺最优。出现实现错误可修复并重跑受影响任务，必须留下版本与失败记录；效果差不是修改参数/删结果的依据。

## B：主模型交替训练验证

A完成且实现/预算审计通过后直接执行，不要求A某方法获胜才运行。

- degree30、3个父ERM seeds，各分支从完全相同父Actor/Critic开始。
- 512外层updates，batch64，相同新uniform中心流；主优化器Adam lr=1e-5、gradient clip1.0；ρ沿用同degree/seed父WDRO记录，λ从对应值开始按原公式更新。
- 六分支：random_top4、critic_calibrated4、rl_frozen4、rl_joint4、random64、es64，共18个新主模型。
- rl_frozen4保持A攻击器固定；rl_joint4每4个主updates，用当前主模型再做一次攻击者batch64 REINFORCE更新。攻击者额外训练调用必须单独报告，不能声称与冻结版总计算相同。
- critic_calibrated4每32个主updates，用本轮真实评分标签做短校准并记录目标快照，避免把旧代理永久当成当前主模型。
- 不引入每pin一个agent、长rollout、centralized critic或对手池；本轮验证最小双策略交替结构，并明确完整MARL尚未实现。
- 外层每64步checkpoint；保存Actor、主Critic、攻击策略/代理、所有optimizer、λ、RNG、数据游标与账目。
- 独立validation每64步，每分布128net；在相对起始ERM nominal退化≤0.002条件下，按OOD worst-group选择best；无满足者保留起始模型并明确报告。
- 新evaluation每分布512net；独立CEM64攻击每分布64net、最大化regret（λ=0），同等攻击全部18模型。无ISPD调参。

## 公平性、诊断与统计

- 按degree、父模型、seed、distribution、searcher、budget完整输出逐网结果和长表。
- A报告原f、Δf、regret、真实cost、改善比例、合法/修复比例、调用数、reference miss、代理/策略训练成本与时间；同中心同λ配对比较。
- 主比较：es64对random64/cem64；critic_calibrated4和rl4对random_top4；校准Critic对raw；64档学习方法对random64。
- B报告训练前后mean/P95/OOD worst-group、nominal门槛、独立攻击、λ/mean cost/ρ轨迹，rl_joint与rl_frozen区别及额外成本。
- 95%配对bootstrap，2000次，固定统计seed；报告逐seed方向和条件于本次数据/父模型的CI，不宣称无限总体或多重检验后的显著性。
- 所有方法即使失败或不改善也保留。合法性/预算/可恢复性属于工程验收；是否更强属于实验结果，不混用。

## 并发、资源与恢复

起始快照：8×RTX6000Ada；2/3空闲，4/5部分占用，0/1显存接近满，6/7高利用。附加盘约2.6TiB可用，CPU配额nproc=31，另有公开芯片数据准备任务。

启动前多次采样GPU，先用可用卡做覆盖degree10/30/50的显存/吞吐smoke；按实测峰值加1024MiB context与2048MiB安全余量动态准入。所有8卡均持续参与容量检查；有剩余显存且计算未持续饱和的卡可共享。并发通过独立模型任务增加，评价可分块；不用无效显存占用伪造满载。

长流程放入独立tmux socket；队列记录PID、进程启动时间、心跳、锁、任务identity、命令、日志、checkpoint和退出状态。不终止其他任务。资源不足只等待/恢复，不改变已冻结实验参数。CPU GeoSteiner可能成为瓶颈，根据实测调每GPU并发与CPUworker总数，不用显存占满比例代替吞吐。

结果写.partial，检查后原子发布。task和阶段审计幂等；完成文件必须包含协议hash、代码hash、父模型hash、输入hash、调用账目和结果hash。完成后释放本版本锁并检查进程退出。

## 验收和交付

1. 版本副本/父文件hash、数据分离、合法性、输入梯度、ES正反预算、RL梯度和checkpoint恢复测试通过。
2. A的30个模型任务、9方法、四分布完整；每方法真实预算精确，全部真实最佳不低于其原X。
3. B的18个训练与独立评价齐全；resume状态/账目有证据，无遗失或删减的失败。
4. 逐网NPZ、训练JSONL、汇总CSV/JSON、配对CI、资源/效率表、三路线结论报告、父文件保护审计和completion manifest。
5. 仅全部交付审计通过才标DONE。任何仅smoke/正在训练的任务都标DOING。

## v3.1 数值实现修订（正式重跑前冻结）

首轮v3完整任务审计发现：同一原X在ES的64行推理与其余方法的256行推理中，1024中心有7处greedy选树不同。其余144分块合法性、预算、数据分离与hash检查通过。这是批次形状改变浮点运算后影响离散argmax的问题；首轮产物保留于原v3目录并标为失败证据，不纳入新结果。

本修订保持所有算法、超参数、数据种子命名空间、父模型、比较网格与评分预算相同。greedy Oracle固定每次Actor执行256行，尾部不足时复制该请求末点填满并丢弃填充输出；REST/GeoSteiner与搜索预算仍只计算真实请求。另计actor padding前向行数，避免隐藏实际网络计算。每个评价分块在写盘前与已评分方法的原X基准分数逐项核对；任何不一致立即失败。固定batch的置换/分块一致性必须经真实GPU诊断验证，之后完整重训/重评，旧攻击器权重不复用。
