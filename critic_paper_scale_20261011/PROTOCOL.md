# 扩展训练冻结协议

## 1. 问题与预算

原短程实验中Critic预测在validation和最终test为0；原REST多个best落在update0；23ICCAD基本持平；25ICCAD出现小幅改善，但K3增加耗时。扩大预算回答：这些比较在较长训练下是否保持，以及排除初始零输出后最直接的Critic换算是否有实际价值。

主线21项直接续到2048步。保持degree10、batch128、greedy microbatch128、学习率1e-4、候选32+中心、shortlist4、原rho与lambda学习率、每32步验证、每8步checkpoint、原数据与候选生成命名空间。历史best（含原update0）仍参与同一验证选择。每项继承原latest中的Actor、Critic、Adam、RNG、lambda、步数和best_weights，初始bootstrap不加载任何测试结果。

不更换batch或重置主线优化器，避免将规模扩展混同重新训练。原短程的源目录不修改；本目录保留其runner和训练数据生成代码的快照，外层只修改输出目的地、预算及运维记录。

## 2. 修正Critic独立组

每种子从原REST Geo第128步Actor出发，固定Actor生成校准样本。新的无根集合Critic：二维坐标逐点128维MLP、mean/max聚合、128维MLP输出logit，r_hat=softplus(logit)−1。输出没有ReLU截断区，也没有BatchNorm；初始最后一层权重0、偏置log(expm1(1.01))，因此初值约0.01，且数学上1+r_hat>0。数值非正分母仍由运行时拒绝，不偷偷裁剪。

这是架构修正而不是只增加训练步数。保持与原方案相同的代数换算：reference_hat=L_greedy/(1+r_hat)，候选损失依旧代数上等于r_hat；因此修正输出层并未消除自举偏差、预测与greedy的语义差异或代理排序风险。

### 训练侧校准

- 每种子4096个独立uniform训练net和1024个独立uniform holdout net，使用新命名空间202610112的update1000000+chunk和2000000+chunk；不使用实验validation/test。
- 固定Actor采用eval模式采样4条拓扑，标签为4次相对GeoSteiner误差均值。它近似原Critic的sampled regret语义；Actor训练模式可能产生不同分布，此限制保留。
- Critic拟合256步、batch128、Adam学习率3e-4。样本有放回抽取，标签分块保存，优化器和随机数每16步保存。不根据holdout调参或反复尝试。
- 预先规定的启动门槛：holdout MSE小于零预测MSE，预测标准差大于1e-5，输出有限且分母为正。未达到则该种子的两个修正训练分支gated，不改阈值。
- 同时报告使用训练标签均值的常数预测MSE及相关性；启动门槛不要求优于常数预测，因此通过不等于学到了有用的样本排序。
- 标签计算与拟合时间、GeoSteiner调用量单列，历史父模型亦含oracle训练。不声称完整流程oracle-free。

通过的种子共享同一份校准Actor/Critic checkpoint，分别启动Geo和critic_inverse，各新增2048次Actor更新。两臂新建相同Adam，lambda从0开始，种子及数据/候选命名空间配对。它们的update0是旧Geo第128步模型，因此选择与原main不同；只在修正组内部作方法归因比较。

## 3. 评价和结论边界

真实validation仍四分布各256个net，每32步按最差OOD均值选择最早最佳checkpoint，包含各组起点；训练结束后真实test四分布各1024个net评估best和final。REST用精确GeoSteiner缓存，论文用真实任务物理成本。论文新的repaired组不存在；main论文完全保留原测试生成协议标识，保证与短程相同样本。

保留512、1024、2048里程碑模型及全部验证轨迹；预先固定2048步，不从中间测试表现决定停止或扩展。本轮继续使用已见过的测试基准，是基于前轮发现开展的探索性扩展，不作为独立确认性研究。

报告各seed、三seed均值、配对差、best步数、lambda、原中心选择率、Critic分布/零值、实际参考偏差及真实任务成本。bootstrap仅是三配对种子的描述性区间，不能用net数量替代训练重复数。若修正组种子不齐，报告实际数量和gated原因。

时间分别列出新增训练（样本生成、搜索、参考、优化）、原历史累计训练、校准和总墙钟；原日志训练计时不含周期验证/最终测试。K3教师训练未摊销。没有等总时间比较，不能用等更新结果宣称效率优势。论文K3更改目标归一化，25ICCAD仍为现有local-greedy buffering组件。

## 4. 运行、存储与恢复

30项队列包含21主线、3校准和最多6修正训练，依赖写入queue.json。最多8个GPU任务，一张卡一个本队列worker。每张卡连续3次、间隔20秒采样可用显存>=16GiB且利用率<=35%后接纳，每进程内存上限20%，CPU数学线程1。附加盘少于20GiB可用时不接纳新任务。不中断外部任务。

控制器和每项训练有互斥锁，记录PID/start_ticks/boot_id。main bootstrap保存resume_origin，latest原子写入为提交点；已有latest不再覆盖。latest包含best_weights以便回滚未提交验证记录。每8步保存训练状态，每16步保存校准状态，标签按128网分块存储含RNG。SIGTERM在当前更新/标签块之后保存。校准失败导致依赖gated；容量暂停最多自动重试1次，之后保留状态供恢复。错误不冒充科学负结果。

通过`bash launch.sh`恢复，完成项跳过；旧子进程仍活跃时拒绝重复控制器。source快照/配置/代码在本目录，权重/缓存/数据/日志通过artifacts链接放在附加盘。启动前附加盘约748GB可用，根盘约15GB。
