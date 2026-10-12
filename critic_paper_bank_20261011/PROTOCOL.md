# 理论与实验边界

## 1. Critic 值怎样换算

现有基础 WDRO 的 Actor 采样出拓扑，其线长为 L_sample；GeoSteiner 给出 L*。训练目标为 r=L_sample/L*−1。Critic 拟合该随机策略相对误差的条件期望，而非最优线长，也不是 greedy 线长预测器。

本次按最简单的代数关系构造 reference_hat=L_greedy/(1+critic(X))。只读当前更新前的 Critic，detach，随后用该参考值替代候选评分分母和 sampled 更新分母。每次更新前重新生成参考值；不引入独立标签训练、不做测试集校准或裁剪。

这不是只凭 Critic 就识别出 L*：还使用当前 Actor 的 greedy 线长作为尺度锚。它也不是一个可行树证书或最优值证书。L_greedy/reference_hat−1 在代数上恰等于 critic(X)，因此候选搜索实际上退化成按 Critic 预测排序。外层 sampled 目标则为 L_sample/L_greedy × (1+critic)−1，存在自举漂移，而且 sampled/greedy 的错位本身引入偏差。实验如失败可作为该直接换算的负结果，不能证明所有 value surrogate 都无效。

推理时预测 Critic 用 eval 模式和固定 batch128；训练时 Critic 用 train 模式，与原架构一致。两种模式可能进一步造成偏差。测试保存 Critic、真实线长、L*、反推参考值、参考值偏差、HPWL 违反率以及 oracle 辅助得到的 predicted_current_length=L*×(1+critic)。后者仅用来测量预测误差，不能用于无 oracle 的部署。

## 2. K3 在两篇论文中的意义

23ICCAD 使用项目已有固定 commit TripleGAT Actor，J=[0.5L+0.5R]/18。25ICCAD 使用项目已有 categorical-phase HarRL 组件，J=[1e18 delay+0.25 buffer_count]/10。两者原实现都没有调用 GeoSteiner，也没有可信的全局最优时序/缓冲参考值。

从每个任务已完成的 ERM/WDRO 三种子模型中，按原 validation worst-OOD objective 选取3个 best checkpoint，固定教师库。B_J(X)=min_k J(teacher_k,X)。K3 分支训练与候选搜索使用 J_current/B_J−1；原分支继续使用 J_current。两者均用当前 Actor 的实际拓扑/物理评估器计算 J_current，K3 每个候选额外运行3个教师及各自物理评估器。

这是把“模型库相对参考”迁移到任务成本上，不能称为替代两篇论文中的 GeoSteiner，也不预设加速。B_J 不是最优成本；current 可能优于库，负目标必须保留。归一化改变样本权重和lambda有效尺度，因此效果是整项方法改动的效果。固定原 lambda_lr=1 和原 rho，不从测试结果重新调整。

## 3. 配对、预算与数据

REST 各 seed 从同 seed 的已完成 ERM best Actor/Critic 出发；两篇论文从同 seed 已完成 ERM latest Actor/Critic 出发。所有分支重建同样学习率的 Adam，以避免把不匹配的 optimizer 状态混入不同目标。K3 教师构造保存/恢复随机数，不改变同种子分支的训练 RNG 起点。

训练样本、候选噪声使用新命名空间20261011，分支间配对；batch128。REST做128步，Timing和Buffer各256步，seeds101/102/103。训练扰动32，几何 shortlist4；不使用 ES/RL/Critic梯度攻击器。Critic 分支只改变参考分母，不改变候选生成方法。

REST 的 validation/test 使用历史缓存中的前256/1024个样本/组和对应真实参考，不产生新标签反馈；论文 validation/test 使用本实验命名空间、固定seed261011重新生成。四种分布 uniform/mix_gauss/edge_bias/grid_cluster。所有分支通过真实 validation worst OOD mean 选择最早最佳checkpoint，包含update0；测试仅在训练与选择完成后读取。保留best和final，不掩盖update0被选中的情况。

短程 continued-training 不能回答从头训练、degree30/50、真实芯片或论文整体复现问题。报告每个种子、三种子均值、配对差值及描述性种子bootstrap区间；不把单模型实例数量当训练重复次数。

## 4. 运行与恢复

参数在首轮GPU训练前固定于protocol.json。inputs.json记录教师/父模型路径、历史validation分数、大小和mtime，不重新做内容指纹或大范围验收。历史输入的内容不修改。

控制器最多并行3任务；每张GPU先连续3次采样（间隔20秒），空闲显存至少16GiB且利用率不超过35%才接纳。每进程显存上限20%，CPU数学线程1。采用tmux、控制器/任务独占锁、PID/start_ticks/boot_id身份、heartbeat、原子检查点。控制器重启如果旧子进程尚活跃则拒绝重复启动。已有complete跳过，失败保留日志。

训练中断后最多回退8次更新，latest含当前与best权重、Adam、RNG、lambda和步数，恢复时裁掉未提交日志。测试文件原子写入；训练已结束而评估中断时，从最后训练checkpoint重新评估。父目录与代码留在DRO/Codex，大产物经artifacts链接存入附加盘。
