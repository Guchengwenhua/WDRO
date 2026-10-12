# 方向 1 / 2：搜索区域标签与保留尺度的代理目标

本文件在新实验运行前定义假设、方法与结论边界。实现为本目录的 `surrogate.py`；旧 v3.1 源码和其历史结果均不改写。

## 一、共同数学对象与实验边界

固定 degree 30 的 ERM Actor 参数 \(\theta\)，分别使用父模型 seed 101、102、103。对于合法归一化 net \(Z\)，真实 greedy regret 为

\[
r_\theta(Z)=\frac{L_{\mathrm{REST},\theta}^{\mathrm{greedy}}(Z)}{L^\star(Z)}-1.
\]

\(L^\star\) 由项目固定 GeoSteiner 参考实现与缓存取得，\(L_{\mathrm{REST},\theta}^{\mathrm{greedy}}\) 由冻结 Actor 和 REST 几何评价器取得。代理 \(C_\psi\) 拟合该 regret。给定中心 \(X\)、固定非负价格 \(\lambda\)，搜索目标为

\[
f_{\theta,\lambda}(X,Z)=r_\theta(Z)-\lambda c(X,Z),\quad
\hat f_{\psi,\lambda}(X,Z)=C_\psi(Z)-\lambda c(X,Z),
\]

其中 \(c\) 始终为已有 Hungarian-L1 mean 成本。Critic 是 Z-only 的 regret 代理；运输成本显式依赖中心，不被塞进 Z-only 标签。主模型训练采用 sampled REINFORCE 时对应的成本与这里的 greedy 攻击标签是两个对象，本固定模型研究不声称改变了外层训练目标。

所有新增代理均从同一 seed 的已校准 v3.1 `InputCritic` 深拷贝初始化，保持原网络宽度/深度与线性输出头；不更新 Actor。旧代理作为 `old` 保留，新增四个代理为 `uniform_mse`、`region_mse`、`region_huber`、`region_rankdiff`。每个新增代理固定 6 epochs、Adam lr \(10^{-4}\)、batch size 128、1,024 个训练标签。最终攻击固定每中心 64 次真实评价，中心占 1 次，63 个候选使用同一搜索配置与同一价格；具体测试分布、样本量与 paired bootstrap 由主协议统一冻结。

标签成本、代理训练计算和测试攻击成本单独报告。不同方法使用相同攻击预算并不意味着总计算成本相同。

## 二、方向 1：让训练数据覆盖搜索区域

### 2.1 动机与可检验假设

基于 uniform 数据的风险

\[
R_P(\psi)=\mathbb E_{Z\sim P}[C_\psi(Z)-r_\theta(Z)]^2
\]

不自动约束梯度搜索诱导分布 \(Q_{\psi_0,\lambda}\) 上的风险。搜索会主动访问预测高分区域，模型偏差可能在这些点被放大。已有模型驱动优化研究讨论过这类分布转移造成的高估；此处只借用问题动机，没有实施其保守目标，也没有移植其理论保证。[Trabucco et al., 2021](https://proceedings.mlr.press/v139/trabucco21a.html)

待检验的经验假设是：在相同新增真实标签数下，来自实际合法化梯度轨迹的样本能降低局部差分误差，并提高固定预算的真实攻击分数。即使验证 MSE 降低，该假设仍需由真实攻击检验。

### 2.2 两个相同标签预算的数据池

**Uniform refresh**：在新训练 namespace 内生成 1,024 个合法 uniform nets，每个真实评价一次。它控制“仅增加训练数据与 6 epochs”的效应。

**Search-region refresh**：新训练 namespace 内生成 256 个独立 uniform 中心，每中心固定取四个点：

1. 原中心 \(X_i\)。
2. Gaussian std 0.02 起点经旧合法化后的 \(Z_{i,0}\)。
3. 从该起点运行固定初始代理的第 4 步 \(Z_{i,4}\)。
4. 同一轨迹的第 8 步 \(Z_{i,8}\)。

所有点使用相同初始代理 \(\psi_0\)、价格 \(\lambda\)、最大预合法化位移 0.08、RMS 归一化步长 0.01。四点共计 \(256\times4=1024\) 次真实标签。重复点仍计为真实评价请求；不会为追求唯一率而临时改变采样。按每个 net 的坐标字典序分配随机扰动，再恢复原 pin 顺序，以保持固定 seed 下的排列一致性。

两池使用不同 namespace，不复用最终测试中心。混合分布可写为

\[
Q=\tfrac14P_X+\tfrac14P_{Z_0}+\tfrac14P_{Z_4}+\tfrac14P_{Z_8}.
\]

本轮是固定初始代理收集的一次 refresh，不含自适应多轮高误差挖掘。这样隔离数据来源效应，避免额外 oracle 预算与迭代次数混杂。多轮循环/高预测误差选择仍是未实施扩展，不能由本轮代替验收。

### 2.3 控制与统计对比

主因果对比为 `region_mse - uniform_mse`；`uniform_mse - old` 衡量本轮额外标签和更新相对于旧代理的综合效应。所有代理从同一参数初始化，不串行继承另一个变体；所有 Actor labels 对同一固定父模型有效。

最终每个中心保留已真实评价候选的最高 \(r-\lambda c\)。记录其分数提升、regret、Hungarian 成本以及逐中心差值。方向 1 支持性证据要求最终真实攻击差值，而不是只报告训练 MSE。不同 seed 分开列示，不能将所有样本混为独立模型重复。

### 2.4 预期失败与解释边界

搜索区域可能太窄、拟合初始代理的特定偏差，或者丢失 uniform 泛化；仅有 256 个相关样本组意味着有效独立几何覆盖少于 1,024。MSE 最优不保证输入梯度有用；重复点和碰撞修复也会降低信息量。新数据并非模型在线变化后的最新标签，因此固定 Actor 结论不能直接用于外层训练。

若在某个有限候选集合 \(\mathcal S\) 上满足 \(\sup_{Z\in\mathcal S}|C_\psi(Z)-r_\theta(Z)|\le\epsilon\)，代理最大值选择相对于该**同一集合**的真实最佳值至多损失 \(2\epsilon\)：在真实最优点及代理最优点分别应用误差界即可。我们没有验证该一致误差条件，更没有证明梯度搜索覆盖连续域，因此它仅解释为什么局部校准可能有帮助。

## 三、方向 2：比较 MSE、Huber、局部差分与排序

### 3.1 相同数据与尺度约束

三个目标共享完全相同的 1,024 个 search-region labels、训练顺序、初始参数、epochs、batch size 和学习率。每个 minibatch 按中心整组取样，以确保四点之间的六个局部无序对均可参与损失。不同中心之间不生成排序对。

预测值和标签始终采用原始 regret 单位，不标准化后直接用于搜索。若把代理乘上 \(a>0\) 而保持 \(\lambda\) 不变，实际上将搜索权衡改成 \(C-(\lambda/a)c\)，因此不能把任意排序分数当作数值校准的 regret。

### 3.2 预先固定的三个目标

MSE 基线：

\[
\mathcal L_\mathrm{MSE}=\frac1N\sum_i(C_\psi(Z_i)-y_i)^2.
\]

Huber 回归：令 \(e=C_\psi(Z)-y\)，固定 \(\delta=0.01\)：

\[
h_\delta(e)=\begin{cases}\tfrac12 e^2,&|e|\le\delta,\\
\delta(|e|-\tfrac12\delta),&|e|>\delta.\end{cases}
\]

它限制单个大残差的导数绝对值到 \(\delta\)，可能减少异常样本影响，也可能降低恰好需要拟合的高 regret 尾部的权重。Huber 和 MSE 在相同学习率下的梯度量级不同；结果解释为这两个预定义完整训练配置的对比，不声称已找出各目标的最佳学习率。

局部差分/排序复合目标：令 \(\mathcal P\) 为同中心内的无序对，\(d_{ij}=y_i-y_j\)、\(\hat d_{ij}=C_i-C_j\)，固定温度 \(\tau=.01\)，定义

\[
\mathcal L_\mathrm{local}=\mathcal L_\mathrm{MSE}
+0.25\,\mathbb E_{(i,j)\in\mathcal P}(\hat d_{ij}-d_{ij})^2
+0.1\tau^2\,\mathbb E_{(i,j)\in\mathcal P:d_{ij}\ne0}
\log(1+\exp[-\operatorname{sign}(d_{ij})\hat d_{ij}/\tau]).
\]

真实差分绝对值不超过 \(10^{-12}\) 的对不计排序项；差分项仍保留。\(\tau^2\) 使排序项与平方 regret 的尺度相容。MSE 锚定绝对数值；差分项训练有限移动收益；排序项区分局部谁更难。任何两点差分都不是非光滑真实评分器的精确梯度标签。

对排序/差分项，全组预测同时平移常数不改变结果；MSE 会惩罚该偏移。回归锚点是此复合损失区别于纯排序代理的关键。复合项仍可能鼓励过大的局部 margin，必须报告预测均值/标准差与验证 MSE，不能只报告排序正确率。

### 3.3 选模与验证

使用独立 validation namespace 生成 **128 个中心**，每中心同样四点，共 512 个真实标签。所有数据/目标变体共享该验证池。每个变体仅在 6 个 epoch checkpoints 中按 validation MSE 选择最小者，平局取较早者。这一轮保持选模标准不变，以隔离训练数据/损失效应；“按真实攻击收益选代理”尚未作为另一条训练消融实施。

验证还记录 MAE、全局 Spearman、中心内平均 Spearman、非平局局部对排序准确率、局部差分 MSE、预测与标签的均值/标准差。以上指标均不替代独立 test 上的真实攻击结果。主协议中五个代理全部进入最终测试，不依据测试表现删除变体，不用最终测试指标选择 checkpoint。

### 3.4 计算与 oracle 账本

| 项目 | 每 seed 唯一新增真实标签 | 共享关系 |
|---|---:|---|
| Uniform train | 1,024 | 仅 uniform_mse |
| Search-region train | 1,024 | region_mse/huber/rankdiff 共用 |
| Validation | 128 centers × 4 = 512 | 全部四个训练变体共用 |
| 本模块合计 | 2,560 | 不包括初始 v3.1 训练与最终攻击 |

每个训练变体 6×1,024=6,144 行训练前向及反向；正常运行每 epoch 512 行验证，结束时再 512 行指标核验。搜索区域训练采集需 256×8=2,048 行输入梯度前向/反向，共享验证轨迹另需 128×8=1,024 行。两种训练池的标注预算相同，但区域池多出梯度采集成本。实际 wall time 与合法化计数入产物，不由预期行数冒充。

所有 oracle 请求由已有 `Oracle` 持久账本计数，包括固定 256 行推理批形状引入的 padding；本模块另外写 `label_requests.jsonl`，在真实评价之前记 requested，标签原子落盘后记 completed。中断后只复用已提交标签，未提交请求允许重试并记录额外开销。`manifest.json` 绑定标签文件、数组内容、Actor 参数、初始代理参数、namespace、价格和请求日志哈希。训练保存每 epoch 的优化器状态与模型，并采用 epoch 独立随机种子恢复 dropout 序列。

## 四、产物与接口

顶层接口：

```python
result = train_variants(oracle, parent_critic, base_model, seed, outdir,
                        degree=30, price=frozen_lambda,
                        namespace="critic_search_20260915", heartbeat=heartbeat)
models = result["models"]
```

`outdir/data/{uniform,search_region,shared_validation}` 保存 `labels.npz`、标注进度、请求日志和 manifest；`outdir/models/{variant}` 保存 `best.pt`、`latest.pt` 和 `summary.json`。父目录 summary 汇总共享预算。所有文件位于本新研究分支内。

CPU 单元测试覆盖数值尺度锚定、局部对边界、Huber 导数、组完整性、标签共享/哈希校验、中断恢复与初始代理不可变。GPU acceptance 由主 worker 执行真实 Actor/GeoSteiner 标签与独立最终攻击；单元测试通过本身不表示本研究的性能假设成立。
