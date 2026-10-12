# 对抗生成框架 v3 实验结果

本文件由 adv_v3.report 从完整已绑定产物生成；没有新增训练或 oracle 评分。

A：30个父模型任务、1080个评价格完整，逐网复核 276480 个选中net的合法性、Hungarian成本、真实分数恒等式及原X下界。

| 比较（左减右） | score均值稳定为正的格数 | 稳定为负的格数 | 其余格数 |
|---|---:|---:|---:|
| es64 − random64 | 40 | 0 | 0 |
| es64 − cem64 | 18 | 0 | 22 |
| critic_calibrated4 − random_top4 | 12 | 7 | 21 |
| rl4 − random_top4 | 2 | 22 | 16 |
| critic_calibrated4 − critic_raw4 | 7 | 1 | 32 |
| critic_calibrated64 − random64 | 36 | 0 | 4 |
| rl64 − random64 | 22 | 0 | 18 |

每格为固定父方法×degree×分布、三个已训练seed的配对比较；正值代表攻击分数更高。稳定方向要求95%配对CI不跨零且三个seed同向。cost变化单列，不能把走得更远直接称为更好。

B：18个交替训练任务、216行训练前后及独立攻击统计完整；0个任务选中起始模型（best_update=0）。

训练前后mean/P95、逐seed最差OOD组和独立CEM64比较见 joint_cells.csv、joint_worst_group.csv、joint_paired.csv。RL联合版新增奖励调用单独计账，其总训练预算不同于冻结攻击器版。

全部CI采用2000次按中心配对bootstrap，三个seed共享同一中心重采样；条件于本次固定数据和已训练模型，未作多重比较校正。未证明连续空间最坏分布、Wasserstein风险证书或新方法的普遍优势。

调用账目保留请求/完成上下界及恢复额外调用。共享服务器和跨恢复分段耗时不支持受控加速比；逐项效率范围见efficiency表。输入分离审计覆盖训练来源中心和Critic标签集，未保留的RL动作轨迹不在逐内容交集审计内。

原precision_v2实验与代码均保留，父源文件hash再次核验通过；阶段9论文工作没有启动。
