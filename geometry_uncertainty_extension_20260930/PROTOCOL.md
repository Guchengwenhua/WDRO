# 条件几何不确定性独立开发/测试接续协议
遵循 /remote-home/Guilin/Skill.md；仅科研推理、统计及运行管理，不运行工程tests、smoke或hash。所有新增产物仅在本目录。冻结2026-09-30，无WDRO训练自动gate通过。

## 分阶段矩阵
主阶段d30：seed102 ERM M0900、WDRO M1118；seed103 ERM M0902、WDRO M1120。
每模型四族 edge_bias/grid_cluster/mix_gauss/uniform；dev各1024来自validation seed202；test各2048来自test seed303。共49152推理实例。
次阶段d10：seed102 ERM M0888、WDRO M1108，每族dev512/test1024，共12288。总61440，非seed×degree×T全笛卡尔积。
数据集：d30 dev D0238/D0243/D0248/D0253，test D0218/D0223/D0228/D0233；d10 dev D0236/D0241/D0246/D0251，test D0216/D0221/D0226/D0231。
dev/test来自不同生成seed；记录net_ids、indices，拒绝身份重叠；test排除pilot既用实例。每个模型共享同一组实例和候选块，不能把多模型轨迹当独立N。
局限：validation原用于checkpoint选择，test语料在历史benchmark用过；这是本次指标开发与测试的独立拆分，不宣称全新外部确认数据。当前不读取pilot数值调参数或选checkpoint。

## 冻结模型及概率几何
读取既有catalog，冻结路径、state_key、结构、seed及所有源文件size/mtime，拒绝源改动；不hash。沿用pilot已实施的forward hook observer，不修改模型。
p(j,o | greedy i,history)是条件分布，非完整联合。T1、greedy、FP32、TF32关闭，batch256末尾补齐；early/middle/late阶段事先定义。
H/logK、maxp、Q、G、C和first_Hnorm沿用pilot自然log定义；maxp原值相关，但低maxp选困难。
几何为实际L路径弧长0,1/2,1三点的正反向最小平均Manhattan距离，除输入bbox(w+h)。相同路径反向不增加距离。
每实例汇总，步数不是独立样本。无额外联合n倍推理。

## 开发选择与测试
dev仅在Q/G/C三者选一个，以各模型×族等权、固定16候选块中所选真实误差减random的平均差最大为规则；并列Q优先于G再C。写dev_gate.json后才进入test推理和读取test真值。负开发收益也不隐藏。
test主要对照为选中几何指标 vs Hnorm、低maxp、random，共d30 4模型×4族×3=48、d10 2×4×3=24。每个degree内Holm调整；d10明确次级。9999配对符号翻转双侧p，1000块bootstrap区间。
多模型/种子总体区间先按相同候选块平均模型，再按族分层重采样，避免种子伪重复。
所有指标相关、前10%筛选precision及阶段相关保留探索性；高误差阈值测试组90th分位，报告真实基率，容许ties。常量相关null。相关CI按实例bootstrap，不按步。
这只是离线固定预算困难候选筛选，非连续扰动攻击。几何指导WDRO训练须待主任务baseline忠实性与本阶段结果gate，脚本不启动训练。

## 资源及恢复
GPU默认7，显存fraction0.1，CPU/BLAS线程1。所有缓存、tmux socket归本目录。
每batch atomic NPZ、cursor、heartbeat；固定split与模型meta。重启跳过已提交分片；统计可重做，不重跑已提交推理。随机数从固定seed及dataset标识重建，无优化器。
默认磁盘小产物估计<1GB，已查主盘约21GB free。run.log保持进度，heartbeat ETA基于实际吞吐并标记不含未来统计。
启动 bash launch.sh 7；独立tmux geometry-extension-20260930；结束对话前只确认正式进程与首批产物，不等待完成。
