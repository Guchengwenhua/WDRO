# RL 扩大实验验收

生成时间：2026-09-26T10:02:11Z。综合状态：**INCOMPLETE**。

| 实验 | 审计任务 | 实验完整性 | 已完成产物检查 |
|---|---:|---|---|
| 20260921 首轮 | 80/80 | PASS | PASS |
| 20260922 W/T/A 扩展 | 87/104 | INCOMPLETE | PASS |

## 首轮结论

逐文件、调用预算、几何/拓扑/线长、输入隔离和检查点/优化器/RNG/λ递推重新检查；重新生成统计表，并核对原表的逐字节一致性。主检验额外用独立实现枚举符号翻转与Holm校正。

| 阶段 | 指标 | 方法/对照 | 相对改善 | Holm p | 改善seed | 结论 |
|---|---|---|---:|---:|---:|---|
| fixed | lambda5__gain | bestof_ppo/reinforce | 4.32% | 0.06250 | 7/8 | NOT_ESTABLISHED |
| coev | nominal_length | bestof_ppo/random | -0.52% | 1.00000 | 5/8 | NOT_ESTABLISHED |
| coev | random32_length | bestof_ppo/random | -0.18% | 1.00000 | 4/8 | NOT_ESTABLISHED |
| coev | es32_length | bestof_ppo/random | -0.27% | 1.00000 | 4/8 | NOT_ESTABLISHED |
| fixed | lambda5__gain | sequential_ppo/reinforce | -1.51% | 1.00000 | 2/8 | NOT_ESTABLISHED |
| coev | nominal_length | sequential_ppo/random | -2.89% | 1.00000 | 3/8 | NOT_ESTABLISHED |
| coev | random32_length | sequential_ppo/random | -3.51% | 1.00000 | 2/8 | NOT_ESTABLISHED |
| coev | es32_length | sequential_ppo/random | -3.46% | 1.00000 | 2/8 | NOT_ESTABLISHED |

正数表示改善。固定模型搜索gain更高与联合训练后布线策略更好是不同结论；未达到门槛不等于证明无效或等效。

## 扩展轮

已审计新增updates：532,480/638,976。
输入隔离：独立重建训练输入 12,582,464，评价输入 20,480，交集 0。

下表只展示8个seed齐全的比较。未齐全的比较不删seed、不补均值；全实验完成前不发布六项主检验的最终优势判定。

| 比较 | 方法 | 可用seed | 相对改善 | 描述性95%区间（绝对改善） | 判定 |
|---|---|---:|---:|---|---|
| post_only | bestof_ppo | 8/8 | 7.49% | [0.165691, 0.480694] | PENDING_FULL_EXPERIMENT |
| pre_only_at_long_post | bestof_ppo | 8/8 | 1.84% | [0.030268, 0.125417] | PENDING_FULL_EXPERIMENT |
| ppo_vs_random_at_long_pre_long_post | bestof_ppo | 8/8 | -0.02% | [-0.014583, 0.009555] | PENDING_FULL_EXPERIMENT |
| post_only | sequential_ppo | 8/8 | 10.50% | [0.264160, 0.664147] | PENDING_FULL_EXPERIMENT |
| pre_only_at_long_post | sequential_ppo | 7/8 | 待齐全 | 待齐全 | PENDING_FULL_EXPERIMENT |
| ppo_vs_random_at_long_pre_long_post | sequential_ppo | 7/8 | 待齐全 | 待齐全 | PENDING_FULL_EXPERIMENT |

各方法的完整8-seed鲁棒线长均值（越小越好）：

| W | 方法 | T=2048 | T=4096 | T=8192 | 后训练延长的相对改善 |
|---:|---|---:|---:|---:|---:|
| 2048 | erm | 4.323408 | 4.015729 | 3.926862 | 9.17% |
| 2048 | random | 4.188891 | 3.957229 | 3.871299 | 7.58% |
| 2048 | bestof_ppo | 4.213622 | 3.989286 | 3.897896 | 7.49% |
| 2048 | sequential_ppo | 4.331834 | 3.979636 | 3.876894 | 10.50% |
| 8192 | erm | 3.919578 | 3.896411 | 3.884631 | 0.89% |
| 8192 | random | 3.912313 | 3.895725 | 3.825212 | 2.23% |
| 8192 | bestof_ppo | 3.908606 | 3.887758 | 3.826085 | 2.11% |

固定模型上的攻击器延长训练（λ=5，次要描述性比较）：

| 方法 | 可用seed | gain相对改善 | 描述性95%区间（gain绝对差） |
|---|---:|---:|---|
| bestof_ppo | 0/8 | 待齐全 | 待齐全 |
| sequential_ppo | 0/8 | 待齐全 | 待齐全 |

尚未完成：

- post_pre8192_sequential_ppo_s308
- attacker8192_bestof_ppo_s301
- attacker8192_sequential_ppo_s301
- attacker8192_bestof_ppo_s302
- attacker8192_sequential_ppo_s302
- attacker8192_bestof_ppo_s303
- attacker8192_sequential_ppo_s303
- attacker8192_bestof_ppo_s304
- attacker8192_sequential_ppo_s304
- attacker8192_bestof_ppo_s305
- attacker8192_sequential_ppo_s305
- attacker8192_bestof_ppo_s306
- attacker8192_sequential_ppo_s306
- attacker8192_bestof_ppo_s307
- attacker8192_sequential_ppo_s307
- attacker8192_bestof_ppo_s308
- attacker8192_sequential_ppo_s308

## 存储、恢复与验收范围

附加盘初次复制校验：PASS。初次复制记录保存在storage_migration.json；最终采用混合存储，保留原DRO/Codex访问路径。
- 初次复制 `/remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922/artifacts/runs` 至 `/attached/remote-home1/Guilin/DRO/Codex/rl_prepost_scaleup_20260922/artifacts/runs`；1699个文件，12.56 GiB。
- 初次复制 `/remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922/lineage` 至 `/attached/remote-home1/Guilin/DRO/Codex/rl_prepost_scaleup_20260922/lineage`；224个文件，1.01 GiB。
- 初次复制 `/remote-home/Guilin/DRO/Codex/rl_scaleup_20260921/artifacts/runs` 至 `/attached/remote-home1/Guilin/DRO/Codex/rl_scaleup_20260921/artifacts/runs`；712个文件，3.80 GiB。

恢复原权重路径检查后的布局：PASS。约0.405 GiB必需检查点及小文件保留主盘，其余完成档案和大型评价数组保留附加盘；剩余17项任务的输出检查点也留在本实验根目录内。
原始权重检查实测：PASS，加载18个必需父检查点，外部权重仍被拒绝。
直接链接整个目录曾触发权重路径检查，相关任务在初始化阶段退出，未进入训练更新；这些退出记录保留，修复采用文件布局调整，没有放宽检查或修改冻结代码。

迁移未改变训练源码、协议、步数、seed、选择规则和GPU预算；使用原tmux队列续跑。验收脚本本身只读训练产物。
实验结论限定为degree30、8个训练seed、自身布线长度和有限候选攻击；不是最优性差距或连续Wasserstein鲁棒性认证。
首轮重算产物：`../rl_scaleup_20260921/artifacts/acceptance_20260926/`。本目录old/new记录本次首轮及扩展子集审计；final/new记录扩展完成后的自动终验。
