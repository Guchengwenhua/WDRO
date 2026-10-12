# 三路线实验最终交付状态

**DONE**。完成核验时间：2026-09-12T14:41:15.655833+00:00。

A30/B18、正式科学审计、自动结果提取、独立交付审计和中文研究解读均已完成。全部实验进程退出，53把锁释放；本次恢复新增科研任务0、评分调用0。

- [三路线研究结论与交付](研究结论与交付.md)：方法定义、A的degree/父模型/分布拆分、B的mean/P95和独立攻击、预算轨迹及结论边界。
- [人工解读完成凭证](manifests/research_interpretation_completion.json)：绑定研究报告、预算图和全部上游PASS证据。
- [完整科学审计](manifests/all_completion.json)：A30/1080格、B18/216行，27,108项绑定。
- [独立交付审计](experiments/diagnostics/delivery_audit.json)：36,376项递归绑定；801个父源文件、新旧各810个冻结文件保护通过。
- [恢复后独立核验](experiments/diagnostics/recovery_check_20260912/recovery_delivery_verification.json)：192项任务记录和两条队列事件流未变，0重复启动。
- [预算轨迹图](experiments/interpretation/research_review/joint_budget_trajectories.png)及[预算数据凭证](experiments/interpretation/research_review/research_budget_completion.json)。

主要判断：ES在本轮固定模型、64次真实评分下增强攻击的证据最清楚；Critic效果依赖预算和分布；RL少调用及在线交替训练优势尚未建立。B没有证实新路线相对对应基线的稳定优越性。ES完整512步平均成本约为1.02951ρ，累计超额保留，不将有限实验写成Wasserstein证书。

本次恢复修复了审计层的逻辑/物理路径匹配和tmux进程识别误报。科研代码、协议、任务完成凭证和原始结果未改，失败日志与旧审计代码均保留。范围仍为冻结三路线A30/B18；没有新增ISPD实验或推进阶段9。

自动生成的 `AUTO_RESULTS.md`、`DELIVERY_STATUS.md` 和 autofinish凭证保留机器收尾时的快照，其中人工复核标志为false；后续人工解读完成状态以本页和单独人工凭证为准。不要因这些历史快照或历史failure文件再次启动任务。
