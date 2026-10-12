# 2026-09-16 独立完整性与退出验收

**PASS。未发现科学产物、源码、父模型或动态导航的hash漂移；未启动训练、推理或GeoSteiner评价。**

## 绑定与任务

- 实际回读6,520个canonical文件，对应13,140条manifest绑定引用；不匹配0。
- 覆盖原completion_audit的6,493项审计产物和19项发布输出、freeze的12项、automation的4项及final_delivery的4项，并扩展到16任务各自complete/identity。
- 16个正式任务齐全：1个固定模型validation、3个固定模型test、12个128步继续训练。父Actor/WDRO checkpoint与既有冻结父模型索引匹配。
- 动态docs/STATUS独立分类，当前也匹配原完成绑定。

## 实际预算和退出

- greedy Actor、REST、reference请求及完成均887,296；sampled Actor/REST调用24,576。
- GeoSteiner cache miss开始/完成均768,368；Actor实际前向行1,346,560，其中padding459,264。
- 所有任务实际调用恰好等于本轮已提交预算；16个只读SQLite quick_check均通过，SQLite calls与持久账本一致。
- 16个worker及scheduler记录PID均不存在；autofinish PID也不存在。18把task/queue/autofinish锁均已释放。
- 专用tmux socket没有server；没有记录PID占用GPU。额外按脚本路径/cwd扫描，未发现残留研究worker或multiprocessing子进程。
- 原completion_audit记录scheduler仍活跃是统计程序运行时快照，final_delivery和本次live复核证明其随后退出，二者不冲突。

## 扩展数据隔离

使用float32坐标、按x/y字典序排列pin、SHA-256逐网哈希，纳入3,905个NPZ数组，共875,904行几何。

|角色|累计读取行|唯一net|
|---|---:|---:|
|train|129,024|83,424|
|validation|193,152|165,216|
|test|553,728|484,112|

新增覆盖原隔离审计未完整纳入的固定搜索support：validation188,416点，test540,672点。还包含joint所有输入、全部已评分训练候选、代理两个训练pool和shared validation。三个跨角色交集均为0；同角色不同方法/seed重复中心允许。

## 复核范围

本目录只复核字节绑定、来源、正式任务覆盖、实际调用账本、几何隔离及live资源退出；没有重新计算性能统计或重新求解拓扑/参考长度。统计独立复算由本轮验收的其他模块负责。

## 文件

- `integrity_report.json`：总判定、来源控制hash、live进程/锁/tmux和Oracle计数。
- `binding_checks.json`：全部13,140条绑定及实际SHA；`mismatches.json`为空列表。
- `tasks_and_ledgers.json`：16任务预算和SQLite复核。
- `expanded_data_isolation.json`：全部几何来源清单、计数和交集。
- `remaining_processes.json`：补充的worker/多进程子进程扫描。
- 三份Python脚本及同名日志可复现本次复核，输出限定在本目录。
