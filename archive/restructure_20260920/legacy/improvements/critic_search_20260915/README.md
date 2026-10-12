# tmp2 七方向研究：2026-09-15

本目录落实根目录 `tmp2` 的七个候选方向。实验状态见 [STATUS.md](STATUS.md)，原始结果见 [RESULTS.md](RESULTS.md) 及其完成清单。

**2026-09-16 独立验收已完成：** [统一验收报告](artifacts/review_20260916/REPORT.md)。执行、完整性和数值复算通过，仅方向3达到本轮改善判据。报告同时更正完整统计清单的冻结时序声明，方向7保留探索性解释；引用原结果时应一并采用该更正。

## 理论与范围

- [需求原文](REQUEST_tmp2.md)
- [方向1、2：搜索区域数据与代理目标](THEORY_01_02.md)
- [方向3–6：步长、起点、轨迹和合法化](THEORY_03_06.md)
- [方向7：训练中刷新与λ节奏](THEORY_07.md)
- [冻结协议](protocol.json)，[配置与源码快照](artifacts/freeze.json)

本轮是 degree30、三个训练seed的独立机制实验。复用precision_v2 ERM Actor和v3.1校准代理；主WDRO和历史v3.1产物保留。
新数据命名空间独立；合成validation选参数；最终合成test确认；本轮不使用外部芯片数据调参。
固定模型每中心64次真实评分，代理额外标签和梯度计算另记；方向7为128步、batch16、4真实评分的配对继续训练。

## 文件说明

| 文件 | 内容 |
|---|---|
| bootstrap.py | 只读复用父分支模块 |
| search.py | 方向诊断和方向3–6提案/真实评分 |
| surrogate.py | 方向1、2数据池、损失和代理训练 |
| fixed.py | validation及三个seed独立测试 |
| training.py | 方向7的2×2继续训练 |
| scheduler.py | 动态GPU准入、每卡一任务、持久队列 |
| report.py | 独立数值复算、配对统计和完成审计 |
| autofinish.py / closeout.py | 服务器端自动等待、验收收尾、进程与锁退出核对 |
| tests/ | 预算、几何、损失、训练恢复等检查 |
| artifacts/ | 输入、checkpoint、日志、逐候选真实结果、统计 |

`artifacts` 逻辑路径在 `DRO/Codex` 下，实际解析到已有附加盘
`/attached/remote-home1/Guilin/DRO/Codex/experiments/critic_search_20260915`。
不向旧主线目录写新实验结果。

## 运行与恢复

```bash
cd /remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915
# 查看任务列表、当前GPU和失败信息
cat artifacts/operations/queue.json
# 查看持久会话
tmux -S artifacts/operations/tmux.sock list-sessions
# 自动收尾状态：complete 表示最终验收完成，failed_* 表示需要检查对应日志
cat artifacts/operations/autofinish_status.json
# 在确认旧scheduler和worker均退出后，按相同协议恢复
bash run_env.sh scheduler.py
```

调度器拒绝重复启动；worker使用文件锁和输入/配置/源码hash。每步训练保存恢复状态，固定攻击按16中心分块保存。
`tmp2_research` 负责队列和统计；`tmp2_autofinish` 等待它退出后核验最终状态并发布 `artifacts/audit/final_delivery.json`。
正常完成不需要当前对话保持连接。失败会保留日志及 `failed_*` 状态，不会自动宣称DONE。
若scheduler异常退出且worker尚活着，不重复启动，先按 `launch.json` 的PID及 `/proc/PID/stat` 启动时间核对。
实际评分账本不随checkpoint回退；中断后额外调用作为恢复开销保留。
每GPU最多一任务，单进程显存分配上限10%，启动要求至少6GiB空余，每worker三个精确参考进程。
利用率满载时允许有界共享，但不终止其他进程，不预分配占满剩余显存。

## 解读原则

较高的真实惩罚分数说明有限预算搜索更强；不能直接推出训练后的模型更好。
方向7分别报告初始、validation选优模型和最终更新模型；初始也可被validation选择，故需同时看训练动态。
多方向探索使用配对CI并报告多重比较边界。三个seed不足以证明广泛可重复性，degree30不能推广到所有degree。
所有不利方向/seed/分布照常报告，smoke不计科学验收。
