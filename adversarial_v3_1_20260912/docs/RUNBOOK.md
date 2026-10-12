# 对抗生成框架 v3：运行与恢复

> 2026-09-12 14:33 UTC 恢复检查：A30/B18 全部科学审计、自动提取、独立交付与 autofinish 已 PASS；当前没有待恢复的科研任务。见 `shared/experiments/diagnostics/recovery_check_20260912/recovery_delivery_verification.json`。以下原运行命令保留为历史恢复手册，不能据此重复启动。人工科学解读状态以 `研究结论与交付.md` 及最终交付凭证为准。

本版本目录：`/remote-home/Guilin/DRO/Codex/adversarial_v3_1_20260912`。正式流程依次执行 A 固定模型 30 任务、A 审计、B 交替训练 18 任务和最终报告。协议为 [../protocol.json](../protocol.json)，科学代码与参数已绑定到任务identity；操作性监控不会改变模型或实验参数。

## 查看进度

在版本目录执行下列只读命令：

```bash
cd /remote-home/Guilin/DRO/Codex/adversarial_v3_1_20260912
cat STATUS.md
cat experiments/pipeline/state.json
cat experiments/queue/fixed/heartbeat.json
tmux -S experiments/tmp/adv_v3.sock list-sessions
tail -n 60 experiments/logs/pipeline_driver.log
```

正式主会话是 `adv_v3_pipeline`，使用本版本独立socket `shared/experiments/tmp/adv_v3.sock`。当前监控运行在此会话的 `monitor` 窗口，标准输出为 `shared/experiments/logs/monitor.log`。查看会话：

```bash
tmux -S experiments/tmp/adv_v3.sock attach-session -t adv_v3_pipeline
```

主进程将输出重定向到日志，tmux窗格可能没有文本。按 `Ctrl-b d` 只分离终端，任务继续运行。

```bash
tmux -S experiments/tmp/adv_v3.sock list-windows -t adv_v3_pipeline
tail -n 20 experiments/logs/monitor.log
```

`shared/experiments/pipeline/events.jsonl` 记录各阶段命令、退出码和对应日志路径；`shared/experiments/queue/{fixed,joint}/events.jsonl` 记录每个任务的启动、GPU、PID、尝试次数和日志路径。具体任务状态位于 `shared/experiments/{fixed,joint}/任务ID/{launch,heartbeat,failure,complete}.json`；实际评分账目在同任务的 `oracle/oracle_ledger.json`。例如：

```bash
tail -n 8 experiments/queue/fixed/events.jsonl
cat experiments/fixed/erm_d50_s101/heartbeat.json
cat experiments/fixed/erm_d50_s101/oracle/oracle_ledger.json
tail -n 5 experiments/fixed/erm_d50_s101/rl/training.jsonl
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
```

`STATUS.md` 每60秒刷新，`shared/experiments/monitor/samples.jsonl` 每15秒记录只读采样。吞吐使用相邻采样的实际完成计数差，单位为net/s；首次采样没有速率。完成数要求task completion为pass且进程退出；监控不会替代最终逐文件hash审计。

## 启动或恢复监控

监控只读取实验文件与GPU状态，写入自身的 `shared/experiments/monitor` 和 `STATUS.md`。其独占锁阻止重复监控，不操作任务锁或worker。

```bash
bash run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python -B -m adv_v3.monitor --once
```

当前已有常驻监控；不要重复启动。只有确认 `shared/experiments/monitor/launch.json` 对应PID已退出，才需要恢复它。下面是可选的独立监控恢复会话，和当前 `adv_v3_pipeline:monitor` 窗口的部署方式不同：

```bash
tmux -S experiments/tmp/adv_v3.sock new-session -d -s adv_v3_monitor -c /remote-home/Guilin/DRO/Codex/adversarial_v3_1_20260912 'bash run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B -m adv_v3.monitor > experiments/monitor/driver.log 2>&1'
```

监控遇到pipeline完成，或pipeline失败且控制PID退出，会保存最后状态后结束；自身退出记录为 `shared/experiments/monitor/exit.json`。worker不会被它终止。

## 网络中断后的检查

先恢复SSH并查看tmux、pipeline/queue的launch与heartbeat。不要根据终端断连重复启动任务，也不要仅用 `kill -0` 判断任务仍有效。以下检查同时验证PID启动时间和僵尸状态：

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python -B - <<'PY'
import json
from pathlib import Path
for name in ('pipeline', 'queue/fixed', 'queue/joint', 'monitor'):
    path=Path('experiments')/name/'launch.json'
    if not path.exists():
        continue
    launch=json.loads(path.read_text())
    proc=Path(f"/proc/{launch['pid']}/stat")
    if not proc.exists():
        print(name, launch['pid'], 'exited')
        continue
    raw=proc.read_text(); fields=raw[raw.rfind(')')+2:].split()
    same=fields[19] == str(launch['process_start_ticks'])
    print(name, launch['pid'], 'alive' if same and fields[0] not in ('Z','X') else 'exited_or_reused', fields[0])
PY
```

若pipeline与queue进程仍有效，保持原任务运行即可。控制器异常退出时，worker可能继续持锁运行；queue恢复时会根据launch PID与启动时间接管这些worker。先保留 `failure.json`、队列事件、日志和checkpoint，确认失败原因；代码或协议identity不一致必须修复来源，不能删除记录后混合续跑。

## 恢复正式pipeline

仅在确认原pipeline控制PID已退出、原queue控制PID也已退出后恢复。现存worker由queue接管，不应单独重复启动。若旧tmux会话仍显示为存活，先检查它对应的真实进程，再决定恢复位置；不运行 `kill-server` 或删除锁文件。

当前正式命令固定为：A并发上限16、B并发上限8、准入峰值2500MiB；调度仍受实时GPU/CPU/磁盘准入限制。确认可恢复后，新建独立恢复会话并使用新日志名：

```bash
tmux -S experiments/tmp/adv_v3.sock new-session -d -s adv_v3_pipeline_resume -c /remote-home/Guilin/DRO/Codex/adversarial_v3_1_20260912 'bash run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python -u -m adv_v3.pipeline --fixed-concurrency 16 --joint-concurrency 8 --peak-mib 2500 --share-busy-gpus > experiments/logs/pipeline_resume_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1'
```

pipeline会重新检查阶段，queue会校验已有完成产物并跳过有效任务。A学习器分别从自己的latest恢复；B每个外层update保存Actor、主Critic、攻击器/代理、全部优化器、RNG和λ。真实Oracle账目持续累加，恢复不回滚已发生调用。临时 `.partial` 文件不算完成产物，不手工改名为完成文件。

## 完成交付与诊断

正式完成应同时具有 `shared/experiments/pipeline/complete.json`、两阶段queue completion、全部30/18 task completion和最终report产物。不要用GPU空闲、单个smoke通过或监控显示一个任务pass替代整轮验收。

已完成的真实GPU恢复诊断位于 [joint_gpu_recovery/comparison.json](../experiments/diagnostics/joint_gpu_recovery/comparison.json)；其中连续4步与独立进程2+2步的模型、主优化器、RNG、λ和真实调用账目精确一致。它验证恢复实现，不代替正式512步实验。

本轮资源调度在2026-09-12增加 `--share-busy-gpus`：允许在计算繁忙但显存有容量的GPU上分时共享；仍保留显存实测峰值、3072MiB额外余量、所有已准入任务未来峰值预留与CPU worker预算28。当时尝试仅重启控制器并接管现有worker，但实际发生了下段记录的会话退出，随后由queue从检查点恢复；不能将这次操作记为无中断接管。实验参数和科学代码未变；修改前操作代码存放operations/admission_share_20260912。

2026-09-12资源控制器重启时发现旧worker继承tmux窗格会话，窗格关闭会导致worker退出；本次已自动从既有checkpoint恢复，新旧PID不重叠。操作代码已补充start_new_session=True，后续新控制器和任务采用独立session。当前A控制器在此修改前已载入，将保持运行至结束；不要为应用隔离改动再次重启它。若必须恢复，仍先核对PID/锁/检查点，并把重复已评分但未提交分块的计算记为恢复开销。证据operations/worker_session_isolation_20260912。

## 无人值守收尾（2026-09-12 09:54 UTC 已上线）

用户要求剩余流程自主运行后结束对话。现有 pipeline/queue 未重启；同一独立 socket 中增加 `adv_v3_autofinish` 会话，执行：

```bash
bash run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python -u diagnostics/autofinish.py --poll-seconds 15
```

它仅在完整 pipeline PASS 且原控制器/monitor 退出后，执行结果提取与独立交付审计。自动状态位于 `DELIVERY_STATUS.md` 和 `shared/experiments/autofinish/state.json`；最终自动摘要为 `AUTO_RESULTS.md`，完成凭证为 `shared/experiments/autofinish/complete.json`。错误写入 `failure.json` 并停止，不改参数或重启科研流程。人工中文科学解读保留为待下次复核。

6 项收尾程序测试通过；上线 PID/启动时间、心跳、防重复锁、协议 hash 及原控制器未重启证据位于 `shared/experiments/diagnostics/autonomy_handoff_verification.json`。下次先读 [无人值守恢复交接](无人值守_恢复交接.md)，不要重复启动现有 finalizer。


## 2026-09-12 最终审计恢复记录

13:44 UTC 的 final_audit 失败来自逻辑路径/附加盘物理路径的字符串不一致：B 输入其实已绑定。仅修复 `adv_v3/report.py` 的路径规范化，哈希、缺失文件与重复绑定检查保留；7项新增回归及5项既有报告测试通过。48个已完成科研任务被原queue验证后跳过，未重训或重新评分。

恢复后独立交付检查曾把tmux保留的pipeline命令文本误报为活动科研进程。`diagnostics/delivery_audit.py` 改为检查实际可执行文件和结构化Python入口，保留launch PID/启动时间、退出和锁门禁；9项新增回归及7项既有自测通过。仅重开autofinish的交付步骤，复用已PASS的提取表。

失败控制记录、旧源码和日志均归档到 `shared/experiments/diagnostics/recovery_check_20260912/`；路径修复原件与测试另见 `operations/report_path_binding_20260912/`。最终核验程序为 `diagnostics/recovery_closeout_check.py`，证明192项任务记录及两条队列事件流未变、0新增科研worker、0新增评分调用，并复核自动产物绑定和全部进程/锁释放。

自动生成的 `AUTO_RESULTS.md`、`DELIVERY_STATUS.md`、`shared/experiments/autofinish/complete.json` 是机器收尾时的不可变快照，其中人工解读标志仍为false；人工解读的后续完成应由单独凭证记录，避免改动这些已绑定文件。
