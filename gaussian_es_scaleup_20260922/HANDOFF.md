# 服务器交接：Gaussian-ES 扩规模实验

更新时间：2026-09-22T05:41:59.368067+00:00。

当前状态：**RUNNING**。当前已展开任务计数：{'running': 8, 'complete': 11, 'pending': 8}；完整自动工作流将展开为97项，其中49项训练（12个预训练父模型、4个独立seed901调参、33个正式后训练），另有15项搜索/预算诊断和33项统一评价。

本次已完成方案、实现、CPU/GPU工程验证、来源冻结与tmux启动，尚未完成科学实验，不能声称模型性能已经改善。

## 实测验证与时间

- CPU数学/预算/置换/调度/任务门槛测试5项通过，见 artifacts/tests/unit.log。
- 实际GPU上d20/d30/d40、K64/K256预算、真实sampled更新、双尺度GeoSteiner、greedy批形状一致性、评估入口通过，见 artifacts/tests/gpu_gate.json。
- 连续6步与3步保存后恢复至6步的Actor、Critic、Adam、lambda、RNG逐项完全一致，见 artifacts/tests/resume.json。
- 峰值reserved显存1358 MiB；每任务2个参考求解进程+1个Torch CPU线程；同时最多8 GPU任务。
- ES64 batch64 单步实测d20=5.80s、d30=11.46s、d40=19.25s；d30 ES256=48.11s。在线d30预训练约0.55–0.65s/步。
- 当前容量推算总训练约455.2 worker-hours；8卡容量下界约56.9小时，包含评价、验证与争用余量的预计范围为65–114小时，即约3–5天。这是当前吞吐外推，不是完工保证；未来资源排队时间另计。status.py会用已发生的预训练速度更新估计。

## 查看与恢复

查看状态和估时：

```bash
bash /remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/run_env.sh status.py
```

进入tmux控制器：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/tmux.sock attach -t gaussian_es_scaleup
```

控制器把日志写入 artifacts/logs/pipeline.log，tmux画面可能没有持续输出；进度以status命令、heartbeat和progress.jsonl为准。

断网不影响运行。仅当原tmux会话不存在时使用以下恢复命令；入口发现已有会话会直接返回，控制器也有flock和PID启动tick核对，能接管自己仍存活的worker，不重复启动：

```bash
bash /remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/launch.sh
```

## 自动执行与产物

1. 已冻结9组mu/step的validation搜索、Random64/CEM64对照，并自动追加K64/K256预算诊断。
2. 12个共同父模型预训练。seed901专供4个外层调参，正式degree30五种子101–105、degree20/40各101–103。
3. 只用validation选择外层参数，再启动33个配对ERM/Random64/ES64后训练任务。
4. 所有训练完成后生成model_freeze.json，才启动33个统一测试任务；每分布4096个测试net和128个CEM64压力测试中心。
5. 自动生成RESULTS.md、artifacts/metrics.csv、statistics.json、validation_curves.csv、PNG/SVG/PDF图、completion.json及来源哈希。

每64步保存完整latest；best由validation选择且包含起点。心跳5秒更新。资源不足保留断点等待；单任务错误最多重试两次，超过后标NEEDS_ATTENTION，失败证据保留，不把未完成任务标成成功。该情况下需检查错误日志后再处理，控制器不会修改科学协议绕过错误。

## 存储与来源

方案和代码：/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/。

所有产物入口：同目录artifacts。实际存储在项目NVMe路径 /attached/remote-home1/Guilin/DRO/Codex/experiments/gaussian_es_scaleup_20260922/，由Codex/shared/experiments链接接入；见artifacts/manifests/storage.json。原根盘10次fsync约3.1秒，NVMe约0.003秒，迁移仅涉及本分支新产物。

artifacts/tests/gpu_failure.json记录的是为迁移磁盘主动中断的旧工程验证；后续NVMe运行已完成并通过gpu_gate.json，旧记录保留。

来源冻结：artifacts/manifests/source_freeze.json，71个代码/配置/初始checkpoint/求解器文件。历史全库结构检查的1148个冻结原件和49283个文件身份均通过；仅README.md和summaries/README.md相对9月20日维护清单发生变化（本任务未编辑这两份文件），见layout_verification.log，不将其写成全库检查通过。

科学范围严格限于本方案：固定模型搜索更强不等于后训练更好，原始负结果与所有seed均保留，无连续Wasserstein最优或普遍优越性声明。
