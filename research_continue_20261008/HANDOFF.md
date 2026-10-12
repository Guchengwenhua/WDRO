# 后续训练启动交接

任务：d30模型库补齐与长预算对照；d10几何熵、普通熵、原始和随机筛选四臂训练。41项队列任务（26训练、14独立评估、1最终汇总），另有只读CPU相关性监测。

tmux会话：research-continue-20261008；socket：artifacts/parallel.sock；恢复入口：bash launch.sh。运行时保持当前会话，勿重复启动控制器。

方案见PROTOCOL.md，冻结参数见protocol.json/jobs.json。所有产物位于附加盘，项目artifacts符号链接提供访问；旧失败与已完成实验保持原样。

|任务|物理GPU|已完成更新|目标|batch|
|---|---:|---|---:|---:|
|geometry_geometry_s101|5|1|125|4096|
|geometry_entropy_s101|4|16|125|4096|
|d30_geo125_s103|7|3|125|4096|
|d30_lagged3125_s103|6|11|125|4096|
|geometry_geometry_s102|3|1|125|4096|
|geometry_original_s101|2|18|125|4096|

剩余时间：普通熵和原始筛选按当前正式日志估计约40–50分钟/当前运行；d30恢复阶段约1–4小时。bank后续长预算根据上轮125更新耗时约2.2–3.7小时，完整625更新对应约11–18小时/运行；本轮全部bank新增工作约百余GPU进程小时，六卡理想并行也需约一天，实际取决于资源和依赖。几何熵两个种子均已完成首个更新，首轮约331–334秒，按此粗估各剩约11.4–11.5小时；单步估计尚不稳定。整批计划暂估1–3天，资源等待可能延长。初期估计含启动成本且共享GPU负载会变化。

日志：artifacts/controller.log和artifacts/logs/；状态：artifacts/state.json；相关性：artifacts/geometry_monitor/REPORT.md；全部任务终结后自动汇总到artifacts/REPORT.md。详尽PID、断点和实际batch见LAUNCH_RECEIPT.json。
