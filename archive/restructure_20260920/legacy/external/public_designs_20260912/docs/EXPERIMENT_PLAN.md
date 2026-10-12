# 公开芯片物理设计外部测试扩展

授权日期：2026-09-12。所有新增文件位于本目录。原阶段0–8的冻结配置、模型、结果保持为独立历史实验；本目录为新公开数据的零样本外部测试。

1. 使用有效 reference_precision_v2 的75个冻结模型：ERM、Mixed ERM、Hard Mining、GroupDRO、WDRO，各degree 10/20/30/40/50，seed 101/102/103。模型选择来源仍为合成validation，不训练、不调参。
2. 从官方竞赛网站和作者公开仓库下载已有物理布局、逐net pin坐标与连接。每个源记录URL、时间、SHA256、格式、设计身份和许可说明。已有ISPD18/19不重复算作新增设计；ISPD07/08重叠设计只纳入一次。
3. 保留每条原始网络的全部端点。仅原始degree属于模型支持集合的net进入几何校验，重复XY导致degree变化按既有契约拒绝；缺失连接/坐标整条拒绝，不截断、不补点。报告全部degree计数和无效原因。没有随机抽样或根据模型结果选数据。
4. 复用冻结bbox最长边等比例归一化到[0.1,0.9]，输入为float32，记录offset和scale，校验inverse。3D竞赛输入投影为2D XY，层/障碍/容量不进入本研究的RSMT指标。
5. 所有合法nets使用precision_v2 GeoSteiner，全部分别在2^27和2^28尺度求解，要求optimal状态、整数坐标、连通拓扑、长度自洽和双尺度严格相等。CPU多进程计算，可按原子分片恢复。
6. GPU统一batch256、T=1、greedy。动态检查所有卡，在剩余显存和计算余量允许时并发；已满载的共享任务不被终止。提高有效吞吐，显存仅按模型实际需要分配。
7. 保留所有逐net预测及拓扑，独立Python拓扑/线长复算，每个任务验证data/model/reference SHA256。失败与重试单独留档。
8. 分数据集、设计、degree、seed报告mean/P95、net-weighted mean、design-macro mean、worst-design mean。整体指标先在每seed汇总再对3seed等权平均，不先把3seed逐net预测平均后取P95。不同技术节点的相同逻辑设计保留design-family标识，整体汇总仅作描述。
9. WDRO与四基线配对差值，以设计/设计家族为重采样单位给mean的95% bootstrap区间（2000次，固定seed20260912）；报告原始分seed差值。网络相关性通过设计聚类处理；测试中的有限设计集合不是对任意芯片分布的保证。
10. 验收需来源/输入/reference/model/output绑定一致、75任务完整、无缺失预测、无负regret异常、独立拓扑核验、统计复算、幂等恢复检查、退出进程及释放锁。归档可恢复命令和完整结果后更新根目录项目进度。

本测试测量设计衍生的二维无障碍rectilinear Steiner tree相对regret；不是全芯片布线竞赛评分，也不等同于签核PPA。开源芯片的研究工艺实现不能一概称为已流片商用芯片。
