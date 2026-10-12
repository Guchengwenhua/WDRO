# 第二阶段有限扰动搜索
独立脚本 attack.py；不修改或重启诊断。
冻结模型 M0900/M1118，32中心来自stage8 attack_center（D0372/377/382/387），每分布8中心。没有借用这些中心的不存在缓存参考。
每个中心8个随机扰动（seed20260931，每坐标Uniform[-.02,.02]），统一bbox最长边缩放至0.8并平移至下界0.1，保持L1横纵相同比例。最终扰动不声称仍满足L∞=.02；输出实际mean indexed L1及max L∞。名义输入保留原始center。
原始规范最小pin L1间距1e-6；不合法候选重新采样，最多100次；不哈希、不去重改变degree。
8个扰动候选+nominal共9条轨迹，每中心一组。指标与random只在8个扰动候选内选择，nominal是额外基线且不占这8次搜索预算。与第一阶段相同模型/温度/归一化/分数规则，无从第一阶段结果选择指标或调整方向。
所有模型预测、分数、selected索引落盘后，才建立GeoSteinerEvaluator，并离线精评288实例，require_integral=True，以既有2**27 binary-scale规范查最优状态和嵌入。没有调用reference_identity/hash。
每模型每批预测、每中心参考都是原子NPZ；恢复跳过。无长期训练optimizer状态。中心重采样95%区间1000次，同中心候选从不当独立N。
比较random、各指标、oracle upper bound，输出相对random与nominal误差差值，按总体/4族。首次小样本探索，无确认性显著结论。
启动由主代理统一安排，命令 bash launch_attack.sh，独立tmux geom-attack-20260930、项目socket artifacts/attack.sock。输出artifacts/attack、ATTACK_REPORT.md。

首次运行将模型/数据清单与各源文件size/mtime冻结到inputs_metadata.json；恢复只读此冻结metadata，源发生变动即拒绝，避免拼接不同checkpoint。GeoSteiner显式使用shared/legacy_dro/WDRO-REST/algorithms/libeval.so。
