from analyze_results import ROOT,OUT,read
import numpy as np
from datetime import datetime,timezone,timedelta

b=read(OUT/'bank_analysis.json');g=read(OUT/'geometry_analysis.json');s=read(OUT/'supplement.json');inv=read(OUT/'inventory.json')
def link(name,path):return f'[{name}]({path})'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def metric(arm,kind):return [r for r in b['metrics'] if r['arm']==arm and r['checkpoint']==kind]
def contrast(arm,kind,seed):return next(r for r in b['contrasts'] if r['arm']==arm and r['checkpoint']==kind and r['seed']==str(seed))
def ci(r):return f"{r['delta_pp']:+.6f} [{r['ci_low_pp']:+.6f}, {r['ci_high_pp']:+.6f}]"
text=[]
text.append('国庆两条实验的验收报告\n\n截至 '+datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M UTC+8')+'。')
text.append('已完成的数据支持两个有限结论：checkpoint bank 在 d10、2000 更新的既定配置中降低了独立参考误差和本次运行耗时；几何指标 G 在冻结候选池中更能筛出高误差样本。整周扩展未完成：总控制器异常退出，后续 45 项 bank 作业和 21 项几何作业全部未启动。因此没有 10000 更新、d30 bank 训练、全新受约束扰动筛选或几何指导 WDRO 训练的结果。')
text.append(table(['实验阶段','实际完成的数据','失败或未完成对结论的影响'],[
 ['Bank 初步续训','5 方法 × 2 种子 × 200 更新；10/10 完成，128000 sample-updates','所有 final200 均劣于初始父模型；refresh3 没有实际更换成员'],
 ['Bank 滚动续训','2/2 完成，各 200 更新；每种子实际换库 3 次','有换库机制数据，但没有相对 Geo 的稳定显著优势'],
 ['Bank 正式 d10','3 方法 × 3 种子 × 2000 更新；9/9 完成，4608000 sample-updates','仅 d10；更长训练与 d30 没有结果'],
 ['几何初步诊断','2 模型 × 4 分布 × 512，共 4096 轨迹；8/8 单元完成','探索性多指标比较，不能作训练收益结论'],
 ['几何有限扰动','32 中心 × 9 个输入，288 个新参考；2 模型共 576 轨迹','归一化后扰动可能超原始半径；非既定半径的 WDRO 攻击'],
 ['几何开发/测试扩展','d30 49152 + d10 12288 = 61440 轨迹，2/2 阶段完成','固定模型离线筛选；无几何指导训练'],
 ['整周扩展','Bank 0/45；几何 0/21 作业启动','控制器失败，不是实验负门控，也不是 66 个训练失败']]))
text.append('上述 66 项包含训练、评估、门控和汇总，不能当作 66 次独立训练。原清单的 63 项 skipped_dependency、3 项 not_run 是旧快照；上游正式 bank 和几何扩展实际均已完成。完整未运行清单见 '+link('unrun_jobs.csv',OUT/'unrun_jobs.csv')+'，运行失败详情见 '+link('运行状态回执',OUT/'OPERATIONS.md')+'。')
text.append('Bank 的主指标使用 r=L/L*−1，L* 为 precision_v2 缓存的 geosteiner_exact_binary_scale_v2 参考，与 bank 代理分母分离。每个模型使用相同的 4 个分布 × 10000 个实例，共 40000 个测试 ID；9 模型 × latest/best 共保存 720000 个误差记录，但仍只有 40000 个独特测试实例。保存的 reference 与原始缓存一致；由 length/reference−1 重算的误差及组均值与原记录一致。这里复用已生成的线路长度与参考，没有新增模型推理或重新求解 GeoSteiner。')
text.append('正式训练从相同的 legacy actor 初始化，每种子使用匹配的新 critic/Adam，batch256、2000 更新；32 个扰动加 nominal，固定 shortlist 后精评 4 个候选。Geo 使用精确参考；K3 使用历史验证集选出的固定 3 个 teacher；lagged3 使用固定历史锚点及最近 2 个窗口快照。主比较为 final2000，验证集选出的 best 单独作为次级结果。这里“独立参考”指评价分母独立于 bank，不表示全新盲测：同一历史测试池已被既有研究使用，本次正式测试也包含初步 bank 已评估的每族前256个实例，共1024/40000（2.56%）。')
rows=[]
for arm in ['geo','k3','lagged3']:
    r=metric(arm,'latest');means=[x['mean_pct'] for x in r]
    rows.append([arm,f'{np.mean(means):.6f} ± {np.std(means,ddof=1):.6f}',f"{np.mean([x['p95_pct'] for x in r]):.6f}",f"{np.mean([x['worst_ood_pct'] for x in r]):.6f}",'参考' if arm=='geo' else ci(contrast(arm,'latest','mean_of_3'))])
text.append(table(['方法','平均真实误差 % ± 种子SD','平均模型P95 %','平均最差OOD组误差 %','相对Geo差值 pp [95%条件区间]'],rows))
text.append('固定 K3 的平均误差下降 0.009662 个百分点，相对 Geo 的误差降幅为 4.33%；lagged3 下降 0.007955 个百分点，相对降幅为 3.57%。最差 OOD 指 mix_gauss、edge_bias、grid_cluster 中的最大组均值；表中先对每个模型求最差组或 P95，再对三个种子求均值，不是将不同模型样本混合后的分位数。')
rows=[]
for seed in [101,102,103]:
    for arm in ['geo','k3','lagged3']:
        r=next(x for x in metric(arm,'latest') if x['seed']==seed)
        rows.append([seed,arm,f"{r['mean_pct']:.6f}",f"{r['worst_ood_pct']:.6f}",'参考' if arm=='geo' else ci(contrast(arm,'latest',seed))])
text.append(table(['种子','方法','final2000误差 %','最差OOD %','相对Geo差值 pp [95%条件区间]'],rows))
text.append('K3 三个种子的实例条件区间均低于 0；lagged3 的 seed101 区间跨 0，另外两个低于 0。区间为本次根据已保存结果计算的 2000 次按固定分布分层、同 ID 配对的百分位 bootstrap。三个种子的总体区间先在同 ID 上平均模型差值，再重采样实例；它描述这三个已训练模型的测试实例不确定性，不是训练种子总体的置信区间。只有 3 个训练种子；这些 bank 区间没有多重比较校正，也没有预先规定等效/非劣界值。因此应称“此配置下的正向数值证据”，不能称统计等效或跨种子总体的普遍优越性。')
rows=[]
for arm in ['geo','k3','lagged3']:
    r=metric(arm,'best');rows.append([arm,', '.join(str(x['update']) for x in r),f"{np.mean([x['mean_pct'] for x in r]):.6f}",f"{np.mean([x['worst_ood_pct'] for x in r]):.6f}",'参考' if arm=='geo' else ci(contrast(arm,'best','mean_of_3'))])
text.append(table(['方法','best更新数，种子101/102/103','best平均误差 %','best平均最差OOD %','相对Geo-best差值 pp [95%条件区间]'],rows))
text.append('best 结果同样有正向均值，但不同方法被选中的训练步不同，不能替代等更新预算的 final2000 主比较。final 的固定 K3 均值优于 lagged3；best 的顺序相反，现有结果不足以认定滚动换库优于固定 K3。完整的各分布均值、P95 和各 seed/best 差值保存在 '+link('bank_metrics.csv',OUT/'bank_metrics.csv')+' 与 '+link('bank_contrasts.csv',OUT/'bank_contrasts.csv')+'。')
text.append('WDRO 机制有明确限制：固定 K3 和 lagged3 各自的全部 6000 次更新后 λ 都是 0；Geo 的三个种子分别有 99.95%、100%、99.90% 的更新后 λ 为 0，最大 λ 仅 0.000962。平均所选 Hungarian L1 cost，Geo 约 0.02385，K3 约 0.01809，lagged3 约 0.01514，均低于 ρ=0.029607。故运行了原软对偶更新公式，但约束在这些轨迹中基本不活跃；这批收益不能归因于持续生效的 Wasserstein 惩罚，也没有训练后独立攻击的鲁棒性数据。')
text.append('Bank 不是精确参考的无偏替代。若 B=min_j L_j≥L*，则 L/B−1≤L/L*−1，代理误差向下偏，也会改变候选排名。初步实验的独立 32 中心 × 9 候选诊断中，K1/K3/K5 的平均 bank gap 分别为 0.234915%/0.059461%/0.052715%，代理减真实误差分别为 −0.234915/−0.059557/−0.052761 个百分点，argmax 一致率为 21.875%/87.5%/87.5%。这些是初步实验 best 模型的诊断，不能冒充正式 final2000 的参考偏差测量。正式 K3 的负训练目标比例为 2.506%–2.553%，lagged3 为 3.442%–3.488%；负目标保留，critic 的 ReLU 未改变。')
text.append('早期负结果仍成立，不能被正式阶段覆盖：5 方法 × 2 种子的所有验证 best 均为未更新父模型 step0，平均测试误差 0.234418%；全部 final200 更差。K1/K3/K5/refresh3 相对同期 Geo 的两种子平均差分别为 −0.005370/+0.002122/+0.043780/+0.002122 个百分点；K1 两种子反向，K5 更差。refresh3 每个种子只有一组成员，实际未换库；后续独立 lagged3 才每种子换库 3 次，其误差为 0.302789%/0.295187%，相对 Geo 的区间仍跨 0。正式研究恢复原始初始化、batch256 和 2000 更新，不能把两阶段当成同一设置下的重复。原始逐项结果见 '+link('pilot_metrics.csv',OUT/'pilot_metrics.csv')+'。')
text.append('运行成本按本次共享服务器实际作业 wall time 比较，包括训练、周期验证、换库及最终 latest/best 测试；不等于独占 GPU 时或能耗。')
times={arm:sum(x['seconds'] for x in b['queue']['finished'] if x['job'].startswith(arm+'_')) for arm in ['geo','k3','lagged3']}
rows=[]
for arm in times:
    ts=[v for k,v in b['training'].items() if k.startswith(arm+'_')];acc=[v['final_accounting'] for v in ts]
    rows.append([arm,f'{times[arm]:.2f}',f'{times[arm]/3600:.3f}',f'{100*(1-times[arm]/times["geo"]):.2f}%',f"{sum(x['exact_train_nets'] for x in acc):,}",f"{sum(x['teacher_forward_nets'] for x in acc):,}",f"{sum(x['teacher_seconds'] for x in acc):.2f}",f"{sum(x['refresh_seconds'] for x in acc):.2f}"])
text.append(table(['方法，3种子合计','作业秒','小时','比Geo少耗时','在线Geo求解net数','teacher前向net数','teacher秒','换库秒'],rows))
text.append('九个正式作业合计 22991.97 秒，即 6.387 小时；单作业 Geo 为 2601.58–3378.45 秒，K3 为 2343.31–2484.11 秒，lagged3 为 2352.75–2407.08 秒。Geo 最早作业明显较慢，且本协议禁用了标量精确参考缓存；因此 16.13%/17.26% 是本次队列 wall time 降幅，不能推广为相对缓存优化 Geo 的固定加速比。每个 lagged3 正式作业实际换库 39 次，前 50 步 2 个成员，随后 3 个，故 teacher 调用略少于固定 K3。teacher 秒已包含在作业秒内，不应再重复相加。记录的峰值 allocated 显存约 Geo 360.76 MiB、bank 378.26 MiB，不代表共享 GPU 总占用。')
text.append('Bank 在线训练求解数为 0，但历史 teacher 的训练/选择和验证、测试仍依赖 oracle；表中没有计入历史 teacher 与标签库的构建成本。因此可以说“免在线新 GeoSteiner 标签”，不能说“全流程无 oracle”。正式阶段每模型周期验证 21×8000=168000 次，最终测试 80000 次，均复用既有参考。初步 bank 队列 981.02 秒，滚动补充 189.00 秒；各完成作业另有 288 个离线参考求解，两阶段合计 3456 次，不能混进零在线调用口径。')
text.append('几何实验计算的是 greedy 第一端点和既有历史条件下，第二端点/方向的条件概率，不是完整联合动作熵。G 将动作路径之间的几何距离与概率不确定性结合；Q/G/C 仅在开发部分三选一。d30 开发集相对随机的差分别为 +0.4487/+0.6000/+0.4617 个百分点，d10 为 +0.1414/+0.2442/+0.1096，两个阶段均冻结 G。测试阶段每个候选块恰有 16 个候选，选到更大真实误差表示更善于寻找困难样本，不表示输出线路更短。')
text.append('d30 主阶段包含 ERM/WDRO × seed102/103 的 4 个固定模型，每族 dev1024/test2048；d10 次级阶段仅有 seed102 的 ERM/WDRO 两模型，每族 dev512/test1024。61440 是模型轨迹数，独特实例为 dev6144、test12288；开发/测试 ID 无交集，扩展测试与初步几何诊断已用 ID 无交集。d30 统计单位是 4 族各128个候选块，d10 是 4 族各64块，多模型先在同一块内平均，不能作为额外独立样本。历史 validation/test 曾用于模型选择或其他 benchmark，不是全新外部确认数据。')
rows=[]
for stage,label in [('d30_multiseed','d30主阶段'),('d10_degree_control','d10次级')]:
    for base,v in g[stage]['aggregate'].items():
        rows.append([label,base,f"{v['delta_pp']:+.4f}",f"[{v['ci_pp'][0]:.4f}, {v['ci_pp'][1]:.4f}]",f"{v['positive']}/{v['cells']}",f"{v['holm_significant_positive']}/{v['cells']}"])
text.append(table(['阶段','G的对照','所选误差差 pp','总体配对95%区间 pp','组级差值为正','组级Holm显著正向'],rows))
text.append('总体点值、上述总体区间，以及从保存 p 值重新计算的 Holm 调整均与原报告一致。总体区间是按固定分布分层重采样候选块的未校正区间；组级比较使用 9999 次配对符号翻转的双侧 p 值，分别在 d30 的48项、d10的24项内 Holm 校正。这两种统计层级不能混称“所有组显著”。相对 random 的 d30 信号较强（15/16 组通过），相对 Hnorm/低maxp 只有 3/16 和 3/16 组通过；d10 相对低maxp 无一组通过。负组保留：d30 M1118/D0218 相对 Hnorm 为 −0.002131 pp；d10 M0888/D0221、M1108/D0226 相对低maxp 为 −0.061805、−0.091077 pp，三项调整 p 均为1。组级细节见 '+link('geometry_cells.csv',OUT/'geometry_cells.csv')+'。')
text.append('有限扰动实验只有 32 个中心，每分布8个，每中心8个扰动加 nominal；指标只在8个扰动中选，nominal 单独作对照。G 在 M0900 上比 random 高 +0.564 pp，95%区间 [0.156,0.940]；M1118 上仅 +0.125 pp，区间 [−0.233,0.476]。它没有跨模型稳定的攻击优势。候选经 bbox 归一化后，所选 G 的最大 L∞ 分别约 '+f"{next(x for x in s['attack'] if x['model']=='M0900' and x['method']=='G')['max_postnorm_linf']:.5f}"+' 与 '+f"{next(x for x in s['attack'] if x['model']=='M1118' and x['method']=='G')['max_postnorm_linf']:.5f}"+'，不能把生成前的 ±0.02 当作最终严格半径。按中心重采样的区间不能升级为288个独立攻击样本的区间；本阶段未做多重比较校正。')
text.append('几何初步诊断耗时 322.02 秒，有限扰动阶段 68.77 秒，开发/测试扩展 3200.79 秒（53.35 分钟）。扩展耗时包括推理、几何特征和统计；未记录逐指标独立计时，不能推出 G 相对纯熵的额外时间或吞吐。扩展和初步诊断复用参考，有限扰动新求解288个输入供两模型共用。当前产物逻辑体积约：初步 bank 369.8 MiB、滚动 102.1 MiB、正式 bank 2.082 GiB，初步几何（含扰动）5.75 MiB、扩展几何12.90 MiB；不含共享历史数据、teacher 和参考库。')
text.append('验收结论：正式 d10 bank 可以接受为“在当前参考、初始化、预算及共享服务器条件下，降低在线 oracle 成本且改善测试误差”的数值结果；几何扩展可以接受为“固定16候选预算下，G 提升离线困难检索”的结果。长期替代质量、d30 bank 泛化、主动 Wasserstein 约束下的鲁棒收益、几何指导训练收益均没有本次数据支持。整周研究不能标为全部完成。')
text.append('数据入口：'+link('bank正式协议',ROOT/'checkpoint_bank_scale_20260930/protocol.json')+'；'+link('正式队列结果',ROOT/'checkpoint_bank_scale_20260930/artifacts/queue_status.json')+'；'+link('几何扩展协议',ROOT/'geometry_uncertainty_extension_20260930/PROTOCOL.md')+'；'+link('原始几何报告',ROOT/'geometry_uncertainty_extension_20260930/REPORT.md')+'。本次重算结果：'+link('bank_analysis.json',OUT/'bank_analysis.json')+'、'+link('geometry_analysis.json',OUT/'geometry_analysis.json')+'、'+link('supplement.json',OUT/'supplement.json')+'。重算脚本：'+link('analyze_results.py',OUT/'analyze_results.py')+'、'+link('analyze_supplement.py',OUT/'analyze_supplement.py')+'。')
(OUT/'REPORT.md').write_text('\n\n'.join(text)+'\n')

ops='''国庆实验运行状态回执

截至本目录 inventory.json 的 UTC+8 时间快照。原始部署 HANDOFF、PLAN、ACCEPTANCE_INVENTORY 和实验产物保留；本回执另存，不覆盖历史交接。

正式 bank 队列在 2026-10-01 05:43:51 UTC+8 结束，9/9 退出码0，failures为空；2000步日志连续，每个作业均保存 latest/best 的4族完整测试结果。初步 bank 10/10、滚动补充2/2、初步几何、有限扰动、几何扩展均有完成产物。读取进程列表时没有上述实验的活跃 worker 或 holiday controller。

总控制器 state.json 为 controller_error，active=null，jobs={}。2026-10-01 05:48:11 UTC+8 记录 TypeError：int() 收到 NoneType。调用链 controller.py existing_dependencies → owns_live → live；上游成功退出后 queue_status.json 正常设置 child_pid=null，代码 source.get('child_pid',0) 仍返回 None，随后 int(None) 异常。故上游完成并未触发后续启动；不是 GPU OOM、科学门控拒绝或训练预算耗尽。

两份 jobs.json 合计66项，45项bank（21训练阶段、21评估、2门控、1报告），21项geometry（13训练阶段含100步前缀、4评估、3门控、1报告）。全部完成标记缺失，两个 week_* artifacts 目录均为空，控制器事件中无 job_start。实际应归类为“未开始，控制器失败阻断”。训练阶段可续跑同一模型，不能当作独立模型数。

旧 acceptance_inventory.json 生成于 2026-09-30 23:48:09 UTC+8，早于正式队列完成，内含63项skipped_dependency和3项not_run，attempts均0；它对应早先 idle controller 停止时的快照。当前 state 中 dependency_queue 也是最后一次轮询留下的旧状态，不能用其中“lagged3_s103运行中”判断当前状态。当前上游状态以各自完成文件与queue_status为准。

未完成的科学内容包括 qualified-best3 的三种子2000更新，d10四方法10000更新及测试，d30 Geo/选中bank三种子训练与测试，全新几何dev/test筛选及门控，100步训练成本前缀、后续4方法×3种子2000更新，以及两条整周报告。当前没有真实negative_gate、paused训练或deadline-skipped实验结果。

历史日志另保留：控制器最初启动曾有换行字符串SyntaxError，后来可进入等待阶段；rolling的report.log保留一次汇总脚本SyntaxError，但ROLLING_COMPARISON.md/json已存在且可从逐实例结果复算，不能把旧汇总失败当作两次训练失败。初步bank旧closure记载过queue.py遮蔽标准库的导入失败，后续10个成功产物均保留。

本次只分析已生成数据、状态与日志，新增CPU统计和报告，没有重启控制器、修改实验代码或启动训练。原计划停止训练时间为2026-10-06 23:29 UTC+8，最终截止2026-10-07 23:29 UTC+8；截止时间尚未来临不等于进程仍在运行。若后续恢复，应先修复None PID处理，沿用已有成功产物和冻结门控，并重新评估剩余预算，不能将原整周计划视作仍有完整七天可用。
'''
ops+='\n'+link('逐项状态 CSV',OUT/'unrun_jobs.csv')+'；'+link('状态快照 JSON',OUT/'inventory.json')+'；'+link('控制器日志',ROOT/'holiday_research_20260930/artifacts/controller.log')+'；'+link('控制器源码',ROOT/'holiday_research_20260930/controller.py')+'。\n'
(OUT/'OPERATIONS.md').write_text(ops)
print('Wrote REPORT.md and OPERATIONS.md')
