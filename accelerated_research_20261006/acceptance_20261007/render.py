import csv,json,collections
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;P=O.parent;R=P.parent
def read(p):return json.loads(p.read_text())
def rows(n):return list(csv.DictReader((O/n).open()))
def interval(x):return '['+', '.join(f'{v:.6f}' for v in x)+']'
a=rows('bank_aggregate.csv');contrasts=read(O/'bank_contrasts.json');life=read(O/'lifecycle.json');diag=rows('training_diagnostics.csv');coverage=rows('monitor_coverage.csv');monitor=rows('monitor_final.csv');geom=rows('geometry_screen.csv');num=read(O/'numeric_reconciliation.json')
cost=read(R/'accelerated_bank_20261006/REPORT.json');(O/'cost_accounting.json').write_text(json.dumps({'total':cost['new_work_cost_total'],'by_job':cost['new_work_cost_by_job']},indent=2))
lines=['# 六卡大 batch 实验验收报告','',
'验收对象是独立的 batch4096 加速协议。结论：d10 bank 对照矩阵完成，在线训练参考可由 checkpoint bank 替代并降低本次运行耗时，性能差异较小；d30 缺少第三个种子，扩展实验只能部分验收。几何筛选实验完成，但几何训练因预先冻结的开发集门槛未通过而未开展。几何熵相关性监测完成，属于固定验证集上的探索性证据。','',
'队列于北京时间 2026-10-06 19:58 启动，2026-10-07 08:26 结束。84 项中完成 66 项、失败 2 项、门槛跳过 1 项、依赖跳过 15 项；结束状态不等于所有训练成功。28 项 bank 训练完成：d10 四方法×三种子×两个预算共24项，d30 Geo/lagged3×两个种子共4项。28项均有 latest 与 best 测试。','',
'd30_geo125_s103 与 d30_selected125_s103 均在物理 GPU4 尝试两次后 CUDA OOM，报错时剩余显存约98/242 MiB；对应两项测试未运行。几何 prefix7 被门槛拒绝，成本门槛和12项几何训练因此未运行。这些缺失不能作为训练方法无效的结果。完整清单见 inventory.csv。','',
'训练 batch4096，短预算125更新=512,000样本更新，长预算625更新=2,560,000样本更新；长预算从125更新继续，表中长预算耗时已计入父阶段。测试每分布10,000实例、四分布共40,000实例/模型，推理batch1024；latest为主结果，验证集选出的best仅为次级结果。旧batch256实验与本次更新数、采样和刷新节奏不同，不合并，也不能据此声称纯吞吐优化保持数值等价。','',
'## Bank 性能和运行成本','',
'误差定义为100×(L/Lref−1)，单位为%；Lref为独立GeoSteiner参考。均值±标准差以训练种子为单位；最坏OOD为每种子三个OOD分布中最大均值后再跨种子平均。P95为各分布P95的宏平均。耗时是实际并发环境中的训练进程墙钟时间，包含参考、刷新及训练内验证，不等于GPU满负载算力。','',
'| degree | 更新 | 方法 | 种子数 | 平均误差 %（均值±SD） | 最坏OOD % | 宏P95 % | 每种子训练分钟 | 相对Geo耗时变化 |',
'|---:|---:|---|---:|---:|---:|---:|---:|---:|']
for r in a:
 if r['outcome']!='latest':continue
 geo=next(x for x in a if (x['degree'],x['updates'],x['outcome'],x['arm'])==(r['degree'],r['updates'],'latest','geo'))
 change=(float(r['mean_train_seconds'])/float(geo['mean_train_seconds'])-1)*100
 lines.append(f"|{r['degree']}|{r['updates']}|{r['arm']}|{r['n_seeds']}|{float(r['mean_pct']):.6f} ± {float(r['sd_pct']):.6f}|{float(r['worst_ood_pct']):.6f}|{float(r['macro_p95_pct']):.6f}|{float(r['mean_train_seconds'])/60:.2f}|{change:+.2f}%|")
lines+=['','以下差值为bank−Geo，负数更好。实例区间按分布分层、同一实例跨种子先平均，再配对bootstrap2000次；不把共享测试集的三个种子当成三倍独立实例。种子t区间另列，仅是n=3或n=2且依赖分布假设的描述性区间；多方法比较未做确认性多重校正。','',
'| degree | 更新 | 方法 | 差值 pp | 条件于现有模型的实例95%区间 pp | 种子t 95%区间 pp | 各种子差值 pp |','|---:|---:|---|---:|---|---|---|']
for r in contrasts:
 if r['outcome']=='latest':lines.append(f"|{r['degree']}|{r['updates']}|{r['arm']}|{r['delta_pp']:+.6f}|{interval(r['ci95_conditional_instances_pp'])}|{interval(r['ci95_seed_t_pp'])}|{', '.join(f'{v:+.6f}' for v in r['seed_effects_pp'])}|")
lines+=['','三个bank方案在d10两个预算的平均误差均低于Geo，但改善幅度较小，不能把容差门槛通过解释为统计等效。d30的结论仅覆盖seed101/102；seed103失败使三种子方案不完整。','',
'次级best结果完整保存在 bank_aggregate.csv 和 bank_contrasts.json，不能从latest/best中事后择优形成主结论。保存的length/reference共重算'+f"{num['bank_values']:,}"+'个误差值（包含latest/best、重复实例与不同种子，非独立样本数），与保存误差最大绝对差为'+str(num['max_error_reconstruction_difference'])+'。本次未重新进行GPU推理或重新调用求解器。','']
lines+=['长预算采用验证集选出的best后，Geo/K3/lagged3/qualifiedbest3平均误差分别为0.179176%/0.179984%/0.183389%/0.181573%。lagged3−Geo变为+0.004213 pp，条件实例95%区间[+0.001911,+0.006550]；种子t区间跨零。该次级结果说明优势依赖检查点口径，不能概括为bank在所有选模规则下更优。','']
fields=['exact_train_nets','teacher_forward_nets','exact_train_seconds','teacher_seconds','refresh_seconds']
tot=cost['new_work_cost_total']
lines+=['Geo在线精确参考累计 '+f"{int(tot['exact_train_nets']):,}"+' 个net请求；bank在线新增GeoSteiner请求为0。teacher前向累计 '+f"{int(tot['teacher_forward_nets']):,}"+' 个net，参考求解累计 '+f"{tot['exact_train_seconds']/3600:.3f}"+' 小时、teacher前向 '+f"{tot['teacher_seconds']/3600:.3f}"+' 小时、刷新 '+f"{tot['refresh_seconds']/60:.2f}"+' 分钟。这些为日志计数，包含已记录的失败尝试，不含历史teacher训练、离线筛选参考和全部历史缓存成本；失败前未提交日志的工作可能漏计。计时是训练耗时的组成部分，不重复加到总耗时。','',
f"全队列经过 {life['elapsed_hours']:.3f} 小时；GPU任务进程累计 {life['gpu_process_hours']:.3f} 小时（含筛选/测试和失败重试），不能当作能耗。与同一加速协议Geo对照的耗时比较受共享卡负载影响，未额外做隔离测速。",'',
'checkpoint bank仍依赖历史教师及验证/测试参考，并非完全无oracle。训练目标可能为负，因为teacher给出的是近似参考上界；该目标不能直接等同于真实非负regret。','',
'| degree / 方法前缀 | 已记录更新 | λ=0 更新比例 | 负训练目标比例平均 |','|---|---:|---:|---:|']
for degree,label in [(10,a) for a in ['geo','k3','lagged3','qualifiedbest3']]+[(30,a) for a in ['geo','selected']]:
 rr=[r for r in diag if r['job'].startswith(f'd{degree}_{label}')]
 if not rr:continue
 n=sum(int(r['logged_updates']) for r in rr)
 lines.append(f"|{degree} / {label}|{n}|{sum(int(r['lambda_zero']) for r in rr)/n:.2%}|{sum(float(r['negative_target_fraction_mean'])*int(r['logged_updates']) for r in rr)/n:.2%}|")
lines+=['','λ统计包括实际记录的失败尝试及可能的重复更新；长阶段日志只计新增段。λ接近或等于零时，不支持把训练收益归因于持续活跃的Wasserstein运输约束。','',
'## 几何筛选与未开展的几何训练','',
'两种degree各4个固定模型，开发集每分布32中心、测试集每分布64中心，合计768中心；每中心32扰动+名义样本、短名单4个精确参考预算。共101,376条模型候选轨迹，25,344个离线候选参考。四个模型共享中心，统计先跨模型平均、再以中心为重采样单位。','',
'下表为G方案减对照的100×Δ(相对误差−运输成本)，正数表示找到更大的软对偶目标，并非最终路由长度改善。soft λ固定1；训练中的学习λ是另一变量。区间采用原预设中心bootstrap，p为六项对照内Holm校正；已由保存候选长度、参考和短名单重构目标，并复算统计。分布分层的额外区间见 geometry_screen.csv。','',
'| degree | 数据 | 对照 | 100×目标差 | 95%区间 | Holm p |','|---:|---|---|---:|---|---:|']
sig=read(O/'geometry_significance.json')
for r in sig:
 if r['contrast'].startswith('soft_dual/'):
  lines.append(f"|{r['degree']}|{r['split']}|{r['contrast'].split('/')[1]}|{r['delta']*100:+.6f}|{interval([x*100 for x in r['ci']])}|{r['holm']:.4f}|")
lines+=['','d10开发集G−普通熵为+0.022606，95%区间[−0.001420,+0.046902]，Holm p=0.0727；未通过“优于普通熵且优于随机”的预设条件。因此12项几何训练均未执行，不存在几何训练收益结论。d30测试集支持G在本次固定模型/候选预算下优于普通熵，但测试结果不能反向用于推翻开发集门槛。硬半径辅助结果保存在 geometry_screen.csv，不替代软对偶主目标。','',
'## 训练过程几何熵与困难程度','',
f"监测到 {len(coverage)} 个训练目录、{sum(int(r['n_snapshots']) for r in coverage)} 份检查点，含两个失败任务的中间/初始状态；监测评估累计 {sum(float(r['cpu_evaluation_seconds']) for r in coverage)/3600:.3f} 小时单线程CPU进程耗时。没有几何训练曲线，因为几何训练未启动。每分布256个固定历史validation实例，CPU FP32/batch64；启动较晚且仅抽取可见latest，不能补写未观察到的中间曲线。",'',
'下表仅取已完成bank任务的最终检查点。每个格子的ρ先在分布内计算，随后显示种子均值；它不是把各分布合并后的相关系数。按种子ρ及G−H配对实例bootstrap区间见 monitor_final.csv；G、H各自区间保存在原监测目录的检查点JSON。','',
'| degree | 预算更新 | 方法 | 种子数 | 分布 | G Spearman均值 | H Spearman均值 | G富集倍数均值 |','|---:|---:|---|---:|---|---:|---:|---:|']
runmeta={r['job']:r for r in rows('bank_runs.csv') if r['outcome']=='latest'}
cells=collections.defaultdict(list)
for r in monitor:
 if r['job'] not in runmeta:continue
 m=runmeta[r['job']]
 if int(r['update'])!=int(m['updates']):continue
 cells[m['degree'],m['updates'],m['arm'],r['group']].append(r)
for (d,u,arm,g),rr in sorted(cells.items(),key=lambda x:(int(x[0][0]),int(x[0][1]),x[0][2],x[0][3])):
 lines.append(f"|{d}|{u}|{arm}|{len(rr)}|{g}|{np.mean([float(r['G_rho']) for r in rr]):.4f}|{np.mean([float(r['H_rho']) for r in rr]):.4f}|{np.mean([float(r['G_enrichment']) for r in rr]):.3f}|")
lines+=['','d10的96个最终模型×分布格子中，G的ρ为0.123–0.283，50/96高于H；只有11个格子的G−H未校正配对95%区间全为正。d30的16个格子中，G的ρ为0.116–0.409，14/16高于H，9个格子的未校正区间全为正。这些格子共享实例或训练路径，不能把格子计数当作独立显著性检验。G在d10上没有稳定超过普通熵；d30具有较强的探索性信号。','',
'这些相关性描述当前模型在哪些实例上更易出错，不能证明实例具有模型无关的固有难度，亦不能证明按G采样会改善训练。验证集被反复使用，时间点与共享实例不独立，监测区间未作跨时间/多对照校正；不能据此宣布普遍显著优于普通熵。','',
'## 验收结论','',
'd10 checkpoint bank：本次两种训练预算下的三种子对照数据可验收，支持在线参考替换的可行性及本次成本降低；不支持完全无oracle、统计等效或活跃WDRO约束的收益宣称。d30：仅两种子部分结果，第三种子OOM使该扩展尚未完成。几何筛选：固定模型结果可验收，d30测试支持筛选目标改善；几何训练：未开展，无训练效果可验收。相关性监测：已产生可用的探索性数据，不能替代几何训练对照。','',
'数据入口：bank_runs.csv（逐种子与latest/best）、bank_aggregate.csv、bank_contrasts.json、training_diagnostics.csv、geometry_screen.csv、geometry_significance.json、monitor_final.csv、monitor_coverage.csv、cost_accounting.json、inventory.csv。历史batch256结论仍单独保留在 holiday_research_20260930/acceptance_20261006/REPORT.md，本报告不将它并入加速实验。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')
