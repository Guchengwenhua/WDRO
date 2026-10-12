import json,csv
from pathlib import Path
O=Path(__file__).resolve().parent
def read(n):return json.loads((O/n).read_text())
agg=list(csv.DictReader((O/'aggregate.csv').open()));c=read('contrasts.json');r=read('receipt.json');m=read('monitor_summary.json')
def interval(v):return '['+', '.join(f'{x:+.6f}' for x in v)+']'
lines=['# 后续模型库与几何熵训练简短验收报告','',
'41/41项队列任务完成，无失败、无重试；北京时间2026年10月9日03:19结束。包含26项本轮训练、14项单独评估和1项汇总；加上复用的4组历史短预算结果，共30组训练终点、60组latest/best评估。几何12组均完成，d30此前缺失的第三种子已补齐。','',
'结论：模型库带来明显的训练成本优势，但d30长预算下精度略逊Geo；几何熵获得小幅训练收益信号，成本却显著高于普通熵。两者都不能概括为全面优越。','',
'## 模型库 d30','',
'三种子、batch4096；主结果为固定最终检查点。误差为四分布共40,000实例的100×(L/Lref−1)，Lref是独立GeoSteiner参考。表中误差为种子均值±SD；长预算耗时包含125更新父阶段，不重复累计。','',
'| 更新 | 方法 | 平均误差 % | 每种子训练小时 |','|---:|---|---:|---:|']
for u in [125,625]:
 for arm in ['geo','k3','lagged3']:
  x=next(x for x in agg if (x['family'],int(x['updates']),x['arm'],x['outcome'])==('bank',u,arm,'latest'))
  lines.append(f"|{u}|{arm}|{float(x['mean_pct']):.6f} ± {float(x['sd_mean_pct']):.6f}|{float(x['train_hours']):.2f}|")
lines+=['','短预算K3/lagged3均优于Geo，长预算发生反转：K3和lagged3分别比Geo多0.005984、0.005159个百分点误差，成功训练路径耗时少69.91%、65.35%。收益主要是省训练成本，长时间训练的精度仍以Geo更好；不能宣称等效。上述时间受共享GPU负载影响，且不包含历史教师构建、历史失败和独立测试耗时。','',
'| 625更新对照差 | 平均差 pp | 条件实例95%区间 pp | 三个种子差值 pp |','|---|---:|---|---|']
for x in c:
 if x['family']=='bank' and x['updates']==625 and x['outcome']=='latest':lines.append(f"|{x['comparison']}|{x['mean_delta_pp']:+.6f}|{interval(x['conditional_interval_pp'])}|{', '.join(f'{v:+.6f}' for v in x['seed_effects_pp'])}|")
lines+=['','模型库在线GeoSteiner调用为0，但仍依赖历史教师和独立参考评估。最终λ：Geo约0.0344–0.0358，K3约0.0084–0.0092，lagged3为0；不能把所有方法的表现都归因于同样活跃的运输约束。','',
'## 几何熵与普通熵 d10','',
'四组各三种子、batch4096、125更新=512,000样本更新；统一33候选及4个精确参考预算。主指标是三个OOD分布的相对误差宏平均，越低越好；不与上表d30数值直接比较。','',
'| 筛选方法 | OOD误差 %（均值±SD） | 每种子运行小时 |','|---|---:|---:|']
for arm in ['original','random','entropy','geometry']:
 x=next(x for x in agg if (x['family'],x['arm'],x['outcome'])==('geometry',arm,'latest'))
 lines.append(f"|{arm}|{float(x['ood_pct']):.6f} ± {float(x['sd_ood_pct']):.6f}|{float(x['train_hours']):.3f}|")
lines+=['','几何熵比原始筛选降低0.005243个百分点（相对降低2.64%），比普通熵降低0.003352个百分点（相对降低1.70%）；耗时约为普通熵12.64倍。普通熵与几何熵均额外执行16,896,000个候选actor前向/种子，几何距离评分进一步增加了本实现的耗时。四臂每种子精确参考均为2,048,000个net，等参考预算不等于等计算预算。','',
'| 主对照差（负值更好） | 平均差 pp | Bonferroni 98.333%条件实例区间 pp | 三个种子差值 pp |','|---|---:|---|---|']
for x in c:
 if x['family']=='geometry' and x['outcome']=='latest':lines.append(f"|{x['comparison']}|{x['mean_delta_pp']:+.6f}|{interval(x['conditional_interval_pp'])}|{', '.join(f'{v:+.6f}' for v in x['seed_effects_pp'])}|")
gh=next(x for x in c if x['family']=='geometry' and x['outcome']=='latest' and x['comparison']=='geometry minus entropy')
lines+=['',f"G−普通熵的校正区间{'跨零' if gh['conditional_interval_pp'][1]>=0 else '未跨零但上界很接近零'}，属于边界信号；seed102中G反而略差，只有2/3个种子优于普通熵。其种子t 95%区间为{interval(gh['seed_t95_pp'])} pp，不能据此确认跨训练随机性的稳定优势。验证集best次级结果中G的OOD均值仍最低（0.196017%，普通熵0.200730%），但主结论始终使用latest。",'',
'四臂最终λ均为0。本轮证明的是本设置下改变困难样本筛选策略的探索性收益信号，不证明活跃Wasserstein约束带来收益，也不证明同计算时间下更划算。','',
'## 不确定性与结论范围','',
f"从保存的length/reference独立重算{r['saved_error_values']:,}条误差记录（含重复实例、种子和检查点），与保存误差最大差{r['max_reconstruction_difference']}；未重新推理或重新求解。几何主对照采用20,000次分层配对bootstrap，其他对照5,000次；同实例跨种子先平均，区间条件于这三个已训练模型，不是种子总体保证。三种子效应与描述性t区间另存contrasts.json。",'',
f"相关性监测完整结束，共{m['snapshots']}份检查点记录。几何分支最终48个模型×分布格子的G Spearman范围为{m['G_range'][0]:.3f}–{m['G_range'][1]:.3f}，其中{m['G_above_H']}/48高于普通熵；共享验证实例和反复观察不构成独立证据。",'',
'几何训练是在开发门槛未通过后按用户要求直接启动的探索性实验，且历史测试集已被查看。验收结论是数据完成、效果与成本可比较；没有新盲测或普遍优越性的证据。实际用途上，模型库适合成本受限的训练；几何熵值得研究如何降低评分成本，当前实现的小幅精度收益尚不足以直接证明其计算性价比。','',
'数据：aggregate.csv为总体表，runs.csv为逐种子latest/best，contrasts.json为区间和种子效应，budgets.csv为实际预算与最终λ，receipt.json为完成记录。']
(O/'DETAILS.md').write_text('\n'.join(lines)+'\n')
brief=['# 实验验收简报','',
'41/41项任务完成，无失败或重试；d30缺失种子已补齐，几何四臂共12组训练全部完成。本轮历时26.75小时，GPU任务进程累计105.01小时（不等于GPU满负载算力或能耗）。保存的240万条误差记录重算一致。','',
'**模型库：明显省时，但长训练的平均精度略逊Geo。** d30三种子、batch4096、625更新，主结果取最终检查点；误差为四分布平均相对GeoSteiner参考的误差，越低越好。耗时包含短预算父阶段。','',
'| 方法 | 平均误差 % | 每种子训练小时 |','|---|---:|---:|']
for arm in ['geo','k3','lagged3']:
 x=next(x for x in agg if (x['family'],int(x['updates']),x['arm'],x['outcome'])==('bank',625,arm,'latest'))
 brief.append(f"|{arm}|{float(x['mean_pct']):.5f}|{float(x['train_hours']):.2f}|")
brief+=['','K3/lagged3分别省约70%/65%的耗时，但误差多0.00598/0.00516个百分点；两项条件实例95%区间均为正。125更新时模型库曾略优，625更新后反转。它们适合成本受限的训练，不能宣称长训练精度更优或与Geo统计等效。在线新GeoSteiner调用为0，但仍依赖历史教师与独立评估参考。','',
'**几何熵：有小幅训练收益信号，当前实现成本很高。** d10四臂各三种子、125更新，表中为三个OOD分布的平均误差；不能与上表d30直接横向比较。','',
'| 筛选方法 | OOD误差 % | 每种子运行小时 |','|---|---:|---:|']
for arm,name in [('original','原始'),('random','随机'),('entropy','普通熵'),('geometry','几何熵')]:
 x=next(x for x in agg if (x['family'],x['arm'],x['outcome'])==('geometry',arm,'latest'))
 brief.append(f"|{name}|{float(x['ood_pct']):.5f}|{float(x['train_hours']):.2f}|")
brief+=['','几何熵比原始筛选相对降低2.64%的误差，比普通熵相对降低1.70%；耗时约为普通熵12.64倍。G−普通熵差值为−0.003352个百分点，20,000次配对bootstrap、三项比较校正后的98.333%区间为[−0.006626,−0.000192]个百分点。区间虽未跨零，但接近边界，且只在2/3个种子上胜过普通熵；不能确认跨随机种子的稳定优势。','',
'**结论范围：**上述区间条件于现有三个模型；几何训练是在开发门槛未通过后直接启动的探索性研究，历史测试集也已被查看。四个几何训练臂最终λ均为0，因此不能把收益归因于活跃的运输约束。相关性监测中G只在22/48个最终模型×分布格子上超过普通熵，也不支持普遍优势。','',
'实际判断：模型库的主要价值是省训练成本；几何熵值得研究如何降低评分成本，但当前小幅精度收益尚未证明同计算预算下更划算。时间来自共享服务器实际运行，未做隔离测速。','',
'完整统计和种子波动见 [DETAILS.md](DETAILS.md)、[aggregate.csv](aggregate.csv) 与 [contrasts.json](contrasts.json)。']
(O/'REPORT.md').write_text('\n'.join(brief)+'\n')
