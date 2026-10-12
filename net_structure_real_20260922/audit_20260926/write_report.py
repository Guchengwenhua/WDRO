from common import *
from collections import defaultdict

def rows(name):
 with (OUT/name).open() as f:return list(csv.DictReader(f))
def main():
 required=['static','metrics','geometry','physical_source','multidegree_source','generator_replay','gpu_replay_s101','gpu_replay_s102','gpu_replay_s103']
 evidence={k:read(OUT/f'{k}.json') for k in required};assert all(r['status']=='PASS' for r in evidence.values())
 agg=rows('aggregate.csv');contrasts=rows('paired_contrasts.csv');attack=rows('attack_aggregate.csv');cross=rows('crossed_generators.csv');geo=rows('candidate_geometry.csv');dual=rows('dual_summary.csv');gen=rows('generator_test_summary.csv');pressure=rows('fixed_parent_summary.csv');methods=CFG['methods'];labels=dict(erm_a='真实数据 ERM',raw_wdro='RAW-WDRO',vae_shell_wdro='VAE-SHELL',feature_gate='统计门控',laplacian='图平滑扰动',set_dae='集合去噪',graph_dae='图去噪',tile_raw='tile 随机',graph_tile='图去噪 + tile')
 q=read(ART/'queue.json');begin=min(datetime.datetime.fromisoformat(j['started_at']) for j in q['jobs']);end=max(datetime.datetime.fromisoformat(j['finished_at']) for j in q['jobs']);duration=(end-begin).total_seconds()/3600
 lines=['# 真实 net 结构先验实验：独立验收报告（2026-09-26）','',
 '## 验收结论','',
 '**核心实验的工程与数值验收通过；物理可行性与方法优越性只能作有限结论。** 33条REST训练、12次生成器拟合及其测试产物真实存在，数据隔离、模型选择、哈希、统计与GPU重放通过。没有证据支持“图结构或tile约束稳定全面优于VAE/RAW”，也不能据此认定生成net通过完整电路设计规则。','',
 '本次验收新建 audit_20260926/ 与 artifacts/audit_20260926/，保留原源码、训练权重、选择规则及报告。没有重新训练或按测试集调参。补做了冻结父模型候选压力评估、输入归一化一致的几何压力诊断以及独立种子/设计聚类统计；它们明确属于验收补充。','',
 '## 1. 实验与证据清单','',
 f'- 原队列166/166节点完成，33/33条REST训练到2048步；6次生成器调参到7500步、6次正式拟合到22500步。正式队列从{begin.isoformat()}运行至{end.isoformat()}，约{duration:.2f}小时。',
 '- 33条REST轨迹×3预算×2 checkpoint角色=198份预算快照；加3份父模型自然测试，共201份逐net文件。不是198个独立训练模型。',
 f"- 原记录818条哈希绑定对应798个唯一文件，全匹配；892项静态检查全通过。包含源代码、父模型、共享库、输入源、训练快照和测试绑定。",
 '- 源数据重构131209条多degree net（train/cal/val/B合计）；预训练train40630条。degree30训练2810、校准1202、验证1085、B测试3991、C外部测试1944。设计/家族隔离与跨split精确几何重复排除通过。B/C仍为历史已用数据的设计隔离测试。',
 '- 直接从原始16个ISPD .gr文件重解析9088条degree30 net，核对pin坐标、归一化变换、tile、layer与已保存物理数据，一致。',
 '- 完整重算201份自然测试中的1192935条regret；402行主指标与6030行逐设计指标与原CSV逐值一致。',
 '- GPU分别在3张卡重放201份测试文件的首批与末批（60903条）、66次完整A_val评估（71610条）和33次完整RAW-K32攻击（101376条）。这233889条原REST预测全部复现，误差为0。自然测试并非119万条全部重新前向，完整统计重算与GPU抽样重放范围已分开。',
 '- GPU重放6个正式去噪模型在A_val/B/C、5个噪声水平上的210600条输入，960行重建指标完全复现，误差为0。',
 '- 重算11个候选池共227872个候选的Hungarian成本，全一致；352个参考解分别以2^27和2^28复算，共704次oracle调用，结果一致。10个训练池及独立攻击池参考无失效。',
 '- 4个tile池共89920个候选的pin双射见证、tile边界、bbox与层元数据复核通过；最大归一化边界舍入约2.98e-8，低于冻结容差6e-8。',
 '- 所有训练共享每seed的中心序列、父模型、batch64和学习率；BN running stats保持父模型值，训练其affine参数。33条轨迹的更新序列、真实候选调用数、sampled调用数、lambda递推、验证最优步均一致。',
 '- 原监督器及其工作PID/starttime均已退出，监督器锁可获取；未重复启动实验。恢复后部分队列returncode=-99表示无法回收原进程退出码，其完成状态另由完整产物与本次重放确认。','',
 '详细机器证据：[汇总验收清单](artifacts/audit_20260926/acceptance.json)、[静态检查](artifacts/audit_20260926/static.json)、[数值统计](artifacts/audit_20260926/metrics.json)、[候选几何](artifacts/audit_20260926/geometry.json)。','',
 '## 2. 自然测试：末预算视图','',
 '以下为最大训练预算2048步下、仅由A_val选出的checkpoint，生成器固定301，三个REST种子分别计算后等权平均。mean是各测试集按net加权平均，P95先按seed计算再平均；全部为百分比，越低越好。B有3个设计，C有27个设计。此表只是便于阅读的末预算视图；256/1024/2048及last/val_best全部保留，未事后宣布单一主终点。','',
 '|方法|B mean %|B P95 %|C mean %|C P95 %|独立RAW-K32 mean %|','|---|---:|---:|---:|---:|---:|']
 lookup={(r['method'],r['budget'],r['role'],r['split']):r for r in agg};att={r['method']:r for r in attack if r['generator']=='301'}
 for m in methods:
  b=lookup[m,'2048','val_best','B_test'];c=lookup[m,'2048','val_best','C_external'];lines.append(f"|{labels[m]}|{float(b['mean_pct']):.4f}|{float(b['p95_pct']):.4f}|{float(c['mean_pct']):.4f}|{float(c['p95_pct']):.4f}|{float(att[m]['mean_pct']):.4f}|")
 lines+=['','独立RAW-K32仅在B的96个固定中心上进行，每设计32个，取32候选中的最大regret，不能与全部B自然测试的mean直接相减当作同样本因果变化。','',
 '固定last@2048的对照如下，防止只看一个checkpoint角色：','', '|方法|B mean %|B P95 %|C mean %|C P95 %|','|---|---:|---:|---:|---:|']
 for m in methods:
  b=lookup[m,'2048','last','B_test'];c=lookup[m,'2048','last','C_external'];lines.append(f"|{labels[m]}|{float(b['mean_pct']):.4f}|{float(b['p95_pct']):.4f}|{float(c['mean_pct']):.4f}|{float(c['p95_pct']):.4f}|")
 lines+=['','![验收比较](artifacts/audit_20260926/acceptance_comparison.png)','',
 '图中误差线是三个REST种子的标准差，不是置信区间；候选cost是所有非nominal候选平均，不是WDRO约束的被选候选平均。可导出：[PDF](artifacts/audit_20260926/acceptance_comparison.pdf)。','',
 '## 3. 能支持哪些效果判断','',
 '1. 集合去噪在自然测试上表现较均衡：val_best下B mean 2.0379%、C mean 1.2247%，低于RAW的2.0840%/1.2661%。但C上相对旧VAE 1.2296%的差距很小；不能把很小的描述性差异写成稳定优势。',
 '2. 图去噪能降低去噪误差，也在RAW-K32压力下取得主生成器比较中最低的平均regret 4.9679%。但它的自然测试没有全面优于普通集合去噪，结构模块并未显示稳定附加收益。',
 '3. 图+tile在生成器301时，B自然测试优于tile随机对照（2.0670%对2.1277%），但C改善很小且跨生成器不稳定；不能只挑301的优势概括整个方法。',
 '4. 图平滑、门控、tile随机都不是全面改善：门控的B mean较RAW更高；图平滑有个别较好指标但没有跨指标统一优势。',
 '5. 固定末轮与验证最优的相对排序不同，说明小验证集的选模噪声仍然影响外部表现。所有update0纳入了验证选择；最终最优步确实大于0，不能把旧VAE首轮“best=0”的结论套到本次freeze-BN长训练。','',
 '按相同seed与相同net配对，再联合重采样seed和design（2000次），关键结果如下。Δ是方法减基线，单位为百分点；CI对应design-macro差，不能冒充net加权差的CI。区间仅作探索性描述，B只有3设计，未作多重比较校正。','',
 '|比较|测试集|net mean Δ|design macro Δ|design macro 95%区间|改善seed数|','|---|---|---:|---:|---|---:|']
 for r in contrasts:
  if r['budget']=='2048' and r['role']=='val_best' and ((r['method'] in ['set_dae','graph_dae','graph_tile'] and r['baseline']=='raw_wdro') or (r['method'],r['baseline']) in [('graph_dae','set_dae'),('graph_tile','tile_raw')]):
   lines.append(f"|{labels[r['method']]} − {labels[r['baseline']]}|{r['split']}|{float(r['net_mean_delta_pp']):+.4f}|{float(r['design_macro_delta_pp']):+.4f}|[{float(r['joint_seed_design_ci_low_pp']):+.4f}, {float(r['joint_seed_design_ci_high_pp']):+.4f}]|{r['seeds_improved']}/3|")
 lines+=['','相对RAW的关键区间均跨0；不据此宣称统计确定的普适改善。graph_tile对tile_raw的B比较在主生成器301上是一个正信号，但它仍是小样本、多比较、特定生成器下的结果。','',
 '## 4. 生成器、候选真实性与物理约束','',
 '|去噪器|测试集|归一化MSE|相对含噪输入MSE下降|','|---|---|---:|---:|']
 for r in gen:lines.append(f"|{r['kind']}|{r['split']}|{float(r['design_sigma_seed_macro_mse']):.4f}|{float(r['relative_mse_reduction_pct']):.2f}%|")
 lines+=['','这些数值按design×sigma×generator-seed等权平均，只证明对预定义高斯噪声的去噪能力，不是非法电路识别率。真实与生成net的几何/成本统计应与REST成绩分开解释。','',
 '|候选池|非中心候选平均cost|平均有效K|原样本复制比例|','|---|---:|---:|---:|']
 for r in geo:
  if r['pool']=='attack32' or r['pool'].endswith(('g302','g303')):continue
  lines.append(f"|{r['pool']}|{float(r['mean_non_nominal_cost']):.6f}|{float(r['mean_unique_K']):.4f}|{100*float(r['nominal_copy_fraction']):.3f}%|")
 lines+=['','tile的成本约0.015，RAW约0.056，VAE约0.141；不同方法的实际候选局部性差异很大。tile方法并没有整体退化成原样本，训练池每个中心都保留8个不同候选，但仍需防止把缩小扰动本身误判为物理先验收益。','',
 '**物理约束的通过范围：** 16个ISPD设计的这批pin层号全部为1。已验证的是每个候选存在保留原tile/层标签和bbox的pin对应，不是多层pin接入、障碍物避让、拥塞、真实cell ECO合法移动、DRC或时序通过。输入数据本身缺少完整cell/netlist上下文。','',
 '**门控能力有限：** 它只改动RAW非中心候选的0.4423%，影响1.3167%的训练中心。原报告的“整体坍缩100%拒绝”使用了没有重新归一化的点集；将它恢复到真实输入契约后，拒绝率变为2%，与原真实样本一致。均匀缩放被输入归一化消除了，这个旧诊断不能作为学习到电路约束的证据。归一化后的单离群pin压力拒绝率仅0.5%，轴向压扁为100%；这些仍是几何压力案例，没有真实非法电路标签。','',
 '## 5. WDRO预算与机理归因','',
 '|方法|lambda为0的更新比例|平均被选cost|平均cost−rho|','|---|---:|---:|---:|']
 for r in dual:
  if r['generator']!='301':continue
  lines.append(f"|{labels[r['method']]}|{100*float(r['lambda_zero_fraction']):.2f}%|{float(r['mean_cost']):.6f}|{float(r['mean_budget_gap']):+.6f}|")
 lines+=['','共同rho=0.0709042084。RAW、门控、集合/图去噪和两类tile方法在这次训练中lambda始终为0，候选选择实际等价于各自候选池内的最大regret挖掘；这是预算松弛的结果，不是实现故障，但不能将其收益归因为非零运输惩罚。图平滑与VAE有非零lambda作用。VAE平均cost超过rho约3.85e-5，说明有限步对偶更新不是严格可行性证书。','',
 '验收补做固定父模型评估：每个训练设计固定32中心，共288中心；3个原父模型，K8，lambda分别固定0和1，新增69120条GPU候选评分，没有训练或测试选模。lambda=0下的结果为：','',
 '|候选池|固定父模型最大regret %|对应平均cost|','|---|---:|---:|']
 for r in pressure:
  if float(r['lambda_value'])==0 and not r['pool'].endswith(('g302','g303')):lines.append(f"|{r['pool']}|{float(r['max_regret_pct']):.4f}|{float(r['selected_cost']):.6f}|")
 lines+=['','这张表衡量固定模型的候选压力，不代表训练后的模型收益。图平滑产生更难但也更远的候选，tile保持较局部；这些差异必须保留，不能混成同一搜索器比较。','',
 '## 6. 方案偏差与尚不能验收的主张','',
 '- 原最终报告没有独立汇总方案C的固定父模型压力，本次已补做并单独标记。',
 '- 原RAW池使用历史随机种子及jitter/sparse/correlated/axis混合；新去噪/图平滑使用另一个种子的IID Gaussian提案。因此PLAN中的“与RAW同一高斯噪声”表述过强。tile_raw与graph_tile之间确实共享同一噪声，可用于局部约束下的对照；其余不能称严格提案配对消融。',
 '- 已有graph eta=.1/1的强度调参，但没有同一图网络eta=0的对照，不能单独识别结构损失的贡献。set与graph最终宽度均160、数据/更新预算相同，参数量与计算仍不同。',
 '- config.json继承了旧primary_checkpoint=last_256等未使用字段，而新PLAN声明多预算/双checkpoint报告。已核查执行代码按新updates/validation_interval及队列预算运行；本验收展示全部预算，不追认2048 val_best为唯一预注册主终点。',
 '- 原“整体坍缩拒绝”诊断不满足归一化输入契约，其解释已在本报告中纠正；没有改写原测试文件。',
 '- B仅3设计、REST仅3种子。增加训练轮数不能替代更多独立设计；历史已见设计也不是首次盲测。',
 '- 层号保持在本数据中只有单层标签案例。完整制造/电路合法性与稳定方法优越性均不在本次PASS范围。','',
 '## 7. 建议保留的结论与后续优先级','',
 '可保留：新方法实现与数值可信；集合/图去噪有可复现的去噪能力与部分下游改善；tile硬约束确实限制了粗网格几何且保留探索。暂不保留：“更像真实”就一定“更可布线”，或更复杂图结构稳定优于普通集合去噪。','',
 '如果继续实验，优先做严格同提案/同cost分布的比较、graph eta=0对照，以及具有真实cell/多层pin/障碍物信息的合法ECO与router验证。继续单纯延长这批训练的轮数，不能补齐这些归因和物理信息缺口。本次验收没有自动启动上述新训练。','',
 '## 8. 产物入口','',
 '- [所有预算的独立重算](artifacts/audit_20260926/aggregate.csv)；[配对比较及CI](artifacts/audit_20260926/paired_contrasts.csv)。',
 '- [生成器交叉种子](artifacts/audit_20260926/crossed_generators.csv)；[独立攻击汇总](artifacts/audit_20260926/attack_aggregate.csv)。',
 '- [成本与候选几何](artifacts/audit_20260926/candidate_geometry.csv)；[固定父模型补测](artifacts/audit_20260926/fixed_parent_summary.csv)。',
 '- [物理源重解析](artifacts/audit_20260926/physical_source.json)；[多degree数据重构](artifacts/audit_20260926/multidegree_source.json)。',
 '- 验收脚本全部在 audit_20260926/；此目录输出与原2026-09-22训练产物分开保存。']
 (ROOT/'ACCEPTANCE_20260926.md').write_text('\n'.join(lines)+'\n')
 summary=dict(time=now(),status='PASS_CORE_ENGINEERING_AND_NUMERICAL_WITH_SCOPE_LIMITS',source_experiment='net_structure_real_20260922',original_training_modified=False,new_training_started=False,original_completion=read(ART/'completion.json')['time'],rest_runs=33,generator_runs=12,queue_tasks=166,natural_test_files=201,core_static_checks=892,hash_files=798,full_metric_predictions=1192935,gpu_original_rest_replays=233889,gpu_supplemental_parent_scores=69120,gpu_denoising_replays=210600,physical_witness_candidates=89920,source_reparsed_nets=9088,multidegree_reconstructed_nets=131209,dual_scale_oracle_calls=704,failed_core_checks=[],not_accepted=['full circuit/DRC/timing legality','multilayer pin access performance','stable universal graph/tile superiority','isolated structural loss causality','strict shared RAW proposal claim','original unnormalized global-collapse diagnostic interpretation'],evidence={k:str(OUT/f'{k}.json') for k in required},report=str(ROOT/'ACCEPTANCE_20260926.md'))
 write(OUT/'acceptance.json',summary)
 files=list(OUT.glob('*'))+[ROOT/'ACCEPTANCE_20260926.md']+list((ROOT/'audit_20260926').glob('*.py'))
 write(OUT/'audit_manifest.json',dict(time=now(),files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files if p.is_file() and p.name!='audit_manifest.json']))
 print(summary['status'],flush=True)
if __name__=='__main__':main()
