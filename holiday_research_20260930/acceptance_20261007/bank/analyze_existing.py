import json,datetime
from pathlib import Path
import numpy as np
ROOT=Path('/remote-home/Guilin/DRO/Codex');OUT=Path(__file__).resolve().parent
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']
def read(p):return json.loads(p.read_text())
def stats(arr):return {'n':len(arr),'mean':float(np.mean(arr)),'p95':float(np.quantile(arr,.95))}
def event_summary(path):
 init=None;last=None;validations=[];resumes=[];neg=[];train_count=0
 with path.open() as f:
  for line in f:
   try:e=json.loads(line)
   except json.JSONDecodeError:continue
   if e.get('event')=='validation':
    validations.append({'update':e['update'],'selection':e['result']['selection'],'aggregate':e['result']['aggregate']})
    if e['update']==0:init=e['result']
   if e.get('event')=='train':last=e;train_count+=1;neg.append(e.get('negative_target_fraction',0.))
   if e.get('event')=='resume':resumes.append(e)
 return {'initdev':init,'last_train_update':last and last['update'],'accounting':last and last.get('reference_accounting',{}),'last_train_time':last and last.get('at_utc'),'negative_target_mean_fraction':float(np.mean(neg)) if neg else None,'negative_target_max_fraction':max(neg) if neg else None,'train_events':train_count,'resumes':resumes,'validations':validations}
rows=[];arrays={}
for project in ['checkpoint_bank_scale_20260930','week_bank_20260930']:
 for folder in sorted((ROOT/project/'artifacts').iterdir()):
  if not (folder/'logs/result.json').exists():continue
  result=read(folder/'logs/result.json');events=event_summary(folder/'logs/events.jsonl');config=read(folder/'run_config.json');arm=config['arm'];seed=config['seed']
  row={'project':project,'run':folder.name,'source':str(folder),'status':result['status'],'completed_updates':result['completed_updates'],'target_updates':result['expected_updates'],'arm':arm,'seed':seed,'degree':result['degree'],'best_step':result['best']['update'],'best_dev_selection':result['best'],'init_dev':{'overall':events['initdev']['aggregate']['overall'],'selection':events['initdev']['selection']},'last_dev':result['last_validation'],'last_dev_update':events['validations'][-1]['update'],'training_segment_seconds':result['segment_wall_seconds'],'accounting':events['accounting'],'train_events':events['train_events'],'resumes':events['resumes'],'negative_target_mean_fraction':events['negative_target_mean_fraction'],'negative_target_max_fraction':events['negative_target_max_fraction'],'sources':[str(folder/'logs/result.json'),str(folder/'logs/events.jsonl'),str(folder/'run_config.json')]}
  if (folder/'complete.json').exists():row['process_wall_seconds']=read(folder/'complete.json').get('wall_seconds_this_process')
  transitions=[]
  if (folder/'reference_state/refresh.jsonl').exists():
   with (folder/'reference_state/refresh.jsonl').open() as f:
    for line in f:
     try:transitions.append(json.loads(line))
     except json.JSONDecodeError:continue
  row['refresh_events']=transitions;row['observed_bank_changes']=sum(t.get('bank_used_ids')!=t.get('bank_next_ids') for t in transitions)
  for outcome in ['latest','best']:
   if not all((folder/f'test_{outcome}/{g}.npz').exists() for g in GROUPS):continue
   group_arrays={};ids={};groupstats={}
   for g in GROUPS:
    path=folder/f'test_{outcome}/{g}.npz'
    with np.load(path,allow_pickle=False) as z:
     v=z['regret'].copy();ids[g]=z['net_ids'].copy();group_arrays[g]=v
     # Use existing stored exact-reference metric; do not call model or solver.
     groupstats[g]=stats(v)
    row['sources'].append(str(path))
   allv=np.concatenate(list(group_arrays.values()));row[outcome]={'groups':groupstats,'overall':stats(allv),'worst_ood_mean':max(groupstats[g]['mean'] for g in GROUPS[1:]),'checkpoint_step':read(folder/f'test_{outcome}.json')['checkpoint_update']}
   arrays[(arm,seed,outcome)]=(group_arrays,ids)
  rows.append(row)
comparisons=[]
for row in rows:
 if row['project']!='checkpoint_bank_scale_20260930' or row['arm']=='geo':continue
 for outcome in ['latest','best']:
  a,ia=arrays[(row['arm'],row['seed'],outcome)];b,ib=arrays[('geo',row['seed'],outcome)]
  for g in GROUPS:
   if not np.array_equal(ia[g],ib[g]):raise ValueError('unmatched net IDs')
  diffs={g:a[g]-b[g] for g in GROUPS};rng=np.random.default_rng(20261007);boot=[]
  for _ in range(1000):boot.append(np.mean([v[rng.integers(len(v),size=len(v))].mean() for v in diffs.values()]))
  av=np.concatenate(list(a.values()));bv=np.concatenate(list(b.values()));delta=float(np.mean(av-bv));ci=np.quantile(boot,[.025,.975]).tolist()
  comparisons.append({'arm':row['arm'],'seed':row['seed'],'outcome':outcome,'n_paired_instances':len(av),'mean_delta':delta,'relative_mean_change':delta/float(bv.mean()),'ci95_stratified_paired_instance_bootstrap':ci,'replicates':1000,'group_deltas':{g:float(d.mean()) for g,d in diffs.items()},'worst_ood_delta':max(a[g].mean() for g in GROUPS[1:])-max(b[g].mean() for g in GROUPS[1:]),'pooled_p95_delta':float(np.quantile(av,.95)-np.quantile(bv,.95))})
aggregates=[]
for arm in ['geo','k3','lagged3']:
 rr=[r for r in rows if r['project']=='checkpoint_bank_scale_20260930' and r['arm']==arm]
 for outcome in ['latest','best']:
  v=np.array([r[outcome]['overall']['mean'] for r in rr]);a={'arm':arm,'outcome':outcome,'seeds':[r['seed'] for r in rr],'n_seeds':len(rr),'mean_regret':float(v.mean()),'seed_sd':float(v.std(ddof=1)),'seed_range':[float(v.min()),float(v.max())],'mean_worst_ood':float(np.mean([r[outcome]['worst_ood_mean'] for r in rr])),'mean_p95':float(np.mean([r[outcome]['overall']['p95'] for r in rr]))}
  cc=[c for c in comparisons if c['arm']==arm and c['outcome']==outcome]
  if cc:
   delta=np.array([c['mean_delta'] for c in cc]);a.update(paired_seed_deltas=delta.tolist(),mean_delta=float(delta.mean()),seed_delta_sd=float(delta.std(ddof=1)),seed_delta_range=[float(delta.min()),float(delta.max())])
   # Same frozen instances are reused by seeds: average models first, resample instance IDs once/group.
   dgroups={g:np.mean([arrays[(arm,s,outcome)][0][g]-arrays[('geo',s,outcome)][0][g] for s in [101,102,103]],axis=0) for g in GROUPS};rng=np.random.default_rng(20261007);boot=[np.mean([v[rng.integers(len(v),size=len(v))].mean() for v in dgroups.values()]) for _ in range(1000)];a['ci95_instance_for_mean_of_three_models']=np.quantile(boot,[.025,.975]).tolist()
  aggregates.append(a)
planned=[j['id'] for j in read(ROOT/'week_bank_20260930/jobs.json')['jobs'] if j['kind']=='train'];observed={r['run'] for r in rows if r['project']=='week_bank_20260930'}
result={'as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metric':'relative regret (REST length / cached GeoSteiner OPT)-1; tables use percent and delta percentage points','rows':rows,'paired_comparisons':comparisons,'three_seed_aggregates':aggregates,'week_missing_run_artifacts':[j for j in planned if j not in observed],'bootstrap':'1000 paired instances stratified by 4 geometry groups. Three-model aggregate averages same-instance differences across seeds before resampling; not a seed-population interval.','limits':['No prespecified equivalence margin; similarity is not statistical equivalence.','Scale exact solver CPU seconds were not separately recorded; wall time and counts are available.','Teacher seconds include inference and REST evaluation. Historical training and cached validation/test remain oracle-assisted.','No week bank final/test result exists for the partial run; no inference was run for acceptance.']}
(OUT/'bank_acceptance.json').write_text(json.dumps(result,indent=2))
pct=lambda x:f'{100*x:.6f}'
lines=['# Checkpoint-bank 验收数据（2026-10-07）','','实际完成：scale d10三方法×三种子共9组均达到2000步，batch256；每组latest与validation-best均有4种分布×10000个冻结实例的独立GeoSteiner参考评价。week仅qualifiedbest3 s101达到1017/2000，未有最终测试。','','## 三种子汇总','','下表误差、p95及最坏OOD为百分数；seed SD是三个训练种子的标准差。','','| 方法 | 权重 | 平均误差% | seed SD% | seed范围% | 平均最坏OOD% | 平均p95% |','|---|---|---:|---:|---|---:|---:|']
for a in aggregates:lines.append(f"| {a['arm']} | {a['outcome']} | {pct(a['mean_regret'])} | {pct(a['seed_sd'])} | {pct(a['seed_range'][0])}–{pct(a['seed_range'][1])} | {pct(a['mean_worst_ood'])} | {pct(a['mean_p95'])} |")
lines+=['','## 每种子 latest 与 best','','误差、最坏OOD、p95均为百分数；beststep来自验证选优。初始dev是共同legacy模型在8000验证实例上的误差，并非训练前测试。','','| 方法 | seed | latest均值% | latest最坏OOD% | latest p95% | best均值% | best最坏OOD% | best p95% | beststep | 初始dev均值% | 初始dev最坏OOD% |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 if 'latest' not in r:continue
 lines.append(f"| {r['arm']} | {r['seed']} | {pct(r['latest']['overall']['mean'])} | {pct(r['latest']['worst_ood_mean'])} | {pct(r['latest']['overall']['p95'])} | {pct(r['best']['overall']['mean'])} | {pct(r['best']['worst_ood_mean'])} | {pct(r['best']['overall']['p95'])} | {r['best_step']} | {pct(r['init_dev']['overall']['mean'])} | {pct(r['init_dev']['selection']['worst_group_mean'])} |")
lines+=['','## 同实例配对差值：bank减Geo','','差值和95%CI单位为百分点（误差ratio×100），负值有利bank；latest是主比较。1000次按分布分层的实例bootstrap，每个模型40000配对实例。','','| 方法 | seed | 权重 | 均值差百分点 | 95%实例CI百分点 | 相对变化% |','|---|---:|---|---:|---|---:|']
for c in comparisons:lines.append(f"| {c['arm']} | {c['seed']} | {c['outcome']} | {pct(c['mean_delta'])} | [{pct(c['ci95_stratified_paired_instance_bootstrap'][0])}, {pct(c['ci95_stratified_paired_instance_bootstrap'][1])}] | {100*c['relative_mean_change']:.3f} |")
for a in aggregates:
 if 'mean_delta' in a:lines.append(f"\n{a['arm']} {a['outcome']}：三模型平均配对差{pct(a['mean_delta'])}个百分点，固定三模型的实例CI [{pct(a['ci95_instance_for_mean_of_three_models'][0])}, {pct(a['ci95_instance_for_mean_of_three_models'][1])}]，三seed差值范围 [{pct(a['seed_delta_range'][0])}, {pct(a['seed_delta_range'][1])}]。该CI不包含重新训练种子的总体不确定性。")
lines+=['','## 每种子各分布数据','','| 方法/seed/权重 | uniform均值%/p95% | mix_gauss均值%/p95% | edge_bias均值%/p95% | grid_cluster均值%/p95% |','|---|---|---|---|---|']
for r in rows:
 for outcome in ['latest','best']:
  if outcome in r:lines.append('| '+f"{r['arm']}/{r['seed']}/{outcome}"+' | '+' | '.join(pct(r[outcome]['groups'][g]['mean'])+'/'+pct(r[outcome]['groups'][g]['p95']) for g in GROUPS)+' |')
lines+=['','## 实际成本与银行更新','','| 方法/seed | 完成步 | Geo调用nets | teacher forward nets | teacher秒 | refresh秒 | 实际换入事件 | 训练段墙钟秒 | 含评价process秒 | 平均负target比例% |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 a=r['accounting'];lines.append(f"| {r['arm']}/{r['seed']} | {r['completed_updates']} | {a.get('exact_train_nets')} | {a.get('teacher_forward_nets')} | {a.get('teacher_seconds',0):.3f} | {a.get('refresh_seconds',0):.3f} | {r['observed_bank_changes']} | {r['training_segment_seconds']:.3f} | {r.get('process_wall_seconds','NA')} | {100*r['negative_target_mean_fraction']:.4f} |")
lines+=['','Geo求解单独耗时在scale未记录，不能从总墙钟中可靠分离。银行训练Geo调用为0，不代表历史teacher及缓存标签没有oracle。teacher耗时包含前向与REST可行线长计算。各任务按顺序运行、共享GPU负载变化，墙钟差并非受控速度基准。','','## 未完成范围与结论边界']
for r in rows:
 if r['project']=='week_bank_20260930':lines.append(f"\nqualifiedbest3 s{r['seed']}仅{r['completed_updates']}/{r['target_updates']}步：最近dev在u{r['last_dev_update']}，overall={pct(r['last_dev']['aggregate']['overall']['mean'])}%，worstOOD={pct(r['last_dev']['selection']['worst_group_mean'])}%；best u{r['best_step']} worstOOD={pct(r['best_dev_selection']['selection_objective'])}%。仅单seed、无测试，不能判定其是否接近Geo。")
lines+=['','week未发现run产物的训练任务：'+', '.join(result['week_missing_run_artifacts'])+'。没有10000步或d30银行结论。','','scale的数值接近程度应依据上面的实际百分点差值及种子范围判断；未预设等效界限，不能宣称统计等效或普遍替代GeoSteiner。所有结论限d10、2000步和本冻结测试分布。','','## 结果来源']
for r in rows:lines.append('- '+r['source']+'：logs/result.json、logs/events.jsonl、test_latest/*.npz、test_best/*.npz、reference_state/refresh.jsonl（存在时）。')
lines+=['','机器可读逐分布、逐seed与路径证据：bank_acceptance.json。统计脚本：analyze_existing.py。']
(OUT/'BANK_ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'rows':len(rows),'scale_complete':sum('latest' in r for r in rows),'aggregates':aggregates,'week_missing':len(result['week_missing_run_artifacts'])},indent=2))
