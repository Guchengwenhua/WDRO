"""Read saved scientific outputs at campaign close; no new inference or selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,collections,csv,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;A=P/'artifacts';GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']
def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.json.partial');t.write_text(json.dumps(v,indent=2,ensure_ascii=False));t.replace(p)
sources=read(P/'sources.json');rows=[];missing=[];data={}
for source in sources:
 p=Path(source['result_dir'])
 if not (p/'complete.json').exists() or not (p/'test_latest.json').exists():missing.append(source);continue
 for outcome in ['latest','best']:
  if not (p/f'test_{outcome}.json').exists():continue
  gs=[];ids=[]
  for g in GROUPS:
   with np.load(p/f'test_{outcome}'/f'{g}.npz') as z:gs.append(z['length']/z['reference']-1);ids.append(z['net_ids'])
  key=(source['family'],source['updates'],source['arm'],source['seed'],outcome);data[key]=(gs,ids)
  rows.append(dict(**source,outcome=outcome,mean=float(np.mean(gs)),mean_ood=float(np.mean(gs[1:])),worst_ood=max(float(v.mean()) for v in gs[1:]),macro_p95=float(np.mean([np.quantile(v,.95) for v in gs])),n=40000))
contrasts=[]
for family,updates,control,arms in [('geometry',125,'geometry',['original','random','entropy']),('bank',125,'geo',['k3','lagged3']),('bank',625,'geo',['k3','lagged3'])]:
 for outcome in ['latest','best']:
  for arm in arms:
   ds=[];seed_effects={}
   for seed in [101,102,103]:
    left,right=(control,arm) if family=='geometry' else (arm,control)
    ka=(family,updates,left,seed,outcome);kb=(family,updates,right,seed,outcome)
    if ka not in data or kb not in data:continue
    aa,ai=data[ka];bb,bi=data[kb]
    if not all(np.array_equal(x,y) for x,y in zip(ai,bi)):raise ValueError('comparison IDs disagree')
    d=np.stack(aa)-np.stack(bb);ds.append(d);seed_effects[str(seed)]=dict(mean=float(d.mean()),mean_ood=float(d[1:].mean()))
   if not ds:continue
   d=np.mean(ds,axis=0);used=d[1:] if family=='geometry' else d;rng=np.random.default_rng(20261008);boot=[]
   for _ in range(2000):boot.append(float(np.mean([g[rng.integers(len(g),size=len(g))].mean() for g in used])))
   alpha=.05/3 if family=='geometry' and outcome=='latest' else .05
   contrasts.append(dict(family=family,updates=updates,outcome=outcome,comparison=left+' minus '+right,seeds=seed_effects,delta=float(used.mean()),interval=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),level=1-alpha,scope='conditional test-instance paired interval; not training-seed population uncertainty'))
state=read(A/'state.json');counts=dict(collections.Counter(x['status'] for k,x in state['jobs'].items() if k!='completion_report'))
write(A/'summary.json',dict(rows=rows,contrasts=contrasts,missing=missing,status_counts=counts,generated=time.time(),exploratory=True))
with (A/'metrics.csv').open('w') as f:
 if rows:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
lines=['# 后续模型库与几何训练结果','',f'已获得完整训练和latest测试：{sum(r["outcome"]=="latest" for r in rows)}/{len(sources)}。缺失{len(missing)}组。新队列状态：{counts}。','',
'几何训练由用户在开发门槛失败后明确启动，全部结果属于探索性结果；四臂同候选/精确参考/训练样本预算，G和H的额外actor计算成本另计。历史测试已被查看，不宣称新盲测。latest为主，validation-best为次级。','',
'|分支|更新|方法|种子|检查点|平均误差%|OOD宏平均%|最坏OOD%|','|---|---:|---|---:|---|---:|---:|---:|']
for r in rows:lines.append(f"|{r['family']}|{r['updates']}|{r['arm']}|{r['seed']}|{r['outcome']}|{r['mean']*100:.6f}|{r['mean_ood']*100:.6f}|{r['worst_ood']*100:.6f}|")
lines+=['','|分支|更新|检查点|对照差（负值更好）|差值pp|条件实例区间pp|种子数|','|---|---:|---|---|---:|---|---:|']
for r in contrasts:lines.append(f"|{r['family']}|{r['updates']}|{r['outcome']}|{r['comparison']}|{100*r['delta']:+.6f}|{[100*x for x in r['interval']]}|{len(r['seeds'])}|")
lines+=['','几何latest三项主对照区间水平98.333%，其他描述性区间95%；种子效应逐项保存在summary.json，不能用实例数量代替训练重复次数。成本以队列state.json各任务wall_seconds_total为新增进程耗时，长预算还须计入父阶段；几何训练日志call_ledger记录额外actor前向，不能将等参考预算当作等总计算预算。','', '缺失：'+', '.join(Path(r['result_dir']).name for r in missing)]
(A/'REPORT.md').write_text('\n'.join(lines)+'\n')
write(A/'report_complete.json',dict(status='report_generated',completed_sources=sum(r['outcome']=='latest' for r in rows),expected_sources=len(sources),missing=len(missing)))
