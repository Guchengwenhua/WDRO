"""Summarize fixed milestones; test outcomes never feed back into the queue."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv,collections,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;A=P/'artifacts';GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']
def read(p):return json.loads(p.read_text())
def write(p,x):
 q=p.with_suffix('.json.partial');q.write_text(json.dumps(x,indent=2,ensure_ascii=False));q.replace(p)
sources=read(P/'sources.json');state=read(A/'state.json');rows=[];arrays={};missing=[]
for src in sources:
 p=Path(src['result_dir'])
 if not (p/'complete.json').exists():missing.append(src);continue
 c=read(p/'complete.json');run=p.parent.name;jid=run+f"_u{src['updates']}";wall=state['jobs'][jid].get('wall_seconds_total',0)
 if src['updates']==625:wall+=state['jobs'][run+'_u125'].get('wall_seconds_total',0)
 for outcome in ['latest','best']:
  v=read(p/f'test_{outcome}.json');gs=[];ids=[]
  for g in GROUPS:
   with np.load(p/f'test_{outcome}'/f'{g}.npz') as z:gs.append(z['length']/z['reference']-1);ids.append(z['net_ids'])
  arrays[src['degree'],src['updates'],src['arm'],src['seed'],outcome]=(np.stack(gs),ids)
  rows.append(dict(degree=src['degree'],updates=src['updates'],arm=src['arm'],seed=src['seed'],outcome=outcome,mean=float(np.mean(gs)),ood=float(np.mean(gs[1:])),worst_ood=max(float(x.mean()) for x in gs[1:]),macro_p95=float(np.mean([np.quantile(x,.95) for x in gs])),wall_hours=wall/3600,checkpoint_update=v['checkpoint_update'],phase_accounting=c['result'].get('method_state',{}).get('phase_accounting'),final_lambda=c['result'].get('method_state',{}).get('lambda')))
contrasts=[]
for degree in [10,30]:
 for u in [125,625]:
  for outcome in ['latest','best']:
   for base in ['original','random','entropy']:
    ds=[];seed_effects={}
    for seed in [101,102,103]:
     ka=(degree,u,'geometry',seed,outcome);kb=(degree,u,base,seed,outcome)
     if ka not in arrays or kb not in arrays:continue
     a,ai=arrays[ka];b,bi=arrays[kb]
     if not all(np.array_equal(x,y) for x,y in zip(ai,bi)):raise ValueError('Paired instance IDs differ')
     d=(a-b)[1:];ds.append(d);seed_effects[str(seed)]=float(d.mean())
    if not ds:continue
    d=np.mean(ds,axis=0);rng=np.random.default_rng(20261012);boot=[]
    for _ in range(5000):boot.append(float(np.mean([g[rng.integers(len(g),size=len(g))].mean() for g in d])))
    alpha=.05/6 if u==625 and outcome=='latest' else .05
    contrasts.append(dict(degree=degree,updates=u,outcome=outcome,contrast='geometry minus '+base,delta=float(d.mean()),interval=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),level=1-alpha,seeds=seed_effects,scope='conditional test-instance uncertainty; not training-seed population CI'))
write(A/'summary.json',dict(rows=rows,contrasts=contrasts,missing=missing,generated=time.time(),exploratory=True))
with (A/'metrics.csv').open('w') as f:
 if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# GPU几何熵扩大训练结果','',f'完成阶段：{len(sources)-len(missing)}/{len(sources)}；625更新终点：{sum(r["updates"]==625 and r["outcome"]=="latest" for r in rows)}/24。未完成阶段不能计作成功。','',
'两种degree，四臂，三种子；batch4096、125/625更新。G/H仅在GPU评分；4进程GeoSteiner。旧CPU结果仅作历史背景，新旧耗时差包含多项实现变化。','',
'|degree|更新|方法|实际种子数|latest OOD误差均值%|种子SD|平均进程小时|','|---:|---:|---|---:|---:|---:|---:|']
for degree in [10,30]:
 for u in [125,625]:
  for arm in ['original','random','entropy','geometry']:
   rr=[r for r in rows if (r['degree'],r['updates'],r['arm'],r['outcome'])==(degree,u,arm,'latest')]
   if rr:lines.append(f"|{degree}|{u}|{arm}|{len(rr)}|{np.mean([x['ood'] for x in rr])*100:.6f}|{np.std([x['ood']*100 for x in rr],ddof=1) if len(rr)>1 else 'NA'}|{np.mean([x['wall_hours'] for x in rr]):.3f}|")
lines+=['','主比较是625更新的G−original/random/H，共6项，使用99.1667%条件实例区间；125与best为次级描述结果。种子效应及区间见summary.json。GPU训练保留FP32；G数学定义保留，但浮点计算顺序发生变化，不宣称逐位等价。历史测试已见，全部属于探索性研究。','', '缺失阶段：'+', '.join(str(x['result_dir']) for x in missing)]
(A/'REPORT.md').write_text('\n'.join(lines)+'\n');write(A/'report_complete.json',dict(status='report_generated',complete_stages=len(sources)-len(missing),missing=len(missing)))
