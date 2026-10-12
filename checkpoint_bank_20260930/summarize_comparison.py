"""Independent post-experiment report; run once after the queue finishes."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
ARMS=['geo','k1','k3','k5','refresh3']
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']
def read(p):return json.loads(p.read_text())
def paired(a,b,seed):
 deltas=[];groups={}
 with np.load(a) as aa,np.load(b) as bb:
  for g in GROUPS:
   ids=aa[g+'_ids'];other=bb[g+'_ids']
   if len(ids)!=len(other) or not np.array_equal(ids,other):raise ValueError('paired instance IDs mismatch')
   d=aa[g+'_regret']-bb[g+'_regret'];deltas.append(d);groups[g]=float(d.mean())
 rng=np.random.default_rng(seed)
 # Stratified instance bootstrap: each independent net is a resampling unit.
 # 500 bootstrap replicates, NOT 500 observed independent instances.
 replicates=[]
 for _ in range(500):replicates.append(float(np.mean([d[rng.integers(len(d),size=len(d))].mean() for d in deltas])))
 all_d=np.concatenate(deltas)
 return {'n_instances':len(all_d),'mean_delta':float(all_d.mean()),'group_deltas':groups,'ci95_instance_bootstrap':np.quantile(replicates,[.025,.975]).tolist(),'bootstrap_replicates':500}
def main():
 rows=[];missing=[]
 for seed in [101,102]:
  geo=ROOT/'artifacts'/f'geo_s{seed}'
  for arm in ARMS:
   out=ROOT/'artifacts'/f'{arm}_s{seed}'
   if not (geo/'complete.json').exists() or not (out/'complete.json').exists():missing.append(f'{arm}_s{seed}');continue
   complete=read(out/'complete.json');train=[json.loads(l) for l in (out/'train.jsonl').read_text().splitlines() if l.strip()]
   # Resume logs can repeat rolled-back updates; count only last completed update record per step.
   by_step={r['step']:r for r in train}
   queue=[]
   if (ROOT/'artifacts/queue.jsonl').exists():queue=[json.loads(l) for l in (ROOT/'artifacts/queue.jsonl').read_text().splitlines() if l.strip()]
   successful=[r for r in queue if r['arm']==arm and r['seed']==seed and r['exit_code']==0]
   validation=[json.loads(l) for l in (out/'validation.jsonl').read_text().splitlines() if l.strip()]
   matched=[v['step'] for v in validation if v.get('groups')==complete['best_validation'].get('groups') and v.get('score')==complete['best_validation'].get('score')]
   best_step=min(matched) if matched else None
   row={'best_selected_step':best_step,'arm':arm,'seed':seed,'best':paired(out/'best_test.npz',geo/'best_test.npz',seed),'final':paired(out/'final_test.npz',geo/'final_test.npz',seed),'ledger':complete['ledger'],'training_update_seconds_sum':sum(r['seconds'] for r in by_step.values()),'successful_queue_seconds':successful[-1]['seconds'] if successful else None,'offline_oracle_nets':read(out/'offline_bank_diagnostic.json')['oracle_calls']}
   rows.append(row)
 aggregate=[]
 for arm in ARMS:
  rr=[r for r in rows if r['arm']==arm]
  if not rr:continue
  a={'arm':arm,'n_seeds':len(rr)}
  for outcome in ['best','final']:
   v=[r[outcome]['mean_delta'] for r in rr];a[outcome]={'mean':float(np.mean(v)),'range':[float(min(v)),float(max(v))]}
  aggregate.append(a)
 result={'pairing':'same fixed test net IDs within each seed/group; error measured against original cached GeoSteiner reference','delta':'arm minus Geo; negative favors bank arm','bootstrap':'500 replicates, stratified by geometry group, independent instance unit; conditional on trained seed; not a across-seed CI','equivalence':'No equivalence margin prespecified; confidence intervals including zero do not establish statistical equivalence','rows':rows,'seed_summary':aggregate,'incomplete':missing}
 tmp=ROOT/'COMPARISON.json.partial';tmp.write_text(json.dumps(result,indent=2));tmp.replace(ROOT/'COMPARISON.json')
 lines=['# Checkpoint bank versus GeoSteiner reference training','','All differences are arm minus Geo on the same held-out instance IDs; negative means lower true-reference regret. PRIMARY comparison is final update200; validation-selected best is secondary and may select the unchanged step0 parent. Identical best-parent results do not show equivalent learning. Both use the original cached reference, not the bank denominator.','','| Arm | Seed | Final200 delta [95% instance CI] | Best selected step | Best delta [95% instance CI] | Training Geo nets | Teacher forward nets | Teacher seconds | Update seconds | Successful job seconds |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 def cell(v):return f"{v['mean_delta']:+.8f} [{v['ci95_instance_bootstrap'][0]:+.8f}, {v['ci95_instance_bootstrap'][1]:+.8f}]"
 for r in rows:
  l=r['ledger'];lines.append(f"| {r['arm']} | {r['seed']} | {cell(r['final'])} | {r['best_selected_step']} | {cell(r['best'])} | {l['geosteiner_train_nets']} | {l['teacher_forward_nets']} | {l['teacher_seconds']:.2f} | {r['training_update_seconds_sum']:.2f} | {r['successful_queue_seconds']} |")
 lines+=['','| Arm | Observed seeds | Best delta mean [range] | Final delta mean [range] |','|---|---:|---:|---:|']
 for a in aggregate:lines.append(f"| {a['arm']} | {a['n_seeds']} | {a['best']['mean']:+.8f} {a['best']['range']} | {a['final']['mean']:+.8f} {a['final']['range']} |")
 lines+=['','Confidence intervals use 500 bootstrap replicates of instances, stratified by the four fixed groups (1024 observed instances per completed comparison). They describe instance uncertainty conditional on each trained model, not seed-to-seed uncertainty. Seed summaries report only actual available seeds; two seeds are too few for a strong training-variance claim. No equivalence margin was specified, so these results cannot establish statistical equivalence.','','Offline Geo-arm diagnostic compares its actor against a separate K3 teacher bank; its agreement score is not a measure of the exact reference agreeing with itself. Other arms use their own banks, so different diagnostic banks must be acknowledged. Training Geo nets exclude the offline candidate diagnostic (288 oracle nets per completed arm) and cached validation/test labels. Teacher-forward nets are nominal scored nets; fixed-batch padding is not added to that counter. Teacher seconds include teacher inference and CPU REST length evaluation. Update seconds exclude periodic validation, test and offline diagnostic; successful queue seconds include the successful process run and can exclude earlier interrupted segments. Historical teachers and validation remain oracle-assisted.','','Incomplete: '+(', '.join(missing) if missing else 'none')]
 tmp=ROOT/'COMPARISON.md.partial';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(ROOT/'COMPARISON.md')
if __name__=='__main__':main()
