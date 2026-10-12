"""One-shot result inventory and direct Geo comparisons; no model inference."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def work_cost(folder):
 fields=['exact_train_nets','exact_train_seconds','teacher_forward_nets','teacher_seconds','refresh_seconds','refreshes','duplicate_snapshots_skipped']
 totals={k:0. for k in fields};baseline={}
 if (folder/'import_cost_baseline.json').exists():baseline=json.loads((folder/'import_cost_baseline.json').read_text())['accounting']
 previous=baseline.copy();history={};path=folder/'logs/events.jsonl'
 if not path.exists():return totals
 with path.open() as stream:
  for line in stream:
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if r.get('event')=='resume':previous=history.get(r.get('from_update'),baseline).copy()
   if r.get('event')!='train' or 'reference_accounting' not in r:continue
   current=r['reference_accounting']
   for k in fields:totals[k]+=max(0.,current.get(k,0.)-previous.get(k,0.))
   previous=current.copy();history[r['update']]=current.copy()
 return totals
def main():
 rows=[]
 for p in sorted((HERE/'artifacts').glob('*/dev.json')):
  d=json.loads(p.read_text());sel=d['last_validation']['selection'];row={'job':p.parent.name,'degree':d['degree'],'arm':d['arm'],'seed':d['seed'],'updates':d['updates'],'dev_worst_ood':sel['worst_group_mean'],'dev_nominal':sel['nominal_mean'],'best_step':d['best']['update'],'reference_accounting':d['reference_state']['accounting']}
  for outcome in ['latest','best']:
   path=p.parent/f'test_{outcome}.json'
   if path.exists():row['test_'+outcome]=json.loads(path.read_text())
  rows.append(row)
 # Read external matched Geo only for final report, not for training decisions.
 for seed in [101,102,103]:
  folder=HERE.parent/'checkpoint_bank_scale_20260930/artifacts'/f'geo_s{seed}'
  if not (folder/'complete.json').exists():continue
  d=json.loads((folder/'logs/result.json').read_text());sel=d['last_validation']['selection']
  row={'job':f'external_scale_geo2000_s{seed}','result_dir':str(folder),'external_reference':True,'degree':10,'arm':'geo','seed':seed,'updates':2000,'dev_worst_ood':sel['worst_group_mean'],'dev_nominal':sel['nominal_mean'],'best_step':d['best']['update']}
  for outcome in ['latest','best']:
   path=folder/f'test_{outcome}.json'
   if path.exists():row['test_'+outcome]=json.loads(path.read_text())
  rows.append(row)
 costs={folder.name:work_cost(folder) for folder in (HERE/'artifacts').iterdir() if folder.is_dir() and (folder/'logs/events.jsonl').exists()}
 comparisons=[]
 for r in rows:
  if r['arm']=='geo':continue
  controls=[g for g in rows if g['arm']=='geo' and (g['degree'],g['seed'],g['updates'])==(r['degree'],r['seed'],r['updates'])]
  if not controls:continue
  control=controls[0]
  for outcome in ['latest','best']:
   left=Path(r.get('result_dir',str(HERE/'artifacts'/r['job'])))/f'test_{outcome}';right=Path(control.get('result_dir',str(HERE/'artifacts'/control['job'])))/f'test_{outcome}'
   groups=[]
   for name in ['uniform','mix_gauss','edge_bias','grid_cluster']:
    if not (left/f'{name}.npz').exists() or not (right/f'{name}.npz').exists():break
    with np.load(left/f'{name}.npz') as a,np.load(right/f'{name}.npz') as b:
     if not np.array_equal(a['net_ids'],b['net_ids']):raise ValueError('paired IDs mismatch')
     groups.append(a['regret']-b['regret'])
   if len(groups)!=4:continue
   rng=np.random.default_rng(20260930);boot=[np.mean([x[rng.integers(len(x),size=len(x))].mean() for x in groups]) for _ in range(500)]
   comparisons.append({'job':r['job'],'geo':control['job'],'outcome':outcome,'mean_delta':float(np.concatenate(groups).mean()),'ci95_instance_bootstrap':np.quantile(boot,[.025,.975]).tolist(),'n_instances':sum(map(len,groups)),'bootstrap_replicates':500})
 expected=[j['id'] for j in json.loads((HERE/'jobs.json').read_text())['jobs'] if j['kind']=='train'];done={r['job'] for r in rows}
 seed_effects=[]
 for degree in [10,30]:
  for updates in [2000,10000]:
   for arm in ['k3','lagged3','qualifiedbest3']:
    for outcome in ['latest','best']:
     names={r['job'] for r in rows if r['degree']==degree and r['updates']==updates and r['arm']==arm};v=[r['mean_delta'] for r in comparisons if r['job'] in names and r['outcome']==outcome]
     if v:seed_effects.append({'degree':degree,'updates':updates,'arm':arm,'outcome':outcome,'actual_n_seeds':len(v),'mean_delta':float(np.mean(v)),'range':[float(min(v)),float(max(v))],'seed_standard_deviation':float(np.std(v,ddof=1)) if len(v)>1 else None})
 total_cost={k:sum(c.get(k,0.) for c in costs.values()) for k in ['exact_train_nets','exact_train_seconds','teacher_forward_nets','teacher_seconds','refresh_seconds','refreshes']}
 result={'new_work_cost_by_job':costs,'new_work_cost_total':total_cost,'actual_seed_effects':seed_effects,'rows':rows,'paired_geo_comparisons':comparisons,'incomplete_or_gated':[j for j in expected if j not in done],'interpretation':'Final/latest is primary, selected best secondary; no equivalence margin predefined, no statistical equivalence claim; per-instance bootstrap is conditional on each trained seed; actual seed ranges only'}
 for name,value in [('REPORT.json',json.dumps(result,indent=2)),('REPORT.md', '# Weekly bank experiment results\n\n'+ '\n'.join(f"- {r['job']}: updates={r['updates']}, dev worst OOD={r['dev_worst_ood']:.8f}, best step={r['best_step']}; test completed={bool(r.get('test_latest'))}." for r in rows)+'\n\nNew-work oracle/teacher/refresh costs: '+json.dumps(total_cost)+'\n\nActual seed effects: '+json.dumps(seed_effects)+'\n\nPaired comparisons: '+json.dumps(comparisons)+'\n\nIncomplete or gated: '+', '.join(result['incomplete_or_gated'])+'\n\n'+result['interpretation']+'\n')]:
  dst=HERE/name;tmp=dst.with_suffix(dst.suffix+'.partial');tmp.write_text(value);tmp.replace(dst)
 (HERE/'report_complete.json').write_text(json.dumps({'complete':True,'status':'complete','reported_runs':len(rows),'incomplete':len(result['incomplete_or_gated'])}))
if __name__=='__main__':main()
