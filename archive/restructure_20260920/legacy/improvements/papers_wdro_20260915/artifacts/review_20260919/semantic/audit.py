"""Read-only semantic audit of frozen runs; writes only this audit directory."""
from pathlib import Path
import json,sys,hashlib,datetime
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT))
torch.set_num_threads(2)
P=json.loads((ROOT/'protocol.json').read_text())
FAMILIES=P['families'];SEEDS=P['seeds'];DISTS=list(P['distributions'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jload(p):return json.loads(p.read_text())
def state_equal(a,b):return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
rows=[];objectives=[];paired=[];initializations=[]
for family in FAMILIES:
 for seed in SEEDS:
  initial_by_method={}
  for method in P['methods']:
   run=ROOT/'artifacts/runs'/f'{family}_{method}_s{seed}'
   m=jload(run/'metrics.json')
   logs=[json.loads(x) for x in (run/'progress.jsonl').read_text().splitlines()]
   assert [x['update'] for x in logs]==list(range(1,P['updates']+1))
   validation=[json.loads(x) for x in (run/'validation.jsonl').read_text().splitlines()]
   assert [x['update'] for x in validation]==list(range(0,P['updates']+1,P['validation_every']))
   scores=[max(x['summary'][d]['cost']['mean'] for d in DISTS if d!='uniform') for x in validation]
   chosen=int(np.argmin(scores));choice_update=validation[chosen]['update']
   assert choice_update==m['best']['update']
   rho=m['identity']['rho']
   lam=np.array([x['lambda'] for x in logs]);c=np.array([x['transport_mean'] for x in logs]);prior=np.r_[P['lambda_initial'],lam[:-1]]
   row={'family':family,'method':method,'seed':seed,'selected_update':choice_update,
        'selection_objective':scores[chosen],'initial_validation_objective':scores[0],
        'latest_validation_objective':scores[-1],'selected_initialization':choice_update==0,
        'selected_terminal_update':choice_update==P['updates'],'rho':rho,
        'progress_rows':len(logs),'validation_rows':len(validation)}
   if method=='wdro':
    expected=np.maximum(0,prior+P['lambda_lr']*(c-rho))
    row.update(lambda_update_max_abs_error=float(abs(lam-expected).max()),
       lambda_pre_zero_fraction=float(np.mean(prior==0)),lambda_post_zero_fraction=float(np.mean(lam==0)),
       lambda_final=float(lam[-1]),lambda_max=float(lam.max()),
       mean_transport=float(c.mean()),mean_transport_over_rho=float(c.mean()/rho),
       last256_transport_over_rho=float(c[-256:].mean()/rho),batch_overbudget_fraction=float(np.mean(c>rho)),
       center_selection_fraction=float(np.mean([np.mean(np.array(x['selected_columns'])==0) for x in logs])))
   attack=[]
   for d in DISTS:
    a=m['attack'][d]
    attack.append({'distribution':d,'risk':a['risk'],'average_cost':a['average_cost'],
       'budget_ratio':a['average_cost']/rho,'risk_minus_nominal':a['risk']-a['nominal'],
       'unconstrained_max_minus_risk':a['unconstrained_max']-a['risk']})
   row['test_lp']=attack;rows.append(row)
   initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)['adapter']
   initial_by_method[method]=initial
   selected=torch.load(run/'best.pt',map_location='cpu',weights_only=False)['adapter']
   row['selected_parameters_equal_initial']=state_equal(initial,selected)
   del selected
   for filename in ['test_best.npz','test_initial.npz']:
    with np.load(run/filename,allow_pickle=False) as ar:
     for d in DISTS:
      get=lambda k:ar[d+'__'+k]
      if family=='rest':value=get('length')/get('lstar')-1
      elif family=='timing':value=((1-P['parameters']['timing']['alpha'])*get('wirelength')+P['parameters']['timing']['alpha']*get('max_path'))/(2*(P['degree']-1))
      elif family=='hub':value=(P['parameters']['hub']['beta']*get('wirelength')+(1-P['parameters']['hub']['beta'])*get('skew'))/(2*(P['degree']-1))
      else:value=(get('delay_raw_seconds')*P['parameters']['buffer']['delay_scale']+P['parameters']['buffer']['buffer_penalty']*get('buffers'))/P['parameters']['buffer']['objective_scale']
      error=float(np.max(np.abs(get('cost')-value.astype(np.float32).astype(np.float64))))
      raw_error = float(np.max(np.abs(get('cost')-(get('raw_objective')/(2*(P['degree']-1))).astype(np.float32).astype(np.float64)))) if family=='hub' else error
      relative_error = float(np.max(np.abs(get('cost')-value) / np.maximum(np.abs(value),1e-30)))
      objectives.append({'run':run.name,'file':filename,'distribution':d,'nets':len(value),'max_abs_component_reconstructed_error_after_float32':error,'max_relative_component_reconstructed_error':relative_error,'max_abs_stored_raw_objective_error_after_float32':raw_error})
      assert raw_error==0,(run,filename,d,raw_error)
      assert np.allclose(get('cost'),value,rtol=4*np.finfo(np.float32).eps,atol=0),(run,filename,d,relative_error)
   with np.load(run/'test_best.npz') as best,np.load(run/'test_initial.npz') as init:
    row['best_minus_initial']={d:{'cost_mean_difference':float((best[d+'__cost']-init[d+'__cost']).mean()),'prediction_equal_fraction':float(np.mean(best[d+'__cost']==init[d+'__cost']))} for d in DISTS}
    if family=='hub':row['selected_hub_mean']={d:float(best[d+'__hub_count'].mean()) for d in DISTS}
    if family=='buffer':row['buffer_metrics']={d:{k:float(best[d+'__'+k].mean()) for k in ['delay_scaled','buffers','length','skew_scaled']} for d in DISTS}
  initializations.append({'family':family,'seed':seed,'paired_initial_parameters_equal':state_equal(initial_by_method['erm'],initial_by_method['wdro'])})
  del initial_by_method
  with np.load(ROOT/'artifacts/runs'/f'{family}_erm_s{seed}'/'test_best.npz') as erm,np.load(ROOT/'artifacts/runs'/f'{family}_wdro_s{seed}'/'test_best.npz') as wdro:
   paired.append({'family':family,'seed':seed,'distributions':{d:{'cost_wdro_minus_erm':float((wdro[d+'__cost']-erm[d+'__cost']).mean()),'prediction_equal_fraction':float(np.mean(wdro[d+'__cost']==erm[d+'__cost']))} for d in DISTS}})
# Re-run existing CPU hand calculations without modifying its historic output.
source=ROOT/'test_adapter_buffer.py'
ns={'__file__':str(OUT/'test_adapter_buffer.py'),'__name__':'semantic_readonly_probe'}
exec(compile(source.read_text(),str(source),'exec'),ns)
physical=ns['run']()
summary={'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'PASS_WITH_SCIENTIFIC_LIMITATIONS','scope':'semantic interpretation and component checks; not standalone full lifecycle/statistical acceptance',
 'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'protocol.json',ROOT/'runner.py',ROOT/'core.py',ROOT/'adapter_buffer.py',ROOT/'adapter_timing.py',ROOT/'adapter_hub.py',ROOT/'vendor_harrl/evaluator.py',ROOT/'artifacts/report/REPORT.md']},
 'formal_runs':rows,'paired_initializations':initializations,'paired_selected_predictions':paired,
 'physical_component_checks':physical,'cost_formula_checks':objectives,
 'test_predictions_formula_checked':sum(x['nets'] for x in objectives),
 'critical_limitations':['All three REST WDRO selected update 0: no learned pure-wirelength gain established.',
 'Timing/Hub/Buffer use different fixed-scale objectives and action spaces; compare ERM/WDRO only within family.',
 'Buffer uses categorical phases and upstream local delay-gain greedy insertion, not the paper full optimal DP.',
 'Training sampled policy cost and inner-search greedy cost share physical formula but differ in policy evaluation.',
 'Finite-row-support LP is exact only for each-center designated candidate transport edges, not a continuous certificate.',
 'Equal updates are not equal compute; WDRO has four extra true greedy evaluations per net per step.',
 'Only three training seeds; net-conditional intervals do not resolve training variability.',
 'Training projected-dual updates do not enforce exact per-step or finite-time average feasibility.']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print('WROTE',OUT/'summary.json','formal',len(rows),'formula_predictions',summary['test_predictions_formula_checked'])
