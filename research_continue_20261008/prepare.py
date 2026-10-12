"""Freeze the explicitly authorized continuation, without editing previous runs."""
import json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
B=R/'bank_continuation_20261008';G=R/'geometry_training_20261008';OLD=R/'accelerated_bank_20261006'
PY='/remote-home/Guilin/miniconda3/envs/py311/bin/python'
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
if (P/'jobs.json').exists():raise SystemExit('Preparation already frozen; use launch.sh to resume')
for folder in [P,B,G]:
 folder.mkdir(exist_ok=True);target=Path('/attached/remote-home1/Guilin/DRO/Codex')/folder.name/'artifacts';target.mkdir(parents=True,exist_ok=True)
 (folder/'artifacts').symlink_to(target)
for dst,src in [(B,OLD),(G,R/'accelerated_geometry_20261006')]:
 for f in src.glob('*.py'):shutil.copy2(f,dst/f.name)
 for f in src.glob('inputs*.json'):shutil.copy2(f,dst/f.name)
 shutil.copy2(src/'protocol.json',dst/'protocol.json')
# Keep the previous bank RNG namespace and exact scientific settings for continuation.
# Geometry is an explicit exploratory cohort with all four arms newly initialized.
config=json.loads((G/'protocol.json').read_text());config.update(protocol_id='geometry_exploratory_20261008',geometry_training_gate='explicit user override; exploratory training after failed development gate',geometry_prefix_updates=None)
write(G/'protocol.json',config)
f=G/'accelerated_config.py';s=f.read_text().replace("Path(__file__).resolve().parent.parent/'accelerated_research_20261006/protocol.json'","Path(__file__).resolve().parent/'protocol.json'");f.write_text(s)
write(G/'artifacts/gate.json',dict(allow=True,metric='G',reason='User explicitly requested direct geometry and ordinary entropy training after acceptance; original failed gate retained unchanged',exploratory=True,original_gate=str(R/'accelerated_geometry_20261006/artifacts/gate.json')))
f=G/'run_train.py';s=f.read_text().replace("run_id=f'week_geometry_{args.arm}_s{args.seed}'","run_id=f'geometry_exploratory_20261008_{args.arm}_s{args.seed}'")
s=s.replace("'arm':args.arm,'seed':args.seed,'protocol':protocol","'arm':args.arm,'seed':args.seed,'exploratory':True,'gate_override':gate,'protocol':protocol")
f.write_text(s)
# Bound the ordinary entropy expression at a single feasible action.
f=G/'geometry_policy.py';s=f.read_text().replace("h=-(probability*np.log(np.clip(probability,1e-300,None))).sum(-1)/np.log(np.isfinite(logits).sum(-1))","kvalid=np.isfinite(logits).sum(-1);raw=-(probability*np.log(np.clip(probability,1e-300,None))).sum(-1)\n        h=np.divide(raw,np.log(np.maximum(kvalid,2)),out=np.zeros_like(raw),where=kvalid>1)")
s=s.replace("indices=np.concatenate([np.zeros((b,1),int)","if not np.isfinite(scores).all():raise RuntimeError('nonfinite shortlist scores')\n  indices=np.concatenate([np.zeros((b,1),int)")
f.write_text(s)
policy=dict(run_until_complete=True,deadline_utc=None,gpus=[2,3,4,5,6,7],max_gpu_jobs=6,max_cpu_jobs=1,gpu_min_free_mib=28000,gpu_max_utilization=20,cuda_fraction=.4,foreign_gpu_reserve_mib=8192,root_min_free_gib=3,attached_min_free_gib=200,new_artifact_cap_gib=200,max_attempts=2,poll_seconds=10)
write(P/'policy.json',policy)
protocol=dict(cohort='continuation_and_exploratory_geometry_20261008',authorization='User requested autonomous bank continuation and direct geometry/ordinary-entropy training',bank=dict(degree=30,arms=['geo','k3','lagged3'],seeds=[101,102,103],batch=4096,short_updates=125,long_updates=625,short_samples=512000,long_samples=2560000,repairs='import failed seed103 checkpoints into new output dirs; original failures retained',rng_namespace='accelerated_largebatch_20261006',selection='fixed latest primary; validation-best secondary; no choice based on test',changes='d30 fixed K3 added; all three arms extended equally to625'),geometry=dict(degree=10,arms=['original','random','entropy','geometry'],seeds=[101,102,103],batch=4096,updates=125,samples=512000,candidates=33,exact_shortlist=4,metric='G versus Hnorm',rng_namespace=config['protocol_id'],gate_override=True,classification='exploratory after failed development gate; no claim of pre-gate confirmation',primary='latest125 macro OOD true regret; geometry minus each of other3 arms',statistics='paired per-instance bootstrap on seed-averaged differences; 98.333% Bonferroni intervals for3 primary contrasts; seed effects separately',secondary='validation-selected best, nominal/worstOOD/P95, lambda, cost, difficulty correlations',budget_scope='equal sample/update/candidate/exact-reference budgets; entropy and geometry require extra actor work, report separately'),storage='All artifacts and temporary caches on attached disk, project symlinks; root free threshold3GiB because only small source files on root',limitations=['Historical test data have been inspected; this is an exploratory continuation, not a new blinded holdout.','No population claim from three seeds; no equivalence margin invented.','d30 admission requires44GiB free and low current utilization; future foreign jobs can still change availability.'])
write(P/'protocol.json',protocol)
jobs=[];sources=[]
def bank(arm,seed,updates,src=None,deps=(),priority=20):
 jid=f'd30_{arm}{updates}_s{seed}';out=B/'artifacts'/jid
 argv=[PY,'-u','-B','run_week.py','--arm',arm,'--seed',str(seed),'--degree','30','--updates',str(updates),'--job-id',jid]
 if src:argv+=['--import-from',str(src)]
 meta=dict(family='bank',arm=arm,seed=seed,degree=30,updates=updates,result_dir=str(out),parent=str(src) if src else None)
 j=dict(id=jid,kind='train',cwd=str(B),argv=argv,complete=str(out/'complete.json'),depends_on=list(deps),priority=priority,max_wall_seconds=172800,gpu_min_free_mib=44000,cuda_fraction=.55,**meta);jobs.append(j)
 jobs.append(dict(j,id=jid+'_test',kind='evaluate',argv=argv+['--evaluate-only'],complete=str(out/'evaluation_complete.json'),depends_on=[jid],priority=5,gpu_min_free_mib=28000,cuda_fraction=.4,max_wall_seconds=14400))
 sources.append(meta);return jid,out
# Schedule geometry/entropy first so the requested new arms start immediately.
for seed in [101,102,103]:
 for arm in ['geometry','entropy','original','random']:
  jid=f'geometry_{arm}_s{seed}';out=G/'artifacts/train'/f'{arm}_s{seed}'
  meta=dict(family='geometry',arm=arm,seed=seed,degree=10,updates=125,result_dir=str(out),parent=None)
  jobs.append(dict(id=jid,kind='train',cwd=str(G),argv=[PY,'-u','-B','run_train.py','--arm',arm,'--seed',str(seed)],complete=str(out/'complete.json'),depends_on=[],priority=0 if seed==101 and arm in ['geometry','entropy'] else 10,max_wall_seconds=172800,gpu_min_free_mib=28000,cuda_fraction=.4,**meta));sources.append(meta)
parents={}
for arm,oldarm in [('geo','geo'),('lagged3','selected')]:
 for seed in [101,102,103]:
  old=OLD/'artifacts'/f'd30_{oldarm}125_s{seed}'
  if seed==103:
   jid,out=bank(arm,seed,125,old,priority=1);parents[arm,seed]=(jid,out)
  else:
   parents[arm,seed]=(None,old);sources.append(dict(family='bank',arm=arm,seed=seed,degree=30,updates=125,result_dir=str(old),parent=None,historical=True))
for seed in [101,102,103]:
 jid,out=bank('k3',seed,125,priority=15);parents['k3',seed]=(jid,out)
for seed in [101,102,103]:
 for arm in ['geo','k3','lagged3']:
  dep,src=parents[arm,seed];bank(arm,seed,625,src,[dep] if dep else [],priority=20)
jobs.append(dict(id='completion_report',kind='summarize',cwd=str(P),argv=[PY,'-u','-B','summarize.py'],complete=str(P/'artifacts/report_complete.json'),depends_on=[j['id'] for j in jobs],priority=50,always_run=True,max_wall_seconds=7200))
write(P/'jobs.json',dict(jobs=jobs));write(P/'sources.json',sources)
# Persistent scheduler reuse, with per-job resource limits and avoiding a failed GPU on retry.
s=(R/'accelerated_research_20261006/controller.py').read_text()
s=s.replace("ROOT/'accelerated_bank_20261006/artifacts',ROOT/'accelerated_geometry_20261006/artifacts'","ROOT/'bank_continuation_20261008/artifacts',ROOT/'geometry_training_20261008/artifacts'")
s=s.replace('def gpus():','def gpus(j):')
s=s.replace("r[0] in CFG['gpus'] and r[0] not in reserved", "r[0] in CFG['gpus'] and r[0] not in reserved and r[0] not in state['jobs'][j['id']].get('failed_gpus',[])")
s=s.replace("CFG['gpu_min_free_mib'],r[2]*CFG['cuda_fraction']", "j.get('gpu_min_free_mib',CFG['gpu_min_free_mib']),r[2]*j.get('cuda_fraction',CFG['cuda_fraction'])")
s=s.replace("RESEARCH_CUDA_FRACTION=str(CFG['cuda_fraction'])", "RESEARCH_CUDA_FRACTION=str(j.get('cuda_fraction',CFG['cuda_fraction']))")
s=s.replace("  r.update(status='pending' if transient", "  if 'CUDA out of memory' in tail:r.setdefault('failed_gpus',[]).append(a['gpu'])\n  r.update(status='pending' if transient")
s=s.replace('available=gpus()','available=gpus(j)')
(P/'controller.py').write_text(s)
# Independent correlation sidecar follows this new job manifest without feedback.
shutil.copy2(R/'accelerated_research_20261006/monitor_geometry.py',P/'monitor_geometry.py')
launch='''#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/research_continue_20261008
tmux -S "$PWD/artifacts/parallel.sock" new-session -d -s research-continue-20261008 "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B controller.py >> artifacts/controller.log 2>&1"
tmux -S "$PWD/artifacts/parallel.sock" new-window -t research-continue-20261008 -n correlation "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B monitor_geometry.py >> artifacts/geometry_monitor.log 2>&1"
'''
(P/'launch.sh').write_text(launch)
print(json.dumps(dict(jobs=len(jobs),train=sum(j['kind']=='train' for j in jobs),evaluation=sum(j['kind']=='evaluate' for j in jobs),sources=len(sources))))
