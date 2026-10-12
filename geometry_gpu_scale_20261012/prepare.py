"""Create an isolated GPU entropy scale-up with matched controls."""
import json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;SRC=R/'geometry_training_20261008'
PY='/remote-home/Guilin/miniconda3/envs/py311/bin/python'
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
if (P/'jobs.json').exists():raise SystemExit('Already prepared; resume through launch.sh')
target=Path('/attached/remote-home1/Guilin/DRO/Codex')/P.name/'artifacts';target.mkdir(parents=True,exist_ok=True)
(P/'artifacts').symlink_to(target)
for name in ['adapters.py','faithful_trainer.py','accelerated_config.py']:shutil.copy2(SRC/name,P/name)
cfg=dict(protocol_id='geometry_gpu_scale_20261012',classification='exploratory',degrees=[10,30],arms=['original','random','entropy','geometry'],seeds=[101,102,103],training_batch=4096,updates=625,milestones=[125,625],sample_updates_per_run=2560000,total_run_count=24,total_sample_updates=61440000,
 score_microbatch=1024,ranking_microbatch=4096,validation_batch=1024,validation_interval=6,test_batch=1024,
 geosteiner_workers=4,learning_rate=5e-5,actor_gradient_precision='FP32, TF32 disabled',score_precision='FP64 probabilities, distances and accumulation; FP32 route construction as historical code',
 score_implementation='only G or H on CUDA; streaming decoder hooks; upper-triangle action pairs; one host transfer at end of candidate scoring',
 selection='original plus top3; unchanged 33 candidates and 4 exact labels per center',
 primary='latest625 OOD macro relative regret; G minus original/random/H at each degree',
 statistics='six main contrasts: paired instance bootstrap conditional on seed-averaged models, Bonferroni99.1667%; seed effects separately',
 secondary='latest125, validation-best at125/625, worstOOD, P95, runtime, final lambda, G/H difficulty correlation',
 limitation='same mathematical G definition, different floating-point evaluation order; no bitwise equivalence claim; old tests already seen; no blind generalization claim',
 speed_scope='formal training phase timings; no separate benchmark. Historical timing combines GPU scoring, microbatch and oracle parallelism, not an isolated kernel comparison',
 authorization='User requests geometry optimization and larger training; four matched arms and previous gate override retained')
write(P/'protocol.json',cfg)
# Degree-general initialization, safe spawned oracle workers, reference timing.
f=P/'adapters.py';s=f.read_text().replace("LIB.parents[1]/'save/DAC21/rsmt10b.pt'","LIB.parents[1]/f\"save/DAC21/rsmt{SETTINGS['degree']}b.pt\"")
s=s.replace('rest.build_actor(10,self.device)',"rest.build_actor(SETTINGS['degree'],self.device)")
s=s.replace("geosteiner.GeoSteinerEvaluator(LIB) if SETTINGS.get('arm','geo')=='geo' else None","geosteiner.ParallelGeoSteinerEvaluator(SETTINGS['geosteiner_workers'],LIB) if SETTINGS.get('arm','geo')=='geo' else None")
s=s.replace(' def close(self):pass\n\ndef make_geosteiner()', ' def close(self):\n  if self.solver is not None:self.solver.close()\n\ndef make_geosteiner()')
s=s.replace("self.accounting['exact_train_nets']+=len(pins);return evaluator.lengths(pins),len(pins)","t=time.monotonic();values=evaluator.lengths(pins);self.accounting['exact_train_nets']+=len(pins);self.accounting['exact_train_seconds']=self.accounting.get('exact_train_seconds',0.)+time.monotonic()-t;return values,len(pins)")
f.write_text(s)
f=P/'faithful_trainer.py';s=f.read_text().replace('expected_updates != 125','expected_updates != 625')
s=s.replace("identity['geometry_shortlist'] = dict(arm=SETTINGS['geometry_arm'],metric=SETTINGS['geometry_metric'],candidate_budget=32,exact_budget=4)","identity['geometry_shortlist'] = dict(arm=SETTINGS['geometry_arm'],metric=SETTINGS['geometry_metric'],candidate_budget=32,exact_budget=4,implementation='gpu_upper_triangle_fp64_v1',ranking_microbatch=SETTINGS['ranking_microbatch'])")
s=s.replace('validation_due = update % validation_interval == 0 or update == expected_updates','validation_due = update % validation_interval == 0 or update in (125,625)')
s=s.replace('    for update in range(start_update, stop_after + 1):\n        batch =', '    for update in range(start_update, stop_after + 1):\n        update_started=time.monotonic()\n        batch =')
s=s.replace('        center_batch = batch\n', '        materialize_seconds=time.monotonic()-update_started\n        center_batch = batch\n')
s=s.replace('        outputs, log_probs = actor(batch.pins, False)', '        optimizer_started=time.monotonic()\n        outputs, log_probs = actor(batch.pins, False)')
s=s.replace('        save_best = False\n', '        save_best = False\n')
s=s.replace('                    "actor_loss": float(actor_loss.detach().cpu()),','                    "materialize_seconds": materialize_seconds,\n                    "update_compute_seconds": time.monotonic()-update_started,\n                    "optimization_seconds": time.monotonic()-optimizer_started,\n                    "actor_loss": float(actor_loss.detach().cpu()),')
s=s.replace('                    "reference_accounting": reference_cache.accounting.copy(),\n                    "latest_checkpoint"', '                    "reference_accounting": reference_cache.accounting.copy(),\n                    "shortlist_seconds": method_detail.get("shortlist_wall_seconds"),\n                    "search_seconds": method_detail.get("search_wall_seconds"),\n                    "update_seconds_including_validation": time.monotonic()-update_started,\n                    "latest_checkpoint"')
f.write_text(s)
policy=dict(run_until_complete=True,deadline_utc=None,gpus=[2,3,4,5,6,7],max_gpu_jobs=6,max_cpu_jobs=1,gpu_min_free_mib=35000,gpu_max_utilization=20,cuda_fraction=.55,foreign_gpu_reserve_mib=8192,root_min_free_gib=8,attached_min_free_gib=150,new_artifact_cap_gib=150,max_attempts=2,poll_seconds=10)
write(P/'policy.json',policy)
controller=(R/'research_continue_20261008/controller.py').read_text().replace("[ART,ROOT/'bank_continuation_20261008/artifacts',ROOT/'geometry_training_20261008/artifacts']","[ART]")
# Long official updates may need more than180s to finish and checkpoint on SIGTERM.
controller=controller.replace("time.time()-a['stop_requested']>180","time.time()-a['stop_requested']>1800")
(P/'controller.py').write_text(controller)
jobs=[];sources=[]
for seed in [101,102,103]:
 for degree in [10,30]:
  for arm in ['geometry','entropy','original','random']:
   run=f'd{degree}_{arm}_s{seed}';out=P/'artifacts/train'/run
   for stop in [125,625]:
    jid=run+f'_u{stop}';source=dict(family='geometry',degree=degree,arm=arm,seed=seed,updates=stop,result_dir=str(out/f'u{stop}'),parent=None)
    jobs.append(dict(id=jid,kind='train',cwd=str(P),argv=[PY,'-u','-B','run_train.py','--degree',str(degree),'--arm',arm,'--seed',str(seed),'--stop-after',str(stop)],complete=str(out/f'u{stop}/complete.json'),checkpoint_dir=str(out/'checkpoints'),depends_on=[] if stop==125 else [run+'_u125'],priority=0 if seed==101 and arm in ['geometry','entropy'] and stop==125 else 10 if stop==125 else 20,max_wall_seconds=172800,gpu_min_free_mib=44000 if degree==30 else 35000,cuda_fraction=.55,**source));sources.append(source)
jobs.append(dict(id='completion_report',kind='summarize',cwd=str(P),argv=[PY,'-u','-B','summarize.py'],complete=str(P/'artifacts/report_complete.json'),depends_on=[j['id'] for j in jobs],always_run=True,priority=50,max_wall_seconds=7200))
write(P/'jobs.json',dict(jobs=jobs));write(P/'sources.json',sources)
# Reuse the independent CPU monitor, using the explicit shared checkpoint directory.
monitor=(R/'research_continue_20261008/monitor_geometry.py').read_text()
monitor=monitor.replace("if j['kind']=='train':targets[str(Path(j['complete']).parent)]=Path(j['complete']).parent","if j['kind']=='train':\n    folder=Path(j['checkpoint_dir']).parent;targets[str(folder)]=folder")
(P/'monitor_geometry.py').write_text(monitor)
(P/'launch.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012
tmux -S "$PWD/artifacts/parallel.sock" new-session -d -s geometry-gpu-scale-20261012 "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B controller.py >> artifacts/controller.log 2>&1"
tmux -S "$PWD/artifacts/parallel.sock" new-window -t geometry-gpu-scale-20261012 -n correlation "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B monitor_geometry.py >> artifacts/geometry_monitor.log 2>&1"
''')
print(json.dumps(dict(runs=24,stage_jobs=48,total_updates=15000,total_sample_updates=61440000)))
