import json,pathlib,sys,fcntl
p=pathlib.Path(__file__).resolve().parent
old=p.parent/'checkpoint_bank_20260930'
sys.path.insert(0,str(old))
from summarize_comparison import paired
lock=(p/'artifacts/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
rows=[];missing=[]
for seed in [101,102]:
 d=p/'artifacts'/f'lagged3_s{seed}'
 if not (d/'complete.json').exists():missing.append(seed);continue
 c=json.loads((d/'complete.json').read_text())
 f=json.loads((d/'final_test.json').read_text())
 train=[json.loads(x) for x in (d/'train.jsonl').read_text().splitlines()]
 updates=[json.loads(x) for x in (d/'bank_updates.jsonl').read_text().splitlines()]
 vals=[json.loads(x) for x in (d/'validation.jsonl').read_text().splitlines()]
 beststep=min(x['step'] for x in vals if x['groups']==c['best_validation']['groups'])
 rows.append({'seed':seed,'final_mean':sum(x['mean'] for x in f['groups'].values())/4,'vs_geo':paired(d/'final_test.npz',old/'artifacts'/f'geo_s{seed}'/'final_test.npz',seed),'vs_k3':paired(d/'final_test.npz',old/'artifacts'/f'k3_s{seed}'/'final_test.npz',seed),'best_step':beststep,'ledger':c['ledger'],'updates':updates,'update_seconds':sum(x['seconds'] for x in train),'negative_target_fraction_mean':sum(x['negative_target_fraction'] for x in train)/len(train)})
(p/'ROLLING_COMPARISON.json').write_text(json.dumps({'rows':rows,'missing':missing},indent=2))
lines=['# Rolling-bank continuation supplement','','Fixed rule: historical top1 anchor plus two most recent distinct completed-window snapshots, updated after50/100/150 for the NEXT window. Initial empty slots filled by historical teachers. Not global validation top3. Frozen independent eval copies; no online new oracle labels. The first refresh3 policy legitimately rejected all candidates; this supplement does not reinterpret that result.','','|Seed|Final200 mean regret %|Delta vs Geo pp [95% CI]|Delta vs fixedK3 pp [95% CI]|Actual changes|Best step|Refresh seconds|Update seconds|Mean negative-target fraction|','|---|---:|---|---|---:|---:|---:|---:|---:|']
def cell(v):return f"{v['mean_delta']*100:+.6f} [{v['ci95_instance_bootstrap'][0]*100:+.6f},{v['ci95_instance_bootstrap'][1]*100:+.6f}]"
for r in rows:
 lines.append(f"|{r['seed']}|{r['final_mean']*100:.6f}|{cell(r['vs_geo'])}|{cell(r['vs_k3'])}|{r['ledger']['bank_turnovers']}|{r['best_step']}|{r['ledger']['refresh_seconds']:.2f}|{r['update_seconds']:.2f}|{r['negative_target_fraction_mean']:.5f}|")
 lines+=['',f"Seed{r['seed']} actual membership transitions:",'']
 for u in r['updates']:lines.append(f"- After step{u['completed_window_step']}, effective step{u['effective_from_step']}: "+', '.join(u['after']))
lines+=['','500 paired instance bootstrap replicates stratified by4 families;1024 existing independent-reference test instances per comparison, conditional on each model, not seed-level confidence. Primary is final200; best can return unchanged step0 and cannot establish training benefit. Negative targets are retained; original ReLU critic unchanged. Refresh walltime includes periodic validation and snapshot write/load, and is separate from update time. Detailed call accounting and exact membership in JSON. No long-run equivalence claim.','',f'Incomplete seeds: {missing}']
(p/'ROLLING_COMPARISON.md').write_text(chr(10).join(lines)+chr(10))
print('Wrote ROLLING_COMPARISON.md/json',flush=True)
