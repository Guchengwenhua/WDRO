import itertools,json,csv
from pathlib import Path
import numpy as np
import geometry as geo
from engine import ART,P

def signflip(d):
    d=np.asarray(d);null=np.asarray(list(itertools.product([-1,1],repeat=len(d))))@d/len(d)
    return float(np.mean(null>=d.mean()-1e-14))
def holm(ps):
    order=np.argsort(ps);out=np.zeros(len(ps));prior=0.
    for rank,i in enumerate(order):prior=max(prior,(len(ps)-rank)*ps[i]);out[i]=min(1,prior)
    return out
def make_report():
    out=ART/'report';out.mkdir(exist_ok=True)
    tasks=json.loads((ART/'operations/manifest.json').read_text())
    ev=tasks['evaluation'];records={};rows=[];errors=[]
    for t in ev:
        f=ART/'evaluation'/t['id']/'complete.json'
        if not f.exists():errors.append(f'missing {t["id"]}');continue
        r=json.loads(f.read_text());records[t['id']]=r
        if geo.sha256(ART/'runs'/t['run_id']/t['checkpoint'])!=r['model_sha256']:errors.append(f'model changed {t["id"]}')
        for name,h in r['shards'].items():
            if geo.sha256(f.parent/name)!=h:errors.append(f'shard changed {t["id"]}/{name}')
        for key,m in r['metrics'].items():rows.append(dict(run_id=t['run_id'],checkpoint=t['checkpoint'],degree=t['degree'],seed=t['seed'],method=t['method'],suite=key,**m))
    if rows:
        with (out/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    comparisons=[]
    for a,b in P['primary_pairs']:
        differences=[];rel=[];nominal=[];attack={s:[] for s in ('random64','es64')}
        for seed in P['degrees']['30']['seeds']:
            ra=records.get(f'post_d30_{a}_s{seed}_best');rb=records.get(f'post_d30_{b}_s{seed}_best')
            if not ra or not rb:continue
            ma=ra['metrics'];mb=rb['metrics']
            va=np.mean([ma[f'{d}_natural']['cvar90'] for d in P['distributions'][1:]])
            vb=np.mean([mb[f'{d}_natural']['cvar90'] for d in P['distributions'][1:]])
            differences.append(vb-va);rel.append((vb-va)/vb)
            nominal.append(ma['uniform_natural']['mean']/mb['uniform_natural']['mean']-1)
            for s in attack:attack[s].append(np.mean([ma[f'{d}_{s}']['mean']-mb[f'{d}_{s}']['mean'] for d in P['distributions']]))
        if len(differences)!=8:errors.append(f'incomplete primary {a} vs {b}');continue
        rng=np.random.default_rng(20260922);d=np.asarray(differences);boot=d[rng.integers(0,8,(10000,8))].mean(1)
        comparisons.append(dict(method=a,reference=b,improvement=float(d.mean()),relative_improvement=float(np.mean(rel)),improving_seeds=int((d>0).sum()),nominal_change=float(np.mean(nominal)),attack_change={s:float(np.mean(v)) for s,v in attack.items()},p=signflip(d),ci95=np.quantile(boot,[.025,.975]).tolist(),seed_differences=d.tolist()))
    if comparisons:
        for c,p in zip(comparisons,holm([c['p'] for c in comparisons])):
            c['holm_p']=float(p);c['scientific_pass']=bool(p<.05 and c['relative_improvement']>=.005 and c['improving_seeds']>=6 and c['nominal_change']<=.005 and all(v<=0 for v in c['attack_change'].values()))
    for t in tasks['training']:
        r=json.loads((ART/'runs'/t['id']/'complete.json').read_text())
        if r['step']!=t['updates']:errors.append(f'incomplete step {t["id"]}')
        for fname,key in [('best.pt','best_sha256'),('latest.pt','latest_sha256')]:
            if geo.sha256(ART/'runs'/t['id']/fname)!=r[key]:errors.append(f'checkpoint mismatch {t["id"]}')
        expected=t['updates']*P['batch']
        if r['calls']['sampled']!=expected:errors.append(f'sampled budget {t["id"]}')
        if r['calls']['search']!=(0 if t['method']=='erm' else expected*P['candidates']):errors.append(f'search budget {t["id"]}')
        if r['calls']['teacher']!=(expected*P['candidates'] if t['method']=='excess' else 0):errors.append(f'teacher budget {t["id"]}')
        if t['method'] not in ('erm','hard','soft_dual'):
            for line in (ART/'runs'/t['id']/'progress.jsonl').read_text().splitlines():
                v=json.loads(line)
                if v['expected_cost']>P['rho']+1e-10 or abs(v['gap'])>v['gap_tolerance']:errors.append(f'constraint {t["id"]} step{v["step"]}');break
    audit=dict(engineering_pass=not errors,errors=errors,training_tasks=len(tasks['training']),evaluation_tasks=len(records),comparisons=comparisons)
    geo.atomic_json(out/'audit.json',audit)
    lines=['# 熵预算与无GeoSteiner研究结果','',f'工程验收：{"PASS" if not errors else "FAIL"}。训练{len(tasks["training"])}项，评价{len(records)}项。','',
           '目标为自身动作线长；教师相对损失不是真正最优regret。有限候选期望预算满足不代表连续Wasserstein认证。','',
           '| 方法 | 对照 | 尾部相对改善 | 改善种子 | Holm p | 本协议科学门槛 |','|---|---|---:|---:|---:|---|']
    for c in comparisons:lines.append(f'| {c["method"]} | {c["reference"]} | {c["relative_improvement"]:.3%} | {c["improving_seeds"]}/8 | {c["holm_p"]:.5f} | {"PASS" if c["scientific_pass"] and not errors else "未通过"} |')
    lines+=['','全部degree、seed、方法、best/last和自然/独立攻击指标见 metrics.csv；统计和负结果见 audit.json。',*errors]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return audit
if __name__=='__main__':make_report()
