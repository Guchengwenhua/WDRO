"""Preregistered eight comparisons and checkpoint/dual audit, no training imports."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/report'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def signflip_p(values):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Invalid seed differences')
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=len(values))))
    null=(signs*values).mean(1)
    return float(np.mean(null>=values.mean()-1e-12))

def holm(ps):
    order=np.argsort(ps,kind='stable');out=np.empty(len(ps));last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,(len(ps)-rank)*ps[i]));out[i]=last
    return out.tolist()

def main():
    p=json.loads((ROOT/'protocol.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    if audit['status']!='PASS':raise RuntimeError('Prerequisite full artifact audit did not PASS')
    seeds=p['seeds'];dists=p['distributions'];s=p['statistics'];rows=[];budget=[];runtime=[]
    # Load only requested arrays; retain all raw arrays in hash-bound run archives.
    def values(kind,method,seed,key):
        with np.load(ROOT/'artifacts/runs'/f'{kind}_{method}_s{seed}'/'results.npz') as z:
            return z[key].copy()
    for method in s['primary_methods']:
        for kind,suite,reference in [('fixed','lambda5__gain',s['fixed_reference'])]+[
                ('coev',suite,s['coev_reference']) for suite in s['coev_suites']]:
            differences=[];references=[]
            for seed in seeds:
                dv=[];rv=[]
                for dist in dists:
                    mk=f'{dist}__lambda5__{method}__gain' if kind=='fixed' else f'{dist}__{suite}'
                    rk=f'{dist}__lambda5__{reference}__gain' if kind=='fixed' else f'{dist}__{suite}'
                    a=values(kind,method,seed,mk);b=values(kind,reference,seed,rk)
                    if a.shape!=b.shape:raise RuntimeError('Unpaired arrays')
                    dv.append(float((a-b if kind=='fixed' else b-a).mean()));rv.append(float(b.mean()))
                differences.append(float(np.mean(dv)));references.append(float(np.mean(rv)))
            delta=float(np.mean(differences));ref=float(np.mean(references))
            threshold=s['fixed_minimum_gain_difference'] if kind=='fixed' else s['coev_minimum_relative_improvement']*ref
            rows.append(dict(kind=kind,suite=suite,method=method,reference=reference,seed_count=len(seeds),
                mean_improvement=delta,reference_mean=ref,relative_improvement=delta/ref,
                seed_improvements=json.dumps(dict(zip(seeds,differences))),improving_seeds=sum(v>0 for v in differences),
                p_one_sided=signflip_p(differences),minimum_absolute_improvement=threshold))
    if len(rows)!=8 or len(seeds)!=8:raise RuntimeError('Primary family changed')
    for row,adjusted in zip(rows,holm([r['p_one_sided'] for r in rows])):
        row['p_holm']=adjusted
        row['conclusion']='SUPPORTED' if adjusted<=.05 and row['improving_seeds']>=s['minimum_improving_seeds'] and row['mean_improvement']>=row['minimum_absolute_improvement'] else 'NOT_ESTABLISHED'
    errors=[]
    for rid in audit['expected_tasks']:
        folder=ROOT/'artifacts/runs'/rid
        m=json.loads((folder/'complete.json').read_text())
        c=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        logs=[json.loads(line) for line in (folder/'progress.jsonl').read_text().splitlines()]
        n=p[m['kind']+'_updates']
        if [r['step'] for r in logs]!=list(range(1,n+1)):errors.append(rid+': log steps')
        if c['step']!=n or c['binding']!=m['binding']:errors.append(rid+': checkpoint binding/step')
        if any(k not in c for k in ['rng','actor_optimizer','value_optimizer','actor','value','calls','history','progress_bytes']):errors.append(rid+': missing resume fields')
        if c['progress_bytes']!=(folder/'progress.jsonl').stat().st_size:errors.append(rid+': checkpoint log position')
        if not all(torch.isfinite(v).all() for state in ['actor','value'] for v in c[state].values()):errors.append(rid+': nonfinite model')
        if m['method'] in p['learned_methods'] and c['attacker'] is None:errors.append(rid+': missing attacker')
        for key in ['search_greedy_scores','sampled_actor_scores','oracle_calls']:
            if c['calls'][key]!=m['calls'][key]:errors.append(rid+': checkpoint training counters')
        lam=p['fixed_lambda'] if m['kind']=='fixed' else p['lambda_init']
        for r in logs:
            if abs(r['lambda_before']-lam)>1e-9:errors.append(rid+': lambda continuity');break
            if m['kind']=='coev' and m['method']!='erm':lam=max(0.,lam+p['lambda_lr']*(r['selected_cost']-p['rho']))
            if abs(r['lambda_after']-lam)>1e-9:errors.append(rid+': lambda recurrence');break
            if not np.isfinite([r.get('gain',0.),r.get('sampled_length',1.),r['lambda_after']]).all():errors.append(rid+': nonfinite training')
        if abs(c['lambda']-lam)>1e-9:errors.append(rid+': checkpoint lambda')
        if m['kind']=='fixed':
            warm=torch.load(folder.parent/f"warmup_s{m['seed']}"/'latest.pt',map_location='cpu',weights_only=False)
            if any(not torch.equal(v,warm['actor'][k]) for k,v in c['actor'].items()):errors.append(rid+': frozen Actor changed')
            del warm
        if m['kind']=='coev' and m['method']!='erm':
            costs=np.array([r['selected_cost'] for r in logs])
            budget.append(dict(run_id=rid,method=m['method'],seed=m['seed'],rho=p['rho'],mean_cost=float(costs.mean()),
                mean_gap=float(costs.mean()-p['rho']),last64_gap=float(costs[-64:].mean()-p['rho']),
                last256_gap=float(costs[-256:].mean()-p['rho']),fraction_above_rho=float((costs>p['rho']).mean()),
                final_lambda=lam,no_move_fraction=float(np.mean([r['no_move_fraction'] for r in logs]))))
        attacker_std=(c['attacker']['actor']['log_std'].clamp(np.log(.002),np.log(.2)).exp().tolist() if c['attacker'] else None)
        runtime.append(dict(run_id=rid,kind=m['kind'],method=m['method'],seed=m['seed'],final_attacker_std=json.dumps(attacker_std),
            elapsed_seconds=m['elapsed_seconds'],training_seconds=logs[-1]['elapsed_seconds'],
            evaluation_seconds=m['elapsed_seconds']-logs[-1]['elapsed_seconds'],peak_allocated_bytes=m['peak_allocated_bytes'],**m['calls']))
        del c
    dump_csv(OUT/'primary_comparisons.csv',rows);dump_csv(OUT/'training_budget.csv',budget);dump_csv(OUT/'runtime_and_calls.csv',runtime)
    report=['# 扩大训练的预注册结论','',f"工程与附加审计：{'PASS' if not errors else 'FAIL'}。",'',
        '正数表示改善；八项单侧 seed 符号翻转检验共同做 Holm 校正。NOT_ESTABLISHED 包括无效应、效应不足或证据不足。',
        '', '| 阶段 | 指标 | 方法 / 对照 | 改善 | 相对改善 | Holm p | 正向 seeds | 判定 |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        report.append(f"| {r['kind']} | {r['suite']} | {r['method']} / {r['reference']} | {r['mean_improvement']:.6f} | {r['relative_improvement']:.2%} | {r['p_holm']:.5f} | {r['improving_seeds']}/8 | {r['conclusion']} |")
    report+=['','运输成本见 training_budget.csv；未满足 ρ 的情况必须保留。有限候选测试不提供连续 WDRO 保证。',
             '完整次要对照、分布分项和描述性 95% CI 见 REPORT.md、metrics.csv、comparisons.csv。',
             '纯线长、degree30、8 seeds、最后 checkpoint；不代表最优性 regret、跨 degree 或芯片设计泛化。']
    (OUT/'PRIMARY_REPORT.md').write_text('\n'.join(report)+'\n')
    result=dict(status='PASS' if not errors else 'FAIL',errors=errors,primary_comparisons=rows,
                protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=sha(__file__),upstream_audit_sha256=sha(OUT/'audit.json'),
                scope='Numerical/protocol acceptance is separate from supported improvement',
                artifact_sha256={n:sha(OUT/n) for n in ['primary_comparisons.csv','training_budget.csv','runtime_and_calls.csv','PRIMARY_REPORT.md']})
    tmp=OUT/'confirmatory.partial';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(OUT/'confirmatory.json')
    print(json.dumps({'status':result['status'],'errors':errors,'supported':sum(r['conclusion']=='SUPPORTED' for r in rows)}))
    return 0 if not errors else 1

if __name__=='__main__':raise SystemExit(main())
