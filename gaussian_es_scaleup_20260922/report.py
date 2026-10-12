"""Numerical delivery checks and preregistered paired statistics."""
import csv
import io
import itertools
import json
from pathlib import Path
import time
import numpy as np
from common import ROOT,ART,PROTOCOL as P,DISTS,atomic_json,artifact,checked,sha256,costs,legalize,verify_freeze,data,pin_set_sha256
from design import done
from adv_v3.search import _check_pins

def write_csv(path,rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    partial=path.with_suffix('.partial');partial.write_text(stream.getvalue());partial.replace(path)

def load_eval(task,kind):
    out=ART/'runs'/task['id'];arrays={}
    for dist in DISTS:
        chunks=[];centers=[]
        for path in sorted(out.glob(f'{dist}_{kind}_*.npz')):
            with np.load(path,allow_pickle=False) as f:
                if str(f['model_sha256'])!=task['model']['sha256']:raise RuntimeError('evaluation model mismatch')
                np.testing.assert_allclose(f['regret'],f['length']/f['lstar']-1,rtol=1e-12,atol=1e-12)
                if np.any(f['lstar']<=0) or not np.isfinite(f['regret']).all() or np.any(f['regret']< -1e-7):raise RuntimeError('invalid evaluation')
                _check_pins(f['pins'][:,None],task['degree'])
                if kind=='pressure':
                    np.testing.assert_allclose(f['score'],f['regret'],atol=1e-12)
                    if np.any(f['score']<f['original_score']-1e-12):raise RuntimeError('pressure below nominal')
                chunks.append(f['regret'].copy());centers.append(f['centers'].copy())
        expected=P['test_count'] if kind=='nominal' else P['pressure_count']
        values=np.concatenate(chunks)
        if len(values)!=expected:raise RuntimeError('missing evaluation rows')
        arrays[dist]=(values,np.concatenate(centers))
    return arrays

def sign_flip(values):
    values=np.asarray(values);observed=values.mean()
    null=np.asarray([np.mean(values*np.asarray(signs)) for signs in itertools.product([-1,1],repeat=len(values))])
    return float(np.mean(null<=observed+1e-15))

def crossed_p95_ci(left,right,reps=2000):
    # Shapes: seed, distribution, center. Center resampling shared across seeds/methods.
    rng=np.random.default_rng(20260922);s,g,n=left.shape;values=[]
    for _ in range(reps):
        seeds=rng.integers(s,size=s);v=[]
        for group in range(g):
            ix=rng.integers(n,size=n)
            v.append(np.mean(np.quantile(left[seeds,group][:,ix],.95,axis=1)-np.quantile(right[seeds,group][:,ix],.95,axis=1)))
        values.append(np.mean(v))
    return list(map(float,np.quantile(values,[.025,.975])))

def finalize(tasks):
    verify_freeze()
    manifest=[];rows=[];values={};center_identity={};audit_count=0
    for task in tasks:
        receipt=done(task['id'])
        if not receipt or receipt['task']!=task:raise RuntimeError('task reconciliation failed')
        manifest.append(artifact(ART/'runs'/task['id']/'complete.json'))
        for key in ('best','latest'):
            if key in receipt:checked(receipt[key]);manifest.append(receipt[key])
        for binding in receipt.get('artifacts',[]):checked(binding);manifest.append(binding)
        if task['kind']=='train':
            out=ART/'runs'/task['id'];committed={}
            for line in (out/'progress.jsonl').read_text().splitlines():
                r=json.loads(line);committed[r['step']]=r
            if set(committed)!=set(range(1,task['updates']+1)):raise RuntimeError('missing committed training steps')
            expected=0 if task['method']=='erm' else task['batch']*64
            if any(r['search_calls']!=expected for r in committed.values()):raise RuntimeError('training budget mismatch')
            for path in (out/'search_audit').glob('*.npz'):
                with np.load(path) as f:
                    _check_pins(f['support'],task['degree']);c=costs(f['centers'],f['support'])
                    score=f['regrets']-float(f['price'])*c
                    np.testing.assert_allclose(score.max(axis=1),f['score'],atol=1e-12)
                    ix=score.argmax(axis=1)
                    np.testing.assert_array_equal(f['selected'],f['support'][np.arange(len(ix)),ix])
                    audit_count+=len(ix)
            manifest.append(artifact(out/'progress.jsonl'))
        if task['kind']!='eval':continue
        for kind in ('nominal','pressure'):
            evaluated=load_eval(task,kind)
            for dist,(regret,centers) in evaluated.items():
                key=(task['degree'],kind,dist)
                content=sha256_bytes(centers)
                if key in center_identity and center_identity[key]!=content:raise RuntimeError('unpaired evaluation centers')
                center_identity[key]=content
                values[(task['degree'],task['seed'],task['method'],kind,dist)]=regret
                rows.append(dict(degree=task['degree'],seed=task['seed'],method=task['method'],suite=kind,distribution=dist,
                    n=len(regret),mean=float(regret.mean()),p95=float(np.quantile(regret,.95))))
    es=np.asarray([[values[(30,s,'es','nominal',d)] for d in DISTS[1:]] for s in P['seeds']['30']])
    random=np.asarray([[values[(30,s,'random','nominal',d)] for d in DISTS[1:]] for s in P['seeds']['30']])
    es_tail=np.quantile(es,.95,axis=2).mean(axis=1);random_tail=np.quantile(random,.95,axis=2).mean(axis=1)
    diff=es_tail-random_tail;ci=crossed_p95_ci(es,random,P['bootstrap_replicates']);p=sign_flip(diff)
    relative=float((random_tail.mean()-es_tail.mean())/max(random_tail.mean(),1e-12))
    nominal_gap=float(np.mean([values[(30,s,'es','nominal','uniform')].mean()-values[(30,s,'erm','nominal','uniform')].mean() for s in P['seeds']['30']]))
    pressure_diff=float(np.mean([values[(30,s,'es','pressure',d)].mean()-values[(30,s,'random','pressure',d)].mean() for s in P['seeds']['30'] for d in DISTS]))
    supported=bool(p<=.05 and relative>=.01 and np.sum(diff<0)>=4 and nominal_gap<=P['nominal_tolerance'] and pressure_diff<=0)
    statistics=dict(primary='ES64 minus Random64 OOD P95, degree30',seed_differences=diff.tolist(),mean_difference=float(diff.mean()),
        descriptive_crossed_bootstrap_95ci=ci,one_sided_seed_signflip_p=p,relative_reduction=relative,improving_seeds=int(np.sum(diff<0)),
        nominal_mean_gap_vs_erm=nominal_gap,cem64_mean_gap_vs_random=pressure_diff,primary_gain_supported=supported)
    write_csv(ART/'metrics.csv',rows);atomic_json(ART/'statistics.json',statistics)
    # Validate evaluation split disjointness from every validation namespace used for selection.
    split_audit=[]
    for degree in P['degrees']:
        sets={}
        for split,count in [('model-validation',P['validation_count']),('pilot-validation',P['validation_count']),
                            ('search-validation',P['search_validation_count']),('budget-validation',P['search_validation_count']),
                            ('test',P['test_count']),('pressure',P['pressure_count'])]:
            sets[split]={pin_set_sha256(x) for dist in DISTS for x in data(degree,count,dist,split)}
        for left,right in itertools.combinations(sets,2):
            overlap=len(sets[left]&sets[right])
            if overlap:raise RuntimeError(f'data split overlap: {degree}, {left}, {right}')
            split_audit.append(dict(degree=degree,left=left,right=right,overlap=overlap))
    atomic_json(ART/'split_audit.json',split_audit)
    plot_results(rows,tasks)
    receipt=dict(status='PASS_ENGINEERING_NUMERICAL',at=time.time(),task_count=len(tasks),evaluation_models=33,
        audited_support_centers=audit_count,statistics=statistics,artifacts=manifest+[artifact(ART/'metrics.csv'),artifact(ART/'statistics.json')])
    atomic_json(ART/'completion.json',receipt)
    text=f'''# Gaussian-ES 扩规模结果\n\n工程与数值验收通过，33 个后训练模型全部完成统一测试。\n\n预注册主收益判定：{'支持' if supported else '未建立'}。\n\n- ES64−Random64 的 OOD P95 平均差：{diff.mean():.8g}（负数有利于ES）。\n- 相对 P95 降幅：{relative:.2%}；改善种子：{int(np.sum(diff<0))}/5。\n- seed 符号翻转单侧 p={p:.6g}；描述性交叉bootstrap95%CI={ci}。\n- uniform mean相对ERM差：{nominal_gap:.8g}；独立CEM64相对Random差：{pressure_diff:.8g}。\n\n逐degree/seed/分布完整数据在 artifacts/metrics.csv，全部任务与产物绑定在 artifacts/completion.json。负结果全部保留；不能推广为连续Wasserstein证书、普遍优越性或完整预训练因果结论。\n'''
    (ROOT/'RESULTS.md').write_text(text)

def sha256_bytes(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def plot_results(rows,tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ART/'figures';folder.mkdir(exist_ok=True)
    for degree in P['degrees']:
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,metric in zip(axes,['mean','p95']):
            for j,method in enumerate(['erm','random','es']):
                vals=[np.mean([r[metric] for r in rows if r['degree']==degree and r['method']==method and r['suite']=='nominal' and r['distribution']==dist])*100 for dist in DISTS]
                ax.bar(np.arange(4)+(j-1)*.25,vals,.25,label=method.upper())
            ax.set_xticks(range(4),DISTS,rotation=15);ax.set_ylabel(f'{metric} relative regret (%)');ax.set_title(f'Degree {degree}');ax.legend()
        for suffix in ['png','svg','pdf']:fig.savefig(folder/f'degree{degree}.{suffix}',dpi=180)
        plt.close(fig)
    curve=[]
    for t in tasks:
        if t.get('phase')!='post':continue
        for path in sorted((ART/'runs'/t['id']).glob('validation_*.json')):
            val=json.loads(path.read_text());step=int(path.stem.split('_')[1])
            curve.append(dict(id=t['id'],degree=t['degree'],seed=t['seed'],method=t['method'],step=step,
                nominal=val['uniform']['mean'],worst_group=max(val[d]['mean'] for d in DISTS[1:]),
                ood_p95=float(np.mean([val[d]['p95'] for d in DISTS[1:]]))))
    write_csv(ART/'validation_curves.csv',curve)
