"""Independent budget, geometry, lifecycle and preregistered comparison audit."""
import csv,hashlib,itertools,json,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import audit_utils as a
import scheduler as s
import ops as o
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/report'

def signflip(values):
    x=np.asarray(values,float);null=np.asarray(list(itertools.product([-1.,1.],repeat=len(x))))@x/len(x)
    return float(np.mean(null>=x.mean()-1e-12))
def holm(ps):
    order=np.argsort(ps,kind='stable');out=np.empty(len(ps));last=0.
    for rank,i in enumerate(order):last=max(last,min(1.,(len(ps)-rank)*ps[i]));out[i]=last
    return out.tolist()
def finite(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,np.ndarray):return bool(np.isfinite(value).all())
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,(list,tuple)):return all(finite(x) for x in value)
    if isinstance(value,float):return bool(np.isfinite(value))
    return True

def audit_task(task):
    torch.set_num_threads(1);p=s.config();out=ROOT/'artifacts/runs'/task['id'];errors=[]
    if not s.verify_complete(task):raise RuntimeError('Incomplete task '+task['id'])
    m=s.read(out/'complete.json');c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    rows=[json.loads(x) for x in (out/'progress.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(task['start']+1,task['end']+1)):errors.append('Progress step coverage')
    if c['step']!=task['end'] or c['binding']!=m['binding']:errors.append('Checkpoint metadata')
    if c['progress_bytes']!=(out/'progress.jsonl').stat().st_size:errors.append('Checkpoint log offset')
    if c['calls']!=m['calls']:errors.append('Checkpoint counters')
    if not finite(c):errors.append('Nonfinite checkpoint/optimizer/RNG')
    for key in ['actor_optimizer','value_optimizer','rng','history']:
        if key not in c:errors.append('Missing '+key)
    parent_role='resume' if 'resume' in task['parents'] else 'warmup'
    parent=torch.load(s.task_path(task['parents'][parent_role]),map_location='cpu',weights_only=False)
    lam=parent['lambda'] if parent_role=='resume' else p['lambda_init']
    if task['kind']=='fixed' and any(not torch.equal(v,parent['actor'][k]) for k,v in c['actor'].items()):errors.append('Fixed router changed')
    for r in rows:
        if abs(r['lambda_before']-lam)>1e-9:errors.append('Lambda continuity');break
        if task['kind']=='coev' and task['method']!='erm':lam=max(0.,lam+p['lambda_lr']*(r['selected_cost']-p['rho']))
        if abs(r['lambda_after']-lam)>1e-9:errors.append('Lambda recurrence');break
    if abs(c['lambda']-lam)>1e-9:errors.append('Final lambda')
    learned=task['method'] in p['statistics']['primary_methods']
    if learned:
        prior=parent['attacker']['updates'] if parent_role=='resume' else 0
        if parent_role!='resume' and 'attacker' in task['parents']:
            ap=torch.load(s.task_path(task['parents']['attacker']),map_location='cpu',weights_only=False);prior=ap['attacker']['updates'];del ap
        if c['attacker']['updates']!=prior+(task['end']-task['start'])*p['attacker']['epochs']:errors.append('PPO optimizer epoch count')
    del parent
    metrics=[];stats={};input_hashes={};heldout=set();expected_eval=0;eval_seconds=0.
    for step in task['milestones']:
        path=out/'evaluations'/f'step_{step}.npz';receipt=s.read(path.with_suffix('.json'))
        cp=out/'checkpoints'/f'step_{step}.pt';cm=s.read(cp.with_suffix('.json'))
        if cm['sha256']!=o.sha256(cp) or cm['binding']!=m['binding']:errors.append(f'{step}: checkpoint binding')
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        if saved['step']!=step or not finite(saved):errors.append(f'{step}: invalid checkpoint')
        del saved
        if receipt['sha256']!=o.sha256(path):errors.append(f'{step}: result hash')
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        errors.extend(f'{step}: '+e for e in a.audit_arrays(arrays,m,p))
        stats[str(step)]={}
        if task['kind']=='warmup':
            for d in p['distributions']:
                if arrays.get(d+'__nominal_pins',np.zeros(0)).shape!=(p['test_per_distribution'],p['degree'],2):errors.append('Missing warmup nominal arrays')
        for key,v in arrays.items():
            if key.endswith('_pins'):
                group=(task['kind'] if task['kind']=='fixed' else 'router')+'__'+key
                digest=hashlib.sha256(v.tobytes()).hexdigest()
                if group in input_hashes and input_hashes[group]!=digest:errors.append('Input order changed across checkpoints')
                input_hashes[group]=digest;ids={a.sample_id(pin) for pin in v}
                if len(ids)!=len(v):errors.append('Duplicate heldout input')
                heldout.update(ids)
            if key.endswith('_length') or key.endswith('__gain') or key.endswith('_cost'):
                metrics.append(dict(task=task['id'],kind=task['kind'],method=task['method'],seed=task['seed'],step=step,key=key,n=len(v),mean=float(v.mean()),p95=float(np.quantile(v,.95))))
                stats[str(step)][key]=v
        grouped={}
        for key,v in stats[str(step)].items():grouped.setdefault(key.split('__',1)[1],[]).append(v)
        for key,groupset in grouped.items():
            metrics.append(dict(task=task['id'],kind=task['kind'],method=task['method'],seed=task['seed'],step=step,
                key='equal_group_aggregate__'+key,n=sum(len(v) for v in groupset),mean=float(np.mean([v.mean() for v in groupset])),
                p95=float(np.quantile(np.concatenate(groupset),.95)),worst_group_mean=float(max(v.mean() for v in groupset))))
        groups=len(p['distributions'])
        count=groups*p['fixed_test_per_distribution']*p['candidates']*len(p['test_lambdas'])*3 if task['kind']=='fixed' else groups*p['test_per_distribution']
        if task['kind']=='coev':count+=groups*p['attack_per_distribution']*p['attack_candidates']*2
        if receipt['greedy_scores']!=count:errors.append('Evaluation call budget')
        expected_eval+=count;eval_seconds+=receipt['elapsed_seconds']
        del arrays
    updates=task['end']-task['start'];expected=dict(oracle_calls=0,evaluation_greedy_scores=expected_eval,
        search_greedy_scores=updates*p['batch_size']*p['candidates'] if task['kind']!='warmup' and task['method']!='erm' else 0,
        sampled_actor_scores=updates*p['batch_size'] if task['kind']!='fixed' else 0)
    for key,v in expected.items():
        if m['calls'][key]!=v:errors.append(f'Budget {key}: {m["calls"][key]} != {v}')
    actual=a.verify_call_log(out,m,errors)
    budget=None
    if task['kind']=='coev' and task['method']!='erm':
        costs=np.array([r['selected_cost'] for r in rows]);budget=dict(task=task['id'],start=task['start'],end=task['end'],
            added_segment_mean_gap=float(costs.mean()-p['rho']),last64_gap=float(costs[-64:].mean()-p['rho']),last256_gap=float(costs[-256:].mean()-p['rho']),
            fraction_above_rho=float((costs>p['rho']).mean()),final_lambda=lam)
    return dict(task=task['id'],errors=errors,metrics=metrics,stats=stats,input_hashes=input_hashes,heldout=list(heldout),budget=budget,
        runtime=dict(task=task['id'],kind=task['kind'],method=task['method'],added_updates=updates,elapsed_seconds=m['elapsed_seconds'],evaluation_seconds=eval_seconds,peak_allocated_bytes=m['peak_allocated_bytes'],**m['calls']),actual_calls=actual)

def write_csv(name,rows):a.write_csv(OUT/name,rows)
def main():
    p=s.config();OUT.mkdir(parents=True,exist_ok=True);errors=[];results=[]
    if not o.verify_source_freeze(ROOT):raise RuntimeError('Source freeze missing')
    o.atomic(OUT/'progress.json',dict(stage='task_audits',time=time.time()))
    with ProcessPoolExecutor(max_workers=p['resources']['statistics_workers']) as pool:
        for i,r in enumerate(pool.map(audit_task,p['tasks'])):
            results.append(r);errors.extend(r['task']+': '+e for e in r['errors'])
            o.atomic(OUT/'progress.json',dict(stage='task_audits',complete=i+1,total=len(p['tasks']),time=time.time()))
    common={};heldout=set();byid={r['task']:r for r in results}
    for r in results:
        for group,digest in r['input_hashes'].items():
            if group in common and common[group]!=digest:errors.append('Cross-run input ordering mismatch '+group)
            common[group]=digest
        heldout.update(r.pop('heldout'))
    o.atomic(OUT/'progress.json',dict(stage='independent_training_input_regeneration',time=time.time()))
    training=a.regenerate_training_ids(p)
    if heldout.intersection(training):errors.append('Heldout/training input overlap')
    training_count=len(training);del training
    seeds=p['seeds'];dists=p['distributions'];comparisons=[];secondary=[]
    def vector(rid,step,d,suite='robust'):
        z=byid[rid]['stats'][str(step)]
        return (z[d+'__random32_length']+z[d+'__es32_length'])/2 if suite=='robust' else z[d+'__'+suite]
    def contrast(name,method,left_fmt,left_step,right_fmt,right_step,suite='robust'):
        groups={};refs=[]
        for seed in seeds:
            groups[seed]={};rv=[]
            for d in dists:
                left=vector(left_fmt.format(seed=seed,method=method),left_step,d,suite)
                right=vector(right_fmt.format(seed=seed,method=method),right_step,d,suite)
                groups[seed][d]=right-left;rv.append(right.mean())
            refs.append(np.mean(rv))
        seed_delta=[np.mean([v.mean() for v in groups[seed].values()]) for seed in seeds]
        bs=a.bootstrap(groups,replicates=p['statistics']['replicates'])
        ref=float(np.mean(refs));delta=float(np.mean(seed_delta))
        return dict(contrast=name,method=method,suite=suite,mean_improvement=delta,reference_mean=ref,relative_improvement=delta/ref,
            improving_seeds=sum(v>0 for v in seed_delta),seed_improvements=json.dumps(dict(zip(seeds,seed_delta))),
            p_one_sided=signflip(seed_delta),ci_low=bs['ci_low'],ci_high=bs['ci_high'])
    specs=[('post_only','post_pre2048_{method}_s{seed}',8192,'post_pre2048_{method}_s{seed}',2048),
           ('pre_only_at_long_post','post_pre8192_{method}_s{seed}',8192,'post_pre2048_{method}_s{seed}',8192),
           ('ppo_vs_random_at_long_pre_long_post','post_pre8192_{method}_s{seed}',8192,'post_pre8192_random_s{seed}',8192)]
    for method in p['statistics']['primary_methods']:
        for name,left,ls,right,rs in specs:
            comparisons.append(contrast(name,method,left,ls,right,rs))
            for suite in ['nominal_length','random32_length','es32_length']:secondary.append(contrast(name,method,left,ls,right,rs,suite))
        secondary.append(contrast('equal_REST_update_allocation',method,'post_pre8192_{method}_s{seed}',2048,'post_pre2048_{method}_s{seed}',8192))
        for pre in [2048,8192]:
            secondary.append(contrast('post_4096_vs_2048_pre'+str(pre),method,f'post_pre{pre}_{{method}}_s{{seed}}',4096,f'post_pre{pre}_{{method}}_s{{seed}}',2048))
        gains={};refs=[]
        for seed in seeds:
            z=byid[f'attacker8192_{method}_s{seed}']['stats'];gains[seed]={};rv=[]
            for d in dists:
                key=f'{d}__lambda5__{method}__gain';gains[seed][d]=z['8192'][key]-z['2048'][key];rv.append(z['2048'][key].mean())
            refs.append(np.mean(rv))
        delta=[np.mean([v.mean() for v in gains[seed].values()]) for seed in seeds];bs=a.bootstrap(gains,p['statistics']['replicates']);ref=float(np.mean(refs))
        secondary.append(dict(contrast='attacker_8192_vs_2048',method=method,suite='fixed_lambda5_gain',mean_improvement=float(np.mean(delta)),
            reference_mean=ref,relative_improvement=float(np.mean(delta))/ref,improving_seeds=sum(v>0 for v in delta),
            seed_improvements=json.dumps(dict(zip(seeds,delta))),p_one_sided=signflip(delta),ci_low=bs['ci_low'],ci_high=bs['ci_high']))
    for method in ['erm','random']:
        for name,left,ls,right,rs in specs[:2]:secondary.append(contrast(name,method,left,ls,right,rs))
    for row,adjusted in zip(comparisons,holm([r['p_one_sided'] for r in comparisons])):
        row['p_holm']=adjusted;row['conclusion']='SUPPORTED' if adjusted<=.05 and row['relative_improvement']>=.005 and row['improving_seeds']>=6 else 'NOT_ESTABLISHED'
    write_csv('primary_comparisons.csv',comparisons);write_csv('secondary_comparisons.csv',secondary)
    write_csv('metrics.csv',[x for r in results for x in r['metrics']]);write_csv('training_budget.csv',[r['budget'] for r in results if r['budget']]);write_csv('runtime_and_calls.csv',[r['runtime'] for r in results])
    lineage=[]
    for path in sorted((ROOT/'lineage').glob('*/complete.json')):
        source=json.loads(path.read_text());receipt=s.read(path.parent/'receipt.json')
        if o.sha256(path)!=receipt['source_complete_sha256']:errors.append('Lineage manifest hash changed '+path.parent.name)
        lineage.append(dict(source_run=source['run_id'],kind=source['kind'],method=source['method'],seed=source['seed'],step=source['step'],
            elapsed_seconds=source['elapsed_seconds'],checkpoint_sha256=receipt['checkpoint_sha256'],**source['calls']))
    write_csv('lineage_budget.csv',lineage)
    status='PASS' if not errors else 'FAIL'
    lines=['# 预训练与后训练扩展：最终报告','',f'工程/数值审计：{status}；完整任务 {len(results)}/{len(p["tasks"])}。','',
        '正数为线长改善；主指标是两类独立攻击的四分布等权均值。六项单侧检验统一Holm校正，.5%效应和6/8正向seed门槛。','',
        '| 比较 | 方法 | 改善 | 相对改善 | Holm p | seeds | 判定 |','|---|---|---:|---:|---:|---:|---|']
    for r in comparisons:lines.append(f"| {r['contrast']} | {r['method']} | {r['mean_improvement']:.6f} | {r['relative_improvement']:.2%} | {r['p_holm']:.5f} | {r['improving_seeds']}/8 | {r['conclusion']} |")
    lines+=['','完整checkpoint曲线和分布/P95在metrics.csv；次要区间未校正，见secondary_comparisons.csv。',
        '运输成本统计仅针对本任务新增训练片段；每run的inherited_budget.json记录父checkpoint中的计数。lineage_budget.csv按唯一继承源run记录完整历史阶段成本，包含源完成manifest的评价调用；共享祖先不重复计入此表。',
        '更强固定模型搜索不等于路由器收益；degree30、自身线长、有限攻击，不提供最优性或连续WDRO保证。',
        '旧轮test已被查看；本轮所有模型在新命名空间相同测试集评价，预算/最后checkpoint预先固定。']
    if errors:lines+=['','## 审计错误']+errors
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=['REPORT.md','primary_comparisons.csv','secondary_comparisons.csv','metrics.csv','training_budget.csv','runtime_and_calls.csv','lineage_budget.csv']
    o.atomic(OUT/'audit.json',dict(status=status,errors=errors,tasks=len(results),expected_tasks=len(p['tasks']),training_unique_inputs=training_count,
        heldout_unique_inputs=len(heldout),protocol_sha256=o.sha256(ROOT/'protocol.json'),source_sha256=o.sha256(__file__),
        artifact_sha256={n:o.sha256(OUT/n) for n in files},call_audits={r['task']:r['actual_calls'] for r in results}))
    print(json.dumps(dict(status=status,errors=len(errors),tasks=len(results))))
    return 0 if not errors else 1
if __name__=='__main__':raise SystemExit(main())
