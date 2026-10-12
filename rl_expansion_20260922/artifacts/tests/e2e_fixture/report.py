"""Automatic numerical audit and frozen comparisons. All negative candidates remain visible."""
from concurrent.futures import ProcessPoolExecutor
import json
import math
import numpy as np
import geometry as G
import engine as E
import pipeline as P
from common import ROOT,GROUPS,read,verify_complete,load_data,now,summary
from audit_utils import independent_span,compare,metric,csvwrite,lock_free

def audit_calls(path,ledger):
    starts={}; totals={k:0 for k in ('sampled','rollout','calibration','validation','test')}
    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if r['event']=='start':
                if r['id'] in starts: raise AssertionError('duplicate call')
                starts[r['id']]=r
            else:
                prev=starts.pop(r['id'])
                if (prev['phase'],prev['rows'],prev['step'])!=(r['phase'],r['rows'],r['step']): raise AssertionError('call mismatch')
                totals[r['phase']]+=r['rows']
    if starts: raise AssertionError('unclosed call')
    for k,n in totals.items():
        if n!=ledger[k+'_rows']: raise AssertionError('call ledger mismatch '+k)
    return totals

def audit_training(t):
    path=ROOT/t['outdir']; c=verify_complete(path/'complete.json'); lock_free(path/'run.lock')
    n=t['target']-t['start']
    if c['new_updates']!=n or c['ledger']['backward_updates']!=n: raise AssertionError('update budget')
    if any(c['ledger'][k]!=n*t['config']['batch'] for k in ('sampled_rows','rollout_rows')): raise AssertionError('sample budget')
    if c['ledger']['test_rows'] or c['ledger']['oracle_calls']: raise AssertionError('training leakage')
    audit_calls(path/'actual_calls.jsonl',c['ledger']); curves=[]; last=t['start']; best=None; qprev=None
    for line in (path/'progress.jsonl').open():
        r=json.loads(line)
        if r['type']=='validation':
            curves.append(dict(id=t['id'],arm=t['arm'],seed=t['seed'],step=r['step'],mean=r['metrics']['mean'],worst=r['metrics']['worst']))
            if r['eligible'] and (best is None or r['metrics']['worst']<best['metrics']['worst']): best=r
        else:
            if r['step']!=last+1: raise AssertionError('nonsequential updates')
            if qprev is not None and not np.allclose(r['q_used'],qprev,atol=1e-12): raise AssertionError('wrong prior weights')
            if t['config']['objective']=='group':
                # Invert the known uniform mix, then independently recompute the exponentiated update.
                alpha=t['config']['uniform_mix']; raw=(np.asarray(r['q_used'])-alpha/4)/(1-alpha)
                lr=np.log(np.maximum(raw,1e-300))+t['config']['eta']*np.asarray(r['objective_risks'])
                v=np.exp(lr-lr.max()); q=(1-alpha)*v/v.sum()+alpha/4
                if not np.allclose(q,r['q_next'],atol=2e-12): raise AssertionError('group weight update mismatch')
                qprev=r['q_next']
            last=r['step']
    if last!=t['target'] or best['step']!=c['best_step']: raise AssertionError('endpoint/selection mismatch')
    with np.load(path/'results.npz') as z:
        val=load_data('validation')
        for g in GROUPS:
            np.testing.assert_allclose(independent_span(val[g],z[g+'__topology']),z[g+'__length'],rtol=0,atol=1e-10)
    return dict(id=t['id'],arm=t['arm'],seed=t['seed'],start=t['start'],end=t['target'],new_updates=n,
        batch=t['config']['batch'],sampled=n*t['config']['batch'],seconds=c['elapsed_seconds'],
        peak_mib=c['peak_reserved_bytes']/2**20,selected_step=c['best_step'],
        replayed_call_files=len(list(path.glob('replayed_calls_*.jsonl')))),curves

def audit_eval(t):
    path=ROOT/t['outdir']; c=verify_complete(path/'complete.json'); lock_free(path/'run.lock'); audit_calls(path/'actual_calls.jsonl',c['ledger'])
    data=load_data('test' if t['degree']==30 else f"test_degree{t['degree']}")
    rows=[]; values={}; count=0
    for label,info in t['models'].items():
        receipt=read(path/f'{label}.json')
        if receipt['model_sha256']!=info['sha256']: raise AssertionError('wrong model')
        arrays={}
        with np.load(path/f'{label}.npz') as z:
            for g in GROUPS:
                length=z[g+'__length']; topology=z[g+'__topology']
                G.lengths(data[g],topology)  # Independent connectivity/range validation.
                np.testing.assert_allclose(independent_span(data[g],topology),length,rtol=0,atol=1e-10)
                arrays[g]=length; count+=len(length)
        m=summary(arrays)
        for k in ('mean','worst','p95'):
            if abs(m[k]-receipt['metrics'][k])>1e-10: raise AssertionError('summary mismatch')
        rows.append(dict(run=t['id'],stage=t['stage'],arm=t['arm'],seed=t['seed'],degree=t['degree'],
                         checkpoint=label,step=info['step'],**{k:m[k] for k in ('mean','worst','p95')},
                         **{g:m['groups'][g]['mean'] for g in GROUPS}))
        if t['degree']==30: values[label]=np.stack([arrays[g] for g in GROUPS])
    return t['id'],rows,values,count

def statistical_job(payload):
    name,b,c,metric_name,reps,seed=payload
    result=compare(b,c,metric_name,reps,np.random.default_rng(seed))
    result['metric']=metric_name; result['seeds']=len(b)
    result['mean_deterioration_percent']=100*(metric(c,'mean').mean()-metric(b,'mean').mean())/metric(b,'mean').mean()
    result['support']=bool(result['improvement_percent']>=.5 and result['improved_seeds']>len(b)/2 and
                           result['simultaneous_ci99'][0]>0 and (metric_name!='worst' or result['mean_deterioration_percent']<=1))
    if len(b)==5:
        result['new_seed_104_105_improvement_percent']=(100*(metric(b[-2:],metric_name)-metric(c[-2:],metric_name))/metric(b[-2:],metric_name)).tolist()
    return name,result

def main():
    p=read(ROOT/'protocol.json'); out=ROOT/'artifacts/report'; out.mkdir(parents=True,exist_ok=True)
    choice=read(ROOT/'artifacts/promotion.json'); frozen=read(ROOT/'artifacts/selection.json')
    train=sum((P.training_tasks(s) for s in ('screen','parents','final')),[]); ev=P.training_tasks('evaluation')
    budgets=[]; curves=[]; rows=[]; arrays={}; ncheck=0
    workers=min(8,len(train))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for budget,cv in pool.map(audit_training,train): budgets.append(budget); curves.extend(cv)
        for key,r,v,count in pool.map(audit_eval,ev): rows.extend(r); arrays[key]=v; ncheck+=count
    def get(stage,arm,seeds,step):
        return np.stack([arrays[f'eval_{stage}_{arm}_s{s}_d30'][f'step{step}'] for s in seeds])
    seeds=p['screening_seeds']+p['new_seeds']
    origins=np.concatenate([get('origin','ME',p['screening_seeds'],p['parent_step']),get('parents','ME',p['new_seeds'],p['parent_step'])])
    baseline=get('final','E50',seeds,p['final_target']); reps=p['bootstrap_replicates']; jobs=[]
    jobs += [('budget_mean',origins,baseline,'mean',reps,p['bootstrap_seed']),('budget_worst',origins,baseline,'worst',reps,p['bootstrap_seed']+1)]
    for i,m in enumerate(('mean','worst')):
        arm=choice['winners'][m]
        if arm: jobs.append((f'promoted_{m}_winner',baseline,get('final',arm,seeds,p['final_target']),m,reps,p['bootstrap_seed']+2+i))
    if choice['architecture']:
        ss=p['architecture_seeds']; target=p['architecture_final_target']
        jobs.append(('architecture_winner',get('final','BASE',ss,target),get('final',choice['architecture'],ss,target),'worst',reps,p['bootstrap_seed']+4))
    with ProcessPoolExecutor(max_workers=min(5,len(jobs))) as pool: stats=dict(pool.map(statistical_job,jobs))
    for slot in p['primary_slots']: stats.setdefault(slot,dict(status='not_selected_no_claim'))
    csvwrite(out/'models.csv',rows); csvwrite(out/'training_budget.csv',budgets); csvwrite(out/'validation_curves.csv',curves)
    G.atomic_json(out/'statistics.json',stats)
    checked_updates=sum(x['new_updates'] for x in budgets)
    if checked_updates>p['max_new_updates']: raise AssertionError('Preregistered compute ceiling exceeded')
    for path,h in frozen['training_completions'].items():
        if G.sha256(ROOT/path)!=h: raise AssertionError('Post-freeze training changed')
    audit=dict(status='PASS_FOR_PREREGISTERED_EXPLORATORY_EXPANSION',training_tasks=len(train),evaluation_tasks=len(ev),
        audited_updates=checked_updates,new_sampled=sum(x['sampled'] for x in budgets),
        independently_recomputed_test_lengths=ncheck,oracle_calls=0,protocol_sha256=G.sha256(ROOT/'protocol.json'),
        global_selection_sha256=G.sha256(ROOT/'artifacts/selection.json'),promotion=choice['winners'],architecture=choice['architecture'])
    G.atomic_json(out/'audit.json',audit)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6))
    for arm in choice['selected']:
        rr=[x for x in curves if x['arm']==arm and x['seed'] in p['screening_seeds']]
        steps=sorted(set(x['step'] for x in rr))
        ax.plot(steps,[np.mean([x['mean'] for x in rr if x['step']==s]) for s in steps],label=arm)
    ax.set(xlabel='Global updates',ylabel='Validation mean span length'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    for suffix in ('png','svg','pdf'): fig.savefig(out/f'learning_curves.{suffix}',dpi=180)
    lines=['# 纯RL扩展：自动验收报告','',f"状态：{audit['status']}。工程/数值检查通过，结论限于本次协议。",'',
           f"完成{len(train)}个训练任务、{len(ev)}个测试任务；新增{checked_updates:,}次更新、{audit['new_sampled']:,}条采样轨迹。",'',
           '所有筛选方案（含负结果）见models.csv；预算匹配请结合training_budget.csv和理论方案，batch与模型容量比较不能只看步数。',
           '测试在全局训练/选模冻结后打开；各个统计槽使用99%双侧区间，五槽名义族覆盖95%。小种子数及验证选择仍限制结论。','',
           '|预注册比较|相对改善|99%绝对改善区间|改善种子|达到探索门槛|','|---|---:|---|---:|---|']
    for slot,r in stats.items():
        if 'improvement_percent' in r:
            lines.append(f"|{slot}|{r['improvement_percent']:+.3f}%|{r['simultaneous_ci99']}|{r['improved_seeds']}/{r['seeds']}|{r['support']}|")
        else: lines.append(f'|{slot}|未晋级|||不提出结论|')
    lines += ['', 'degree20/40为探索性跨规模迁移，使用相同模型权重、单次贪心，无新训练。原始长度不是最优相对误差，也不是连续Wasserstein证书。',
              '', '![验证曲线](learning_curves.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    G.atomic_json(out/'complete.json',dict(status='complete',acceptance=audit['status'],finished=now(),
        artifacts={x.name:G.sha256(x) for x in out.iterdir() if x.is_file() and x.name!='complete.json'}))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__': main()
