"""Numerical closeout and preregistered statistics; never trains or calls a solver."""
import csv
import fcntl
import json
import math
from pathlib import Path
import numpy as np
import geometry as G
from common import ROOT,GROUPS,read,now,verify_complete,load_data

def independent_span(pins,topology):
    """Independent masked interval reduction, separate from scatter-based scorer."""
    result=[]
    for start in range(0,len(pins),128):
        x=np.asarray(pins[start:start+128],dtype=np.float64)
        edges=topology[start:start+128].reshape(len(x),-1,2)
        row=np.arange(len(x))[:,None]; total=np.zeros(len(x))
        for anchor in range(x.shape[1]):
            vertical=np.where(edges[:,:,0]==anchor,x[row,edges[:,:,1],1],x[:,anchor,1,None])
            horizontal=np.where(edges[:,:,1]==anchor,x[row,edges[:,:,0],0],x[:,anchor,0,None])
            total+=np.maximum(vertical.max(1),x[:,anchor,1])-np.minimum(vertical.min(1),x[:,anchor,1])
            total+=np.maximum(horizontal.max(1),x[:,anchor,0])-np.minimum(horizontal.min(1),x[:,anchor,0])
        result.append(total)
    return np.concatenate(result)

def metric(v,name):
    # v shape: seed, group, net. Worst-group is per trained model before seed mean.
    means=v.mean(-1)
    return means.mean(1) if name=='mean' else means.max(1)

def compare(base,candidate,name,replicates,rng):
    b=metric(base,name); c=metric(candidate,name)
    point=float(b.mean()-c.mean()); deltas=[]
    for _ in range(replicates):
        seed_idx=rng.integers(len(base),size=len(base))
        net_idx=rng.integers(base.shape[-1],size=(base.shape[1],base.shape[-1]))
        # Each bootstrap replicate uses paired nets for all selected training seeds.
        bb=base[seed_idx[:,None,None],np.arange(base.shape[1])[None,:,None],net_idx[None,:,:]]
        cc=candidate[seed_idx[:,None,None],np.arange(base.shape[1])[None,:,None],net_idx[None,:,:]]
        deltas.append(float((metric(bb,name)-metric(cc,name)).mean()))
    return dict(baseline_value=float(b.mean()),candidate_value=float(c.mean()),improvement=point,
        improvement_percent=100*point/float(b.mean()),improved_seeds=int(np.sum(c<b)),
        seed_improvement=(b-c).tolist(),ci95=np.quantile(deltas,[.025,.975]).tolist(),
        simultaneous_ci99=np.quantile(deltas,[.005,.995]).tolist())

def csvwrite(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def lock_free(path):
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(f,fcntl.LOCK_UN)

def calls(path):
    starts={}; ends=set()
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r['event']=='start':
            if r['id'] in starts: raise AssertionError('duplicate call id')
            starts[r['id']]=r
        else:
            if r['id'] not in starts or r['id'] in ends: raise AssertionError('invalid completed call')
            ends.add(r['id'])
    return {phase:dict(completed=sum(x['rows'] for k,x in starts.items() if k in ends and x['phase']==phase),
                        attempted_upper=sum(x['rows'] for x in starts.values() if x['phase']==phase))
            for phase in ('sampled','rollout','validation','test')}

def main():
    out=ROOT/'artifacts/report'; out.mkdir(parents=True,exist_ok=True)
    p=read(ROOT/'protocol.json'); data=load_data('test'); arrays={}; runs=[]; groups=[]; bindings={}; curves=[]
    validation_by_run={}
    for arm in p['arms']:
        for seed in p['seeds']:
            key=f'{arm}_s{seed}'; tr=ROOT/'artifacts/runs'/key; ev=ROOT/'artifacts/eval'/key
            train=verify_complete(tr/'complete.json'); evaluation=verify_complete(ev/'complete.json')
            lock_free(tr/'run.lock'); lock_free(ev/'run.lock')
            if train['step']!=p['targets'][arm] or train['new_updates']!=p['targets'][arm]-p['start_steps'][arm]:
                raise AssertionError('Incomplete training updates')
            if train['ledger']['oracle_calls'] or evaluation['ledger']['oracle_calls']: raise AssertionError('Oracle calls')
            committed=train['new_updates']*64
            if train['ledger']['sampled_rows']!=committed or train['ledger']['rollout_rows']!=committed:
                raise AssertionError('Committed trajectory accounting')
            actual=calls(tr/'actual_calls.jsonl'); evalcalls=calls(ev/'actual_calls.jsonl')
            if any(actual[phase]['completed']<committed for phase in ('sampled','rollout')):
                raise AssertionError('Actual call ledger below committed work')
            bindings[str(tr/'complete.json')]=G.sha256(tr/'complete.json')
            bindings[str(ev/'complete.json')]=G.sha256(ev/'complete.json')
            rows=[json.loads(x) for x in (tr/'progress.jsonl').read_text().splitlines()]
            updates=[x for x in rows if x['type']=='update']; validations=[x for x in rows if x['type']=='validation']
            if [x['step'] for x in updates]!=list(range(p['start_steps'][arm]+1,p['targets'][arm]+1)):
                raise AssertionError('Training step sequence incomplete')
            best=min((x for x in validations if x['eligible']),key=lambda x:(x['metrics']['worst'],x['step']))
            if best['step']!=train['best_step']: raise AssertionError('Best selection mismatch')
            validation_by_run[key]={x['step']:x['metrics'] for x in validations}
            for x in validations:
                curves.append(dict(arm=arm,seed=seed,step=x['step'],mean=x['metrics']['mean'],worst=x['metrics']['worst']))
            for label,model in train['models'].items():
                receipt=read(ev/f'{label}.json')
                if receipt['model_sha256']!=model['sha256']: raise AssertionError('Evaluated wrong model')
                with np.load(ev/f'{label}.npz') as pred:
                    lengths=[]
                    for g in GROUPS:
                        topology=pred[g+'__topology']; saved=pred[g+'__length']
                        # Validate topology legality and independently reduce its span length.
                        G.lengths(data[g],topology)
                        independent=independent_span(data[g],topology)
                        np.testing.assert_allclose(saved,independent,rtol=0,atol=1e-12)
                        np.testing.assert_allclose(saved.mean(),receipt['metrics']['groups'][g]['mean'],rtol=0,atol=1e-12)
                        lengths.append(saved.copy())
                        groups.append(dict(arm=arm,seed=seed,model=label,step=model['step'],group=g,
                            mean=float(saved.mean()),p95=float(np.quantile(saved,.95))))
                arrays[(arm,label,seed)]=np.stack(lengths)
            attempt_results=[read(x) for x in (ROOT/'artifacts/operations').glob(f'result_{key}_a*.json')]
            timed=[x for x in attempt_results if x.get('status')=='exited' and 'elapsed_seconds' in x]
            worker_seconds=sum(x['elapsed_seconds'] for x in timed) if timed else None
            runs.append(dict(arm=arm,seed=seed,new_updates=train['new_updates'],cumulative_sampled=train['step']*64,
                research_new_sampled=committed,actual_sampled=actual['sampled']['completed'],
                sampled_replay=actual['sampled']['completed']-committed,actual_rollout=actual['rollout']['completed'],
                validation_rows=actual['validation']['completed'],test_rows=evalcalls['test']['completed'],
                seconds=train['elapsed_seconds'],actual_worker_seconds=worker_seconds,
                exited_worker_attempts=len(timed),update_seconds=train['update_seconds'],best_step=train['best_step'],
                peak_reserved_bytes=train['peak_reserved_bytes']))
    if sum(r['new_updates'] for r in runs)!=p['train_new_updates_total']: raise AssertionError('Total budget mismatch')
    for gpu_lock in (ROOT/'artifacts/operations').glob('gpu_*.lock'): lock_free(gpu_lock)
    def get(spec):
        arm,label=spec.split('/'); return np.stack([arrays[(arm,label,s)] for s in p['seeds']])
    comparisons=[]; rng=np.random.default_rng(p['bootstrap_seed'])
    specs=p['primary_comparisons']+[p.get('supplemental_comparison',dict(name='posttraining_scale_exploratory',baseline='SG/step6144',candidate='SG/step12288',metric='worst'))]
    for i,spec in enumerate(specs):
        b=get(spec['baseline']); c=get(spec['candidate'])
        result=compare(b,c,spec['metric'],p['bootstrap_replicates'],rng)
        mean_degrade=100*(metric(c,'mean').mean()/metric(b,'mean').mean()-1)
        passed=(result['improvement_percent']>=p['primary_min_improvement_percent'] and
                result['improved_seeds']>=p['primary_min_improved_seeds'] and result['simultaneous_ci99'][0]>0)
        if spec['name']=='robust_finetuning': passed=passed and mean_degrade<=1
        result.update(spec,mean_degradation_percent=float(mean_degrade),primary=i<5,
                      conclusion=('supports_preregistered_exploratory_hypothesis' if passed else 'not_established_at_this_budget') if i<5 else 'exploratory_only')
        comparisons.append(result)
        print(json.dumps(result),flush=True)
    G.atomic_json(out/'statistics.json',dict(comparisons=comparisons,bootstrap_replicates=p['bootstrap_replicates'],
                  caveat='Three training seeds; paired hierarchical bootstrap intervals are exploratory.'))
    csvwrite(out/'metrics.csv',groups); csvwrite(out/'training_budget.csv',runs); csvwrite(out/'validation_curves.csv',curves)
    flat=[{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in x.items()} for x in comparisons]
    csvwrite(out/'comparisons.csv',flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm in p['arms']:
        steps=sorted({r['step'] for r in curves if r['arm']==arm})
        for ax,name in zip(axes,('mean','worst')):
            values=np.array([[validation_by_run[f'{arm}_s{s}'][step][name] for step in steps] for s in p['seeds']])
            line,=ax.plot(steps,values.mean(0),label=arm)
            ax.fill_between(steps,values.min(0),values.max(0),alpha=.10,color=line.get_color())
            ax.set(xlabel='Total training updates',ylabel=f'Validation {name} own-action length')
            ax.grid(alpha=.2)
    axes[0].legend(ncol=3); fig.suptitle('Degree 30, 3 seeds; bands show seed range (not confidence intervals)')
    for ext in ('png','svg','pdf'): fig.savefig(out/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)
    lines=['# 纯 RL 扩大训练：自动数值核验与结果','',f'完成于 {now()}。18 个训练、18 个新测试任务均完成；独立线长重算、哈希、预算及运行锁核验通过。',
        '', '三种子探索实验，不能证明普遍优越或最优性；原始 span 线长不是相对最优 regret。',
        '', '|主要比较|相对改善|配对99%区间（绝对线长）|改善种子|判断|','|---|---:|---|---:|---|']
    names=dict(training_budget='延长 uniform 基础训练',data_coverage='扩大数据覆盖',robust_finetuning='稳健后训练',
               pretraining_order='预训练顺序',stage_allocation='短/长预训练分配',posttraining_scale_exploratory='延长后训练（补充）')
    for c in comparisons:
        verdict='达到预注册支持标准' if c['conclusion'].startswith('supports') else ('补充探索' if not c['primary'] else '本预算下尚未确立')
        lines.append(f"|{names[c['name']]}|{c['improvement_percent']:.3f}%|[{c['simultaneous_ci99'][0]:.6f}, {c['simultaneous_ci99'][1]:.6f}]|{c['improved_seeds']}/3|{verdict}|")
    h=next(x for x in comparisons if x['name']=='training_budget')
    lines+=['',('延长基础训练达到预注册改善标准，支持训练不足是旧结果的一个原因；仍不能解释全部历史差距。' if h['conclusion'].startswith('supports') else
                  '本次尚未按预注册标准确立延长基础训练的收益。需结合逐种子曲线区分平台期、种子不确定性和退化，不能直接断言训练规模无效。'),
        '', '所有路线均使用自身动作线长，无外部求解器/最优标签。五项主要比较的99%区间使用Bonferroni名义族覆盖；三种子限制仍然存在。',
        '', f"新增 {sum(r['research_new_sampled'] for r in runs):,} 条 sampled 轨迹，等量 committed rollout；实际重放 {sum(r['sampled_replay'] for r in runs):,} 条 sampled。checkpoint 累计训练进程时间 {sum(r['seconds'] for r in runs)/3600:.2f} 小时。调度器有退出记录的实际训练 worker 尝试合计 {sum(r['actual_worker_seconds'] or 0 for r in runs)/3600:.2f} 小时（小型夹具没有调度器计时；缺失或丢失进程的时间不推算）。这些都不能当作队列墙钟时间。",
        '', '![验证曲线](learning_curves.png)', '', '逐组均值/P95：metrics.csv；完整预算：training_budget.csv；逐比较区间：comparisons.csv。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    G.atomic_json(out/'audit.json',dict(status='PASS',training_runs=18,evaluation_runs=18,
        independent_length_reduction=True,all_worker_locks_released=True,all_oracle_calls_zero=True,
        new_updates=sum(r['new_updates'] for r in runs),run_bindings=bindings))
    artifacts={x.name:G.sha256(x) for x in out.iterdir() if x.is_file() and x.name!='complete.json'}
    G.atomic_json(out/'complete.json',dict(status='complete',acceptance='audited_three_seed_exploratory_experiment',
        finished=now(),protocol_sha256=G.sha256(ROOT/'protocol.json'),artifacts=artifacts,run_bindings=bindings))

if __name__=='__main__': main()
