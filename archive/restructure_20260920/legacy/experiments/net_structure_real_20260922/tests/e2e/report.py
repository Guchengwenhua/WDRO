"""Generate partial/final evidence reports; never equate completion with benefit."""
from exp_common import *
import hashlib

def main(final=False):
    rows=[];design_rows=[];paired=[];files=list((ART/'test').glob('*.npz'));lookup={}
    for p in files:
        f=np.load(p);method=str(f['method'][0]);seed=int(f['seed'][0]);g=int(f['generator'][0]);budget=int(f['budget'][0]);role=str(f['role'][0]);r=f['regret'];design=f['design_ids'];lookup[method,seed,g,budget,role]=p
        for split in ['B_test','C_external']:
            take=f['split']==split
            if not take.any():continue
            means=[r[take&(design==d)].mean() for d in np.unique(design[take])]
            rows.append(dict(method=method,seed=seed,generator=g,budget=budget,role=role,split=split,nets=int(take.sum()),mean=float(r[take].mean()),p95=float(np.quantile(r[take],.95)),design_macro=float(np.mean(means)),worst_design=float(np.max(means)),update=int(f['update'][0]),path=str(p)))
            for d in np.unique(design[take]):
                v=r[take&(design==d)];design_rows.append(dict(method=method,seed=seed,generator=g,budget=budget,role=role,split=split,design=d,nets=len(v),mean=float(v.mean()),p95=float(np.quantile(v,.95))))
    for key,p in lookup.items():
        m,s,g,b,role=key;base=lookup.get(('raw_wdro',s,301,b,role))
        if m in ['parent','raw_wdro'] or not base:continue
        f=np.load(p);raw=np.load(base);assert np.array_equal(f['net_ids'],raw['net_ids'])
        for split in ['B_test','C_external']:
            take=f['split']==split;delta=f['regret'][take]-raw['regret'][take];ds=f['design_ids'][take];means=np.array([delta[ds==d].mean() for d in np.unique(ds)]);rng=np.random.default_rng(7701)
            ci=np.quantile(means[rng.integers(len(means),size=(2000,len(means)))].mean(1),[.025,.975])
            paired.append(dict(method=m,seed=s,generator=g,budget=b,role=role,split=split,net_mean_delta=float(delta.mean()),design_macro_delta=float(means.mean()),design_cluster_ci_low=float(ci[0]),design_cluster_ci_high=float(ci[1]),designs=len(means)))
    write_csv(ROOT/'tables/main.csv',rows);write_csv(ROOT/'tables/per_design.csv',design_rows);write_csv(ROOT/'tables/paired_vs_raw.csv',paired)
    done=list((ART/'done').glob('main_*_u2048.json'));q=read_json(ART/'queue.json') if (ART/'queue.json').exists() else {};status='工程完成；科学结论见实际指标' if final else '运行中，未完成正式验收'
    lines=['# net 结构先验真实芯片实验报告','',f'更新：{utc()}；{status}。',f'完成最大训练预算 {len(done)}/33；已有自然测试文件 {len(files)}。','',
           '理论、数据隔离、方法边界见 PLAN.md。指标为真实 REST/L*−1；固定 K 候选不是连续 WDRO 证明。tile 保持仅适用于全局布线抽象，不证明详细布线或真实 cell 移动合法。','',
           '|方法|REST seed|生成器|预算|checkpoint|测试集|mean %|P95 %|设计macro %|选择步|','|---|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['budget']==2048 and r['generator']==301:lines.append(f"|{r['method']}|{r['seed']}|{r['generator']}|{r['budget']}|{r['role']}|{r['split']}|{100*r['mean']:.4f}|{100*r['p95']:.4f}|{100*r['design_macro']:.4f}|{r['update']}|")
    lines+=['','完整各预算数据：tables/main.csv；逐设计：tables/per_design.csv；配对差与按设计bootstrap：tables/paired_vs_raw.csv。B仅3设计，CI不可支持通用显著性主张。','',
            '生成器调参/预训练/后训练曲线：artifacts/generators；几何诊断：artifacts/diagnostics；独立攻击：artifacts/attack；实际数据规模：artifacts/data_counts.csv。所有失败与负结果保留。','',
            '生成器使用扩展degree数据，历史VAE仅用d30；set_dae与graph_dae提供同数据预算对照。CPU/GPU耗时受并行负载影响，按实际调用数和训练暴露报告，不声称严格总算力公平。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if final:
        if len(done)!=33:raise RuntimeError('missing formal runs')
        missing=[j['id'] for j in q.get('jobs',[]) if j.get('status')!='done']
        # report job itself is allowed to be running.
        missing=[s for s in missing if s!='final_report']
        if missing:raise RuntimeError('incomplete queue: '+str(missing))
        bindings=[]
        for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.json'))+[ROOT/'PLAN.md']+list((ART/'done').glob('*.json'))+list((ART/'test').glob('*.npz')):
            h=hashlib.sha256(p.read_bytes()).hexdigest();bindings.append(dict(path=str(p),sha256=h,bytes=p.stat().st_size))
        atomic_json(ART/'completion.json',dict(status='ENGINEERING_COMPLETE_NOT_SUPERIORITY',time=utc(),formal_runs=len(done),artifacts=bindings))
if __name__=='__main__':
    import sys
    main('--final' in sys.argv)
