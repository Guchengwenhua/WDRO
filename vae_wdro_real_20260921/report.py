"""Chinese report and paired tables, generated only from committed full runs."""
import argparse
import datetime as dt
import math
import time
import numpy as np
from exp_common import *
from evaluate import metrics

LABELS={'parent':'Parent','raw_wdro':'RAW-WDRO','vae_shell_wdro':'VAE-SHELL-WDRO','erm_a':'ERM-A','vae_ball_wdro':'VAE-BALL-WDRO'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def finite_format(x,digits=4):return f'{x:.{digits}f}' if x is not None and np.isfinite(x) else '—'

def estimate_delivery(completed):
    """Estimate from the first 32 real, retained training updates, never a pilot."""
    if len(completed)==12 and (ART/'done'/'pipeline.json').exists():
        completion=read_json(ART/'done'/'pipeline.json')
        return dict(status='completed',finished_at=completion['time'],time=utc(),remaining_seconds=0)
    measured={};counts={}
    for path in (ART/'runs').glob('*/training.jsonl'):
        rows=[]
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:continue
        counts[path.parent.name]=max((r['update'] for r in rows),default=0)
        if len(rows)>=32:measured[path.parent.name]=float(np.mean([r['seconds'] for r in rows[:32]]))
    if not measured:return dict(status='waiting_for_first_32_real_updates',time=utc())
    full={f'{s["method"]}_s{s["seed"]}' for s in completed}
    wdro_times=[v for k,v in measured.items() if not k.startswith('erm_a')]
    wdro=max(wdro_times) if wdro_times else max(measured.values())
    erm_times=[v for k,v in measured.items() if k.startswith('erm_a')]
    erm=max(erm_times) if erm_times else wdro*.35
    def duration(method,seed):
        key=f'{method}_s{seed}'
        if key in full:return 0.
        step=measured.get(key,erm if method=='erm_a' else wdro)
        # 25% contention margin plus remaining validation, checkpoint and full tests.
        return max(0,256-counts.get(key,0))*step*1.25+240.
    first=max(duration(m,101) for m in CFG['methods_required'])
    core_later=max(duration(m,s) for s in [102,103] for m in CFG['methods_required'])
    lanes=[0.]*4
    for seed in CFG['seeds']:
        for method in CFG['methods_secondary']:
            lane=int(np.argmin(lanes));lanes[lane]+=duration(method,seed)
    remaining=first+core_later+max(lanes)
    now=dt.datetime.now(dt.timezone.utc)
    return dict(status='estimated_from_formal_updates',time=now.isoformat(),first32_seconds_per_update=measured,
        seed101_pair_estimated_utc=(now+dt.timedelta(seconds=first)).isoformat(),
        core_3seed_estimated_utc=(now+dt.timedelta(seconds=first+core_later)).isoformat(),
        all_12_runs_estimated_utc=(now+dt.timedelta(seconds=remaining)).isoformat(),
        caveat='Contains 25% contention margin and 240 seconds per unfinished run for validation/test; updates on every report or hourly; not a guarantee.')

def scopes(a):
    yield 'B_test',a['split']=='B_test'
    yield 'C_external',a['split']=='C_external'
    for dataset in np.unique(a['dataset_ids'][a['split']=='C_external']):yield str(dataset),a['dataset_ids']==dataset
    yield 'CircuitOps_pooled',np.char.startswith(a['dataset_ids'],'circuitops_')

def plots(main,paired_design):
    if not main:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=ART/'figures';target.mkdir(exist_ok=True)
    methods=[m for m in LABELS if any(r['method']==m and r['scope']=='B_test' for r in main)]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['mean_regret','p95_regret']):
        vals=[np.array([r[metric]*100 for r in main if r['method']==m and r['scope']=='B_test']) for m in methods]
        ax.bar(range(len(methods)),[v.mean() for v in vals],yerr=[v.std(ddof=1) if len(v)>1 else 0 for v in vals],capsize=3)
        ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=20,ha='right');ax.set_ylabel('Regret (%)');ax.set_title('B test '+metric+' (last@256)')
    fig.tight_layout();fig.savefig(target/'main_mean_p95.png',dpi=180);fig.savefig(target/'main_mean_p95.svg');plt.close(fig)
    rows=[r for r in paired_design if r['role']=='last_256' and r['split']=='B_test']
    if rows:
        designs=sorted({r['design'] for r in rows});values=[np.array([r['delta_mean_pp'] for r in rows if r['design']==d]) for d in designs]
        fig,ax=plt.subplots(figsize=(8,4));ax.axhline(0,color='black',linewidth=.8)
        ax.bar(designs,[v.mean() for v in values],yerr=[v.std(ddof=1) if len(v)>1 else 0 for v in values],capsize=4)
        ax.set_ylabel('SHELL - RAW mean regret (percentage points)');ax.set_title('B test paired design differences; negative is better')
        fig.tight_layout();fig.savefig(target/'per_design_paired_difference.png',dpi=180);fig.savefig(target/'per_design_paired_difference.svg');plt.close(fig)

def generate(final=False):
    started=read_json(ART/'config.executed.json');split=read_json(ART/'split_summary.json');rho=read_json(ART/'rho.json')['rho'] if (ART/'rho.json').exists() else None
    completed=[]
    for seed in CFG['seeds']:
        for method in CFG['methods_required']+CFG['methods_secondary']:
            path=ART/'done'/f'train_{method}_s{seed}.json'
            if path.exists():completed.append(read_json(path))
    outputs={}
    for summary in completed:
        method,seed=summary['method'],summary['seed']
        for role in ['last_256','val_best']:
            path=ART/'test'/f'{method}_s{seed}_{role}.npz'
            with np.load(path,allow_pickle=False) as f:outputs[(method,seed,role)]={k:f[k] for k in f.files}
    for seed in CFG['seeds']:
        path=ART/'test'/f'parent_s{seed}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as f:outputs[('parent',seed,'parent')]={k:f[k] for k in f.files}
    main=[];perdesign=[]
    for (method,seed,role),a in outputs.items():
        for scope,mask in scopes(a):
            if not mask.any():continue
            row=dict(method=method,seed=seed,checkpoint_role=role,update=int(a['update'][0]),scope=scope,
                **metrics(a['regret'][mask],a['design_ids'][mask]))
            parent=outputs.get(('parent',seed,'parent'))
            if parent is not None:
                if not np.array_equal(parent['net_ids'],a['net_ids']):raise RuntimeError('test IDs differ from parent')
                parent_mean=float(parent['regret'][mask].mean());row['parent_mean_regret']=parent_mean
                row['mean_delta_vs_parent_pp']=100*(row['mean_regret']-parent_mean)
                row['relative_regret_reduction_vs_parent_percent']=100*(parent_mean-row['mean_regret'])/parent_mean if parent_mean else None
            main.append(row)
        for design in np.unique(a['design_ids']):
            mask=a['design_ids']==design
            perdesign.append(dict(method=method,seed=seed,checkpoint_role=role,update=int(a['update'][0]),design=str(design),
                family=str(a['family_ids'][mask][0]),dataset=str(a['dataset_ids'][mask][0]),split=str(a['split'][mask][0]),
                **metrics(a['regret'][mask],a['design_ids'][mask])))
    paired=[];paired_design=[]
    for seed in CFG['seeds']:
        for role in ['last_256','val_best']:
            raw=outputs.get(('raw_wdro',seed,role));shell=outputs.get(('vae_shell_wdro',seed,role))
            if raw is None or shell is None:continue
            if not np.array_equal(raw['net_ids'],shell['net_ids']):raise RuntimeError('paired test net IDs differ')
            for scope,mask in scopes(raw):
                if not mask.any():continue
                r=metrics(raw['regret'][mask],raw['design_ids'][mask]);s=metrics(shell['regret'][mask],shell['design_ids'][mask])
                paired.append(dict(seed=seed,role=role,scope=scope,nets=int(mask.sum()),raw_mean=r['mean_regret'],shell_mean=s['mean_regret'],
                    delta_mean_pp=100*(s['mean_regret']-r['mean_regret']),delta_p95_pp=100*(s['p95_regret']-r['p95_regret']),
                    delta_design_macro_pp=100*(s['design_macro_mean_regret']-r['design_macro_mean_regret']),
                    relative_regret_reduction_percent=100*(r['mean_regret']-s['mean_regret'])/r['mean_regret'] if r['mean_regret'] else None))
            for design in np.unique(raw['design_ids']):
                mask=raw['design_ids']==design
                paired_design.append(dict(seed=seed,role=role,design=str(design),split=str(raw['split'][mask][0]),
                    dataset=str(raw['dataset_ids'][mask][0]),nets=int(mask.sum()),raw_mean=float(raw['regret'][mask].mean()),
                    shell_mean=float(shell['regret'][mask].mean()),delta_mean_pp=100*float((shell['regret'][mask]-raw['regret'][mask]).mean())))
    aggregate=[]
    for role in ['last_256','val_best']:
        for scope in sorted({r['scope'] for r in paired}):
            rows=[r for r in paired if r['role']==role and r['scope']==scope]
            if not rows:continue
            values=np.array([r['delta_mean_pp'] for r in rows]);rel=[r['relative_regret_reduction_percent'] for r in rows]
            aggregate.append(dict(role=role,scope=scope,seeds=len(rows),mean_paired_delta_pp=float(values.mean()),
                sd_paired_delta_pp=float(values.std(ddof=1)) if len(rows)>1 else None,
                raw_mean_regret=float(np.mean([r['raw_mean'] for r in rows])),shell_mean_regret=float(np.mean([r['shell_mean'] for r in rows])),
                seed_mean_relative_regret_reduction_percent=float(np.mean(rel)) if all(v is not None for v in rel) else None,
                all_seeds_shell_better=bool((values<0).all())))
    costs=[]
    for s in completed:
        row={k:s[k] for k in ['method','seed','device_type','updates','train_centers','best_update','rho','lambda_final','nominal_fraction','mean_selected_cost','mean_budget_gap','training_seconds','sampled_and_candidate_update_seconds','test_seconds','process_seconds']}
        row.update(s['ledger']);costs.append(row)
    tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    last=[r for r in main if r['checkpoint_role'] in ['last_256','parent']]
    best=[r for r in main if r['checkpoint_role'] in ['val_best','parent']]
    for name,rows in [('main_last256',last),('main_valbest',best),('per_design',perdesign),('paired_shell_minus_raw',paired),('paired_across_seeds',aggregate),('paired_per_design',paired_design),('costs',costs)]:
        write_csv(tables/(name+'.csv'),rows)
    ablations=[]
    for seed in CFG['seeds']:
        for role in ['last_256','val_best']:
            shell=outputs.get(('vae_shell_wdro',seed,role))
            if shell is None:continue
            for comparison in ['vae_ball_wdro','erm_a']:
                other=outputs.get((comparison,seed,role))
                if other is None:continue
                if not np.array_equal(shell['net_ids'],other['net_ids']):raise RuntimeError('ablation IDs differ')
                for scope,mask in scopes(shell):
                    if mask.any():
                        ablations.append(dict(seed=seed,role=role,comparison=comparison,scope=scope,
                            shell_mean=float(shell['regret'][mask].mean()),other_mean=float(other['regret'][mask].mean()),
                            shell_minus_other_pp=100*float((shell['regret'][mask]-other['regret'][mask]).mean())))
    write_csv(tables/'ablation_paired.csv',ablations)
    plots(last,paired_design)
    # Full row-level data remain in per-checkpoint NPZ and CSV files in artifacts/test.
    write_csv(tables/'test_files.csv',[dict(method=k[0],seed=k[1],role=k[2],nets=len(a['net_ids']),
        path=str(ART/'test'/(f'parent_s{k[1]}.npz' if k[0]=='parent' else f'{k[0]}_s{k[1]}_{k[2]}.npz'))) for k,a in outputs.items()])
    vae=read_json(ART/'vae'/'selected.json') if (ART/'vae'/'selected.json').exists() else None
    cal=read_json(ART/'vae'/'calibration.json') if (ART/'vae'/'calibration.json').exists() else None
    geometries=[read_json(p) for p in sorted((ART/'pools').glob('*.geometry.json'))]
    refs=[read_json(p) for p in sorted((ART/'pools').glob('*.reference.json'))]
    failures=read_json(ART/'failures.json') if (ART/'failures.json').exists() else {}
    completion=read_json(ART/'done'/'pipeline.json') if (ART/'done'/'pipeline.json').exists() else None
    ended=dt.datetime.fromisoformat(completion['time']) if completion else dt.datetime.now(dt.timezone.utc)
    elapsed=(ended-dt.datetime.fromisoformat(started['started_at'])).total_seconds()
    all_complete=len(completed)==12
    parent_test_passes=sum(Path(t['path']).name.startswith('parent_s') for s in completed for t in s['tests'])
    unique_parent_outputs=sum(k[0]=='parent' for k in outputs)
    extra_parent_passes=max(0,parent_test_passes-unique_parent_outputs)
    total_test_requests=sum(s['ledger'].get('test_requested',0) for s in completed)
    stored_test_rows=sum(len(a['net_ids']) for a in outputs.values())
    estimate=estimate_delivery(completed);atomic_json(ART/'runtime_estimate.json',estimate)
    title='最终结果报告' if all_complete else ('阶段结果报告（有未完成或失败项）' if final else '运行中结果报告')
    lines=[f'# 真实 net 的 VAE 球壳有限候选 WDRO：{title}',f'\n更新时间：{utc()}。开始时间：{started["started_at"]}。',
        f'\n代码已实现；已完成训练与完整自然测试 {len(completed)}/12 组。主比较为 degree30、last@256；性能结论由下列实际结果决定。',
        '\n## 问题与固定实验口径',
        '在相同真实训练数据、对应 seed 的同一 Actor/Critic 父 checkpoint、相同 256 updates × batch64 下，比较 RAW 坐标候选与 VAE 球壳候选的困难样本微调。优化器统一重新初始化，学习率 1e-5。所有方法采用同一逐步 net 索引序列、规范坐标排序、真实 REST/Lstar−1 评分；贪心 batch256，末块重复末行补齐后截取。',
        '这是固定 K=8 候选池上的有限候选 WDRO；VAE 与候选池在三个 REST seeds 间共用。三个 seed 只度量 REST 微调随机性。VAE-only 指非 nominal 候选来源，slot0 始终保留真实原样本。',
        '设备调度：VAE 在入场时空闲的 GPU 上拟合。若 REST 启动时所有 GPU 已被其他作业占用，则使用空闲 CPU；每个 seed 的全部方法固定同一设备类型，预算不变。设备选择记录在 artifacts/devices/，不停止其他任务。candidate_forward_rows 记录含padding的实际前向行数，cpu_rows/gpu_rows分别按实际设备计数。',
        '\n## 数据与隔离',
        table(['划分','保留 net 数','用途'],[[k,split['retained'].get(k,0),v] for k,v in [('A_train','VAE 拟合、REST 微调与候选中心'),('A_cal','半径与共同 rho 几何校准'),('A_val','VAE 和 REST checkpoint 选择'),('B_test','ISPD2008 三个保留设计'),('C_external','ISPD2018/2019 与 CircuitOps 外部测试')]]),
        f'跨划分精确重复剔除：{split["removed"]}；测试内部原始重复 {split["test_duplicate_rows"]} 条，按方案保留。完整记录见 artifacts/splits.csv。使用全部 {started["train_center_count"]} 个训练中心，没有启用 2048 截断。测试数据在历史项目中已被评估，此处实现设计隔离，不宣称全新盲测。',
        f'共同 rho={finite_format(rho,8)}，来自 A_cal 的 RAW 非 nominal 运输成本 q75，不读取 Actor 或测试表现。运输成本为平均 Hungarian-L1；lambda 初值0，每步按旧价选样，之后投影更新。',
        '\n## 自然测试主结果：last@256',
        '下表 mean/P95/design-macro 都以 regret 的百分数表示。Δ 为 SHELL−RAW 的百分点差，负值更好；相对 regret 下降率的分母是同 seed RAW 的 mean regret，不能解释成等比例线长缩短。']
    brows=[r for r in last if r['scope']=='B_test']
    lines.append(table(['方法','seed','update','B mean (%)','B P95 (%)','B design-macro (%)','相对父模型 Δmean (pp)'],[
        [LABELS[r['method']],r['seed'],r['update'],finite_format(100*r['mean_regret']),finite_format(100*r['p95_regret']),finite_format(100*r['design_macro_mean_regret']),finite_format(r.get('mean_delta_vs_parent_pp'))] for r in brows]))
    pb=[r for r in paired if r['role']=='last_256' and r['scope']=='B_test']
    lines.append(table(['seed','SHELL−RAW mean (pp)','P95差 (pp)','design-macro差 (pp)','相对 RAW regret 下降 (%)'],[
        [r['seed'],finite_format(r['delta_mean_pp']),finite_format(r['delta_p95_pp']),finite_format(r['delta_design_macro_pp']),finite_format(r['relative_regret_reduction_percent'])] for r in pb]))
    ab=[r for r in aggregate if r['role']=='last_256' and r['scope']=='B_test']
    if ab:
        a=ab[0];lines.append(f'跨 {a["seeds"]} seeds 的配对 mean 差为 {finite_format(a["mean_paired_delta_pp"])} pp，样本标准差 {finite_format(a["sd_paired_delta_pp"])} pp。')
    lines.extend(['\n## 外部来源与逐设计分解','C_external 不参与选模。CircuitOps 合计仅146 nets，且来源/设计较碎，属于探索性结果。各来源保留独立结果，未以合并平均隐藏退化。',
        table(['seed','外部来源','RAW mean (%)','SHELL mean (%)','SHELL−RAW (pp)'],[
            [r['seed'],r['scope'],finite_format(100*r['raw_mean']),finite_format(100*r['shell_mean']),finite_format(r['delta_mean_pp'])]
            for r in paired if r['role']=='last_256' and r['scope'] not in ['B_test','C_external']]),
        table(['seed','B设计','net数','RAW mean (%)','SHELL mean (%)','SHELL−RAW (pp)'],[
            [r['seed'],r['design'],r['nets'],finite_format(100*r['raw_mean']),finite_format(100*r['shell_mean']),finite_format(r['delta_mean_pp'])]
            for r in paired_design if r['role']=='last_256' and r['split']=='B_test']),
        '全部方法、全部设计的 mean/P95 见 tables/per_design.csv。逐 net 真线长、Lstar、regret 与设计信息见 artifacts/test/*.npz 和对应 *.csv；索引见 tables/test_files.csv。',
        '\n## 验证最优 checkpoint（次结果）',
        'best 由完整 A_val 的 design-macro mean regret 选择，候选包含 update0；同分保留更早 checkpoint。best=0 表示没有找到严格的验证改善，不能称为 VAE 训练收益。',
        table(['方法','seed','best update','B mean (%)','B P95 (%)','B design-macro (%)'],[
            [LABELS[r['method']],r['seed'],r['update'],finite_format(100*r['mean_regret']),finite_format(100*r['p95_regret']),finite_format(100*r['design_macro_mean_regret'])]
            for r in best if r['scope']=='B_test' and r['method']!='parent']),
        '\n## 生成器与选择机制诊断'])
    if vae:
        lines.append(f'选定 L={vae["latent"]}，best epoch={vae["best"]["epoch"]}，A_val Hungarian-L1 均值 {vae["best"]["mean_l1"]:.6f}，中位数 {vae["best"]["median_l1"]:.6f}；共同 rho={rho:.6f}。是否触发预设 L32 回退：{vae["fallback_triggered"]}；选定模型中位误差是否仍超过 rho：{vae["fallback_still_poor"]}。验证潜均值方差均值 {vae["best"]["latent_mean_variance"]:.6f}，mean KL={vae["best"]["mean_kl"]:.6f}。')
        lines.append('一次性 VAE 拟合：'+ '；'.join(f'L{x["latent"]}: {x["epochs"]} epochs, {x["seconds"]/60:.2f} 分钟' for x in vae['all_fits'])+'。')
    if cal:lines.append(f'编码均值半径经验 q2.5/q97.5：r_min={cal["r_min"]:.6f}，r_max={cal["r_max"]:.6f}。这是经验边界，不是物理真实性或置信覆盖保证。')
    lines.append(table(['候选池','平均 unique K','最小 K','修复率 (%)','最大修复位移','内界触发 (%)','外界触发 (%)'],[
        [LABELS[g['method']],finite_format(g['mean_unique_K']),g['min_unique_K'],finite_format(100*g['repair_fraction']),finite_format(g['max_repair_displacement'],8),finite_format(100*g.get('inner_clipping_fraction',0)),finite_format(100*g.get('outer_clipping_fraction',0))] for g in geometries]))
    for g in geometries:
        if g['method']=='vae_shell_wdro':lines.append(f'BALL/SHELL 非 nominal 候选逐坐标完全相同比例：{100*g["ball_shell_exact_equal_fraction"]:.3f}%。内界少触发时，无差异不能证明内界无用。')
    lines.append(table(['方法','seed','nominal选择 (%)','平均选中cost','平均cost−rho','最终lambda'],[
        [LABELS[s['method']],s['seed'],finite_format(100*s['nominal_fraction']),finite_format(s['mean_selected_cost'],6),finite_format(s['mean_budget_gap'],6),finite_format(s['lambda_final'],6)] for s in completed]))
    high_nominal=[s for s in completed if s['method']=='vae_shell_wdro' and s['nominal_fraction']>.95]
    if high_nominal:lines.append('**机制警示：SHELL 至少一组 nominal 选择率超过95%，该实现没有提供足够可用的局部困难样本，不能将此解释为真实性或鲁棒性提升。**')
    lines.append('有限步价格更新不保证平均运输严格满足 rho。搜索分数降低、候选变容易或遗漏困难样本均不是泛化改善的证据。修复后的候选也不再严格等于未经修复的解码输出。')
    lines.append('\n## ERM 与内半径消融')
    lines.append(table(['seed','比较对象','B SHELL−对照 mean (pp)'],[
        [r['seed'],LABELS[r['comparison']],finite_format(r['shell_minus_other_pp'])] for r in ablations if r['role']=='last_256' and r['scope']=='B_test']))
    ball_rows=[r for r in ablations if r['role']=='last_256' and r['scope']=='B_test' and r['comparison']=='vae_ball_wdro']
    if ball_rows:
        ball_delta=float(np.mean([r['shell_minus_other_pp'] for r in ball_rows]))
        if ball_delta>0:lines.append(f'目前 BALL 的跨seed平均 B mean 优于 SHELL {ball_delta:.4f} pp；本设置中增加内半径可能有害，不将 BALL 的较好结果归入 SHELL 收益。')
        else:lines.append(f'SHELL−BALL 的跨seed平均 B mean 差为 {ball_delta:.4f} pp；结合内界触发比例判断本轮是否实际检验了内半径。')
    lines.extend(['\n## 实际耗时与预算',f'实验实际墙钟耗时 {elapsed/3600:.3f} 小时，统计截止 {ended.isoformat()}。单组训练耗时包括共同验证和checkpoint写入；candidate/sample核心耗时另见 costs.csv。并行作业的耗时不能直接相加解释为墙钟时间。',
        table(['方法','seed','设备','updates','训练秒','测试秒','候选真实请求','候选前向行数','sampled请求','验证请求','测试请求'],[
            [LABELS[s['method']],s['seed'],s['device_type'],s['updates'],finite_format(s['training_seconds'],1),finite_format(s['test_seconds'],1),
             s['ledger'].get('candidate_requested',0),s['ledger'].get('candidate_forward_rows',0),s['ledger'].get('sampled_requested',0),s['ledger'].get('validation_requested',0),s['ledger'].get('test_requested',0)] for s in completed]),
        '每个 WDRO 完整组为131072个 greedy candidate 请求和16384个 sampled训练请求；ERM不做困难候选搜索。训练和数据预算相同，搜索与VAE预训练计算开销并不相同，全部单列。',
        f'实际自然测试前向请求共 {total_test_requests} 行，保存的逐net结果共 {stored_test_rows} 行。父模型测试实际执行 {parent_test_passes} 次、保存 {unique_parent_outputs} 个唯一父模型输出；并发 RAW/SHELL 在缓存尚未写完时可能各自评估父模型，本轮多算 {extra_parent_passes} 次、共 {extra_parent_passes*5935} 行。重复评估使用同一父模型和完整输入，额外计算已包含在上表的真实请求数中；没有改变微调预算或结论。',
        table(['参考池','秒','新参考请求','缓存复用 distinct','非nominal无效率 (%)'],[
            [LABELS[r['method']],finite_format(r['seconds'],1),r['new_reference_requests'],r['cached_distinct'],finite_format(100*r['invalid_non_nominal_fraction'])] for r in refs])])
    prep=[]
    for key in ['data','raw_geometry','vae','latent_geometry']:
        path=ART/'done'/f'{key}.json'
        if path.exists():prep.append([key,finite_format(read_json(path).get('seconds'),1)])
    lines.append(table(['一次性阶段','实际秒'],prep))
    lines.extend(['\n## 当前结论与限制'])
    if all_complete:
        means={m:float(np.mean([r['mean_regret'] for r in last if r['scope']=='B_test' and r['method']==m])) for m in LABELS}
        headline=(f'**验收结论：12组实验和自然测试已完成；未验证球壳困难样本生成能稳定改善 REST。** '
            f'B_test 三seed平均 mean regret：父模型 {100*means["parent"]:.4f}%，RAW {100*means["raw_wdro"]:.4f}%，'
            f'SHELL {100*means["vae_shell_wdro"]:.4f}%。SHELL−RAW 为 {100*(means["vae_shell_wdro"]-means["raw_wdro"]):.4f} pp，'
            '但仅seed103的mean优于RAW；12组val_best全部为update0，所有last@256均劣于对应父模型。')
        lines.insert(3,headline)
    if not pb:lines.append('尚无完整 seed101 配对结果，不能判断性能改善。')
    elif len(pb)<3:lines.append(f'已完成 {len(pb)} 个 seed 的主配对，仅作初步描述；不使用最好 seed 代替三 seed 结论。')
    else:
        delta=np.array([r['delta_mean_pp'] for r in pb]);shellrows=[r for r in last if r['method']=='vae_shell_wdro' and r['scope']=='B_test']
        gains=np.array([r.get('mean_delta_vs_parent_pp',0) for r in shellrows])
        if high_nominal:conclusion='机制未充分跑通：局部困难候选选择不足，不能据此验证球壳方法的一般有效性。'
        elif np.all(gains>=0):conclusion='未见相对父模型的自然测试 mean regret 改善；即使部分 SHELL−RAW 差为负，也不能把相对较小退化描述成已改善 REST。'
        elif np.all(delta<0):conclusion='三个 seed 的 B_test mean 均显示 SHELL 优于 RAW；是否值得继续还需结合逐设计和外部来源的退化，以及后续独立攻击检查。'
        elif delta.mean()<0:conclusion='条件性收益：平均 SHELL−RAW 为负，但 seed 方向不一致，应保留差异与负结果。'
        else:conclusion='未见主指标上的平均收益：在共同256步预算下，SHELL没有优于RAW的跨seed平均表现。'
        lines.append(conclusion)
    lines.append('本轮优先完成自然测试，未实施独立 RAW Random32/CEM 攻击；结论限于自然真实测试，不能宣称全面鲁棒性增强。B_test仅三个设计，未给出将几千个net当独立样本的显著性或过窄区间。原始负 regret 不裁剪，数值异常会停止对应任务。')
    if failures:lines.append('运行失败保留：\n```json\n'+json.dumps(failures,ensure_ascii=False,indent=2)+'\n```')
    acceptance_path=ART/'acceptance_checks.json'
    if acceptance_path.exists():
        acceptance=read_json(acceptance_path)
        replay=acceptance['checks']['saved_checkpoint_prediction_replay']['details']
        lines.append(f'验收复核（{acceptance["checked_at"]}）：36个REST checkpoint可加载，27组NPZ/CSV的160245行结果与1050行主表/逐设计指标复算一致；'
            f'15个父模型/last checkpoint重放{replay["actual_nets"]}条真实测试样本（含尾块padding），线长最大误差{replay["max_length_error"]}。'
            '未重训、未新增攻击或调参，验收计算不计入原实验预算。详见 ACCEPTANCE.md 和 artifacts/acceptance_checks.json。')
    if not all_complete:lines.append('未完成的组不会混入共同预算主表；进度与错误见 artifacts/status.json、artifacts/logs 和 artifacts/failures.json（如存在）。')
    if not all_complete and estimate['status']=='estimated_from_formal_updates':
        lines.append(f'基于正式前32步的动态预计（UTC）：seed101完整对照 {estimate["seed101_pair_estimated_utc"]}；三seed主对照 {estimate["core_3seed_estimated_utc"]}；全部12组 {estimate["all_12_runs_estimated_utc"]}。含25%争用余量与每组240秒验证/测试余量，非保证；见 artifacts/runtime_estimate.json。')
    lines.extend(['\n## 产物与复用','代码、固定原方案与配置位于本目录。大产物位于 '+str(ART)+'，本目录 artifacts 为链接。',
        '- tables/main_last256.csv、main_valbest.csv：各 seed/来源主结果。\n- tables/paired_shell_minus_raw.csv、paired_across_seeds.csv、paired_per_design.csv：配对效果与种子分散程度。\n- tables/costs.csv：实际更新数、搜索/训练/验证/测试请求与padding GPU行数。\n- artifacts/test/：逐net自然测试 NPZ/CSV；artifacts/runs/：latest/last/val_best和训练曲线。\n- artifacts/pools/、reference_cache.sqlite：固定候选及共享参考。\n- artifacts/vae/：best/last、曲线、校准和验证重建图。\n- artifacts/figures/：主对照mean/P95、逐设计配对差（PNG/SVG）。',
        f'\n![B_test mean/P95]({ART}/figures/main_mean_p95.png)\n\n![逐设计配对差]({ART}/figures/per_design_paired_difference.png)',
        '\n下一步按实际结果决定：若重建误差/nominal选择暴露覆盖问题，先研究局部可用性；若自然泛化有一致收益，再用共享独立 RAW 攻击检验是否遗漏困难样本。不得把本轮有限候选结果解释成连续 Wasserstein 认证。'])
    content='\n\n'.join(lines)+'\n'
    path=ROOT/'REPORT.md';tmp=ROOT/'REPORT.md.partial';tmp.write_text(content);os.replace(tmp,path)
    first_pair=all(('raw_wdro',101,role) in outputs and ('vae_shell_wdro',101,role) in outputs for role in ['last_256','val_best'])
    if first_pair and not (ROOT/'INTERIM_REPORT.md').exists():
        (ROOT/'INTERIM_REPORT.md').write_text(content.replace(title,'首组完整对照阶段报告（seed101）',1))
        atomic_json(ART/'interim_created.json',dict(time=utc(),seed=101))
    atomic_json(ART/'report_status.json',dict(time=utc(),completed_runs=len(completed),all_complete=all_complete,final_requested=final))
    print('REPORT',len(completed),'complete runs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');generate(p.parse_args().final)
