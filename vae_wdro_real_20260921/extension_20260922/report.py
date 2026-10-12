"""Automatic Chinese reports from committed real runs, including tuning costs."""
import argparse
import csv
import datetime as dt
import fcntl
import time
import numpy as np
from exp_common import *

LABEL={'parent':'父模型','raw_wdro':'RAW','local_shell':'LOCAL-SHELL','erm_a':'ERM-A',
    'legacy_shell':'LEGACY-SHELL','mix_legacy':'MIX-LEGACY','local_ball':'LOCAL-BALL','mix_local':'MIX-LOCAL'}

def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def fmt(x,d=4):return f'{x:.{d}f}' if x is not None and np.isfinite(x) else '—'
def scope_masks(a):
    yield 'B_test',a['split']=='B_test'
    yield 'C_external',a['split']=='C_external'
    for source in np.unique(a['dataset_ids'][a['split']=='C_external']):yield str(source),a['dataset_ids']==source
    yield 'CircuitOps_pooled',np.char.startswith(a['dataset_ids'],'circuitops_')
def metric(v,design):
    return dict(nets=len(v),mean=float(v.mean()),p95=float(np.quantile(v,.95)),macro=float(np.mean([v[design==d].mean() for d in np.unique(design)])))

def diagnostic_report():
    rows=[]
    for path in sorted((ART/'diagnostics').glob('*/summary.json')):
        s=read_json(path);val=s['validations'][-1];zero=s['validations'][0]
        macro=val.get('macro',val.get('design_macro_mean_regret'));base=zero.get('macro',zero.get('design_macro_mean_regret'))
        rows.append(dict(recipe=s['recipe'],method=s['method'],seed=s['seed'],update=s.get('update',256),
            val_macro=macro,parent_macro=base,delta_pp=100*(macro-base),reused=bool(s.get('reused_from')),new_training_seconds=s.get('new_training_seconds',0.)))
    write_csv(ROOT/'tables'/'diagnostic_runs.csv',rows)
    lines=['# 共同微调与BN诊断（仅A_val）',f'更新：{utc()}。以下不是测试结果，全部选择只看A_val。',
        '6种设置：BN正常更新/冻结运行统计×学习率1e-5、3e-6、1e-6；ERM/RAW各用seeds101/102，共24项，4个标准设置对照引用前轮、20项新训练。每项256updates，冻结BN仍训练其仿射参数。',
        mdtable(['设置','方法','seed','验证macro (%)','相对父模型(pp)','旧结果复用'],[[r['recipe'],r['method'],r['seed'],fmt(r['val_macro']*100),fmt(r['delta_pp']),r['reused']] for r in rows])]
    path=ART/'recipe_selected.json'
    if path.exists():
        selected=read_json(path);lines.append(f'共同设置选为 **{selected["recipe"]}**，四项update256验证macro平均={100*selected["mean_val_macro"]:.5f}%；父模型相应平均={100*selected["parent_mean_val_macro"]:.5f}%。全部设置均退化：{selected["all_recipes_degrade"]}。不为各方法单独选择超参。')
        lines.append(mdtable(['设置','四项验证macro平均 (%)'],[[x['recipe'],fmt(100*x['mean_val_macro'])] for x in selected['ranking']]))
    else:lines.append(f'当前有{len(rows)}/24项完整记录；所有项目完成后才固定共同设置。')
    path=ART/'done'/'bn_diagnosis.json'
    if path.exists():
        rows2=read_json(path)['rows'];lines.extend(['## BN统计交叉替换诊断',
            '下列模型只在内存中替换统计，在A_val诊断；不作为主训练模型或测试选模候选。固定权重时的变化说明统计的贡献，不能代替重新训练对照。',
            mdtable(['seed','原方法','权重','BN统计','验证mean (%)','验证macro (%)'],[[r['seed'],r['method'],r['weights'],r['bn_statistics'],fmt(100*r['mean']),fmt(100*r['macro'])] for r in rows2])])
    (ROOT/'DIAGNOSTIC_REPORT.md').write_text('\n\n'.join(lines)+'\n')
    return rows

def estimate(summaries):
    rates={};counts={}
    for folder in [ART/'runs',ART/'diagnostics']:
        for path in folder.glob('*/training.jsonl'):
            rows=[]
            for line in path.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue
            if not rows:continue
            counts[path.parent.name]=rows[-1]['update']
            if len(rows)>=32:rates[path.parent.name]=float(np.mean([r['seconds'] for r in rows[:32]]))
    if (ART/'done'/'pipeline.json').exists():return dict(status='completed',time=utc(),finished=read_json(ART/'done'/'pipeline.json')['time'])
    w=[v for k,v in rates.items() if 'raw_wdro' in k or 'local_shell' in k];e=[v for k,v in rates.items() if 'erm_a' in k]
    if not w:return dict(status='waiting_for_32_formal_updates',time=utc())
    wdro=max(w);erm=max(e) if e else wdro*.35
    diag_remaining=0.
    for recipe in CFG['recipes']:
        for seed in [101,102]:
            for method in ['erm_a','raw_wdro']:
                key=f'{recipe}_{method}_s{seed}'
                if (ART/'done'/f'diag_{key}.json').exists():continue
                diag_remaining+=(256-counts.get(key,0))*(erm if method=='erm_a' else wdro)+30
    primary_remaining=0.
    for seed in CFG['seeds']:
        for method in CFG['primary_methods']:
            key=f'{method}_g0_s{seed}';steps=counts.get(key,0)
            primary_remaining+=max(0,1024-steps)*(erm if method=='erm_a' else wdro)
            primary_remaining+=sum(not (ART/'done'/f'main_{key}_u{u}.json').exists() for u in [256,512,1024])*100
    generator_remaining=0.
    for g in [1,2]:
        for seed in CFG['seeds']:
            key=f'local_shell_g{g}_s{seed}';generator_remaining+=max(0,1024-counts.get(key,0))*wdro+200
    now=dt.datetime.now(dt.timezone.utc)
    # Shared four-job CPU capacity, contention reserve, CVAE/reference/attack overhead.
    core_seconds=(diag_remaining+primary_remaining)/4*1.25+1800
    all_seconds=core_seconds+generator_remaining/4*1.25+1800
    pair=all((ART/'done'/f'main_{m}_g0_s101_u256.json').exists() for m in ['raw_wdro','local_shell'])
    first_seconds=0 if pair else diag_remaining/3*1.2+256*wdro*1.25+1200
    return dict(status='estimated_from_formal_steps',time=now.isoformat(),wdro_step_seconds=wdro,erm_step_seconds=erm,
        first_pair_estimated_utc=(now+dt.timedelta(seconds=first_seconds)).isoformat(),
        primary1024_estimated_utc=(now+dt.timedelta(seconds=core_seconds)).isoformat(),
        full_with_optional_estimated_utc=(now+dt.timedelta(seconds=all_seconds)).isoformat(),
        caveat='Conservative CPU throughput estimate; optional generator/attack admission follows PLAN time gates; not guaranteed.')

def plot(main,paired_design):
    if not main:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ART/'figures';out.mkdir(exist_ok=True)
    for budget in [256,512,1024]:
        rows=[r for r in main if r['budget']==budget and r['role']=='last' and r['scope']=='B_test' and r['generator']==0]
        if not rows:continue
        methods=[m for m in CFG['primary_methods'] if any(r['method']==m for r in rows)]
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for ax,key in zip(axes,['mean','p95']):
            values=[np.array([100*r[key] for r in rows if r['method']==m]) for m in methods]
            ax.bar(range(len(methods)),[v.mean() for v in values],yerr=[v.std(ddof=1) if len(v)>1 else 0 for v in values],capsize=3)
            parent=[r for r in main if r['method']=='parent' and r['scope']=='B_test']
            if parent:ax.axhline(100*np.mean([r[key] for r in parent]),color='black',linestyle='--',label='Parent seed mean');ax.legend()
            ax.set_xticks(range(len(methods)),[LABEL[m] for m in methods],rotation=25,ha='right');ax.set_ylabel('Regret (%)');ax.set_title(f'B test {key}, last@{budget}')
        fig.tight_layout();fig.savefig(out/f'main_{budget}.png',dpi=170);fig.savefig(out/f'main_{budget}.svg');plt.close(fig)
        pair=[r for r in paired_design if r['budget']==budget and r['role']=='last' and r['method']=='local_shell' and r['generator']==0 and r['split']=='B_test']
        if pair:
            designs=sorted({r['design'] for r in pair});values=[np.array([r['delta_pp'] for r in pair if r['design']==d]) for d in designs]
            fig,ax=plt.subplots(figsize=(8,4));ax.axhline(0,color='black',lw=.7)
            ax.bar(designs,[v.mean() for v in values],yerr=[v.std(ddof=1) if len(v)>1 else 0 for v in values],capsize=3)
            ax.set_ylabel('LOCAL-SHELL - RAW mean regret (pp)');ax.set_title(f'Last@{budget}; negative is better')
            fig.tight_layout();fig.savefig(out/f'paired_design_{budget}.png',dpi=170);plt.close(fig)

def generate(final=False):
    diag=diagnostic_report();summaries=[]
    for path in sorted((ART/'done').glob('main_*.json')):summaries.append(read_json(path))
    outputs={};test_index=[]
    for s in summaries:
        for role in ['last','val_best']:
            path=ART/'test'/f'{s["method"]}_g{s["generator"]}_s{s["seed"]}_u{s["update"]}_{role}.npz'
            with np.load(path) as f:a={k:f[k] for k in f.files}
            key=(s['method'],s['generator'],s['seed'],s['update'],role);outputs[key]=a
            test_index.append(dict(method=key[0],generator=key[1],seed=key[2],budget=key[3],role=key[4],path=str(path),nets=len(a['length'])))
    for seed in CFG['seeds']:
        path=ART/'test'/f'parent_s{seed}.npz'
        if path.exists():
            with np.load(path) as f:outputs[('parent',0,seed,0,'parent')]={k:f[k] for k in f.files}
            test_index.append(dict(method='parent',generator=0,seed=seed,budget=0,role='parent',path=str(path),nets=5935))
    main=[];design_rows=[];paired=[];paired_design=[]
    for key,a in outputs.items():
        method,g,seed,budget,role=key
        for scope,take in scope_masks(a):
            if not take.any():continue
            row=dict(method=method,generator=g,seed=seed,budget=budget,role=role,checkpoint_update=int(a['update'][0]),scope=scope,**metric(a['regret'][take],a['design_ids'][take]))
            parent=outputs.get(('parent',0,seed,0,'parent'))
            if parent is not None:
                if not np.array_equal(parent['net_ids'],a['net_ids']):raise ValueError('parent test alignment')
                row['delta_vs_parent_pp']=100*(row['mean']-float(parent['regret'][take].mean()))
            main.append(row)
        for design in np.unique(a['design_ids']):
            take=a['design_ids']==design
            design_rows.append(dict(method=method,generator=g,seed=seed,budget=budget,role=role,design=str(design),
                dataset=str(a['dataset_ids'][take][0]),split=str(a['split'][take][0]),**metric(a['regret'][take],a['design_ids'][take])))
        raw=outputs.get(('raw_wdro',0,seed,budget,role))
        if method in ['parent','raw_wdro'] or raw is None:continue
        if not np.array_equal(raw['net_ids'],a['net_ids']):raise ValueError('paired test alignment')
        for scope,take in scope_masks(a):
            if not take.any():continue
            rawm=metric(raw['regret'][take],raw['design_ids'][take]);newm=metric(a['regret'][take],a['design_ids'][take])
            paired.append(dict(method=method,generator=g,seed=seed,budget=budget,role=role,scope=scope,
                raw_mean=rawm['mean'],method_mean=newm['mean'],delta_mean_pp=100*(newm['mean']-rawm['mean']),
                delta_p95_pp=100*(newm['p95']-rawm['p95']),delta_macro_pp=100*(newm['macro']-rawm['macro']),
                relative_regret_reduction_percent=100*(rawm['mean']-newm['mean'])/rawm['mean'] if rawm['mean'] else None))
        for design in np.unique(a['design_ids']):
            take=a['design_ids']==design
            paired_design.append(dict(method=method,generator=g,seed=seed,budget=budget,role=role,design=str(design),
                split=str(a['split'][take][0]),delta_pp=100*float((a['regret'][take]-raw['regret'][take]).mean())))
    tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    for name,rows in [('all_metrics',main),('per_design',design_rows),('paired_vs_raw',paired),('paired_per_design',paired_design),('test_files',test_index)]:write_csv(tables/f'{name}.csv',rows)
    for budget in [256,512,1024]:
        for role in ['last','val_best']:
            rows=[r for r in main if r['generator']==0 and ((r['budget']==budget and r['role']==role) or r['method']=='parent')]
            write_csv(tables/f'main_{role}_{budget}.csv',rows)
    costs=[]
    for s in summaries:
        costs.append({k:s[k] for k in ['method','generator','seed','recipe','device_type','update','initial_update','new_updates','new_training_seconds','test_seconds','process_seconds','nominal_fraction','mean_selected_cost','mean_budget_gap','lambda_final','best_update']}|s['ledger_delta'])
    write_csv(tables/'costs.csv',costs)
    crossed=[r for r in paired if r['method']=='local_shell'];write_csv(tables/'crossed_generator_rest_seeds.csv',crossed)
    curves=[];calibrations=[]
    for g in [0,1,2]:
        path=ART/'cvae'/f'g{g}'/'selected.json'
        if path.exists():
            selected=read_json(path)
            for fit in selected['all_fits']:curves.append(dict(generator=g,latent=fit['spec']['latent'],beta=fit['spec']['beta_max'],epochs=fit['epochs'],selected_epoch=fit['best']['epoch'],seconds=fit['seconds'],chosen=fit['path']==selected['path'],**fit['best']))
        else:
            for path in (ART/'cvae'/f'g{g}').glob('*/done.json'):
                fit=read_json(path);curves.append(dict(generator=g,latent=fit['spec']['latent'],beta=fit['spec']['beta_max'],epochs=fit['epochs'],seconds=fit['seconds'],chosen=False,**fit['best']))
        path=ART/'cvae'/f'g{g}'/'calibration.json'
        if path.exists():calibrations.append(read_json(path))
    write_csv(tables/'cvae_tuning.csv',curves)
    geometry=[dict(bank=p.name.removesuffix('.geometry.json'),**read_json(p)) for p in sorted((ART/'pools').glob('*.geometry.json')) if not p.name.startswith('attack')]
    geometry_rows=[dict(bank=g['bank'],mean_cost=g.get('mean_cost',g.get('cost_mean')),cost_le_rho_fraction=g.get('cost_le_rho_fraction'),
        centers_with_cost_le_rho=g.get('centers_with_cost_le_rho'),unique_K=g.get('mean_unique_K'),inner_fraction=g.get('inner_fraction',g.get('inner_clipping_fraction')),
        outer_fraction=g.get('outer_fraction',g.get('outer_clipping_fraction')),repair_fraction=g.get('repair_fraction'),alpha=g.get('alpha')) for g in geometry]
    write_csv(tables/'candidate_diagnostics.csv',geometry_rows)
    estimates=estimate(summaries);atomic_json(ART/'runtime_estimate.json',estimates)
    finished=(ART/'done'/'pipeline.json').exists();start=read_json(ART/'config.executed.json')['started_at']
    end=read_json(ART/'done'/'pipeline.json')['time'] if finished else utc();wall=(dt.datetime.fromisoformat(end)-dt.datetime.fromisoformat(start)).total_seconds()
    selected=read_json(ART/'recipe_selected.json') if (ART/'recipe_selected.json').exists() else None
    optional=read_json(ART/'optional_schedule.json') if (ART/'optional_schedule.json').exists() else None
    failures=read_json(ART/'failures.json') if (ART/'failures.json').exists() else {}
    title='最终报告' if finished else ('阶段报告（存在未完成项）' if final else '运行中报告')
    lines=[f'# 局部条件VAE与真实数据WDRO扩展：{title}',f'更新：{utc()}；运行开始：{start}。',
        '主生成器固定为20260922；先调共同微调设置与CVAE，再冻结选择并评估测试。原始球壳、局部条件球壳、混合候选是不同方法，不混称同一个解码流形。',
        f'已有{len(summaries)}个完整“配置×预算”训练与自然测试结果。主实验目标7方法×3REST种子×256/512/1024，另预定2生成器×3REST种子的LOCAL-SHELL交叉扩展。执行完成与性能改善分开判定。',
        '## 调参和数据隔离',
        '调参全部使用A_train/A_cal/A_val；B/C仅输出结果。本轮根据已观察到的前轮结果设计，因此不能声称全新盲测。划分保持A_train2810、A_cal1202、A_val1085、B3991、C1944，按family隔离，父模型、参考值、排序和中心序列一致。',
        f'REST调参：{len(diag)}/24项完整，20项新训练、4项旧标准对照复用；BN更新/冻结×lr[1e-5,3e-6,1e-6]，各ERM/RAW×seeds101/102，以共同update256验证design-macro平均选设置。',
        f'共同设置：**{selected["recipe"]}**；全部候选设置均劣于父模型：{selected["all_recipes_degrade"]}。' if selected else '共同设置尚未选定；主测试不会先于验证选择启动。',
        'BN交叉替换及逐项调参数据见DIAGNOSTIC_REPORT.md；它们不属于自然测试成绩。所有主方法使用同一选定设置，新Adam，batch64，原始真实regret训练。',
        'CVAE以局部近邻真实对/去噪对拟合条件位移。潜维[8,16,32]×KL权重[1e-4,1e-3]共6个主生成器配置，固定验证输入和先验噪声，以先验best-of-8重建Hungarian-L1选checkpoint/超参。该指标不同于前轮无条件VAE的后验均值重建指标，不能直接相减声称重建改进。',
        mdtable(['生成器','L','beta','epochs','best epoch','先验best8 L1','后验L1','选中','秒'],[[r['generator'],r['latent'],r['beta'],r['epochs'],r.get('selected_epoch',r['epoch']),fmt(r['prior_best8_mean_l1'],6),fmt(r['posterior_mean_l1'],6),r['chosen'],fmt(r['seconds'],1)] for r in curves]),
        '## 自然测试：共同预算的last与验证最优',
        '以下mean/P95/macro以regret百分数表示；百分点差=方法−RAW，负数更好。相对下降率的分母为同seed同预算RAW mean regret，不等于同百分比线长缩短。父模型同时列出。best包含update0，且只根据A_val选择。']
    for budget in [256,512,1024]:
        lines.append(f'### 共同{budget}步')
        for role in ['last','val_best']:
            rows=[r for r in main if r['scope']=='B_test' and r['generator']==0 and r['budget']==budget and r['role']==role]
            lines.append(mdtable([role,'seed','实际checkpoint update','B mean (%)','B P95 (%)','B macro (%)','相对父模型mean(pp)'],
                [[LABEL[r['method']],r['seed'],r['checkpoint_update'],fmt(100*r['mean']),fmt(100*r['p95']),fmt(100*r['macro']),fmt(r.get('delta_vs_parent_pp'))] for r in rows]))
        pairs=[r for r in paired if r['scope']=='B_test' and r['generator']==0 and r['budget']==budget and r['role']=='last']
        lines.append(mdtable(['方法','seed','相对RAW mean(pp)','P95(pp)','相对regret下降(%)'],[[LABEL[r['method']],r['seed'],fmt(r['delta_mean_pp']),fmt(r['delta_p95_pp']),fmt(r['relative_regret_reduction_percent'])] for r in pairs]))
        for method in ['local_shell','mix_local']:
            p=[r for r in pairs if r['method']==method]
            if len(p)==3:
                dif=np.array([r['delta_mean_pp'] for r in p]);actual=[r for r in main if r['method']==method and r['generator']==0 and r['budget']==budget and r['role']=='last' and r['scope']=='B_test']
                delta_parent=np.array([r['delta_vs_parent_pp'] for r in actual]);msg='三seed同向改善RAW' if np.all(dif<0) else ('平均改善但seed不一致' if dif.mean()<0 else '未见相对RAW平均mean收益')
                lines.append(f'{LABEL[method]}@{budget}：{msg}；mean配对差均值{dif.mean():+.4f}pp，样本SD{dif.std(ddof=1):.4f}pp。相对父模型三seed都改善：{bool(np.all(delta_parent<0))}。该描述不替代逐设计/外部/攻击检查。')
    parents=[r for r in main if r['method']=='parent' and r['scope']=='B_test']
    lines.append(mdtable(['父模型seed','B mean (%)','B P95 (%)','B macro (%)'],[[r['seed'],fmt(100*r['mean']),fmt(100*r['p95']),fmt(100*r['macro'])] for r in parents]))
    lines.extend(['## 外部数据与逐设计',
        '所有ISPD2018、ISPD2019、CircuitOps来源和逐设计mean/P95均保存到tables/all_metrics.csv、per_design.csv、paired_per_design.csv。CircuitOps仅146个net，保留探索性标签；不以混合平均隐藏特定来源退化。',
        mdtable(['预算','方法','seed','来源','相对RAW mean(pp)'],[[r['budget'],LABEL[r['method']],r['seed'],r['scope'],fmt(r['delta_mean_pp'])] for r in paired if r['generator']==0 and r['method'] in ['local_shell','mix_local'] and r['role']=='last' and r['scope'] in ['ispd2018_protocol_a','ispd2019_protocol_a','CircuitOps_pooled']]),
        '## 生成器、球壳和运输成本',
        mdtable(['池','平均cost','cost≤rho比例','有局部候选中心比例','unique_K','内界率','外界率','修复率','alpha'],[[g['bank'],fmt(g['mean_cost'],6),fmt(g['cost_le_rho_fraction']),fmt(g['centers_with_cost_le_rho']),fmt(g['unique_K']),fmt(g['inner_fraction']),fmt(g['outer_fraction']),fmt(g['repair_fraction']),g['alpha']] for g in geometry_rows]),
        '局部性表中的cost≤rho是诊断，WDRO约束为平均预算，不是逐候选硬约束。共同rho固定0.0709042084；alpha仅在A_cal以BALL成本中位数匹配旧RAW中位数，BALL/SHELL共享同一alpha。球壳作用于相对条件先验白化的潜噪声，经验半径不保证真实性。',
        mdtable(['生成器','L','r_min','r_max','alpha'],[[c['generator'],c['latent'],fmt(c['r_min']),fmt(c['r_max']),c['alpha']] for c in calibrations]),
        mdtable(['方法','g','seed','预算','nominal率','选中cost','cost−rho','最终lambda','best update'],[[s['method'],s['generator'],s['seed'],s['update'],fmt(s['nominal_fraction']),fmt(s['mean_selected_cost'],6),fmt(s['mean_budget_gap'],6),fmt(s['lambda_final']),s['best_update']] for s in summaries]),
        '## 生成器×REST种子交叉扩展',
        '主结论固定g0；g1/g2沿用g0所选超参，不择优挑生成器。RAW只训练一次/REST seed，不能把同一RAW在三个生成器比较中重复出现当成独立样本。',
        mdtable(['g','REST seed','预算','LOCAL-SHELL−RAW mean(pp)'],[[r['generator'],r['seed'],r['budget'],fmt(r['delta_mean_pp'])] for r in crossed if r['scope']=='B_test' and r['role']=='last']),
        '## 共享独立RAW-Random32攻击'])
    attack_path=ART/'attack'/'summary.json'
    if attack_path.exists():
        attack=read_json(attack_path);lines.extend(['96个固定B中心，每个32候选，所有模型共享几何/Lstar，lambda=0取真实regret最大值。它是有限候选压力测试，不是连续空间证明，也不参与选模。',
            mdtable(['方法','seed','预算','role','checkpoint update','攻击mean (%)','攻击P95 (%)'],[[LABEL[r['method']],r['seed'],r['budget'],r['role'],r['update'],fmt(100*r['mean']),fmt(100*r['p95'])] for r in attack['rows']])])
    else:lines.append('独立攻击尚未完成或因预定时间门槛未安排；当前不得据自然测试宣称全面鲁棒性提高。')
    refs=[read_json(p) for p in (ART/'pools').glob('*.reference.json') if not p.is_symlink()]
    lines.extend(['## 实际耗时、预算与完成状态',f'墙钟已用{wall/3600:.3f}小时，统计至{end}。实际REST设备及每次续训新增请求见tables/costs.csv；不能累加各预算的累计账本当成新请求。',
        f'新诊断训练已累计{sum(r["new_training_seconds"] for r in diag):.1f}进程秒；主模型各阶段新增训练累计{sum(s["new_training_seconds"] for s in summaries):.1f}进程秒。CVAE全部已结束尝试累计{sum(r["seconds"] for r in curves):.1f}进程秒，另含池几何与参考成本。CPU/GPU不同设备的墙钟效率不作方法优势主张。',
        mdtable(['方法','g','seed','到达预算','新增updates','训练秒','测试秒','新增候选请求','新增sampled请求','新增测试请求'],[[s['method'],s['generator'],s['seed'],s['update'],s['new_updates'],fmt(s['new_training_seconds'],1),fmt(s['test_seconds'],1),s['ledger_delta'].get('candidate_requested',0),s['ledger_delta'].get('sampled_requested',0),s['ledger_delta'].get('test_requested',0)] for s in summaries]),
        mdtable(['参考池','新增参考','缓存复用distinct','无效率','秒'],[[r['method'],r['new_reference_requests'],r['cached_distinct'],r['invalid_non_nominal_fraction'],fmt(r['seconds'],1)] for r in refs]),
        f'可选规模入场决定：{optional}。失败记录：{failures}。未执行/未完成项目不混入共同预算主表。'])
    if estimates['status']=='estimated_from_formal_steps':lines.append('动态预计（UTC，非保证）：首对照 '+estimates['first_pair_estimated_utc']+'；主实验1024步 '+estimates['primary1024_estimated_utc']+'；含可选扩展 '+estimates['full_with_optional_estimated_utc']+'。')
    lines.extend(['## 结论边界',
        '每个结果必须同时检查：相对RAW是否跨seed一致、相对父模型是否改善、逐设计/来源是否集中退化、独立攻击是否保持。若best=0即未发现验证改善；若nominal率>95%、候选重复或距离极小，即生成有效性不足，不能以更低攻击分数解释为更鲁棒。',
        'B仅3个设计、A_val仅2个设计；不把几千个net当成独立设计样本给出虚假的精确显著性。VAE/REST调参的全部尝试与额外计算已单列，当前比较不等同于总算力完全相同。',
        '## 产物',
        'PLAN.md/config.json固定规则；DIAGNOSTIC_REPORT.md记录选择依据；tables/main_last_*.csv和main_val_best_*.csv分别报告共同预算；tables/test_files.csv索引逐net NPZ/CSV；artifacts/runs包含每阶段last/val_best/latest；artifacts/cvae包含所有拟合、校准、曲线和生成图；独立攻击见artifacts/attack和tables/independent_attack.csv。'])
    plot(main,paired_design)
    for budget in [256,512,1024]:
        path=ART/'figures'/f'main_{budget}.png'
        if path.exists():lines.append(f'![自然测试共同{budget}步]({path})')
    text='\n\n'.join(lines)+'\n';tmp=ROOT/'REPORT.md.partial';tmp.write_text(text);os.replace(tmp,ROOT/'REPORT.md')
    first=all((m,0,101,256,role) in outputs for m in ['raw_wdro','local_shell'] for role in ['last','val_best'])
    if first and not (ROOT/'INTERIM_REPORT.md').exists():
        (ROOT/'INTERIM_REPORT.md').write_text(text.replace(title,'首组完整对照报告（g0、seed101、256步）',1));atomic_json(ART/'interim_created.json',dict(time=utc()))
    atomic_json(ART/'report_status.json',dict(time=utc(),completed_configuration_budgets=len(summaries),pipeline_completed=finished,final_requested=final))
    print('REPORT',len(summaries),'committed configuration-budgets',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args()
    with (ART/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        generate(args.final)
