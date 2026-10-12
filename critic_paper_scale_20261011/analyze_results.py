"""Describe the completed experiment from per-instance outputs and training logs."""
import csv
import datetime
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
OLD = HERE.parent / 'critic_paper_bank_20261011'
OUT = HERE / 'acceptance_20261012'
GROUPS = ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster']

def read(p): return json.loads(p.read_text())
def jsonlines(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def bundle(p):
    with np.load(p, allow_pickle=False) as a: return {k: a[k] for k in a.files}
def flat(a, key): return np.concatenate([a[g+'__'+key] for g in GROUPS])
def csvwrite(p, rows):
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows)))); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    state = read(ART/'state.json'); queue = read(HERE/'queue.json')['jobs']
    rows = []; dynamics = []; trajectories = []; arrays = {}; missing = []; calib = []
    for job in queue:
        d = ART/job['relative_output']
        names = ['complete.json','results.json'] if job['cohort']=='calibration' else ['complete.json','results.json','train.jsonl','validation.jsonl','test_best.npz','test_final.npz']
        absent = [name for name in names if not (d/name).exists()]
        if absent: missing.append({'id':job['id'],'missing':absent}); continue
        result = read(d/'results.json')
        if job['cohort']=='calibration':
            data = bundle(d/'holdout_predictions.npz')
            result['recomputed_mse'] = float(np.mean((data['prediction']-data['target'])**2))
            calib.append(result); continue
        cohort,family,arm,seed = job['cohort'],job['family'],job['arm'],job['seed']
        origin = read(d/'resume_origin.json') if (d/'resume_origin.json').exists() else {'step':0,'training_seconds':0}
        training = [t for t in jsonlines(d/'train.jsonl') if t['step']>origin['step']]
        validation = jsonlines(d/'validation.jsonl')
        sec = sum(t['seconds'] for t in training); selected = np.concatenate([t['selected_columns'] for t in training])
        delta = np.asarray([t['target_mean']-t['critic_mean'] for t in training])
        dynamics.append({'cohort':cohort,'family':family,'arm':arm,'seed':seed,'additional_updates':len(training),
                         'origin_step':origin['step'],'last_step':training[-1]['step'], 'training_seconds':sec,
                         'reference_seconds':sum(t['reference_seconds'] for t in training),
                         'center_fraction':float(np.mean(selected==0)), 'center_count':int(np.sum(selected==0)), 'selection_count':len(selected),
                         'lambda_final':training[-1]['lambda'],'lambda_max':max(t['lambda'] for t in training),
                         'target_minus_prediction_mean':float(delta.mean()),'positive_batch_target_gap_fraction':float(np.mean(delta>0))})
        for v in validation:
            tr = {'cohort':cohort,'family':family,'arm':arm,'seed':seed,'step':v['step'],
                  'validation_cost_mean':float(np.mean([g['cost']['mean'] for g in v['summary'].values()])),
                  'validation_worst_ood':max(g['cost']['mean'] for key,g in v['summary'].items() if key!='uniform')}
            if family=='rest': tr['critic_mean'] = float(np.mean([g['critic']['mean'] for g in v['summary'].values()]))
            trajectories.append(tr)
        for cp in ['best','final']:
            a = bundle(d/f'test_{cp}.npz'); arrays[(cohort,family,arm,seed,cp)] = a
            cost = flat(a,'cost')
            row = {'cohort':cohort,'family':family,'arm':arm,'seed':seed,'checkpoint':cp,
                   'selected_step':result['best']['step'] if cp=='best' else training[-1]['step'],
                   'n':len(cost),'mean':float(cost.mean()),'p95_pooled_groups':float(np.quantile(cost,.95)),
                   'worst_ood_mean':max(float(a[g+'__cost'].mean()) for g in GROUPS[1:]),'training_seconds':sec}
            for g in GROUPS: row[g] = float(a[g+'__cost'].mean())
            keys = ['length'] if family=='rest' else ['wirelength','max_path'] if family=='timing' else ['delay_scaled','length','buffers','skew_scaled']
            for key in keys: row[key+'_mean'] = float(flat(a,key).mean())
            if family=='rest':
                c = flat(a,'critic'); error = flat(a,'predicted_reference_relative_error')
                row.update(critic_mean=float(c.mean()),critic_std=float(c.std()),critic_min=float(c.min()),critic_max=float(c.max()),critic_zero_count=int(np.sum(c==0)),
                           reference_bias=float(error.mean()),reference_mae=float(np.abs(error).mean()),hpwl_violation_fraction=float(flat(a,'hpwl_violation').mean()))
            rows.append(row)
    if missing:
        (OUT/'missing.json').write_text(json.dumps(missing,indent=2)); raise RuntimeError('Incomplete results; see missing.json')
    aggregate = []; comparisons = []; budget = []
    for cohort in ['main','repaired']:
        p = read(HERE/cohort/'protocol.json')
        for family,arms in p['families'].items():
            for cp in ['best','final']:
                for arm in arms:
                    rs = sorted([r for r in rows if (r['cohort'],r['family'],r['arm'],r['checkpoint'])==(cohort,family,arm,cp)], key=lambda x:x['seed'])
                    aggregate.append({'cohort':cohort,'family':family,'arm':arm,'checkpoint':cp,'n_seeds':len(rs),
                                      'mean':float(np.mean([r['mean'] for r in rs])),'seed_sd':float(np.std([r['mean'] for r in rs],ddof=1)),
                                      'seed_values':[r['mean'] for r in rs],'steps':[r['selected_step'] for r in rs],
                                      'worst_ood_mean':float(np.mean([r['worst_ood_mean'] for r in rs])),
                                      'mean_seed_p95':float(np.mean([r['p95_pooled_groups'] for r in rs])),
                                      'training_seconds_mean':float(np.mean([r['training_seconds'] for r in rs]))})
                    if cohort=='main':
                        old = []
                        for r in rs:
                            a = arrays[(cohort,family,arm,r['seed'],cp)]; b = bundle(OLD/f"artifacts/{family}_{arm}_s{r['seed']}/test_{cp}.npz")
                            if any(not np.array_equal(a[g+'__ids'],b[g+'__ids']) for g in GROUPS): raise ValueError('Cannot compare budgets with unmatched test IDs')
                            old.append(float(flat(b,'cost').mean()))
                        oldmean=float(np.mean(old)); newmean=float(np.mean([r['mean'] for r in rs]))
                        budget.append({'family':family,'arm':arm,'checkpoint':cp,'short_mean':oldmean,'long_mean':newmean,'relative_change_percent':100*(newmean/oldmean-1),'delta_each':[r['mean']-v for r,v in zip(rs,old)]})
                for arm in arms[1:]:
                    def choose(a): return {r['seed']:r for r in rows if (r['cohort'],r['family'],r['arm'],r['checkpoint'])==(cohort,family,a,cp)}
                    alt,base=choose(arm),choose(arms[0]); seeds=sorted(set(alt)&set(base)); delta=[]
                    for seed in seeds:
                        a,b=arrays[(cohort,family,arm,seed,cp)],arrays[(cohort,family,arms[0],seed,cp)]
                        if any(not np.array_equal(a[g+'__ids'],b[g+'__ids']) for g in GROUPS): raise ValueError('Cannot pair unmatched test IDs')
                        delta.append(float((flat(a,'cost')-flat(b,'cost')).mean()))
                    delta=np.asarray(delta); rng=np.random.default_rng(261011); boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                    am=float(np.mean([alt[s]['mean'] for s in seeds])); bm=float(np.mean([base[s]['mean'] for s in seeds]))
                    comparisons.append({'cohort':cohort,'family':family,'arm':arm,'control':arms[0],'checkpoint':cp,'seeds':seeds,
                                        'alternative_mean':am,'control_mean':bm,'mean_delta':float(delta.mean()),'delta_each':delta.tolist(),
                                        'relative_change_percent':100*(am/bm-1),'wins':int(np.sum(delta<0)),'ties':int(np.sum(delta==0)),'losses':int(np.sum(delta>0)),
                                        'descriptive_seed_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                                        'training_time_ratio':float(np.mean([alt[s]['training_seconds'] for s in seeds])/np.mean([base[s]['training_seconds'] for s in seeds]))})
    events=jsonlines(ART/'events.jsonl'); starts=[e['time'] for e in events if e['event']=='launch']; wall=state['time']-min(starts)
    output={'completed_jobs':len(queue)-len(missing),'training_jobs':len(rows)//2,'calibration_jobs':len(calib),'failed':state['failed'],'gated':state['gated'],'pending':state['pending'],
            'additional_actor_updates':sum(t['additional_updates'] for t in dynamics),'additional_actor_sample_positions':sum(t['additional_updates']*128 for t in dynamics),
            'model_evaluation_rows':sum(r['n'] for r in rows),'queue_wall_seconds':wall,'calibration':calib,'aggregate':aggregate,'comparisons':comparisons,'short_to_long':budget,'dynamics':dynamics,'per_run':rows}
    (OUT/'statistics.json').write_text(json.dumps(output,indent=2)+'\n'); csvwrite(OUT/'per_run.csv',rows); csvwrite(OUT/'training_dynamics.csv',dynamics);csvwrite(OUT/'validation_trajectories.csv',trajectories)
    def ag(c,f,a,cp):return next(r for r in aggregate if (r['cohort'],r['family'],r['arm'],r['checkpoint'])==(c,f,a,cp))
    def co(c,f,a,cp):return next(r for r in comparisons if (r['cohort'],r['family'],r['arm'],r['checkpoint'])==(c,f,a,cp))
    report=['# 2048步扩展实验验收报告', '', '日期：2026-10-12（北京时间）。', '',
            '**30/30项完成：21项原方案续训、3项Critic校准、6项修正Critic训练；失败、gated和未完成均为0。结论：直接Critic换算仍不成立；23ICCAD的K3出现单种子主导的收益；25ICCAD平均略有改善但种子不一致，且更慢。**', '',
            '## 1. 实验范围与数据', '',
            f"degree10、batch128、每方法3个种子（101/102/103）。原方案由128/256步续至累计2048步；修正组从旧Geo第128步Actor出发，配对重置优化器并新增2048步。合计新增{output['additional_actor_updates']}次Actor更新、{output['additional_actor_sample_positions']}个训练样本位置。54份best/final输出共{output['model_evaluation_rows']}条逐模型评估记录；每份为四分布各1024个net，独立训练重复仍只有3次。", '',
            f"队列从首项启动到收尾约{wall/3600:.2f}小时。best由真实validation最差OOD组均值选择，test不参与选择。原组保留历史best（含原update0）；修正组的update0是旧Geo第128步，不能跨两组直接归因比较。", '',
            '## 2. 真实指标：三个种子的均值', '',
            'REST为相对GeoSteiner线长误差（%）；Timing/Buffer为各自固定尺度真实成本，越低越好。耗时为本阶段新增训练的单种子平均秒数，不含周期验证、最终测试或校准。', '',
            '| 组别/任务 | 方法 | best | final | best步数（101/102/103） | 新增训练秒数 |','|---|---|---:|---:|---|---:|']
    for c in ['main','repaired']:
        p=read(HERE/c/'protocol.json')
        for f,arms in p['families'].items():
            for a in arms:
                b,z=ag(c,f,a,'best'),ag(c,f,a,'final');scale=100 if f=='rest' else 1;suffix='%' if f=='rest' else ''
                report.append(f"| {c}/{f} | {a} | {b['mean']*scale:.6f}{suffix} | {z['mean']*scale:.6f}{suffix} | {b['steps']} | {b['training_seconds_mean']:.1f} |")
    report+=['','## 3. Critic：去除零输出后仍然失败','','原组Critic三个final测试输出仍全部为0，延长训练没有恢复预测作用；扩展阶段三个种子几乎100%选择原中心，lambda保持0。best仍全部为原update0，不能用其与Geo的best相同来证明等效替代。', '']
    r=co('main','rest','critic_inverse','final')
    report.append(f"原组Critic的final真实误差为{100*r['alternative_mean']:.6f}%，Geo为{100*r['control_mean']:.6f}%，差{100*r['mean_delta']:+.6f}个百分点（{r['wins']}胜/{r['losses']}负）。短程曾观察到Critic退化较少，该相对排名在长程均值上反转。")
    report+=['','修正组Critic输出已非零，但其参考预测严重向下偏移：','','| 修正预测分支种子 | final Critic均值 | 预测参考相对偏差 | 低于HPWL比例 | 训练批次中目标均值高于预测的比例 |','|---:|---:|---:|---:|---:|']
    for seed in [101,102,103]:
        r=next(r for r in rows if (r['cohort'],r['arm'],r['seed'],r['checkpoint'])==('repaired','critic_inverse',seed,'final'))
        d=next(d for d in dynamics if (d['cohort'],d['arm'],d['seed'])==('repaired','critic_inverse',seed))
        report.append(f"| {seed} | {r['critic_mean']:.6f} | {100*r['reference_bias']:.3f}% | {100*r['hpwl_violation_fraction']:.1f}% | {100*d['positive_batch_target_gap_fraction']:.2f}% |")
    report+=['','三个种子共12288个最终测试参考预测全部低于HPWL。HPWL是该无障碍直角布线问题的线长下界，这些预测不能作为可信的最优线长。最终Critic输出范围在每个种子内几乎为常数；Geo监督对照的Critic均值则约0.008–0.009，未出现同样数量级的上漂。', '',
             '实现中的换算与训练目标是：', '',
             '$$\widehat L^*=L_g/(1+\widehat r),\qquad y=(1+\widehat r)L_{sample}/L_g-1.$$', '',
             '因此对固定输入，若采样动作的平均线长高于greedy线长，则', '',
             '$$\mathbb E[y]-\widehat r=(1+\widehat r)(\mathbb E[L_{sample}]/L_g-1)>0.$$', '',
             '这给出自举目标推动预测上升的机制解释。三个修正预测分支约97%的训练批次记录了正的“目标均值−Critic均值”，与该机制相符；这不是所有输入或所有Critic方法的失败定理。', '']
    for cp in ['best','final']:
        r=co('repaired','rest','critic_inverse',cp)
        report.append(f"- 修正组{cp}：Geo误差{100*r['control_mean']:.6f}%，预测参考为{100*r['alternative_mean']:.6f}%，增加{100*r['mean_delta']:.6f}个百分点，相对误差增幅{r['relative_change_percent']:.2f}%；三个种子均更差。")
    r=co('repaired','rest','critic_inverse','best')
    report += ['',f"预测参考训练耗时约为Geo的{r['training_time_ratio']:.2f}倍，但更快伴随参考失真及真实指标变差。不能判为成功替代。",'',
               '校准阶段仅比零预测MSE小，相关性为0.040/0.049/0.062，与训练均值常数预测的MSE接近。三个校准合计使用15360个GeoSteiner标签和61440条采样拓扑，拟合768步；这项oracle成本独立列于statistics.json。通过初始化门槛并未保证后续自举目标正确。','',
               f'![修正组训练轨迹]({OUT / "critic_trajectory.png"})', '',
               '图左为四组validation的Critic均值（对数轴），实线使用预测参考，虚线为Geo监督；图右为真实validation误差，不能与最终test指标混用。', '',
               '## 4. 两篇论文：长程结果与成本', '']
    for family,label in [('timing','23ICCAD'),('buffer','25ICCAD')]:
        b,f=co('main',family,'k3','best'),co('main',family,'k3','final')
        report += [f"### {label}", '',f"- best：原方法{b['control_mean']:.8f} → K3 {b['alternative_mean']:.8f}，平均相对变化{b['relative_change_percent']:+.3f}%，{b['wins']}胜/{b['ties']}平/{b['losses']}负。",
                   f"- final：原方法{f['control_mean']:.8f} → K3 {f['alternative_mean']:.8f}，平均相对变化{f['relative_change_percent']:+.3f}%，{f['wins']}胜/{f['ties']}平/{f['losses']}负。",
                   f"- 新增训练耗时为原方法的{b['training_time_ratio']:.2f}倍。",'']
        if family=='timing': report.append('23ICCAD的best改善主要来自种子101，种子102持平且仍选update0，种子103略差；三种子best区间跨0。相比短程基本持平，长程出现了有价值的单种子信号，但不能称普遍稳定收益。')
        else: report.append('25ICCAD仍只有2/3种子改善，种子103反而变差；best与final的三种子区间均跨0。短程final曾三种子都改善，这一一致性在长程中消失。')
        report.append('')
    report+=['论文best模型的物理分量（三种子均值）：','','| 任务 | 方法 | 线长 | 最长路径 / delay×1e18 | 缓冲数 |','|---|---|---:|---:|---:|']
    for f in ['timing','buffer']:
        for a in ['original','k3']:
            rs=[r for r in rows if (r['cohort'],r['family'],r['arm'],r['checkpoint'])==('main',f,a,'best')]
            length=np.mean([r['wirelength_mean' if f=='timing' else 'length_mean'] for r in rs]);other=np.mean([r['max_path_mean' if f=='timing' else 'delay_scaled_mean'] for r in rs]);buf='—' if f=='timing' else f"{np.mean([r['buffers_mean'] for r in rs]):.6f}"
            report.append(f'| {f} | {a} | {length:.6f} | {other:.6f} | {buf} |')
    report+=['','25ICCAD的delay使用固定RC和归一化坐标，数值单位1e-18秒，不是实际芯片时序收益。综合成本改善不代表每个物理分量都改善。', '',
             '## 5. 扩大预算到底带来了什么', '', '| 原组方法 | 短程best | 2048步best | 短程final | 2048步final |', '|---|---:|---:|---:|---:|']
    for f,arms in read(HERE/'main/protocol.json')['families'].items():
        for a in arms:
            b=next(r for r in budget if (r['family'],r['arm'],r['checkpoint'])==(f,a,'best'));z=next(r for r in budget if (r['family'],r['arm'],r['checkpoint'])==(f,a,'final'));scale=100 if f=='rest' else 1;suf='%' if f=='rest' else ''
            report.append(f"| {f}/{a} | {b['short_mean']*scale:.6f}{suf} | {b['long_mean']*scale:.6f}{suf} | {z['short_mean']*scale:.6f}{suf} | {z['long_mean']*scale:.6f}{suf} |")
    report+=['','REST原组三个方法的best均未被新增训练改善，final均比短程更差；基础K3的final虽在三个种子上优于Geo，仍未超过历史best。论文两个任务的平均成本随预算扩大而下降，但方法间的配对优势仍依赖种子。', '',
             '## 6. 配对差值和结论边界', '', '差值为改动方法减对照，负值较好。REST单位是百分点，论文为成本原单位；区间为3个配对种子的10000次bootstrap描述性95%区间。n=3且比较较多，不以区间是否跨0宣称广泛统计显著性。', '',
             '| 组别/方法 | checkpoint | 三种子差值 | 平均差值 | 描述性区间 |', '|---|---|---|---:|---|']
    for r in comparisons:
        scale=100 if r['family']=='rest' else 1;ci=r['descriptive_seed_bootstrap95']
        report.append(f"| {r['cohort']}/{r['family']}/{r['arm']} | {r['checkpoint']} | {', '.join(f'{x*scale:.7f}' for x in r['delta_each'])} | {r['mean_delta']*scale:.7f} | [{ci[0]*scale:.7f}, {ci[1]*scale:.7f}] |")
    report += ['', '- 本轮不支持继续把当前自举Critic换算当作有效GeoSteiner替代；不否定有独立真实监督、能控制校准漂移的其他预测方法。',
               '- K3在论文中将J改成J/B_J−1，改变样本权重与lambda有效尺度，并额外进行教师物理评分；不能只归因于checkpoint质量。两篇论文原方法本来不调用GeoSteiner。',
               '- 25ICCAD仍为categorical-phase与local-greedy buffering组件，非完整Shi–Li DP；所有结论限于degree10、固定预算、3种子。',
               '- 主线和修正组初始化/优化器不同，只作各组内比较。本轮是根据前轮结果开展、复用原测试基准的探索性扩展，不能宣称独立确认性结论。',
               '- 没有等总时间对照；K3教师历史训练成本未摊销。加速、真实误差和物理成本分别报告，不把更快或非零预测当作科学成功。', '',
               '数据：[统计JSON](statistics.json)、[逐模型CSV](per_run.csv)、[训练动态](training_dynamics.csv)、[验证轨迹](validation_trajectories.csv)。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for i,seed in enumerate([101,102,103]):
        for arm,ls in [('critic_inverse','-'),('geo','--')]:
            tr=sorted([r for r in trajectories if (r['cohort'],r['family'],r['arm'],r['seed'])==('repaired','rest',arm,seed)],key=lambda r:r['step'])
            label=f'{arm} / seed {seed}'
            ax[0].plot([r['step'] for r in tr],[r['critic_mean'] for r in tr],ls=ls,color=f'C{i}',label=label)
            ax[1].plot([r['step'] for r in tr],[100*r['validation_cost_mean'] for r in tr],ls=ls,color=f'C{i}',label=label)
    ax[0].set_yscale('log');ax[0].set_ylabel('Mean Critic prediction (log scale)');ax[0].set_title('Repaired Critic: prediction drift')
    ax[1].set_ylabel('Mean true validation regret (%)');ax[1].set_title('Actual validation wirelength error')
    for a in ax:a.set_xlabel('Additional actor updates');a.grid(alpha=.25)
    ax[0].legend(fontsize=7);fig.savefig(OUT/'critic_trajectory.png',dpi=180);fig.savefig(OUT/'critic_trajectory.svg');plt.close(fig)
    print(json.dumps({'completed':output['completed_jobs'],'updates':output['additional_actor_updates'],'evaluations':output['model_evaluation_rows'],'wall_hours':wall/3600,'comparisons':comparisons},indent=2))

if __name__=='__main__':main()
