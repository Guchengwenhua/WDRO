"""Summaries keep exact continuation separate from the repaired-Critic cohort."""
import csv, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE/'artifacts'
def read(p): return json.loads(p.read_text())
def write(p, text):
    q = p.with_name(p.name+'.partial'); q.write_text(text); os.replace(q,p)

def main():
    rows = []; contrasts = []; calibration = []
    for seed in [101,102,103]:
        path = ART/f'calibration/s{seed}/results.json'
        if path.exists(): calibration.append(read(path))
    lines = ['# 扩展实验结果', '', '解释与结论见[2026-10-12验收报告](acceptance_20261012/REPORT.md)，包括Critic预测漂移、真实参考误差和短长预算对照。', '', '主线续训与修正Critic实验分开比较。只有已完成训练及真实测试的任务进入下表；初始化、校准、排队不等于训练完成。', '']
    for cohort, expected in [('main',21),('repaired',6)]:
        completed = list((ART/cohort).glob('*/complete.json'))
        lines += [f'## {cohort}：完成 {len(completed)}/{expected}', '', '| 任务 | 方法 | 种子 | checkpoint | 选择步数 | 真实平均成本 | OOD最差组 | 本阶段训练秒数 |', '|---|---|---:|---|---:|---:|---:|---:|']
        for marker in sorted(completed):
            d = marker.parent; result = read(d/'results.json')
            origin = read(d/'resume_origin.json') if (d/'resume_origin.json').exists() else {'training_seconds':0}
            for cp in ['best','final']:
                groups = result['test_'+cp]
                row = {'cohort':cohort,'family':result['family'],'arm':result['arm'],'seed':result['seed'],'checkpoint':cp,
                       'selected_step':result['best']['step'] if cp=='best' else 2048,
                       'mean':float(np.mean([g['cost']['mean'] for g in groups.values()])),
                       'worst_ood':max(g['cost']['mean'] for k,g in groups.items() if k!='uniform'),
                       'incremental_training_seconds':result['training_seconds']-origin['training_seconds'],
                       'training_seconds_including_parent':result['training_seconds']}
                if result['family']=='rest':
                    row['critic_mean']=float(np.mean([g['critic']['mean'] for g in groups.values()]))
                    row['predicted_reference_bias']=float(np.mean([g['predicted_reference_relative_error']['mean'] for g in groups.values()]))
                rows.append(row)
                lines.append(f"| {row['family']} | {row['arm']} | {row['seed']} | {cp} | {row['selected_step']} | {row['mean']:.8f} | {row['worst_ood']:.8f} | {row['incremental_training_seconds']:.2f} |")
        lines += ['', '真实成本越低越好；REST是相对GeoSteiner误差（此表小数乘100为百分数）；Timing/Buffer是任务固定尺度成本。成本不能跨任务排序。', '']
        protocol = read(HERE/cohort/'protocol.json')
        for family,arms in protocol['families'].items():
            for arm in arms[1:]:
                for cp in ['best','final']:
                    def select(a): return {r['seed']:r for r in rows if (r['cohort'],r['family'],r['arm'],r['checkpoint'])==(cohort,family,a,cp)}
                    a,b = select(arm),select(arms[0]); seeds = sorted(set(a)&set(b))
                    if not seeds: continue
                    delta=np.array([a[s]['mean']-b[s]['mean'] for s in seeds]); ci=None
                    if len(seeds)==3:
                        rng=np.random.default_rng(261011); boot=delta[rng.integers(0,3,(10000,3))].mean(1); ci=np.quantile(boot,[.025,.975]).tolist()
                    c={'cohort':cohort,'family':family,'arm':arm,'control':arms[0],'checkpoint':cp,'seeds':seeds,'delta_each':delta.tolist(),'mean_delta':float(delta.mean()),'descriptive_seed_bootstrap95':ci}
                    contrasts.append(c); lines.append(f"- {family} {arm}−{arms[0]} ({cp})：{len(seeds)}种子，差值{c['mean_delta']:.8f}，各自{c['delta_each']}，描述性区间{ci}。")
    lines += ['', '## Critic校准', '', '只用于修正分支的共同初始化；训练/校准/评估标签分开。通过门槛仅排除初始零输出，不代表优于常数均值预测或证明有效对抗排序。', '', '| 种子 | 门槛 | holdout MSE | 零预测MSE | 训练均值常数MSE | 预测标准差 | 标签秒数 | 拟合秒数 |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in calibration:
        lines.append(f"| {r['seed']} | {r['qualified']} | {r['heldout_mse']:.8g} | {r['zero_predictor_mse']:.8g} | {r['training_mean_predictor_mse']:.8g} | {r['prediction_std']:.8g} | {r['label_seconds']:.2f} | {r['fit_seconds']:.2f} |")
    lines += ['', '## 解释限制', '', '- main继承原完整断点，原有best（含最初update0）仍参与选择；repaired从旧Geo第128步Actor和同一校准Critic出发，重置Adam/lambda，报告的是新增2048步。两者不能直接归因比较。',
              '- 修正模型仍通过当前greedy线长反推参考，候选分数仍与Critic有关，存在自举偏差；只有真实指标、预测分布和候选选择统计才能说明效果。',
              '- K3在两篇论文中改变目标归一化且增加教师物理评分；25ICCAD仍是现有组件。3个种子的bootstrap仅作描述。',
              '- 这是在已有试验结论基础上的探索性扩展，继续使用原测试基准，不是独立确认性研究。校准的oracle成本单列，不能声称完整流程无oracle。']
    write(HERE/'RESULTS.md','\n'.join(lines)+'\n')
    write(ART/'contrasts.json',json.dumps(contrasts,indent=2)+'\n')
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows))); tmp=ART/'metrics.csv.partial'
        with tmp.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        os.replace(tmp,ART/'metrics.csv')

if __name__=='__main__':main()
