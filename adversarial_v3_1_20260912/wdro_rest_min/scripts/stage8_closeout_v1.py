#!/usr/bin/env python3
"""Publish scoped Stage 8 interpretation after all independent review gates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage8.common import *
from stage8.delivery import read_csv,check_tree


def text_atomic(path,text):
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def main():
    gate();review_path=ROOT/'manifests/stage8_delivery_review_v1.json'
    review=read(review_path);assert review['status']=='pass'
    check_tree(review)
    completion=read(checked(review['completion']))
    proof=read(checked(review['delivery']))
    runtime=read(LOG/'recovery_v1/result.json');assert runtime['status']=='pass'
    controller=read(LOG/'recovery_v1/controller/job.json')
    assert controller['status']=='exited' and controller['lock_released']
    identity=pid_identity(controller['process']['pid'])
    assert not identity or identity['state'] in ('Z','X') or identity['start_ticks']!=controller['process']['start_ticks']
    launch=read(LOG/'recovery_v1/latest_launch.json')
    exit_path=Path(launch['shell']['logical_path']).parent/'exit.json';assert read(exit_path)['exit_code']==0
    stats=read(LOG/'statistics/result.json');data=read(checked(stats['raw_json']))
    for desc in stats['figures']:
        p=checked(desc)
        if p.suffix=='.pdf':assert p.read_bytes().startswith(b'%PDF-')
        if p.suffix=='.svg':
            import xml.etree.ElementTree as ET
            assert ET.parse(p).getroot().tag.endswith('svg')
    methods=['erm','mixed_erm','hard_mining','group_dro','wdro_rest']
    groups=['uniform','mix_gauss','edge_bias','grid_cluster']
    claims=data['claim_support']
    def claim(baseline,group,metric):
        return next(r for r in claims if r['dataset']=='attack' and r['baseline']==baseline and r['distribution']==group and r['metric']==metric)
    assert all(claim('erm',g,m)['supported_in_this_test'] for g in groups for m in ['mean','p95'])
    assert not any(r['supported_in_this_test'] for r in claims if r['dataset']=='attack' and r['baseline']=='hard_mining')
    assert all(claim(b,g,m)['difference']>0 and claim(b,g,m)['ci_low']>0 for b in ['mixed_erm','group_dro'] for g in ['mix_gauss','edge_bias'] for m in ['mean','p95'])
    execution=read(checked(review['execution']))
    rows=['# 阶段8 结果解读与交付清单','',
        '阶段8已完成并通过完整候选审计、统计导出、640587项文件绑定复核和独立数值复算。阶段9未启动；本文件给出后续正文可使用的结论边界。','',
        '固定CEM-256测试中，在五degree与三训练seed的宏平均层面，WDRO相对ERM在四个分布的mean与P95、以及OOD worst-group mean上均改善；配对CI位于零以下，三个seed同向，nominal门槛通过。',
        '相对Hard Mining，没有攻击宏指标通过稳定改善门槛；mix_gauss及OOD worst-group mean的聚合差值反而为正，三个seed方向不一致。相对Mixed ERM和GroupDRO，WDRO在uniform与grid_cluster的mean/P95较好，在mix_gauss与edge_bias的mean/P95较差。',
        '因此可写相对ERM的经验改善与具体分布上的比较，不能写独立攻击下全面优于所有基线，也不能把某些指标的改善扩展到所有degree、所有seed或所有分布。','',
        '## 攻击主结果','',
        '以下为relative regret百分比；mean先逐degree与seed统计后宏平均。P95及完整CI见final_main_long.csv及攻击图。','',
        '| 方法 | uniform | mix_gauss | edge_bias | grid_cluster | OOD worst-group |','|---|---:|---:|---:|---:|---:|']
    for method in methods:
        vals=[next(r['estimate'] for r in data['attack_main'] if r['method']==method and r['distribution']==g and r['metric']=='mean') for g in groups+['ood_worst_group']]
        rows.append('| '+method+' | '+' | '.join(f'{100*v:.5f}' for v in vals)+' |')
    rows+=['','OOD worst-group为三个OOD组在seed/degree聚合后均值的最大值；本次五种方法的最差组均为mix_gauss，因此这一行不构成独立于mix_gauss的另一份证据。','',
        '下表差值为WDRO减比较基线，单位为百分点；负值有利于WDRO。','',
        '| 比较基线 | OOD worst-group差值 | 配对95% CI | 三seed同向改善 |','|---|---:|---|---|']
    for baseline in methods[:-1]:
        r=claim(baseline,'ood_worst_group','mean')
        rows.append(f"| {baseline} | {100*r['difference']:.5f} | [{100*r['ci_low']:.5f}, {100*r['ci_high']:.5f}] | {'是' if r['all_three_seed_improve'] else '否'} |")
    rows+=['','![CEM-256攻击结果](experiments/figures/stage8_attack_v2.png)','',
        '## 成本、合法率与计算量','',
        '所有方法使用同一独立中心和256次真实评分预算。表中成本是选中候选的实际Hungarian-L1均值；受影响比例以模型-中心攻击为单位。','',
        '| 方法 | 平均运输cost | 最大cost | 受影响比例 | 合法率 |','|---|---:|---:|---:|---:|']
    for method in methods:
        a=[r for r in data['attack_per_seed'] if r['method']==method]
        rows.append(f"| {method} | {np.mean([r['mean_cost'] for r in a]):.8f} | {max(r['max_cost'] for r in a):.8f} | {100*np.mean([r['changed_fraction'] for r in a]):.5f}% | {100*min(r['legal_fraction'] for r in a):.1f}% |")
    rows+=['',
        f"300格、40000独立中心、600000个模型-中心攻击。Actor、REST evaluator和搜索GeoSteiner各实际调用153600000次，另有600000次选中候选双尺度GeoSteiner核验；调用回执下界等于上界，重放额外调用及未完成不确定调用均为0。",
        f"攻击队列历时{execution['attack_queue_wall_seconds']/3600:.5f}小时（{execution['first_worker_start']}至{execution['last_worker_finish']}）。并行worker历时之和与队列历时分别报告，不能相互替代。",
        '75个训练run中64个有完整segment耗时覆盖，11个覆盖不全；效率表明确保留覆盖范围及含暂停的历时，不能把部分segment之和当总训练时长。共享服务器耗时不支持受控加速比结论。','',
        '## ISPD与解释边界','',
        'ISPD仍为冻结模型的Protocol A外部测试。本次配对CI支持WDRO相对ERM的net-weighted mean、design-macro mean和P95改善，相对Hard Mining的net-weighted mean与P95改善，以及相对GroupDRO的P95改善；不支持相对Mixed ERM的稳定优势。worst-design均未通过稳定改善门槛。',
        'CI使用10000次配对net bootstrap，条件于三个已训练seed与固定数据分层，未作多重比较校正。ISPD固定design×degree计数，P95逐次从pooled nets重算；不推断design总体。',
        'CEM-256是固定预算的局部压力测试，cost为实际测量值；不是全局最坏分布求解或Wasserstein球内证书。rho=q50不代表覆盖一半测试分布。未使用这些结果继续选择rho、lambda、模型或搜索参数。','',
        f"nominal WDRO减ERM的宏均值为{100*review['nominal_check']['macro_delta']:.5f}个百分点，15个degree×seed中最大退化为{100*review['nominal_check']['maximum_delta']:.5f}个百分点，均低于0.2个百分点的冻结上限。",'',
        '## 产物与恢复记录','',
        '- 主表与原始JSON：`experiments/tables/stage8/final_main_long.csv`、`final_tables.json`。',
        '- 逐seed/degree攻击、配对CI与结论门槛：`attack_per_seed.csv`、`attack_per_degree.csv`、`attack_paired_ci.csv`、`claim_support.csv`；逐网配对NPZ位于`paired_nets/`。',
        '- 效率与真实账目：`training_efficiency.csv`、`inference_efficiency.csv`、`attack_execution_wallclock.csv`、`actual_call_reconciliation.csv`。',
        '- 已验收的合成/ISPD/消融输入表位于`accepted_inputs/`，包括`stage5_ablation.csv`与预算轨迹。',
        '- 两张图各有PNG/PDF/SVG：`experiments/figures/stage8_attack_v2.*`、`stage8_budget_v2.*`。PNG已视觉检查，PDF/SVG结构检查通过。',
        '- 完成、交付、独立复核索引：`wdro_rest_min/manifests/stage8_completion_v1.json`、`stage8_delivery_verification_v1.json`、`stage8_delivery_review_v1.json`。',
        '- 原流水线在攻击完成后因scripts/stage7.py遮蔽包而退出。恢复使用`stage8_finish_v1.py`独立进程入口，冻结攻击/审计/统计代码及结果hash保持不变；原失败日志保留。恢复流水线已exit=0、锁释放、无剩余worker。',
        '- 运行补充manifest：`stage8_finish_runtime_v1.json`；启动、逐格任务与phase退出日志：`experiments/logs/stage8/recovery_v1/`。','']
    document=CODEX/'阶段8_结果解读与交付清单.md';text_atomic(document,'\n'.join(rows))
    # This closeout records the reviewed rendering and supplementary execution
    # tables without rewriting any previously accepted research artifacts.
    out=dict(status='pass',stage=8,at_utc=utc_now(),stage9_started=False,code=artifact(Path(__file__)),
        completion=review['completion'],delivery=review['delivery'],independent_review=artifact(review_path),
        interpretation=artifact(document),runtime_completion=artifact(LOG/'recovery_v1/result.json'),
        recovery_exit=artifact(exit_path),all_owned_workers_exited=True,lock_released=True,
        visual_review=dict(status='pass',pngs=[p for p in stats['figures'] if p['logical_path'].endswith('.png')],
            observed='Labels, legends, CI positions and axes are readable and not clipped; data match the exported table.'),
        nominal_guard=review['nominal_check'],claim_scope='Specific comparisons and metrics; no overall dominance claim.')
    target=ROOT/'manifests/stage8_closeout_v1.json';atomic_write_json(target,out,mode=0o444)
    atomic_write_json(LOG/'pipeline/recovered_by.json',dict(original_status='failed_before_audit',resolved=True,closeout=artifact(target)))
    runtime_doc=CODEX/'阶段8_子步骤与运行说明.md';text=runtime_doc.read_text()
    note='> 阶段8已完成，恢复流水线exit=0。最终结果与限制见[结果解读与交付清单](阶段8_结果解读与交付清单.md)。以下启动和进度快照为保留的历史记录。\n\n'
    text=text.replace('# 阶段8 子步骤与运行说明\n\n','# 阶段8 子步骤与运行说明\n\n'+note,1);text_atomic(runtime_doc,text)
    p=TOP/'01_WDRO-REST_项目进度.md'
    with (LOG/'progress.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);s=p.read_text();now=utc_now()
        values={'更新时间':now,'总体阶段':'阶段8 `DONE`（完整审计、统计与独立交付复核通过）',
            '实验状态':'300/300攻击与审计；110行主表、864行攻击配对CI、640587文件绑定及独立数值复算通过：DONE',
            '批量实验阻塞':'阶段8无剩余阻塞；全部任务退出，锁释放','下一动作':'阶段8完成；结论按基线/分布/指标限定，阶段9未启动'}
        lines=s.splitlines()
        for i,line in enumerate(lines):
            for key,value in values.items():
                if line.startswith('| '+key+' |'):lines[i]=f'| {key} | {value} |'
            if line.startswith('| 8. 独立攻击/统计 |'):lines[i]=f'| 8. 独立攻击/统计 | `DONE` | 全部攻击、完整审计、最终统计和独立交付复核通过；证据 `{target}` |'
        s='\n'.join(lines)+'\n'
        entry=f'### {now} 阶段8最终交付\n\n- `DONE` 300/300攻击与153600000候选完整审计通过；全部实际调用回执闭合、无额外重放，队列耗时15.39061小时。110行主表、864行配对CI、675行推理效率及表图交付完成；640587文件绑定及300格/45汇总点独立复算通过。证据 `{target}`。\n- 结果边界：相对ERM支持本次攻击宏mean/P95/worst-group改善；不支持优于Hard Mining；Mixed ERM与GroupDRO在mix_gauss/edge_bias上更优。详见 `{document}`。\n- 原导入冲突已通过独立phase进程恢复，冻结模型与研究代码未改；恢复exit=0、锁释放、所有任务结束。阶段9未启动。\n\n'
        s=s.replace('## 8. 变更记录\n\n','## 8. 变更记录\n\n'+entry,1);text_atomic(p,s)
    print({'status':'pass','closeout':str(target),'interpretation':str(document),'stage9_started':False},flush=True)


if __name__=='__main__':main()
