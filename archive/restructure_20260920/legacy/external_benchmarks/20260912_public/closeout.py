"""Publish the audited report and update only this extension's project status."""
import re
import subprocess
from common import *

def main():
    delivery=read(BASE/'delivery_verification.json')
    assert delivery['status']=='pass' and delivery['failure'] is None
    assert read(BASE/'logs/pipeline/job.json')['status']=='exited'
    subprocess.run(['bash',str(ENV),PYTHON,str(BASE/'render_report.py')],check=True,cwd=BASE)
    summary=read(BASE/'summary.json')
    plan=read(BASE/'cache/input_manifest.json')
    performance=read(BASE/'logs/performance_summary.json')
    write(BASE/'logs/final_resources.json',dict(at=now(),gpus=gpu_snapshot(),
        scope='Current public external benchmark tasks have exited; other server workloads are outside this run.'))
    (BASE/'README.md').write_text('# 公开芯片物理设计外部测试\n\n'
        '状态：DONE（独立交付审计PASS）。\n\n'
        '- [中文完整报告、全部方法与结论边界](REPORT.md)\n'
        '- [完成清单](completion_manifest.json) · [独立审计](delivery_verification.json)\n'
        '- [冻结输入](cache/input_manifest.json) · [统计入口](summary.json)\n'
        '- [原始来源](sources/) · [逐net预测](results/) · [表](tables/) · [PNG/PDF图](figures/)\n\n'
        '29个下载设计；28个有目标degree样本；134550条有效nets；75个冻结模型；2018250条预测。\n'
        '全部新增内容经DRO/Codex逻辑路径保存；14个任务结果使用项目已有NVMe符号链接，详见存储记录。\n')
    evidence=[BASE/p for p in ['EXPERIMENT_PLAN.md','README.md','REPORT.md','report_artifacts.json','delivery_verification.json',
        'delivery_artifact_bindings.json','summary.json','cache/input_manifest.json','cache/input_audit.json','cache/reference_manifest.json',
        'logs/queue/completion.json','logs/pipeline/completion.json','logs/performance_summary.json','logs/storage_acceleration.json',
        'logs/final_resources.json','logs/prelaunch_tests.log','tests/test_summarize_cpu.log','logs/test_verify_delivery.log',
        'logs/evaluation_smoke/audit.json','notes/project_code_inventory.json','notes/selected_pin_coordinate_audit.json']]
    evidence.extend(sorted(BASE.glob('*.py')))
    evidence.extend(sorted((BASE/'tests').glob('test_*.py')))
    evidence.extend(sorted((BASE/'tests').glob('*verify*.log')))
    evidence.extend(sorted((BASE/'logs').glob('*verify*test*.log')))
    evidence.extend(sorted((BASE/'logs/delivery_failures').glob('*')))
    completion=dict(status='pass',extension='public_physical_design_zero_shot_v1',completed_at_utc=now(),
        designs_downloaded=plan['design_count'],designs_with_eligible_nets=plan['valid_design_count'],nets=plan['total_nets'],
        frozen_models=75,predictions=plan['expected_predictions'],reference_calls=2*plan['total_nets'],
        datasets=summary['datasets'],gpus_used=sorted(performance['gpus']),maximum_concurrent_evaluations=performance['maximum_concurrent_evaluations'],
        evaluation_queue_wall_seconds=performance['evaluation_queue_wall_seconds'],
        tables={k:v['rows'] for k,v in summary['tables'].items()},independent_delivery_counts=delivery['counts'],
        train_or_retune=False,original_stage0_to_8_results_unchanged=True,
        evidence=[artifact(p) for p in dict.fromkeys(evidence)],
        acceptance=dict(full_inputs_and_sources_verified=True,all_75_model_tasks_complete=True,full_dual_scale_reference=True,
            all_prediction_tree_checks_pass=True,independent_statistics_recomputed=True,all_own_jobs_exited_and_locks_released=True,
            cross_gpu_resume_and_idempotence_pass=True,report_and_figures_exported=True))
    write(BASE/'completion_manifest.json',completion)
    report_relative='external_benchmarks/20260912_public/REPORT.md'
    entry=TOP/'05_公开芯片外部测试扩展_报告.md'
    entry.write_text('# 公开芯片数据外部测试扩展（2026-09-12）\n\n'
        f'状态：**DONE**。完整结果与证据见[外部测试报告]({report_relative})。\n\n'
        '已下载ISPD08全16个设计及CircuitOps13个开源芯片物理实现，覆盖Nangate45、ASAP7、SKY130HD。'
        '29设计中28个有目标degree样本，共134550条合法nets；75个原冻结模型完成2018250条预测。\n\n'
        '新增数据整体mean regret：ERM 0.827848%，WDRO 0.819455%；P95分别3.554503%和3.521469%。'
        'Mixed ERM和GroupDRO整体mean/P95更低；Nangate45子集WDRO mean略高于ERM，不能声称全面优于其他基线。\n\n'
        'GPU2–5最多16任务并发，CPU参考求解24进程；所有本轮作业退出、锁释放。'
        '双尺度269100次reference calls与75模型全量预测通过，独立复算6张统计表及160行配对CI通过。\n\n'
        '[完成清单](external_benchmarks/20260912_public/completion_manifest.json) · '
        '[独立交付审计](external_benchmarks/20260912_public/delivery_verification.json) · '
        '[来源与下载](external_benchmarks/20260912_public/sources/) · '
        '[统计表](external_benchmarks/20260912_public/tables/)\n\n'
        '本次仅新增零样本外部测试，没有重新训练/调参或改动阶段0–8结果；二维单net线长指标不等于整芯片PPA。\n')
    progress=TOP/'01_WDRO-REST_项目进度.md'
    original=progress.read_text()
    backup=BASE/'logs/documentation_before_closeout';backup.mkdir(exist_ok=True)
    (backup/progress.name).write_text(original)
    start=original.index('### 公开芯片数据外部测试扩展（2026-09-12，用户已授权）')
    end=next(m.start() for m in re.finditer(r'^#{1,3} ',original,re.MULTILINE) if m.start()>start)
    section='### 公开芯片数据外部测试扩展（2026-09-12，用户已授权）\n\n'
    section+='- `DONE` 下载ISPD08全16个工业ASIC衍生设计及CircuitOps13个开源芯片物理实现，原始14392871条nets；29设计中28个含目标degree，134550条合法net全量测试。GCD保留零样本行；18条目标net因重复XY按冻结契约拒绝。\n'
    section+='- `DONE` 75个reference_precision_v2冻结模型、2018250条逐net预测、269100次双尺度GeoSteiner calls，全部来源/hash、合法性、最优状态、拓扑/长度和调用数通过；无新训练或外部调参。\n'
    section+='- `DONE` 6张统计表共3245行，含160行design-family聚类配对CI，独立全量复算通过；GPU2–5最多16任务并发，24CPU参考worker，全部本轮任务退出并释放锁。\n'
    section+='- `DONE` 新增数据整体mean regret：ERM 0.827848%，WDRO 0.819455%；P95 3.554503%→3.521469%。Mixed ERM/GroupDRO整体更好；Nangate45的WDRO mean略高于ERM，保留所有不利结果。\n'
    section+='- 证据：[扩展报告](05_公开芯片外部测试扩展_报告.md)、[完成清单](external_benchmarks/20260912_public/completion_manifest.json)、[独立交付](external_benchmarks/20260912_public/delivery_verification.json)。28项CPU实现/统计/审计测试、跨GPU恢复幂等与348507个pin独立坐标核验通过；首次交付JSON序列化故障已修复，失败记录保留并正式复核通过。\n'
    section+='- 所有内容经Codex逻辑路径保存，14个未启动任务输出接项目已有NVMe链接；[路径记录](external_benchmarks/20260912_public/logs/storage_acceleration.json)。本扩展不改变阶段0–8结论，阶段9未启动。\n\n'
    updated=original[:start]+section+original[end:]
    updated=updated.replace('| 下一动作 | 用户2026-09-12已授权公开芯片数据外部测试扩展：下载、输入核验和冻结75模型评测 DOING；阶段9未启动 |',
        '| 下一动作 | 公开芯片外部测试扩展DONE：29设计下载、134550 nets、75模型/2018250预测及独立交付通过；报告已交付，阶段9未启动 |',1)
    updated=re.sub(r'\| 更新时间 \| [^\n]* \|',f'| 更新时间 | {now()} |',updated,count=1)
    progress.write_text(updated)
    index=TOP/'03_项目文件总索引.md'
    text=index.read_text();(backup/index.name).write_text(text)
    needle='| 后续研究提案 |'
    addition='| 新增公开芯片外部测试 | [扩展报告](05_公开芯片外部测试扩展_报告.md)、[完整产物](external_benchmarks/20260912_public/)；29设计下载、134550 nets、75模型/2018250预测，已完成独立审计 |\n'
    assert needle in text
    if '[扩展报告](05_公开芯片外部测试扩展_报告.md)' not in text:
        text=text.replace(needle,addition+needle,1)
    index.write_text(text)
    write(BASE/'logs/documentation_closeout.json',dict(at=now(),completion=artifact(BASE/'completion_manifest.json'),
        updated_documents=[artifact(p) for p in [entry,progress,index]],backups=[artifact(p) for p in backup.iterdir()]))
    print(json.dumps(dict(status='pass',completion=str(BASE/'completion_manifest.json'),report=str(BASE/'REPORT.md'))))

if __name__=='__main__':
    main()
