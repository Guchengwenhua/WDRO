"""Standalone figures and a Chinese evidence-linked external-test report."""
import csv
from collections import Counter, defaultdict
from common import *

LABELS={'erm':'ERM','mixed_erm':'Mixed ERM','hard_mining':'Hard Mining','group_dro':'GroupDRO','wdro_rest':'WDRO-REST'}
DATASETS={'ispd2008_global_routing':'ISPD 2008 GR','circuitops_nangate45':'CircuitOps Nangate45','circuitops_asap7':'CircuitOps ASAP7','circuitops_sky130hd':'CircuitOps SKY130HD','overall':'All eligible nets'}

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=read(BASE/'cache/input_manifest.json')
    summary=read(BASE/'summary.json')
    delivery=read(BASE/'delivery_verification.json')
    assert delivery['status'].lower()=='pass'
    queue=read(BASE/'logs/queue/completion.json')
    refs=read(BASE/'cache/reference_manifest.json')
    audit=read(BASE/'cache/input_audit.json')
    scopes=['dataset:'+x for x in summary['datasets']]
    methods=list(LABELS)
    rows={(r['scope'],r['method']):r for r in summary['summary']}
    out=BASE/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors=['#697386','#55a1c4','#e2a34b','#6f9c71','#b64c68']
    xs=np.arange(len(scopes));width=.15
    for ax,metric,title in zip(axes,['net_weighted_mean','p95'],['Mean relative regret','P95 relative regret']):
        for j,method in enumerate(methods):
            ax.bar(xs+(j-2)*width,[100*rows[(scope,method)][metric] for scope in scopes],width,color=colors[j],label=LABELS[method])
        ax.set_xticks(xs,[DATASETS[s.split(':',1)[1]].replace('CircuitOps ','') for s in scopes])
        ax.set_ylabel('Relative regret (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Frozen models on additional public physical designs (three-seed mean)',fontsize=12)
    figures=[]
    for ext in ('png','pdf'):
        path=out/f'public_external_mean_p95.{ext}';fig.savefig(path,dpi=180);figures.append(artifact(path))
    plt.close(fig)
    with (BASE/'tables/per_design_seed.csv').open() as f:design_rows=list(csv.DictReader(f))
    grouped=defaultdict(dict)
    for row in design_rows:
        if int(row['count']):grouped[row['design_id']][(row['method'],int(row['training_seed']))]=float(row['mean'])
    labels=sorted(grouped)
    delta=np.array([np.mean([grouped[d]['wdro_rest',s]-grouped[d]['erm',s] for s in (101,102,103)])*100 for d in labels])
    fig,ax=plt.subplots(figsize=(9,9),layout='constrained')
    ax.barh(np.arange(len(labels)),delta,color=np.where(delta<=0,'#3b9487','#cc6677'))
    ax.set_yticks(np.arange(len(labels)),[x.replace('circuitops_','').replace('ispd2008_gr_','ISPD08:') for x in labels],fontsize=8)
    ax.axvline(0,color='#444444',linewidth=.8);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('WDRO minus ERM mean regret (percentage points; lower favors WDRO)')
    ax.set_title('Every eligible design; three fitted seeds equally averaged')
    for ext in ('png','pdf'):
        path=out/f'wdro_vs_erm_per_design.{ext}';fig.savefig(path,dpi=180);figures.append(artifact(path))
    plt.close(fig)
    gpu_counts=Counter(row['gpu'] for row in queue['completed'])
    telemetry=[json.loads(line) for line in (BASE/'logs/queue/resources.jsonl').read_text().splitlines()]
    peaks={str(i):dict(tasks=gpu_counts[i],max_gpu_utilization=max(s['gpus'][i]['utilization'] for s in telemetry),
                       max_total_used_mib=max(s['gpus'][i]['used_mib'] for s in telemetry)) for i in gpu_counts}
    performance=dict(gpus=peaks,maximum_concurrent_evaluations=max(len(s['active']) for s in telemetry),
        evaluation_queue_wall_seconds=queue['wall_seconds'],reference_workers=refs['workers'],reference_worker_seconds=refs['summed_worker_seconds'],
        actor_seconds=sum(read(row['result']['logical_path'])['actor_seconds'] for row in queue['completed']),
        peak_reserved_mib=max(read(row['result']['logical_path'])['peak_reserved_mib'] for row in queue['completed']),
        note='NVML total memory/utilization includes co-tenants on shared GPUs; wall timings are concurrent throughput, not isolated method efficiency.')
    write(BASE/'logs/performance_summary.json',performance)
    coverage=[]
    for design in plan['designs']:
        for g in design['groups']:
            coverage.append(dict(dataset=design['dataset'],design=design['design_id'],degree=g['degree'],parsed=g['parsed_count'],valid=g['valid_count'],invalid=g['invalid_count'],invalid_reasons=json.dumps(g['invalid_reasons'],sort_keys=True)))
    with (BASE/'tables/input_coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    overall_erm=rows['overall','erm'];overall_wdro=rows['overall','wdro_rest']
    lines=['# 公开芯片物理设计外部测试报告','',f'完成日期：{now()}。交付审计：**PASS**。','',
        f'新增下载29个设计；原始普通信号网络共{plan["total_source_nets"]:,}条。28个设计有目标degree样本，纳入{plan["total_nets"]:,}条合法网络；75个冻结模型产生{plan["expected_predictions"]:,}条逐net预测。',
        '', '模型来自reference_precision_v2，仅用合成数据训练/选模；本次没有新增训练或外部测试调参。旧阶段0–8产物保持原样。',
        '', f'本轮整体net-weighted mean：ERM {100*overall_erm["net_weighted_mean"]:.6f}% → WDRO {100*overall_wdro["net_weighted_mean"]:.6f}%；P95：{100*overall_erm["p95"]:.6f}% → {100*overall_wdro["p95"]:.6f}%。WDRO相对ERM有小幅改善，但整体mean和P95仍落后于Mixed ERM和GroupDRO。',
        '', '分数据集看，WDRO在ISPD08的mean/design-macro相对ERM差值区间低于0；Nangate45的WDRO mean和P95略高于ERM，且mean差值区间跨0。ASAP7的WDRO mean点估计较低但只有两个设计家族，区间跨0；SKY130HD仅43条net、一个设计，不能据此声称稳定跨芯片优势。结果没有支持WDRO全面优于所有基线。',
        '', '## 数据与覆盖','',
        '- [ISPD08官方来源](https://www.ispd.cc/contests/08/ispd08rc.html)：16个工业ASIC衍生全局布线设计；8个与ISPD07共享，此处只计一次。',
        '- [NVlabs/CircuitOps作者仓库](https://github.com/NVlabs/CircuitOps)：13个公开芯片物理设计研究实现，覆盖Nangate45、ASAP7、SKY130HD；包含AES、Ariane、BlackParrot、Ibex、JPEG、SweRV、NVDLA等。没有将它们一概称为已流片量产芯片。',
        '- CircuitOps全源690,475条net中10,463条缺上游IR未导出的I/O，按完整DEF连接整网拒绝；这些拒绝均不在当前目标degree集合。目标degree合计3341条，几何合法3341条。',
        '- 所有数据按原始degree 10/20/30/40/50筛选，未截断、拆分、补点或抽样。ISPD08有18条目标net因重复XY导致degree改变而拒绝。GCD没有目标degree样本，保留零样本行，不伪造结果。',
        '- 三组相同归一化几何来自不同源net（均在各自同一ISPD设计内部）；保留真实网络计数并明确记录。',
        f'- 与原有{audit["prior_nets_compared"]:,}条合成validation/test及ISPD18/19逐无序pin hash比较，几何交集为0。此项不等同于重新扫描全部历史训练流。',
        '', '| 数据集 | 下载设计数 | 有效设计数 | 合法nets |','|---|---:|---:|---:|']
    for dataset in summary['datasets']:
        ds=[d for d in plan['designs'] if d['dataset']==dataset]
        r=rows[('dataset:'+dataset,'erm')]
        lines.append(f'| {DATASETS[dataset]} | {len(ds)} | {r["design_count"]} | {r["count"]:,} |')
    lines+=['','完整逐design×degree计数：[input_coverage.csv](tables/input_coverage.csv)。来源与许可：[竞赛说明](notes/contests_sources.md)、[开源设计说明](notes/real_design_sources.md)。ICCAD15/17官方失效链接已记录，未计入已测试数据。',
        '', '## 全方法结果','', '下表数值均为relative regret ×100（%），越低越好。每个seed先独立计算指标，再对3个seed等权平均；P95不是先平均每net预测再取分位数。',
        '', '| 数据集 | 方法 | mean (%) | P95 (%) | design-macro (%) | worst-design (%) |','|---|---|---:|---:|---:|---:|']
    for scope in scopes+['overall']:
        for method in methods:
            r=rows[scope,method];label=DATASETS[scope.split(':',1)[-1]]
            lines.append(f'| {label} | {LABELS[method]} | {r["net_weighted_mean"]*100:.6f} | {r["p95"]*100:.6f} | {r["design_macro_mean"]*100:.6f} | {r["worst_design_mean"]*100:.6f} |')
    lines+=['','![各数据集mean和P95](figures/public_external_mean_p95.png)','',
        '整体汇总由ISPD08的大样本量主导，应优先查看分数据集/设计结果。不同方法和degree的不利结果全部保留。',
        '', '## 配对差值与解释边界','',
        'WDRO减各基线的配对mean与design-macro差值、95%区间和各seed结果见[paired_design_cluster_ci.csv](tables/paired_design_cluster_ci.csv)。2000次bootstrap以design-family为簇，同一家族的不同实现、所有nets和三个seed共同重采样；区间反映有限设计家族重采样，不能解释为对任意芯片的保证。SKY130HD只有一个设计家族，其区间退化，不提供有意义的跨设计不确定性估计。',
        '', '![逐设计WDRO减ERM](figures/wdro_vs_erm_per_design.png)','',
        '测试度量是二维无障碍RSMT relative regret。ISPD输入的层、拥塞、容量以及真实设计的障碍、DRC、时序均不进入该指标，因此不能直接声称提高完整芯片布线评分或PPA。',
        '', '## 执行与验收','',
        f'- 双尺度reference：{plan["total_nets"]:,}条网络分别在2^27/2^28求解，共{refs["calls"]:,}次正式GeoSteiner calls，全量optimal状态、整数连通嵌入和长度相等核验通过。',
        f'- 75/75模型任务完成；固定batch256、T=1 greedy，每条预测独立Python拓扑与线长核验，所有输入/模型/reference/输出hash和配对身份通过。',
        f'- 正式GPU分配：{dict(sorted(gpu_counts.items()))}；最大同时{performance["maximum_concurrent_evaluations"]}个评价进程；24个CPU参考worker。GPU队列墙钟约{queue["wall_seconds"]:.1f}秒。',
        f'- 单任务torch最大reserved为{performance["peak_reserved_mib"]:.1f}MiB。显存按实际推理需求分配；已满载共享GPU不被抢占或清理。各卡实测记录见[performance_summary.json](logs/performance_summary.json)。',
        '- CPU单元/统计测试、跨GPU断点恢复和完成幂等检查通过。正式结果全量独立统计复算通过；本轮GPU任务退出、文件锁可获取释放。',
        '- 除连接完整性审计外，独立DEF+LEF复算AES/ibex/JPEG共348,507个pin，包含本轮649条net/7100个pin，最大误差0.875 DBU低于1 DBU阈值；见[独立坐标审计](notes/selected_pin_coordinate_audit.json)。',
        '- [独立交付审计](delivery_verification.json)、[输入审计](cache/input_audit.json)、[双尺度reference](cache/reference_manifest.json)、[队列完成](logs/queue/completion.json)、[统计入口](summary.json)。',
        '', '## 文件与恢复','',
        '统一入口是 `/remote-home/Guilin/DRO/Codex/external_benchmarks/20260912_public/`。原始下载和大部分结果直接存于该目录；部分尚未启动任务的结果通过本目录内的逐任务符号链接接到项目已有 `Codex/experiments/public_external_20260912/` 的NVMe存储，以加速原子写盘。逻辑路径与resolved path逐文件记录，详见[存储加速记录](logs/storage_acceleration.json)。',
        '', '完整入口：[实验计划](EXPERIMENT_PLAN.md)、[来源清单](sources/contests/designs_manifest.json)、[真实设计清单](sources/real_designs/design_manifest.json)、[冻结输入](cache/input_manifest.json)、[逐模型结果](results/)、[6张统计表](tables/)、[图PNG/PDF](figures/)。',
        '', '任务已完成，不需要再次启动。若文件审计失败或中断需恢复，先检查`logs/pipeline/job.json`及任务PID、锁、心跳，再在独立tmux运行：',
        '', '```bash', f'bash {ENV} {PYTHON} {BASE}/pipeline.py', f'bash {ENV} {PYTHON} {BASE}/verify_delivery.py', f'bash {ENV} {PYTHON} {BASE}/render_report.py', '```',
        '', '队列会观察仍在运行的原任务，参考/预测按已校验分片恢复；完成任务再次调用只核验现有产物。', '']
    (BASE/'REPORT.md').write_text('\n'.join(lines))
    write(BASE/'report_artifacts.json',dict(status='pass',at=now(),report=artifact(BASE/'REPORT.md'),figures=figures,
        input_coverage=artifact(BASE/'tables/input_coverage.csv'),performance=artifact(BASE/'logs/performance_summary.json'),delivery=artifact(BASE/'delivery_verification.json')))
    print(str(BASE/'REPORT.md'))

if __name__=='__main__':
    main()
