"""Render the independently computed acceptance evidence; no selection or training."""
import ast
import csv
import datetime as dt
import itertools
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def rows(name):return list(csv.DictReader((OUT/name).open()))
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(header,values):return ['|'+'|'.join(header)+'|','|'+'|'.join(['---']*len(header))+'|']+['|'+'|'.join(map(str,r))+'|' for r in values]
def link(name,path):return f'[{name}]({path})'

def main():
    audit=read(OUT/'audit.json');stats=read(OUT/'statistics_independent.json');probes=read(OUT/'probes.json')
    assert audit['status']=='PASS_CORE_WITH_DEGENERATE_WEIGHT_DECAY_ABLATION' and probes['status']=='PASS'
    values=rows('metrics_independent.csv');curves=rows('validation_curves_independent.csv');training=rows('training_audit.csv');ck=rows('checkpoint_audit.csv');gates=rows('promotion_independent.csv')
    def aggregate(stage,arm,step,degree=30):
        rr=[r for r in values if r['stage']==stage and r['arm']==arm and r['checkpoint']==f'step{step}' and int(r['degree'])==degree]
        assert rr,(stage,arm,step,degree)
        return {k:float(np.mean([float(r[k]) for r in rr])) for k in ('mean','worst','p95')}|dict(seeds=len(rr))
    control=aggregate('screen','E50',20480);capacity_control=aggregate('screen','BASE',20480)
    summaries=[]
    for a in ('E50','E20','E100','COS','WD','BNF','ADV','D4','GD','GDF','GDS','GDN','GDNF','B256','BASE','WIDE','DEEP'):
        v=aggregate('screen',a,20480);b=capacity_control if a in ('BASE','WIDE','DEEP') else control
        summaries.append(dict(arm=a,**v,mean_gain_percent=100*(b['mean']-v['mean'])/b['mean'],worst_gain_percent=100*(b['worst']-v['worst'])/b['worst']))
    late={}
    for s in (32768,40960,45056,49152):
        rr=[r for r in curves if r['id'].startswith('final_E50_') and int(r['step'])==s]
        assert len(rr)==5;late[s]={k:float(np.mean([float(r[k]) for r in rr])) for k in ('mean','worst')}
    sensitivity={}
    for slot in ('budget_mean','budget_worst'):
        d=np.asarray(stats[slot]['seed_improvement']);null=np.asarray([np.mean(d*np.asarray(signs)) for signs in itertools.product((-1,1),repeat=len(d))])
        sensitivity[slot]=dict(two_sided_exact_sign_flip_p=float(np.mean(np.abs(null)>=abs(d.mean())-1e-12)),
            one_sided_exact_sign_flip_p=float(np.mean(null>=d.mean()-1e-12)),scope='posthoc sensitivity only; not the preregistered decision rule')
    group_weights=[]
    for arm in ('GD','GDF','GDS','GDN','GDNF'):
        qq=np.array([ast.literal_eval(r['final_q']) for r in training if r['id'].startswith('screen_'+arm+'_')])
        assert len(qq)==3;group_weights.append(dict(arm=arm,mean_final_q=qq.mean(0).tolist(),mean_max_q=float(qq.max(1).mean())))
    architecture_costs=[]
    for arm in ('BASE','WIDE','DEEP'):
        cc=[read(ROOT/f'artifacts/runs/screen_{arm}_s{s}/complete.json') for s in (201,202,203)]
        architecture_costs.append(dict(arm=arm,parameters=int(next(r['parameter_count'] for r in ck if r['id']==f'screen_{arm}_s201')),
            mean_elapsed_hours=float(np.mean([c['elapsed_seconds'] for c in cc])/3600),
            mean_peak_reserved_mib=float(np.mean([c['peak_reserved_bytes'] for c in cc])/2**20)))
    supplemental=dict(endpoint_summaries=summaries,late_validation=late,seed_level_sensitivity=sensitivity,group_weights=group_weights,
        architecture_costs=architecture_costs,
        late_validation_gain_percent={k:100*(late[45056][k]-late[49152][k])/late[45056][k] for k in ('mean','worst')})
    dump('supplemental.json',supplemental)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for key,color in [('mean','#17608a'),('worst','#bf602f')]:
        base=stats['budget_'+key]['baseline_value'];mid=aggregate('final','E50',32768)[key];end=stats['budget_'+key]['candidate_value']
        axes[0].plot([12288,32768,49152],100*(base-np.array([base,mid,end]))/base,'o-',color=color,label=key)
        for seed in range(101,106):
            rr=[r for r in curves if r['id']==f'final_E50_s{seed}']
            axes[1].plot([int(r['step']) for r in rr],[float(r['mean']) for r in rr],alpha=.65,label=str(seed) if key=='mean' else None) if key=='mean' else None
    axes[0].set(xlabel='Total training updates',ylabel='Test improvement (%)',title='E50 budget scaling: five seeds');axes[0].legend()
    axes[1].set(xlabel='Total training updates',ylabel='Validation mean wirelength',title='Final E50 validation trajectories');axes[1].legend(ncol=3,fontsize=8)
    labels=['Same new samples','Same new updates']
    for i,(key,color) in enumerate([('mean','#17608a'),('worst','#bf602f')]):
        ss=[stats['batch_'+a+'_'+key] for a in ('sample_matched','update_matched')]
        y=np.array([s['improvement_percent'] for s in ss]);ci=np.array([np.array(s['ci95'])*100/s['baseline_value'] for s in ss])
        axes[2].errorbar(np.arange(2)+(i-.5)*.1,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),fmt='o',capsize=5,color=color,label=key)
    axes[2].set_xticks(range(2),labels);axes[2].axhline(0,color='grey',lw=1)
    axes[2].set(ylabel='B256 improvement over E50 (%)',title='Batch comparison: posthoc 95% CI');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'acceptance_summary.{ext}',dpi=180)
    plt.close(fig)
    m=audit['metadata'];total_lengths=audit['independent_test_lengths']+audit['independent_selected_validation_lengths']
    lines=['# 纯 RL 扩量实验独立验收（2026-09-26）','',
        '**结论：核心工程、预算、数值与预注册统计验收通过；权重衰减消融无有效扰动，单独判为无效比较。**',
        '',f"验收状态：`{audit['status']}`。本报告针对 `{ROOT.name}`，没有合并其他并行研究分支的结果。",'',
        '新增证据支持“原有纯 RL 训练预算不足，继续训练仍可改善线长”：12,288 → 49,152 步后，平均线长下降 2.43%，最差组平均线长下降 2.62%，五个种子均改善，两个预注册 bootstrap 区间均完全大于零。该结果不能单独区分预训练和后训练的因果贡献，也不意味着原有误差全部由训练不足造成。',
        '', '训练采用在线新样本，所以步数增加同时增加了数据量。本轮长训练比较验证的是两者联合扩量，不能单独归因于优化器更新次数或训练数据量。',
        '', '本次为保存产物的独立 CPU 验收，没有启动新训练或新 GPU 推理。训练、数据、源码、原自动报告均保持原样。',
        '', '## 1. 完成状态与独立检查','',
        *table(['检查','验收结果'],[
            ['生命周期',f"{m['start_utc']} 至 {m['end_utc']}，约 {m['total_hours']:.2f} 小时"],
            ['训练 / 测试任务',f"{audit['training_tasks']} / {audit['evaluation_tasks']}，全部完成"],
            ['worker',f"{m['successful_workers']} 次成功退出，{m['admission_deferrals']} 次入场延后；任务失败 0"],
            ['运行身份与锁',f"{m['exited_process_identities']} 个已结束身份，{m['released_locks']} 个锁可取得；没有本实验活动队列"],
            ['冻结哈希',f"{m['unique_hashed_files']:,} 个独立文件，{m['hash_references']:,} 个引用全部匹配"],
            ['实际新增优化器更新',f"{audit['audited_updates']:,}，低于冻结预算上限 1,200,128"],
            ['实际采样 / 贪心 baseline',f"各 {audit['sampled']:,} 条"],
            ['继承与 Actor 状态',f"{audit['exact_parent_inheritances']} 次父模型继承，{audit['verified_saved_actors']} 个已登记 Actor 内容摘要核验"],
            ['树与线长',f"独立复算 {total_lengths:,} 条；最大绝对误差 {audit['max_length_error']:.3g}"],
            ['选模及测试隔离',f"最后训练 {audit['last_training_finished']}；首次测试 {audit['first_test_call']}；全部模型先冻结"],
            ['数据重叠',f"{audit['exact_new_split_nets']:,} 条新校准/验证/测试数据内部无精确重复，且与上一轮验证/测试无精确重叠"],
            ['配对训练流',f"{audit['paired_batch64_log_entries']:,} 条相同 seed/step 的 batch64 日志摘要一致；另复现 21 个采样探针"],
            ['外部求解器',f"调用账本为 {audit['oracle_calls']}；训练代码与导入保护核查未发现外部参考求解器路径"],
        ]),
        '',f"累计 worker 时间 {m['worker_hours']:.2f} 小时，其中训练 worker 时间 {m['training_worker_hours']:.2f} 小时。这是并行进程时长之和，不是 GPU 满载小时；共享设备上的实际算力利用率不能由该数直接推出。",'',
        '58 个训练任务由 51 个筛选、2 个新增种子父模型、5 个长训练构成。没有候选通过验证集晋级门槛，所以无需运行条件分支中的额外长训练；任务数少于上限符合冻结协议。',
        '', '计分独立使用保存的树端点检查取值范围、无环和连通性，再用坐标广播/区间极值重算 REST span 线长，不调用原报告的 scorer。核验包含 4,325,376 条测试输出及 475,136 条选中检查点的验证输出；这不是同等数量的独立测试样本，也没有重新执行全部神经网络前向。',
        '', '训练日志逐步复算学习率、group 权重、归一化风险、前一时刻优势方差使用、数据流步号、链式摘要与选模条件；并检查更新/采样/回传账本闭合。父模型检查覆盖 Actor、优化器矩、RNG、BN 以及应继承的目标状态。采样探针复用冻结坐标生成器，本次没有重跑全部训练以逐步再现梯度。',
        '', '## 2. 主结论：扩大训练预算有效','',
        '测试 degree=30，每组 8,192 个 net，共 32,768 个 net；五个训练种子共同使用同一测试集。单次贪心解码，不做测试时搜索。指标为原始坐标下的线长，越小越好。最差组先对每个训练模型取四个组均值的最大值，再跨种子平均。',
        '',*table(['指标','12,288 步','49,152 步','相对改善','99% 绝对改善区间','改善种子'],[
            [label,f"{stats[slot]['baseline_value']:.6f}",f"{stats[slot]['candidate_value']:.6f}",f"{stats[slot]['improvement_percent']:.3f}%",f"[{stats[slot]['simultaneous_ci99'][0]:.6f}, {stats[slot]['simultaneous_ci99'][1]:.6f}]",'5/5']
            for slot,label in [('budget_mean','平均线长'),('budget_worst','最差组平均线长')]]),
        '', '独立使用冻结 RNG 种子、各 10,000 次按训练种子和组内 net 配对重采样，点估计、逐种子差值、95% 与 99% 区间均与原报告相符。五个预留主比较槽各用 99% 区间，名义上按 Bonferroni 控制 95% 族覆盖；未晋级的三个槽维持“不提出结论”。两个实际主比较均满足改善至少 0.5%、多数种子改善且区间下界大于零的预设门槛。',
        '', '独立新增的 104、105 两个种子，其平均线长分别改善 2.491%、2.292%，最差组分别改善 2.477%、2.604%，方向与原三个种子一致。',
        '', '推断限度：训练种子仍只有五个，bootstrap 对种子总体的尾部估计可能不稳定。补充的种子级精确双侧 sign-flip 敏感性检验，两项均为 p=0.0625（单侧 p=0.03125）；这是事后敏感性分析，不替换冻结判据。因此可表述为“达到预注册探索性支持门槛”，不宜表述为各种统计检验下均已确证。',
        '', '## 3. Batch 收益需要按预算解释','',
        *table(['比较','每模型新增样本 E50 / B256','平均改善','最差组改善'],[
            ['相同样本：E50 20,480 vs B256 14,336','524,288 / 524,288',f"{stats['batch_sample_matched_mean']['improvement_percent']:+.4f}%",f"{stats['batch_sample_matched_worst']['improvement_percent']:+.4f}%"],
            ['相同更新：E50 20,480 vs B256 20,480','524,288 / 2,097,152',f"{stats['batch_update_matched_mean']['improvement_percent']:+.4f}%",f"{stats['batch_update_matched_worst']['improvement_percent']:+.4f}%"]]),
        '', '相同样本比较使用相同 seed、相同基础数据块 12,289–20,480；B256 每步打包四个基础块。前者的两项 95% 和 99% 配对区间均跨零；没有显示 batch256 的样本效率改善。后者有正向信号，但更新次数相同伴随四倍新增样本，不能单独归因于 batch 大小。本段 5,000 次 bootstrap 为事后描述性分析，不纳入预注册主检验。B256 原本即不参加晋级。',
        '', '## 4. 调参、GroupDRO 与容量筛选','',
        '下表为固定筛选终点 20,480 步的测试值，完整保留负结果。E50 系列对照种子为 101–103；BASE/WIDE/DEEP 为从头训练的 201–203，必须与 BASE 比较。表中的测试值不用于晋级或重新挑选赢家。',
        '',*table(['方案','平均线长','最差组线长','相对对应对照：平均','最差组'],[
            [r['arm'],f"{r['mean']:.6f}",f"{r['worst']:.6f}",f"{r['mean_gain_percent']:+.3f}%",f"{r['worst_gain_percent']:+.3f}%"] for r in summaries]),
        '', 'E20/E100 分别调整学习率为 2e-5/1e-4；COS 使用预定至 49,152 步的余弦退火；BNF 冻结 BN 统计；ADV 使用上一时刻的优势 EMA 标准差；D4 使用坐标对称增强。GD 为原始 GroupDRO，GDF 加 0.2 均匀混合，GDS 降低组权重学习率，GDN/GDNF 使用固定父模型组成本归一化（后者还加均匀混合）。',
        '', '独立重算验证集晋级要求：至少 0.2% 聚合改善、至少 2/3 种子改善，另一指标恶化不超过 1%。调参和容量候选均未同时满足要求；完整门槛值见 promotion_independent.csv。WIDE 的测试小幅正向平均值不构成通过验证集筛选的证据。所有 GroupDRO 变体在本次固定终点的两个测试聚合指标都弱于 E50。',
        '',*table(['架构','参数量','每模型平均实际运行小时','平均峰值保留显存 MiB'],[[r['arm'],f"{r['parameters']:,}",f"{r['mean_elapsed_hours']:.2f}",f"{r['mean_peak_reserved_mib']:.1f}"] for r in architecture_costs]),
        '', '容量组匹配更新和样本数，未匹配 FLOPs。运行时间含验证/保存及共享设备的调度影响，不是独占 GPU 的标准化速度测试。',
        '', '未晋级限制了结论范围：大模型只验证到 20,480 步，不能排除更长训练或专门调参的收益；COS 也未走完原定 49,152 步退火周期，不能据此断言完整退火无用。',
        '', '## 5. 验收发现：WD 消融实际退化','',
        'WD 使用 AdamW、学习率 5e-5、weight_decay=1e-4，理论乘法因子为 1−5e-9。在当前 FP32 实现中该因子舍入为 1，权重衰减乘法没有改变参数。CPU 探针逐位验证了此现象；同时三个种子的 WD 与 E50 在所有登记检查点的 Actor 哈希完全相同，十步受控 Adam/AdamW 更新也逐位相同。',
        '', '**这一组不是有效的权重衰减消融，不能支持“正则化没有收益”。** 原自动报告的数值没有算错，但将 WD 与其他有效调参负结果并列而不解释会产生误导。该限制不影响 E50 长训练或其他方案的保存产物和统计结论。验收保留原结果，不用测试集调整衰减强度或补训后重新包装本轮结果；新的衰减实验需单独冻结方案和验证流程。',
        '', '## 6. 收益递减与尚未验证的部分','',
        *table(['E50 总步数','五种子测试平均','五种子测试最差组'],[
            ['12,288',f"{stats['budget_mean']['baseline_value']:.6f}",f"{stats['budget_worst']['baseline_value']:.6f}"],
            *[[str(s),f"{aggregate('final','E50',s)['mean']:.6f}",f"{aggregate('final','E50',s)['worst']:.6f}"] for s in (32768,49152)]]),
        '',f"最后 4,096 步（45,056 → 49,152）五种子验证均值仍改善 {supplemental['late_validation_gain_percent']['mean']:.3f}%，最差组改善 {supplemental['late_validation_gain_percent']['worst']:.3f}%，但收益已小于早期扩量。五个验证最佳检查点都在 46,080–48,128 步，不能仅由最后端点断言已经收敛或以后必然继续改善。",'',
        'degree20/40 只用最终 E50 权重做了探索性迁移评估，没有对应的扩量前迁移对照：', '',
        *table(['degree','测试平均','测试最差组','每种子测试 net 数'],[[d,f"{aggregate('final','E50',49152,d)['mean']:.6f}",f"{aggregate('final','E50',49152,d)['worst']:.6f}",'8,192'] for d in (20,40)]),
        '', '这些绝对线长不能跨 degree 直接比较好坏，也不能证明跨规模泛化因扩量而改善。本研究没有最优线长参考，不报告最优相对误差；没有真实芯片合法布线约束验证，也没有连续 Wasserstein 鲁棒性证书。',
        '', '## 7. 交付与复现','',
        '独立证据目录：'+str(OUT)+'。', '',
        '- `verify.py` / `verify.log`：哈希、运行身份、资源分配、逐步日志、继承与全部保存树的数值复算。',
        '- `statistics.py` / `statistics_independent.json` / `*_bootstrap.npy`：独立置信区间与全部重采样差值。',
        '- `probes.py` / `probes.json`：数据块配对、模型回执及 FP32 权重衰减探针。',
        '- `metrics_independent.csv`、`training_audit.csv`、`checkpoint_audit.csv`、`promotion_independent.csv`：可追溯明细。',
        '- `input_hashes.json` / `complete.json`：原输入冻结摘要与本次验收交付摘要。',
        '- `HANDOFF_at_launch.md`：原启动交接的逐字节归档，保留旧 delivery.json 中的 handoff 哈希；根目录 HANDOFF.md 更新为完成状态。',
        '', '复验命令（在实验目录，均为 CPU）：', '', '```bash',
        "CUDA_VISIBLE_DEVICES='' RL_ONLY_CPU_THREADS=1 bash run_env.sh artifacts/acceptance_20260926/verify.py",
        "CUDA_VISIBLE_DEVICES='' RL_ONLY_CPU_THREADS=1 bash run_env.sh artifacts/acceptance_20260926/statistics.py",
        "CUDA_VISIBLE_DEVICES='' RL_ONLY_CPU_THREADS=1 bash run_env.sh artifacts/acceptance_20260926/probes.py",'```','',
        f'![验收图表]({OUT}/acceptance_summary.png)','']
    (ROOT/'ACCEPTANCE_20260926.md').write_text('\n'.join(lines))
    print('Independent report and scientific figures written',flush=True)

if __name__=='__main__':main()
