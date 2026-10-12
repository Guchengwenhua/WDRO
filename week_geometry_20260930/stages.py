import argparse
from week_common import *
def main(args):
 ART.mkdir(exist_ok=True)
 if args.phase=='freeze':
  f=freeze();atomic(ART/'freeze_complete.json',dict(status='complete',choices=f['choices']));return
 if args.phase=='train_gate':
  f=metadata();s=read(ART/'screen_d10_dev/summary.json')
  keys=['soft_dual/entropy','soft_dual/random']
  rows=[s['statistics'][key] for key in keys]
  positive=f['choices']['10']['extension_dev_positive'] and all(r['mean_delta']>0 and r['ci95'][0]>0 and r['holm_p_six_contrasts']<.05 for r in rows)
  atomic(ART/'gate.json',dict(allow=bool(positive),reason='d10 development soft-dual objective improvement vs both entropy and random has positive95% paired CI and Holm p<.05; extension dev positive' if positive else 'development positive-evidence rule failed; geometry training is stopped, no test result consulted',
                            metric=f['choices']['10']['metric'],criteria={key:s['statistics'][key] for key in keys},test_used=False))
  atomic(ART/'gate_complete.json',dict(status='complete'));return
 if args.phase=='report':
  from training_report import render_training
  training_rows,training_contrasts=render_training()
  lines=['# 七天几何研究：实际数据','']
  for path in sorted(ART.glob('screen_d*/summary.json')):
   s=read(path);lines += [f"## d{s['degree']} {s['split']}，指标{s['metric']}，{s['centers']}中心×{s['model_count']}模型",'',
    '| 范围/对照 | 几何减对照的目标差 | 配对95%区间 | Holm p |','|---|---:|---|---:|']
   for key,v in s['statistics'].items():lines.append(f"| {key} | {v['mean_delta']:.6f} | {v['ci95']} | {v['holm_p_six_contrasts']:.4f} |")
   lines+=['','soft_dual是固定lambda=1的真实相对误差减Hungarian运输成本；hard_radius是最终归一化坐标c<=rho内最大误差，二者不能混称。正差表示筛选到更困难的候选。','']
  if (ART/'gate.json').exists():
   gate=read(ART/'gate.json');lines+=[f"训练gate：allow={gate['allow']}；{gate['reason']}",'']
  if (ART/'budget_gate.json').exists():
   budget=read(ART/'budget_gate.json');lines += [f"训练成本gate：allow={budget['allow']}，实际前缀推算整矩阵剩余秒数={budget['projected_remaining_matrix_seconds']}，分支预算秒数={budget['branch_allowance_seconds']}；{budget['reason']}",'']
  if (ART/'train/geometry_s101/segment_0100.json').exists():
   prefix=read(ART/'train/geometry_s101/segment_0100.json');lines += [f"正式2000步训练的可恢复前缀已到{prefix['completed_updates']}步；不是完整训练结果。",'']
  training=[]
  for p in sorted((ART/'train').glob('*/complete.json')):
   cfg=p.parent/'test_latest.json'
   if cfg.exists():training.append((p.parent.name,read(cfg)))
  lines+=['## 已完成训练','']
  for name,v in training:lines.append(f"- {name}: latest2000主结果测试平均误差={v['overall_mean']:.6f}，worstOOD={v['worst_ood_mean']:.6f}；checkpoint update={v['checkpoint_update']}。")
  for name,v in training:
   bp=ART/'train'/name/'test_best.json'
   if bp.exists():
    b=read(bp);lines.append(f"- {name} validation-best次结果：update={b['checkpoint_update']}，平均误差={b['overall_mean']:.6f}，worstOOD={b['worst_ood_mean']:.6f}。")
  if not training:lines.append('尚无完整训练结果，不能推断训练收益。')
  for baseline,contrast in training_contrasts.items():lines.append(f"- 主比较geometry-{baseline}，matched seeds={contrast['seeds']}，macroOOD差={contrast['mean_ood_delta']:.6f}，三重比较校正98.333%区间={contrast['bonferroni_ci']}。")
  lines.append(f'完整训练{len(training)}/12；未完成/负gate任务不计作成功。')
  (ROOT/'REPORT.md').write_text('\n'.join(lines));atomic(ART/'report_complete.json',dict(status='report_generated',branch_status='complete' if len(training)==12 else 'partial_or_gated',screen_cells=len(list(ART.glob('screen_d*/summary.json'))),completed_train_runs=len(training)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','train_gate','report'],required=True);main(p.parse_args())
