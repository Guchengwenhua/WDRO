"""Render only actual scientific results; independent from operational handoff."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
def render():
    art=ROOT/'artifacts';rows=[]
    def fmt(x):
        return '未定义' if x is None else f'{x:.4f}'
    lines=['# REST 条件几何不确定性：探索性实验结果','',
           '固定 d30、FP32 greedy、温度 1；每个模型/分布分别分析。指标观测已解码第一端点条件下的第二端点与方向概率，不是完整联合点对熵。真实误差采用 precision_v2 既有参考。所有区间是按实例或候选块重采样的 95% 百分位区间。','']
    for p in sorted(art.glob('M*/D*/summary.json')):
        s=json.loads(p.read_text());rows.append(s)
        lines += [f"## {s['model']} / {s['dataset']}，n={s['n']}",'',
                  f"平均真实相对误差 {100*s['mean_regret']:.3f}%，P95 {100*s['p95_regret']:.3f}%。",'',
                  '| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |',
                  '|---|---:|---:|---:|---:|']
        for name,v in s['metrics'].items():
            lines.append(f"| {name} | {fmt(v['pearson'])} | {fmt(v['spearman'])} | {v['precision_top10']:.3f} | {v['high_error_prevalence']:.3f} |")
        lines+=['','maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。','',
                '| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |',
                '|---:|---|---:|---:|---|']
        for budget,cell in s['hard_selection'].items():
            for name,v in cell['methods'].items():
                lo,hi=v['delta_ci95']
                lines.append(f"| {budget} | {name} | {100*v['mean_selected_true_regret']:.3f} | {100*v['delta_vs_random']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] |")
        lines+=['','这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。','']
    lines+=['## 数据支持的范围','',f'已生成 {len(rows)} 个模型×分布结果。未开展训练效果实验或新扰动攻击；这里不能推断这两类效果。高熵是否对应高误差应逐组以相关系数和筛选结果判断，不能由指标定义保证。多指标探索未做多重比较校正，历史测试集不是新独立确认集。','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':render()
