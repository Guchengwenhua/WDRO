"""Publish evidence-backed stage 6 completion without starting stage 7."""
import csv,json,sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.rest import file_sha256
from src.job_control import utc_now
from src.io_utils import atomic_write_json

def main():
    cp=ROOT/'manifests/stage6_completion_v1.json';c=json.loads(cp.read_text())
    v=json.loads((ROOT/'manifests/stage6_delivery_verification_v1.json').read_text())
    assert c['status']==v['status']=='pass' and v['completion']['sha256']==file_sha256(cp)
    rows=list(csv.DictReader((EXP/'tables/stage6_main.csv').open()));worst=list(csv.DictReader((EXP/'tables/stage6_worst_group.csv').open()))
    report=['# 阶段6 多种子合成主实验验收报告','',f'验收时间：{utc_now()}。阶段6完成，阶段7–9未启动。','',
        '75个模型齐全：复用纠错重跑的15个ERM、45个基线和阶段5按原规则选中的d30/s101配置；新增14个WDRO各2000 updates、batch256。新增训练的全候选/cost/对偶/真实Hard Mining预算配对审计通过，所有模型在test前按validation best冻结。','',
        '20个共享合成test文件各10000 nets；75模型共300个评价格、300万逐net预测，统一T=1 greedy、batch256及同型号硬件。结果、net ID、模型及数据/reference hash完整保留。','',
        '| 方法 | uniform mean | mix_gauss mean | edge_bias mean | grid_cluster mean | OOD worst-group mean |','|---|---:|---:|---:|---:|---:|']
    for method in ['erm','mixed_erm','hard_mining','group_dro','wdro_rest']:
        group={r['distribution']:float(r['mean']) for r in rows if r['method']==method}
        w=next(float(r['value']) for r in worst if r['method']==method and r['degree']=='macro5')
        report.append('| '+method+' | '+' | '.join(f'{group[g]:.8f}' for g in ['uniform','mix_gauss','edge_bias','grid_cluster'])+f' | {w:.8f} |')
    report+=['','mean按degree和三个训练seed等权平均。P95为逐degree/seed P95的平均，并非混池P95；完整P95与95%CI见CSV。worst-group为三个OOD组mean的最大值。',
        'CI使用10000次paired per-net bootstrap，按degree/distribution分层，所有方法和seed使用相同net抽样；条件于现有三个训练seed，不能解读为广泛训练seed总体的不确定性。多重比较未校正。',
        '本报告只建立合成主实验的完整结果；独立强攻击与ISPD尚未执行，不提前作最终鲁棒性或外部泛化结论。没有因test结果调参或删除不利结果。','',
        '证据：`wdro_rest_min/manifests/stage6_models_frozen_v1.json`、`stage6_completion_v1.json`、`stage6_delivery_verification_v1.json`；`experiments/tables/stage6_{main,per_degree,per_seed,paired_ci,worst_group}.csv`；逐net预测与paired差值均通过完整独立回读。',
        '训练有效轨迹calls与重试成本分开；训练segment/失败日志和调度历史保留。test每net各一次Actor/REST，reference来自阶段1缓存；独立产物审计另计300万REST calls。','',
        f'恢复/重复验收：`bash {ROOT}/scripts/stage3_env.sh {sys.executable} {ROOT}/scripts/launch_stage6.py`；已完成任务不会重复训练。完整回读可运行同wrapper的`scripts/verify_stage6_delivery.py`。']
    path=CODEX/'阶段6_多种子合成主实验验收报告.md';temp=path.with_suffix('.md.partial');temp.write_text('\n'.join(report)+'\n');temp.replace(path)
    progress=CODEX/'01_WDRO-REST_项目进度.md';s=progress.read_text();now=utc_now()
    for key,value in {'更新时间':now,'总体阶段':'阶段0–6 `DONE`；阶段7未启动','实验状态':'阶段6的75模型、300测试格、paired CI与完整交付验收通过','批量实验阻塞':'阶段6无剩余阻塞；后续需按授权边界执行','下一动作':'本次止于阶段6；阶段7 ISPD尚未启动'}.items():
        s=re.sub(r'\| '+re.escape(key)+r' \| .*? \|',f'| {key} | {value} |',s,count=1)
    s=re.sub(r'\| 6\. 多种子合成主实验 \| .*?\n','| 6. 多种子合成主实验 | `DONE` | 75模型冻结、300格300万逐net test、完整预算/产物审计、主表/附表/paired CI通过；见阶段6验收报告与completion/delivery索引 |\n',s,count=1)
    s=re.sub(r'- `(DONE|DOING|TODO)` 6\.4：.*?\n','- `DONE` 6.4：统计、主表/附表/paired CI和完整交付复核通过；见阶段6验收报告与completion/delivery manifests。\n',s,count=1)
    s=s.replace('## 8. 变更记录\n\n',f'## 8. 变更记录\n\n### {now}\n\n- 阶段6完整验收通过：75模型、300格/300万test预测、14个新增WDRO全候选/cost/对偶/HM预算审计、主表/degree附表及paired95%CI。证据 `阶段6_多种子合成主实验验收报告.md` 与 `wdro_rest_min/manifests/stage6_completion_v1.json`、`stage6_delivery_verification_v1.json`。\n- 原protocol/data-contract和阶段5冻结配置未变；没有test选模/回调，所有结果保留；本次止于阶段6。\n\n',1)
    temp=progress.with_suffix('.md.partial');temp.write_text(s);temp.replace(progress)
    atomic_write_json(EXP/'logs/stage6_pipeline/final_report.json',dict(status='pass',at_utc=utc_now(),report=str(path),sha256=file_sha256(path)))
if __name__=='__main__':main()
