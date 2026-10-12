"""Publish Stage 6 DONE in the authoritative root only after new delivery passes."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import utc_now
from run_precision_training_queue import source_gate
ORIGINAL=Path('/remote-home/Guilin/DRO/Codex')

def main():
    source_gate()
    cp=ROOT/'manifests/stage6_completion_v1.json';v=ROOT/'manifests/stage6_delivery_verification_v1.json'
    complete=json.loads(cp.read_text());verified=json.loads(v.read_text())
    assert complete['status']==verified['status']=='pass' and verified['completion']['sha256']==file_sha256(cp)
    freeze=ROOT/'manifests/stage6_models_frozen_v1.json';frozen=json.loads(freeze.read_text())
    assert len(frozen['models'])==75 and frozen['test_evaluated_at_freeze'] is False
    cells=list((EXP/'logs/stage6_test').glob('*/*/result.json'));assert len(cells)==300
    assert all(json.loads(p.read_text())['status']=='complete' and json.loads(p.read_text())['completed_nets']==10000 for p in cells)
    stages=[ROOT/'manifests'/n for n in ('stage2_completion_20260905.json','stage3_delivery_verification_v1.json',
        'stage4_delivery_verification_v1.json','stage5_delivery_verification_v1.json')]
    assert all(json.loads(p.read_text())['status']=='pass' for p in stages)
    report=CODEX/'阶段6_多种子合成主实验验收报告.md';assert report.exists()
    inputs=[*stages,cp,v,freeze,report,ROOT/'configs/stage5_selected_wdro_v1.json',
        ROOT/'manifests/precision_v2_training_source_v1.json',EXP/'logs/reference_cache_v2/completion.json',
        ORIGINAL/'experiments/logs/precision_v2/reference_verification/acceptance.json']
    record=dict(status='pass',at_utc=utc_now(),stage=6,reference_correction='precision_v2',models=75,test_cells=300,test_predictions=3000000,
        stage7_started=False,original_v1_preserved=True,files=[dict(logical_path=str(p),resolved_path=str(p.resolve()),sha256=file_sha256(p)) for p in inputs])
    target=ORIGINAL/'wdro_rest_min/manifests/precision_v2_replay_completion.json';atomic_write_json(target,record,mode=0o444)
    path=ORIGINAL/'01_WDRO-REST_项目进度.md';s=path.read_text();prefix='revisions/reference_precision_v2/Codex'
    values={'更新时间':utc_now(),'总体阶段':'阶段6 `DONE`（precision_v2纠错、受影响轨迹重放与合成主测试全部验收）',
        '实验状态':'纠错版75模型、300格/300万预测、主表/degree附表/paired CI和独立交付通过；原v1仅作历史证据',
        '批量实验阻塞':'阶段6无剩余阻塞；本次授权范围已完成','下一动作':'本次止于阶段6；阶段7 ISPD模型评价保持未启动'}
    for key,value in values.items():s=re.sub(r'\| '+re.escape(key)+r' \| .*? \|',f'| {key} | {value} |',s,count=1)
    s=re.sub(r'\| 6\. 多种子合成主实验 \| .*?\n',f'| 6. 多种子合成主实验 | `DONE` | precision_v2 75模型、300格300万预测及paired CI完整验收；见wdro_rest_min/manifests/precision_v2_replay_completion.json |\n',s,count=1)
    section='### 阶段 6 多种子合成主实验（本次授权）\n\n'
    section+=f'- `DONE` precision_v2参考精度修复、受影响阶段1–5复核/同设置重跑及新validation冻结。\n- `DONE` 6.1–6.4：新75模型冻结、300格/300万预测、完整预算/产物审计、主表/degree附表/paired CI全部通过。\n- 证据：`wdro_rest_min/manifests/precision_v2_replay_completion.json`、`{prefix}/阶段6_多种子合成主实验验收报告.md`。原v1失败/修复历史保留在下方，旧13格结果未混入新主表。阶段7–9未启动。\n\n'
    s=re.sub(r'### 阶段 6 多种子合成主实验（本次授权）\n.*?(?=## 4\. 阶段状态)',section,s,count=1,flags=re.S)
    s=s.replace('## 8. 变更记录\n\n',f'## 8. 变更记录\n\n### {utc_now()} precision_v2阶段6最终验收\n\n- `DONE` 75模型、300格/300万test、主表/degree附表与paired CI通过独立交付。证据`wdro_rest_min/manifests/precision_v2_replay_completion.json`和`{prefix}/阶段6_多种子合成主实验验收报告.md`。原研究设置与原数据文件未变；修复reference后按原validation规则重放，不按test改配置。阶段7未启动。\n\n',1)
    temp=path.with_suffix('.md.partial');temp.write_text(s);temp.replace(path)
    print(json.dumps(dict(status='pass',stage6='DONE',manifest=str(target))))
if __name__=='__main__':main()
