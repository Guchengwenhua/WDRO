"""Publish the acceptance receipt and refresh human handoff; frozen experiment is read-only."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    audit=read(OUT/'audit.json');stats=read(OUT/'statistics_independent.json');probes=read(OUT/'probes.json')
    assert audit['status']=='PASS_CORE_WITH_DEGENERATE_WEIGHT_DECAY_ABLATION'
    assert probes['status']=='PASS' and all(stats[k]['support'] for k in ('budget_mean','budget_worst'))
    inputs=read(OUT/'input_hashes.json')
    # The full 1,224-input hash verification is recorded by verify.py. Recheck compact root seals here.
    for name in ('protocol.json','artifacts/selection.json','artifacts/report/complete.json'):
        p=ROOT/name;assert sha(p)==inputs[str(p)]
    promotion_hash=sha(ROOT/'artifacts/promotion.json')
    assert promotion_hash==read(ROOT/'artifacts/selection.json')['promotion_sha256']
    assert str(ROOT/'HANDOFF.md') not in inputs
    snapshot=OUT/'HANDOFF_at_launch.md';delivery=read(ROOT/'artifacts/operations/delivery.json')
    if not snapshot.exists():
        assert sha(ROOT/'HANDOFF.md')==delivery['handoff_sha256'];snapshot.write_bytes((ROOT/'HANDOFF.md').read_bytes())
    assert sha(snapshot)==delivery['handoff_sha256']
    handoff=f'''# 扩量实验交接：已完成并独立验收

2026-09-23 23:56:55 UTC 正式实验完成；2026-09-26 独立验收完成。

结论：核心工程、预算、数值和预注册统计检查通过；WD 权重衰减消融因 FP32 舍入退化为无效比较。验收状态为 `{audit['status']}`。

58 个训练任务、71 个测试任务全部完成；新增 712,704 次更新、50,331,648 条采样。独立检查 1,224 个文件哈希，复算 4,800,512 条保存线长，最大绝对误差为 0。队列进程已退出，锁已释放，无待运行任务，无需重新启动 tmux。

E50 训练由 12,288 延长到 49,152 步，五个种子的平均线长下降 2.428%，最差组下降 2.622%，达到预注册探索性门槛。训练步数和在线新样本数共同增加；结论不能单独归因于预训练、后训练或更新次数。其他调参和容量方案未通过晋级门槛；batch256 未显示同样本数优势。WD 不能用于判断权重衰减是否有效。

详细报告：[ACCEPTANCE_20260926.md]({ROOT}/ACCEPTANCE_20260926.md)。

独立证据与交付清单：[artifacts/acceptance_20260926/complete.json]({OUT}/complete.json)。

原启动交接保存在 [HANDOFF_at_launch.md]({snapshot})，与原 `artifacts/operations/delivery.json` 的 handoff_sha256 完全一致。该旧 delivery 是 9 月 22 日的启动凭证，当前完成状态请看本文件、独立验收报告和 `artifacts/lifecycle.json`。原协议、源码、模型、数据与自动报告没有修改。
'''
    (ROOT/'HANDOFF.md').write_text(handoff)
    files=[p for p in OUT.iterdir() if p.is_file() and p.name not in ('complete.json','finalize.log') and not p.name.endswith('.partial')]
    files.extend([ROOT/'ACCEPTANCE_20260926.md',ROOT/'HANDOFF.md'])
    receipt=dict(status='complete',acceptance=audit['status'],finished=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_report_complete_sha256=inputs[str(ROOT/'artifacts/report/complete.json')],
        independently_verified_promotion_sha256=promotion_hash,
        original_handoff_sha256=sha(snapshot),input_hash_manifest_sha256=sha(OUT/'input_hashes.json'),
        training_complete=True,numerical_audit_pass=True,frozen_primary_exploratory_support=True,
        weight_decay_ablation_valid=False,no_new_training=True,no_new_gpu_inference=True,
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    temp=OUT/'complete.json.partial';temp.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');os.replace(temp,OUT/'complete.json')
    for path,h in receipt['artifacts'].items():assert sha(ROOT/path)==h
    print('Acceptance receipt PASS:',len(files),'artifacts; archived original handoff verified.')
if __name__=='__main__':main()
