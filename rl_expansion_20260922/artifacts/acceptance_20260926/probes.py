"""Bounded sampler replay and FP32 weight-decay implementation probes, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import geometry as G  # Only frozen geometric coordinate sampler; no solver/model inference.
GROUPS=('uniform','mix_gauss','edge_bias','grid_cluster')
def read(p):return json.loads(p.read_text())
def base(seed,step):return np.stack([G.make_data(16,30,g,f'budget-v1/train/{step}/{g}',seed,True) for g in GROUPS])
def main():
    probes=[]
    for seed in (101,102,103):
        for arm,steps in [('E50',(12289,16384,20480)),('B256',(12289,13312,14336,20480))]:
            records={r['step']:r for line in (ROOT/f'artifacts/runs/screen_{arm}_s{seed}/progress.jsonl').open() if (r:=json.loads(line))['type']!='validation' and r['step'] in steps}
            chunks=1 if arm=='E50' else 4
            for step in steps:
                first=step if chunks==1 else 12288+(step-12289)*4+1
                x=np.concatenate([base(seed,s) for s in range(first,first+chunks)],axis=1).reshape(64*chunks,30,2)
                h=hashlib.sha256(x.tobytes()).hexdigest();assert h==records[step]['data_info']['unaugmented_sha256']
                probes.append(dict(seed=seed,arm=arm,step=step,first_base_step=first,last_base_step=first+chunks-1,sha256=h))
    # Check evaluation receipts name the exact globally frozen model, including degree transfer.
    receipt_count=0
    for p in (ROOT/'artifacts/tasks').glob('eval_*.json'):
        t=read(p)
        for label,m in t['models'].items():
            r=read(ROOT/t['outdir']/(label+'.json'))
            assert r['model_sha256']==m['sha256'];receipt_count+=1
    import torch
    before=torch.linspace(-3,3,1000,dtype=torch.float32);after=before.clone().mul_(1-5e-5*1e-4)
    assert torch.equal(before,after)
    p1=torch.nn.Parameter(before.clone());p2=torch.nn.Parameter(before.clone())
    a=torch.optim.Adam([p1],lr=5e-5);w=torch.optim.AdamW([p2],lr=5e-5,weight_decay=1e-4)
    for step in range(10):
        grad=torch.cos(before+step*.1);p1.grad=grad.clone();p2.grad=grad.clone();a.step();w.step()
    assert torch.equal(p1,p2)
    result=dict(status='PASS',sampler_probes=probes,sample_matched_base_step_range=[12289,20480],
        sample_matched_new_samples_per_model=8192*64,update_matched_B256_new_samples=8192*256,
        update_matched_E50_new_samples=8192*64,verified_eval_model_receipts=receipt_count,
        weight_decay=dict(lr=5e-5,weight_decay=1e-4,mathematical_factor=1-5e-5*1e-4,
            float32_factor=float(np.float32(1-5e-5*1e-4)),multiplication_bitwise_noop=True,
            ten_adam_adamw_updates_equal=True,torch_version=torch.__version__),
        scope='Sampler probes use the frozen coordinate generator; audit is not a full training replay.')
    (OUT/'probes.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(probes),'sampler probes;',receipt_count,'model receipts; FP32 decay is no-op',flush=True)
if __name__=='__main__':main()
