import json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
import geometry as g
from models.actor_critic import Actor
ART=ROOT/'artifacts';OUT=ART/'review_20260926'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.cuda.set_per_process_memory_fraction(.12);g.install_oracle_guard()
rows=[];predictions=0
for t in json.loads((ART/'operations/manifest.json').read_text())['training']:
    if t['method']!='excess':continue
    p=ART/'runs'/t['id'];actor=Actor(t['degree'],torch.device('cuda:0'))
    actor.load_state_dict(torch.load(ART/'runs'/t['parent']/'best.pt',map_location='cpu',weights_only=False)['actor']);actor.eval()
    audit=np.load(p/'candidate_audit.npz');z=audit['candidates'];x=z.reshape(-1,t['degree'],2);lens=[]
    with torch.no_grad():
        for k in range(0,len(x),256):
            v=x[k:k+256];top,_=actor(v,True);lens.append(g.lengths(v,top.cpu().numpy()));predictions+=len(v)
    teacher=np.concatenate(lens).reshape(128,16);ell=audit['length']/teacher-1
    idx=((t['updates']-1)//256)*256;rec=json.loads((p/'progress.jsonl').read_text().splitlines()[idx]);cost=audit['cost']
    a=(ell-rec['lambda_']*cost)/rec['tau'];a-=a.max(1,keepdims=True);pr=np.exp(a);pr/=pr.sum(1,keepdims=True)
    error=float(np.abs(pr-audit['probability']).max());ec=float((pr*cost).sum(1).mean())
    primal=float((pr*ell).sum(1).mean()-rec['tau']*(pr*np.log(np.maximum(pr,1e-300)*16)).sum(1).mean())
    row=dict(task=t['id'],teacher_min=float(teacher.min()),max_probability_error=error,expected_cost=ec,primal_error=abs(primal-rec['primal']),pass_=error<1e-11 and ec<=.04+1e-10 and abs(primal-rec['primal'])<1e-11)
    rows.append(row)
result=dict(status='PASS' if all(v['pass_'] for v in rows) else 'FAIL',rows=rows,predictions=predictions,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),time=time.time())
(OUT/'teacher_replay.json').write_text(json.dumps(result,indent=2))
print(result['status'],predictions,flush=True)
