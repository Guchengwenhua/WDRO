"""Verify the tighter allocator cap on actual continuation training, not a dummy tensor."""
import json,os,shutil,subprocess,time
from pathlib import Path
import engineering as eng
import ops as o
ROOT=Path(__file__).resolve().parent
def main():
    dest=ROOT/'artifacts/allocator_probe';accepted=dest/'acceptance.json'
    if accepted.exists():
        if json.loads(accepted.read_text())['status']!='PASS':raise RuntimeError('Previous allocator probe failed')
        return
    if not dest.exists():
        dest.mkdir(parents=True);p=json.loads((ROOT/'protocol.json').read_text())
        task=next(t for t in p['tasks'] if t['id']=='post_pre2048_sequential_ppo_s301')
        task.update(end=2051,milestones=[2051]);p['tasks']=[task];p['experiment']='ALLOCATOR_ENGINEERING_ONLY'
        p['checkpoint_every']=1;p['resources']['wait_for_origin_training']=False
        (dest/'protocol.json').write_text(json.dumps(p,indent=2))
        for n in eng.FILES:shutil.copyfile(ROOT/n,dest/n)
        (dest/'run_env.sh').write_text((ROOT/'run_env.sh').read_text().replace(str(ROOT),str(dest)))
        src=task['parents']['resume']['run'];shutil.copytree(ROOT/'lineage'/src,dest/'lineage'/src)
    p=json.loads((dest/'protocol.json').read_text());task=p['tasks'][0];history=[]
    while True:
        snap=o.snapshot();history=(history+[snap])[-3:]
        other={}
        for hb in (ROOT/'artifacts/gpu_pilot/artifacts/runs').glob('*/heartbeat.json'):
            h=json.loads(hb.read_text())
            if o.same_identity(o.ident(h['pid']),h):other[hb.parent.name]={'gpu':h['physical_gpu_index']}
        cap,cards,_=o.admission(snap,history,other,dict(p['resources'],max_admission_util=65))
        if len(history)>=3 and cap and cards:break
        time.sleep(5)
    gpu=cards[0];o.atomic(dest/'preflight.json',dict(snapshot=snap,history=history,device=gpu,memory_fraction=p['resources']['memory_fraction']))
    code="import runner,torch,json; code=runner.main(); result={'exit':code,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'status':'PASS' if code==75 else 'FAIL'}; open('acceptance.json','w').write(json.dumps(result,indent=2)); assert code==75"
    cmd=['nice','-n','10','bash',str(dest/'run_env.sh'),'-c',code,'--task-id',task['id'],'--stop-after','2050',
        '--physical-gpu-index',str(gpu['index']),'--gpu-uuid',gpu['uuid'],'--duty-fraction','.5']
    with (dest/'console.log').open('a') as f:
        result=subprocess.run(cmd,cwd=dest,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index'])),stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Allocator continuation probe failed')
    print(accepted.read_text())
if __name__=='__main__':main()
