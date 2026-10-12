"""Create the explicitly authorized, separately reported large-batch experiment."""
import json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent
B=ROOT/'accelerated_bank_20261006';G=ROOT/'accelerated_geometry_20261006'
PY='/remote-home/Guilin/miniconda3/envs/py311/bin/python'
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
for dst,src in [(B,ROOT/'week_bank_20260930'),(G,ROOT/'week_geometry_20260930')]:
 dst.mkdir(exist_ok=True)
 for f in src.glob('*.py'):shutil.copy2(f,dst/f.name)
 for f in src.glob('inputs*.json'):shutil.copy2(f,dst/f.name)
 target=Path('/attached/remote-home1/Guilin/DRO/Codex')/dst.name/'artifacts';target.mkdir(parents=True,exist_ok=True)
 if not (dst/'artifacts').exists():(dst/'artifacts').symlink_to(target)
target=Path('/attached/remote-home1/Guilin/DRO/Codex')/P.name/'artifacts';target.mkdir(parents=True,exist_ok=True)
if not (P/'artifacts').exists():(P/'artifacts').symlink_to(target)
config=dict(protocol_id='accelerated_largebatch_20261006',training_batch=4096,short_updates=125,long_updates=625,
 short_sample_updates=512000,long_sample_updates=2560000,score_microbatch=1024,teacher_microbatch=1024,
 validation_batch=1024,test_batch=1024,screen_batch=1024,validation_interval=6,refresh_interval=3,
 learning_rate=5e-5,optimizer='original Adam; no unvalidated linear LR scaling',seeds=[101,102,103],
 comparisons='new matched Geo/K3/lagged3/qualifiedbest3; old batch256 results historical only',
 limitation='same sample count, different update count, batch composition and refresh cadence; not numerical equivalence',
 geometry_prefix_updates=7,geometry_training_gate='unchanged development-only positive evidence',
 authorization='User explicitly allowed changed training batch with separate synchronized controls and reporting')
write(P/'protocol.json',config)
helper='''import json
from pathlib import Path
CONFIG=json.loads((Path(__file__).resolve().parent.parent/'accelerated_research_20261006/protocol.json').read_text())
def configure(protocol,trainer,baseline,updates):
 protocol['protocol_id']=CONFIG['protocol_id']
 protocol['training']['batch_size']=CONFIG['training_batch']
 protocol['training']['policy_updates']=updates
 trainer['validation_batch_size']=CONFIG['validation_batch']
 trainer['validation_interval_updates']=CONFIG['validation_interval']
 trainer['checkpoint_interval_updates']=CONFIG['validation_interval']
 trainer['structured_log_interval_updates']=1
 baseline['search']['score_microbatch']=CONFIG['score_microbatch']
'''
for dst in [B,G]:
 write(dst/'protocol.json',config);(dst/'accelerated_config.py').write_text(helper)
 for name in ['adapters.py','faithful_trainer.py','run_week.py' if dst==B else 'run_train.py']:
  p=dst/name;s=p.read_text()
  if name=='adapters.py':
   s=s.replace('range(0,len(pins),256)','range(0,len(pins),1024)').replace('pins[i:i+256]','pins[i:i+1024]')
   s=s.replace('update%50','update%3')
  elif name=='faithful_trainer.py':
   s=s.replace('expected_updates < 2000','expected_updates < 125').replace('expected_updates != 2000','expected_updates != 125')
   s=s.replace('at least2000 updates','at least125 accelerated updates')
  else:
   s=s.replace('from adapters import *','from adapters import *\nfrom accelerated_config import configure,CONFIG')
   s=s.replace('torch.cuda.set_per_process_memory_fraction(.1,device)','torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RESEARCH_CUDA_FRACTION","0.55")),device)')
   s=s.replace("baseline=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json')", "baseline=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json');configure(protocol,trainer,baseline,"+('args.updates' if dst==B else '125')+")")
   s=s.replace("range(0,len(data['pins']),256)","range(0,len(data['pins']),1024)").replace("data['pins'][i:i+256]","data['pins'][i:i+1024]")
   s=s.replace('original batch256','accelerated batch1024').replace('memoryfraction.1','memoryfraction0.55')
   if dst==B:s=s.replace('choices=[2000,10000]','choices=[125,625]')
   else:
    s=s.replace('2000','125')
  p.write_text(s)
# The geometric screen is a new invocation with a recorded common inference batch.
p=G/'screen.py';s=p.read_text().replace("Runner(f['models'][mid],256)","Runner(f['models'][mid],1024)").replace('range(0,len(flat),256)','range(0,len(flat),1024)').replace('a+256','a+1024');p.write_text(s)
p=G/'week_common.py';s=p.read_text().replace('torch.cuda.set_per_process_memory_fraction(.1)','torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RESEARCH_CUDA_FRACTION","0.55")))').replace('batch=256','batch=1024').replace("PROJECT/'holiday_research_20260930/policy.json'","PROJECT/'accelerated_research_20261006/policy.json'");p.write_text(s)
for name in ['training_report.py','stages.py']:
 p=G/name;s=p.read_text().replace('latest2000','latest125').replace('2000步','125步').replace('segment_0100.json','segment_0007.json').replace('正式2000','正式125').replace('七天几何研究','大batch几何研究');p.write_text(s)
# The cost prefix is retained for a measured ETA, not a wall-clock stop.
(G/'budget_gate.py').write_text('''from week_common import *
def main():
 dev=read(ART/'gate.json');p=read(ART/'train/geometry_s101/segment_0007.json')
 evidence=p['completed_updates']>=7
 rate=p['wall_seconds_this_process']/7
 atomic(ART/'budget_gate.json',dict(allow=bool(dev.get('allow') and evidence),status='gate_evaluated',test_used=False,reason='development gate and seven official accelerated updates; no calendar cutoff',prefix_completed_updates=p['completed_updates'],projected_remaining_matrix_seconds=1493*rate,branch_allowance_seconds=None,estimate_scope='conservative serial work estimate; six GPUs reduce wall time subject to DAG'))
if __name__=='__main__':main()
''')
# Local matched controls: none of the old batch256 results is a new baseline.
p=B/'gate.py';s=p.read_text();start=s.index('def dev(');end=s.index('def judge(',start)
s=s[:start]+'''def dev(arm,seed,stage):
 horizon='125' if stage=='matched' else '625'
 p=HERE/'artifacts'/f'd10_{arm}{horizon}_s{seed}'/'dev.json'
 v=read(p)['last_validation']
 return {'score':v['selection']['worst_group_mean'],'nominal':v['selection']['nominal_mean']}
'''+s[end:];p.write_text(s)
p=B/'report.py';s=p.read_text();start=s.index(' # Read external matched Geo');end=s.index(' costs=',start);s=s[:start]+s[end:];s=s.replace('for updates in [2000,10000]','for updates in [125,625]').replace('# Weekly bank experiment results','# Accelerated batch4096 bank experiment results');p.write_text(s)
jobs=[]
def add(jid,cwd,script,args,marker,kind,deps=(),gate=None,priority=20,always=False):
 j=dict(id=jid,cwd=str(cwd),argv=[PY,'-u','-B',str(cwd/script),*map(str,args)],complete=str(marker),kind=kind,depends_on=list(deps),priority=priority,estimated_seconds=3600,max_wall_seconds=172800)
 if gate:j['gate_file']=str(gate)
 if always:j['always_run']=True
 jobs.append(j);return jid
arms=['geo','k3','lagged3','qualifiedbest3'];base=[];long=[];evals=[]
def bankrun(degree,arm,seed,updates,deps,gate=None,parent=None):
 jid=f'd{degree}_{arm}{updates}_s{seed}';args=['--job-id',jid,'--arm',arm,'--seed',seed,'--degree',degree,'--updates',updates]
 if parent:args+=['--import-from',str(B/'artifacts'/parent)]
 if arm=='selected':args+=['--selection-file',str(B/'gates/d30.json')]
 add(jid,B,'run_week.py',args,B/'artifacts'/jid/'complete.json','train',deps,gate)
 eid=add(jid+'_test',B,'run_week.py',args+['--evaluate-only'],B/'artifacts'/jid/'evaluation_complete.json','evaluate',[jid],gate,priority=40)
 evals.append(eid);return jid
for seed in [101,102,103]:
 for arm in arms:base.append(bankrun(10,arm,seed,125,[]))
add('bank_gate_matched',B,'gate.py',['--stage','matched'],B/'gates/matched_complete.json','gate',base,priority=5)
for seed in [101,102,103]:
 for arm in arms:long.append(bankrun(10,arm,seed,625,['bank_gate_matched'],B/f'gates/extend_{arm}.json',f'd10_{arm}125_s{seed}'))
add('bank_gate_long',B,'gate.py',['--stage','long'],B/'gates/long_complete.json','gate',long,priority=5,always=True)
d30=[]
for seed in [101,102,103]:
 for arm in ['geo','selected']:d30.append(bankrun(30,arm,seed,125,['bank_gate_long'],B/'gates/d30.json'))
add('bank_report',B,'report.py',[],B/'report_complete.json','summarize',base+long+d30+evals+['bank_gate_long'],priority=100,always=True)
bankjobs=list(jobs)
add('geometry_freeze',G,'stages.py',['--phase','freeze'],G/'artifacts/freeze_complete.json','gate',priority=0)
for split in ['dev','test']:
 for degree in [10,30]:
  deps=['geometry_freeze'] if split=='dev' else [f'geometry_dev_d{degree}','geometry_train_gate']
  add(f'geometry_{split}_d{degree}',G,'screen.py',['--degree',degree,'--split',split],G/f'artifacts/screen_d{degree}_{split}/complete.json','evaluate',deps,priority=10)
add('geometry_train_gate',G,'stages.py',['--phase','train_gate'],G/'artifacts/gate_complete.json','gate',['geometry_dev_d10'],priority=0)
devgate=G/'artifacts/gate.json';costgate=G/'artifacts/budget_gate.json'
add('geometry_prefix7',G,'run_train.py',['--arm','geometry','--seed',101,'--stop-after',7],G/'artifacts/train/geometry_s101/segment_0007.json','train',['geometry_train_gate'],devgate,priority=10)
add('geometry_cost_gate',G,'budget_gate.py',[],costgate,'gate',['geometry_prefix7'],devgate,priority=0)
for seed in [101,102,103]:
 for arm in ['original','random','entropy','geometry']:
  add(f'geometry_train_{arm}_s{seed}',G,'run_train.py',['--arm',arm,'--seed',seed],G/f'artifacts/train/{arm}_s{seed}/complete.json','train',['geometry_cost_gate'],costgate,priority=15)
geojobs=jobs[len(bankjobs):]
add('geometry_report',G,'stages.py',['--phase','report'],G/'artifacts/report_complete.json','summarize',[j['id'] for j in geojobs],priority=100,always=True)
write(B/'jobs.json',{'jobs':bankjobs});write(G/'jobs.json',{'jobs':jobs[len(bankjobs):]});write(P/'jobs.json',{'jobs':jobs})
write(P/'policy.json',dict(run_until_complete=True,deadline_utc=None,training_cutoff_utc=None,gpus=[2,3,4,5,6,7],max_gpu_jobs=6,max_cpu_jobs=1,
 gpu_min_free_mib=35000,gpu_max_utilization=85,cuda_fraction=.55,foreign_gpu_reserve_mib=8192,
 root_min_free_gib=10,attached_min_free_gib=200,new_artifact_cap_gib=200,max_attempts=2,poll_seconds=5))
print('Prepared',len(jobs),'jobs in isolated batch4096 experiment')
