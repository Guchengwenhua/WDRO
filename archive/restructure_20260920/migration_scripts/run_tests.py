from pathlib import Path
import subprocess,os,json,time
R=Path('/remote-home/Guilin/DRO/Codex'); OUT=R/'archive/restructure_20260920/validation';OUT.mkdir(exist_ok=True)
PY='/remote-home/Guilin/miniconda3/envs/py311/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
tasks=[
 ('legacy','wdro_legacy_v1/wdro_rest_min',['-m','unittest','discover','-s','tests']),
 ('precision','wdro_precision_v2/wdro_rest_min',['-m','unittest','discover','-s','tests']),
 ('v31','adversarial_v3_1_20260912',['-m','unittest','discover','-s','tests']),
 ('critic','critic_search_20260915',['-m','unittest','discover','-s','tests']),
 ('critic_search','critic_search_20260915',['-c',"import runpy; m=runpy.run_path('tests/test_search.py'); [f() for n,f in m.items() if n.startswith('test_')]; print('3 function tests passed')"]),
 ('critic_closeout','critic_search_20260915',['test_closeout.py']),
 ('critic_training','critic_search_20260915',['test_training.py']),
 ('oracle_free','oracle_free_20260915',['-m','unittest','test_geometry','test_scheduler']),
 ('rl_only','rl_only_20260915',['-m','unittest','test_environment','test_scheduler']),
 ('rl_research','rl_research_20260915',['-m','unittest','discover','-s','tests']),
 ('papers','papers_wdro_20260915',['-m','unittest','discover','-s','tests']),
 ('comparison','benchmarks/rl_only_vs_mainline_20260919',['-m','unittest','test_comparison','test_scheduler']),
 ('external','benchmarks/public_designs_20260912',['-m','unittest','discover','-s','tests']),
]
results=[]
for name,rel,args in tasks:
 start=time.time();taskenv=dict(env)
 if name=='papers':taskenv['PYTHONPATH']=str(R/rel/'.deps')+':'+str(R/rel)
 with (OUT/(name+'.log')).open('w') as f:
  try:
   p=subprocess.run([PY,*args],cwd=R/rel,env=taskenv,stdout=f,stderr=subprocess.STDOUT,timeout=240)
   code=p.returncode
  except subprocess.TimeoutExpired:code=124
 result=dict(name=name,returncode=code,seconds=round(time.time()-start,2),cwd=str(R/rel),command=[PY,*args])
 results.append(result);(OUT/'tests.json').write_text(json.dumps(results,indent=2))
 print(name,code,flush=True)
