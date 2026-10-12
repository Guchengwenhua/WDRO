import json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
BASE=PROJECT/'wdro_precision_v2'
ART=ROOT/'artifacts'
EXT=PROJECT/'geometry_uncertainty_extension_20260930/artifacts'
PYTHON='/remote-home/Guilin/miniconda3/envs/py311/bin/python'
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']
MODELS={10:['M0888','M1108','M0890','M1110'],30:['M0900','M1118','M0902','M1120']}
def read(p):return json.loads(Path(p).read_text())
def atomic(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.partial')
 with q.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def npz(p,**values):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.partial')
 with q.open('wb') as f:np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def runtime(gpu=True):
 for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
 cache=ART/'cache';cache.mkdir(parents=True,exist_ok=True)
 for name in ['TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TORCH_HOME','TRITON_CACHE_DIR','MPLCONFIGDIR']:os.environ[name]=str(cache)
 os.environ['PYTHONDONTWRITEBYTECODE']='1'
 if gpu:
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  torch.cuda.set_per_process_memory_fraction(.1)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def beat(folder,status,**extra):
 atomic(Path(folder)/'heartbeat.json',dict(pid=os.getpid(),status=status,time=time.time(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),batch=256,**extra))
 print(json.dumps(dict(status=status,**extra)),flush=True)
def paired(delta,seed=20261002):
 delta=np.asarray(delta,float);rng=np.random.default_rng(seed)
 boot=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(2000)]
 signs=rng.choice([-1.,1.],size=(9999,len(delta)))
 p=(1+(np.abs((signs*delta).mean(-1))>=abs(delta.mean())).sum())/10000
 return dict(mean_delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),p_two_sided=float(p),n_centers=len(delta))
def holm(p):
 order=np.argsort(p);v=np.zeros(len(p));previous=0
 for pos,i in enumerate(order):previous=max(previous,(len(p)-pos)*p[i]);v[i]=min(1.,previous)
 return v.tolist()
def freeze():
 if not (EXT/'complete.json').exists():raise RuntimeError('extension not complete')
 output=ART/'freeze.json'
 if output.exists():return read(output)
 cat={m['id']:m for m in read(PROJECT/'benchmarks/all_models_all_data_20260928/artifacts/models.json')}
 models={m:cat[m] for degree in MODELS for m in MODELS[degree]}
 stages={10:'d10_degree_control',30:'d30_multiseed'};choices={}
 for degree,stage in stages.items():
  gate=read(EXT/stage/'dev_gate.json');metric=gate['winner']
  if metric not in ['Q','G','C']:raise RuntimeError('invalid predeclared winner')
  choices[str(degree)]=dict(metric=metric,extension_dev_positive=gate['dev_mean_delta_vs_random'][metric]>0,source=str(EXT/stage/'dev_gate.json'),source_gate=gate)
 sources={}
 for m in models.values():
  p=Path(m['path']);st=p.stat();sources[str(p)]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
 policy=read(PROJECT/'holiday_research_20260930/policy.json')
 value=dict(models=models,choices=choices,sources=sources,dev_seed=202610041,test_seed=202610057,nominal_plus_candidates=33,exact_budget=4,
            test_used=False,deadline=policy.get('deadline_utc'),no_new_training_after=policy.get('training_cutoff_utc'),run_until_complete=policy.get('run_until_complete',False))
 atomic(output,value);return value
def metadata():
 f=read(ART/'freeze.json')
 for p,v in f['sources'].items():
  st=Path(p).stat()
  if st.st_size!=v['size'] or st.st_mtime_ns!=v['mtime_ns']:raise RuntimeError('frozen checkpoint changed '+p)
 return f
