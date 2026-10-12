"""Explicit non-scientific bookkeeping adaptations and replaceable reference provider.
No historical source is edited. Seed hashing is retained only for original RNG namespaces.
Content digests are not computed. Existing digest strings may be used as stored identifiers.
"""
import copy,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/remote-home/Guilin/DRO/Codex');BASE=ROOT/'wdro_precision_v2';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'wdro_rest_min'))
import src.datasets as datasets
import src.candidates as candidates
import src.training_data as training_data
import src.validation as validation
import src.geosteiner as geosteiner
import src.rest as rest
from src.checkpoints import atomic_torch_save,capture_rng_state,restore_rng_state
LIB=Path('/remote-home/Guilin/DRO/DRO/WDRO-REST/algorithms/libeval.so')
rest.LEGACY_ROOT=LIB.parents[1]
SETTINGS={}

def omitted(*args,**kwargs):return 'not_computed_per_skill'
def file_sha256(path):
 p=Path(path);s=p.stat();return {'identity_kind':'path_size_mtime_not_digest','path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def canonical_sha256(value):return {'identity_kind':'literal_config_not_digest','value':value}
def protocol_sha256(value):return {'identity_kind':'literal_protocol_not_digest','value':value}
def validation_manifest_hash(value):return 'not_computed_per_skill'
def canonical_points(pins):
 a=np.asarray(pins,dtype=np.float64);o=np.lexsort((a[:,1],a[:,0]));return tuple(map(tuple,a[o].tolist()))
# Exact coordinate tuples preserve unordered equality; no content digest computation.
datasets.array_bundle_sha256=omitted
training_data.array_bundle_sha256=omitted
training_data.extend_stream_hash=omitted
candidates.array_bundle_sha256=omitted
candidates.pin_set_sha256=canonical_points
import stage5.search as search
search.array_bundle_sha256=omitted;search.pin_set_sha256=canonical_points

def reference_identity(*args,**kwargs):return {'reference_id':'geosteiner_exact_binary_scale_v2','library':str(LIB),'content_hashes':'not_computed'}
geosteiner.reference_identity=reference_identity

def load_trusted_legacy_actor(actor,checkpoint_path,*,map_location):
 # Explicit actual path; same legacy actor tensors, no credential or other file reads.
 path=LIB.parents[1]/'save/DAC21/rsmt10b.pt';c=torch.load(path,map_location=map_location,weights_only=False);actor.load_state_dict(c['actor_state_dict'],strict=True)
 return {'path':str(path),'source_metadata':file_sha256(path),'loaded_components':['actor_state_dict'],'legacy_batch_idx':int(c.get('batch_idx',-1))}

def load_validation_group(protocol,*,degree,distribution,**kwargs):
 count=protocol['data']['sizes_per_degree_distribution']['validation'];seed=protocol['data']['seeds']['validation']
 path=BASE/f'experiments/cache/datasets/protocol_v1/validation/{distribution}_d{degree}_s{seed}_n{count}.npz'
 a,m=datasets.load_dataset(path,verify=False);b,bm=datasets.load_dataset(BASE/'experiments/cache/lstar/protocol_v1'/f"{m['array_bundle_sha256']}.npz",verify=False)
 if not np.array_equal(a['net_ids'],b['net_ids']):raise ValueError('cached validation IDs mismatch')
 return a['pins'],b['lstar'],a['net_ids'],m
validation.load_validation_group=load_validation_group
from src.validation import evaluate_validation
from src.training_data import materialize_training_batch
extend_stream_hash=omitted
array_bundle_sha256=omitted

class Exact:
 def __init__(self):self.solver=geosteiner.GeoSteinerEvaluator(LIB) if SETTINGS.get('arm','geo')=='geo' else None
 def lengths(self,pins):
  if self.solver is None:raise RuntimeError('bank training attempted exact solve')
  return self.solver.lengths(pins)
 def close(self):pass

def make_geosteiner():return Exact()

class ResearchReference:
 def __init__(self,path):
  self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.device=SETTINGS['device'];self.arm=SETTINGS['arm'];self.bank=[];self.models=[]
  self.accounting={'teacher_forward_nets':0,'teacher_seconds':0.,'exact_train_nets':0,'refresh_seconds':0.,'refreshes':0,'duplicate_snapshots_skipped':0}
  pool=SETTINGS['pool'];self.bank=copy.deepcopy(pool[:3] if self.arm=='k3' else pool[:1] if self.arm=='lagged3' else [])
  self._load()
 def initialize(self,actor):
  if self.arm!='lagged3':return
  path=self.path/'snapshot_u0000.pt'
  if not path.exists():atomic_torch_save({'actor_state_dict':actor.state_dict(),'update':0},path)
  self.bank=self.bank[:1]+[{'path':str(path),'run':'lagged_u0','update':0}];self._load()
 def _load(self):
  rng=capture_rng_state();self.models=[]
  for r in self.bank:
   a=rest.build_actor(10,self.device);d=torch.load(r['path'],map_location=self.device,weights_only=False);a.load_state_dict(d['actor_state_dict']);a.eval();a.requires_grad_(False);self.models.append(a)
  restore_rng_state(rng)
 def lengths(self,pins,evaluator):
  if self.arm=='geo':
   self.accounting['exact_train_nets']+=len(pins);return evaluator.lengths(pins),len(pins)
  t=time.monotonic();values=[];self.accounting['last_scored_bank']=[r['run'] for r in self.bank]
  with torch.no_grad():
   for a in self.models:
    out=[]
    for i in range(0,len(pins),1024):
     top,_=a(pins[i:i+1024],True);out.append(SETTINGS['rest'].lengths(pins[i:i+1024],top.cpu().numpy()))
    values.append(np.concatenate(out))
  self.accounting['teacher_forward_nets']+=len(pins)*len(self.models);self.accounting['teacher_seconds']+=time.monotonic()-t
  return np.min(values,axis=0),0
 def refresh(self,actor,update):
  if self.arm!='lagged3' or update%3 or update>=2000:return
  t=time.monotonic();current=actor.state_dict()
  if any(all(torch.equal(v,m.state_dict()[k]) for k,v in current.items()) for m in self.models):
   self.accounting['duplicate_snapshots_skipped']+=1;self.accounting['refresh_seconds']+=time.monotonic()-t;return
  path=self.path/f'snapshot_u{update:04d}.pt';atomic_torch_save({'actor_state_dict':current,'update':update},path)
  anchor=self.bank[0];snapshots=[r for r in self.bank[1:] if r.get('update',-1)>=0]
  self.bank=[anchor]+(snapshots+[{'path':str(path),'run':f'lagged_u{update}','update':update}])[-2:]
  self._load();self.accounting['refreshes']+=1;self.accounting['refresh_seconds']+=time.monotonic()-t
  with (self.path/'refresh.jsonl').open('a') as log:log.write(json.dumps({'after_update':update,'bank_used_ids':self.accounting.get('last_scored_bank',[]),'bank_next_ids':[r['run'] for r in self.bank],'refresh_seconds_total':self.accounting['refresh_seconds']})+'\n')
 def state(self):return {'bank':self.bank,'accounting':self.accounting,'arm':self.arm}
 def restore(self,state):
  if state['arm']!=self.arm:raise ValueError('reference arm mismatch')
  self.bank=state['bank'];self.accounting=state['accounting'];self._load()
 def close(self):pass
