"""Isolated small real-data end-to-end fixture; cannot set production completion."""
import os,sys,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
root=Path('/remote-home/Guilin/DRO/Codex/net_structure_real_20260922');dest=root/'artifacts/tests/e2e';dest.mkdir(exist_ok=True)
for p in root.glob('*.py'):shutil.copy2(p,dest/p.name)
c=json.loads((root/'config.json').read_text());c['artifact_storage']=str(dest/'artifacts');c['generator'].update(tune_pre=2,tune_post=2,formal_pre=2,formal_post=2,validation_every=2,batch=16);c['runtime']['total_reference_workers_default']=2;c['seeds']=[101]
(dest/'config.json').write_text(json.dumps(c));a=dest/'artifacts';a.mkdir(exist_ok=True)
f=np.load(root/'artifacts/data.npz');ids=[]
for split in ['A_train','A_cal','A_val','B_test','C_external']:ids.extend(np.flatnonzero(f['split']==split)[:8])
ids=np.array(ids);np.savez_compressed(a/'data.npz',**{k:f[k][ids] for k in f.files});data=np.load(a/'data.npz');centers=np.flatnonzero(data['split']=='A_train');np.savez_compressed(a/'training_centers.npz',indices=centers)
p=np.load(root/'artifacts/physical.npz');np.savez_compressed(a/'physical.npz',**{k:p[k][ids] for k in p.files})
for sub in ['indices','pools','multidegree']:(a/sub).mkdir(exist_ok=True)
np.savez_compressed(a/'indices/seed101.npz',indices=np.random.default_rng(101).integers(0,8,(2048,64)))
for name in ['rho.json','recipe_selected.json']:shutil.copy2(root/'artifacts'/name,a/name)
for path in (root/'artifacts/multidegree').glob('*.npz'):
 z=np.load(path);np.savez_compressed(a/'multidegree'/path.name,**{k:z[k][:16] for k in z.files})
for method in ['raw_wdro','vae_shell_wdro']:
 z=np.load(root/'artifacts/pools'/f'{method}.npz');payload={k:z[k][:8] for k in z.files};payload['center_index']=centers;np.savez_compressed(a/'pools'/f'{method}.npz',**payload)
log=(root/'artifacts/logs/integration.log').open('w');env={**os.environ,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'NET_RUN_RESOURCE':'1'}
commands=[]
def run(args):
 tick=time.monotonic();r=subprocess.run([c['python'],*args],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT);log.flush();commands.append(dict(args=args,returncode=r.returncode,seconds=time.monotonic()-tick));assert r.returncode==0,args
for kind in ['set','graph']:
 for width in [96,160]:
  for eta in ([0] if kind=='set' else [.1,1]):run(['generator.py','--phase','tune','--key',f'{kind}_w{width}_e{eta:g}'])
run(['generator.py','--select'])
for kind in ['set','graph']:run(['generator.py','--phase','formal','--kind',kind])
for method in ['feature_gate','laplacian','tile_raw','set_dae','graph_dae','graph_tile']:
 run(['pools.py',method]);name=f'{method}_g301' if method in ['set_dae','graph_dae','graph_tile'] else method;run(['oracle.py',name])
for method in c['methods']:run(['rest_train.py','--method',method,'--seed','101','--generator','301','--target','2','--device','cuda'])
run(['attack.py','--prepare']);run(['oracle.py','attack32'])
folder=a/'runs/graph_tile_g301_s101';shutil.copy2(folder/'val_best_2.pt',folder/'val_best_2048.pt')
run(['attack.py','--method','graph_tile','--seed','101','--generator','301']);run(['report.py'])
(root/'artifacts/tests/integration.json').write_text(json.dumps(dict(status='PASS_E2E_FIXTURE_ONLY',commands=commands,time=time.time()),indent=2))
