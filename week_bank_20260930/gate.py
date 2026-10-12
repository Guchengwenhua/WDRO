"""Predeclared development-only stage gates; never opens test files."""
import argparse,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SCALE=HERE.parent/'checkpoint_bank_scale_20260930/artifacts'
ARMS=['k3','lagged3','qualifiedbest3']
def read(p):return json.loads(p.read_text())
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.json.partial');t.write_text(json.dumps(d,indent=2));t.replace(p)
def dev(arm,seed,stage):
 if stage=='matched' and arm!='qualifiedbest3':p=SCALE/f'{arm}_s{seed}/logs/result.json';v=read(p)['last_validation']
 else:
  name=f'd10_{"qbest2000" if stage=="matched" else arm+"10000"}_s{seed}';v=read(HERE/'artifacts'/name/'dev.json')['last_validation']
 return {'score':v['selection']['worst_group_mean'],'nominal':v['selection']['nominal_mean']}
def judge(arm,stage):
 try:
  a=[dev(arm,s,stage) for s in [101,102,103]];g=[dev('geo',s,stage) for s in [101,102,103]]
 except (FileNotFoundError,KeyError) as exc:return {'allow':False,'reason':'incomplete development evidence: '+str(exc)}
 aa=np.array([r['score'] for r in a]);gg=np.array([r['score'] for r in g]);diff=aa-gg
 nominal=all(x['nominal']<=y['nominal']+.002 for x,y in zip(a,g))
 finite=bool(np.isfinite(np.r_[aa,gg]).all());mean_ok=float(diff.mean())<=max(.0002,.05*float(gg.mean()));seeds_ok=int((diff<=.0005).sum())>=2
 allow=finite and nominal and mean_ok and seeds_ok
 return {'allow':allow,'reason':'predeclared dev-only tolerance gate passed' if allow else 'negative dev result; stop this expansion branch','arm':arm,'stage':stage,'mean_worst_ood_regret':float(aa.mean()),'geo_mean':float(gg.mean()),'per_seed_delta':diff.tolist(),'nominal_qualification':nominal,'mean_tolerance':max(.0002,.05*float(gg.mean())),'seeds_with_delta_le_0_0005':int((diff<=.0005).sum()),'warning':'gate is resource allocation, not statistical equivalence'}
def main(stage):
 folder=HERE/'gates';folder.mkdir(exist_ok=True);results={a:judge(a,stage) for a in ARMS}
 if stage=='matched':
  for a,r in results.items():write(folder/f'extend_{a}.json',r)
  write(folder/'extend_geo.json',{'allow':any(r['allow'] for r in results.values()),'reason':'matched Geo continuation only when at least one teacher method warrants extension'})
 else:
  eligible=[r for r in results.values() if r['allow']];winner=min(eligible,key=lambda r:(r['mean_worst_ood_regret'],r['arm'])) if eligible else None
  write(folder/'d30.json',{'allow':winner is not None,'selected_arm':winner['arm'] if winner else None,'reason':'lowest dev mean among eligible long-run bank methods; test unopened' if winner else 'no eligible long-run development result; no d30 expansion','comparisons':results})
 write(folder/f'{stage}_complete.json',{'complete':True,'allow':True,'status':'complete','results':results})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['matched','long'],required=True);main(p.parse_args().stage)
