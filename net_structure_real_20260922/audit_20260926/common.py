from pathlib import Path
import os,sys,json,csv,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';OUT=ART/'audit_20260926';OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'config.json').read_text());sys.path.insert(0,str(ROOT));sys.path.insert(0,CFG['source_package'])
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(r,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');tmp.replace(p)
def csvout(name,rows):
 p=OUT/name
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def npz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
