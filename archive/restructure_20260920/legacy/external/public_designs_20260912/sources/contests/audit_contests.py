"""Full-read coordinate/degree audit independent of downstream evaluation."""
import concurrent.futures, collections, hashlib, json, pathlib, sys, time
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent.parent/'parsers'))
from contests import parse_design
DEGREES=(10,20,30,40,50)
def audit(row):
 start=time.time();seen=set();counts=collections.Counter();selected=collections.Counter();duplicates=collections.Counter();zero=collections.Counter();minxy=np.array([np.inf,np.inf]);maxxy=-minxy;npins=0
 for net_id,xy in parse_design(row['path']):
  if net_id in seen:raise ValueError(f'duplicate net identity {net_id}')
  seen.add(net_id);n=len(xy);counts[n]+=1;npins+=n
  if n: minxy=np.minimum(minxy,xy.min(axis=0));maxxy=np.maximum(maxxy,xy.max(axis=0))
  if n in DEGREES:
   selected[n]+=1
   if len(set(map(tuple,xy)))!=n:duplicates[n]+=1
   if not n or max(np.ptp(xy,axis=0))<=0:zero[n]+=1
 out={'design_id':row['design_id'],'num_nets':sum(counts.values()),'num_pins':npins,'min_xy':minxy.tolist(),'max_xy':maxxy.tolist(),'raw_degree_counts':dict(sorted(counts.items())),'selected_raw_degree_counts':dict(sorted(selected.items())),'selected_repeated_xy_net_counts':dict(sorted(duplicates.items())),'selected_zero_span_net_counts':dict(sorted(zero.items())),'elapsed_seconds':time.time()-start}
 print(json.dumps({k:v for k,v in out.items() if k!='raw_degree_counts'}),flush=True)
 return out
if __name__=='__main__':
 rows=json.loads((BASE/'designs_manifest.json').read_text())['designs']
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as p:results=list(p.map(audit,rows))
 target=BASE/'coordinate_audit.json';tmp=target.with_suffix('.json.partial');tmp.write_text(json.dumps({'schema_version':1,'designs':results,'count':len(results),'all_nets':sum(x['num_nets'] for x in results),'all_pins':sum(x['num_pins'] for x in results),'status':'PASS'},indent=2)+'\n');tmp.replace(target)
