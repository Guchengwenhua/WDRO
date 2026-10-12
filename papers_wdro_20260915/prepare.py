"""CPU preparation shared by all four solver families, resumable by shard."""
import argparse, concurrent.futures, json, os, time
from pathlib import Path
import numpy as np
from core import ROOT,MAIN,atomic_json,atomic_npz,dataset,costs,digest,read_protocol,sha,generate_support,shortlist_support

DATA=ROOT/'artifacts/data'

def shard(seed,first,stop):
    p=read_protocol();out=DATA/f'train_s{seed}_{first:04d}.npz';meta=out.with_suffix('.json')
    identity={'protocol':digest(p),'seed':seed,'first':first,'stop':stop,'preparer':sha(__file__),'core':sha(ROOT/'core.py')}
    if out.exists() and meta.exists():
        old=json.loads(meta.read_text())
        if old['identity']==identity and old['sha256']==sha(out): return {'cached':True,'seed':seed,'first':first}
        raise ValueError('existing data shard identity differs')
    t=time.time();values={k:[] for k in ['pins','shortlist','indices','cost_rooted','cost_unrooted']};support_hashes=[]
    for u in range(first,stop):
        batch=dataset(p,f'train_u{u:06d}','uniform',seed,p['batch_size'])
        support=generate_support(batch,training_seed=seed,update=u,search=p['search'])
        ids,_=shortlist_support(support,p['search']['exact_rescore_top_k'])
        chosen=support.pins[np.arange(len(batch.pins))[:,None],ids]
        values['pins'].append(batch.pins);values['shortlist'].append(chosen);values['indices'].append(ids)
        values['cost_rooted'].append(costs(batch.pins,chosen,True));values['cost_unrooted'].append(costs(batch.pins,chosen,False))
        support_hashes.append(support.sha256)
    atomic_npz(out,**{k:np.asarray(v) for k,v in values.items()})
    record={'identity':identity,'sha256':sha(out),'support_hashes':support_hashes,'seconds':time.time()-t,'bytes':out.stat().st_size}
    atomic_json(meta,record);return {'cached':False,'seed':seed,'first':first,'seconds':record['seconds']}

def prepare_eval(p):
    index={}
    identity={'protocol':digest(p),'preparer':sha(__file__),'core':sha(ROOT/'core.py')}
    for split,count,seed in [('validation',p['validation_count'],6202),('test',p['test_count'],6303),('attack',p['attack_count'],6404)]:
        for dist in p['distributions']:
            out=DATA/f'{split}_{dist}.npz'
            batch=dataset(p,split,dist,seed,count)
            arrays={'pins':batch.pins,'net_ids':batch.net_ids}
            if split=='attack':
                support=generate_support(batch,training_seed=7404,update=1,search=p['search'])
                arrays.update(candidates=support.pins,cost_rooted=costs(batch.pins,support.pins,True),cost_unrooted=costs(batch.pins,support.pins,False))
            if out.exists():
                with np.load(out,allow_pickle=False) as old:
                    if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                        raise ValueError('existing evaluation data differs from independent regeneration: '+str(out))
            else:
                atomic_npz(out,**arrays)
            atomic_json(out.with_suffix('.json'),{'identity':identity,'count':count,'split':split,'distribution':dist,'sha256':sha(out)})
            index[str(out)]=sha(out)
            index[str(out.with_suffix('.json'))]=sha(out.with_suffix('.json'))
    calibration=DATA/'calibration.json'
    if calibration.exists() and json.loads(calibration.read_text())['protocol']!=digest(p):
        raise ValueError('existing rho calibration uses a different protocol')
    if not calibration.exists():
        batch=dataset(p,'train_calibration','uniform',7101,256)
        support=generate_support(batch,training_seed=7101,update=1,search=p['search'])
        values={kind:costs(batch.pins,support.pins[:,1:],rooted) for kind,rooted in [('rooted',True),('unrooted',False)]}
        atomic_json(calibration,{'source':'separate_training_only_centers','seed':7101,'count':256,'quantile':0.75,
            'rooted':float(np.quantile(values['rooted'],.75)),'unrooted':float(np.quantile(values['unrooted'],.75)),
            'support_sha256':support.sha256,'protocol':digest(p)})
    index[str(calibration)]=sha(calibration)
    return index

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--max-shards',type=int,default=0)
    args=ap.parse_args();p=read_protocol();DATA.mkdir(parents=True,exist_ok=True)
    start=time.time();evals=prepare_eval(p)
    jobs=[(s,u,min(u+p['shard_updates'],p['updates']+1)) for s in p['seeds'] for u in range(1,p['updates']+1,p['shard_updates'])]
    all_jobs=len(jobs)
    if args.max_shards: jobs=jobs[:args.max_shards]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        future={pool.submit(shard,*job):job for job in jobs}
        for i,f in enumerate(concurrent.futures.as_completed(future),1):
            print(json.dumps({'completed_shards':i,'submitted_shards':len(jobs),'total_shards':all_jobs,**f.result()}),flush=True)
            atomic_json(ROOT/'artifacts/operations/preparation_status.json',{'status':'running','completed_shards_this_attempt':i,'total_shards':all_jobs,'time':time.time(),'seconds':time.time()-start})
    if len(jobs)==all_jobs:
        index=dict(evals)
        for seed,first,_ in jobs:
            path=DATA/f'train_s{seed}_{first:04d}.npz';index[str(path)]=sha(path);index[str(path.with_suffix('.json'))]=sha(path.with_suffix('.json'))
        atomic_json(DATA/'manifest.json',{'status':'complete','protocol':digest(p),'files':index,'seconds':time.time()-start,'total_shards':all_jobs})
        atomic_json(ROOT/'artifacts/operations/preparation_status.json',{'status':'complete','total_shards':all_jobs,'seconds':time.time()-start,'time':time.time()})

if __name__=='__main__': main()
