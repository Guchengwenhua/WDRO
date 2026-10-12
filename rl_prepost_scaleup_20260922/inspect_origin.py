"""Read-only current progress and hash-bound training trend snapshot."""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    p=json.loads((ROOT/'protocol.json').read_text());origin=Path(p['origin_root']);q=json.loads((origin/'artifacts/operations/queue.json').read_text())
    curves=[];bindings={}
    for rid in q['complete']:
        folder=origin/'artifacts/runs'/rid;m=json.loads((folder/'complete.json').read_text());path=folder/'progress.jsonl'
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=m['artifacts_sha256']['progress.jsonl']:raise RuntimeError('Source progress hash mismatch')
        rows=[json.loads(l) for l in path.read_text().splitlines()];key='gain' if m['kind']=='fixed' else 'sampled_length'
        means=[float(np.mean([x[key] for x in rows[start:end]])) for start,end in [(0,256),(1024,1280),(1536,1792),(1792,2048)]]
        curves.append(dict(run_id=rid,kind=m['kind'],method=m['method'],seed=m['seed'],metric=key,block_means=means));bindings[str(path)]=sha
    grouping={}
    for c in curves:grouping.setdefault(c['kind']+'_'+c['method'],[]).append(c)
    groups={name:dict(seeds=[v['seed'] for v in vals],block_means=np.mean([v['block_means'] for v in vals],axis=0).tolist(),
        last_block_improving=sum((v['block_means'][-1]>v['block_means'][-2] if v['kind']=='fixed' else v['block_means'][-1]<v['block_means'][-2]) for v in vals)) for name,vals in grouping.items()}
    result=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),completed=len(q['complete']),total=q['total'],
        stage_counts={k:sum(r.startswith(k) for r in q['complete']) for k in ['warmup','fixed','coev']},
        retries={k:v for k,v in q['attempts'].items() if v>1},running=q['running'],groups=groups,curves=curves,bindings=bindings,
        scope='Completed-source training trends; not validation improvement or full experiment acceptance')
    (ROOT/'artifacts/operations/origin_progress_snapshot.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['time_utc','completed','total','stage_counts','retries']}))
if __name__=='__main__':main()
