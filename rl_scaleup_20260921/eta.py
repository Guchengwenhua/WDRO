"""Conservative ETA from progress, updated without changing experiment tasks."""
import argparse,json,os,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OPS=ROOT/'artifacts/operations'
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def rate(p):
    try:
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if len(rows)<2:return None
        a,b=rows[-min(32,len(rows))],rows[-1]
        return (b['elapsed_seconds']-a['elapsed_seconds'])/(b['step']-a['step'])
    except (FileNotFoundError,ValueError,KeyError,ZeroDivisionError):return None
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');a=parser.parse_args()
    began=time.time()
    calibration=read(OPS/'calibration.json',{})
    while True:
        from scheduler import tasks
        p=read(ROOT/'protocol.json');queue=read(OPS/'queue.json',{});ts=tasks(p);rates={};observed={}
        for t in ts:
            v=rate(ROOT/'artifacts/runs'/t['id']/'progress.jsonl')
            if v is not None and v>0:rates.setdefault((t['kind'],t['method']),[]).append(v);observed[t['id']]=v
        warm=statistics.median(rates.get(('warmup','erm'),[8.]))
        total=0.;records=[]
        for t in ts:
            out=ROOT/'artifacts/runs'/t['id'];done=read(out/'complete.json')
            if done:continue
            hb=read(out/'heartbeat.json',{});step=hb.get('step',0)
            method=t['method'];kind=t['kind']
            fallback=warm if kind=='warmup' or method=='erm' else warm*(3.0 if method in ('mappo','sequential_ppo') else 2.0)+(warm if kind=='coev' else 0)
            cal=calibration.get(kind+'_'+method,{})
            sec=statistics.median(rates.get((kind,method),[cal.get('seconds_per_update',fallback)]))
            # Separate final evaluation allowance, refined as completed runs appear.
            allowance=0. if kind=='warmup' else cal.get('estimated_formal_evaluation_seconds',600. if kind=='fixed' else 900.)
            left=max(0,p[kind+'_updates']-step)*sec+allowance
            total+=left;records.append(dict(run_id=t['id'],step=step,estimated_remaining_seconds=left,seconds_per_update=sec))
        parallel=p['resources']['max_workers'];hours=(total/parallel+900)/3600
        result=dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),remaining_tasks=len(records),observed_rates=observed,
                    estimated_hours=hours,range_hours=[hours*.65,hours*1.8],assumed_parallel_workers=parallel,
                    note='Rough throughput estimate; unobserved methods extrapolate warmup. GPU contention, serial dependencies, failures and resource waiting can exceed this range.',tasks=records)
        tmp=OPS/'eta.partial';tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,OPS/'eta.json')
        delivery=read(OPS/'delivery.json',{})
        if not a.watch or delivery:return
        time.sleep(30)
if __name__=='__main__':main()
