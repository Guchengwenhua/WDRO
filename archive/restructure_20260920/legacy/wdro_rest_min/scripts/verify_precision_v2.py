"""Independent reference repair acceptance; no model selection or new test inference."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from precision_v2.geosteiner import GeoSteinerEvaluator,ParallelGeoSteinerEvaluator,reference_identity,SCALE
from precision_v2.independent import fst_milp,hanan_milp

OUT=ROOT.parent/'experiments/logs/precision_v2/reference_verification'

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--max-new-cases',type=int);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);gst=GeoSteinerEvaluator();rows=[]
    fixture=ROOT.parent/'experiments/logs/stage6_negative_regret/failed_batch.npz'
    with np.load(fixture,allow_pickle=False) as z:failed=z['pins'][56].copy()
    cases=[('reported_test_regression',failed),('three_terminal_analytic',np.array([[.1,.1],[.9,.1],[.5,.9]],dtype=np.float32))]
    if not a.smoke:
        for n in (4,5,6):
            for seed in range(3):
                cases.append((f'independent_small_d{n}_s{seed}',np.random.default_rng(seed+700+n).uniform(.1,.9,(n,2)).astype(np.float32)))
        for n in (10,20,30,40,50):
            for seed in range(3):
                cases.append((f'independent_formal_degree_d{n}_s{seed}',np.random.default_rng(seed+900+n).uniform(.1,.9,(n,2)).astype(np.float32)))
        witnesses=json.loads((ROOT.parent/'experiments/logs/stage6_negative_regret/feasible_tree_witnesses.json').read_text())
        for w in witnesses['witnesses']:
            cases.append((w['net_id'],np.asarray(w['nodes_integer'][:w['degree']],dtype=np.float64)/SCALE))
    output=OUT/('smoke.json' if a.smoke else 'acceptance.json')
    binding=dict(reference_identity=reference_identity(),cases=[dict(name=name,pins_sha256=hashlib.sha256(pins.tobytes()).hexdigest()) for name,pins in cases],
                 code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'precision_v2/independent.py']})
    if output.exists():
        old=json.loads(output.read_text())
        if 'binding' in old:
            assert old['binding']==binding,'verification input/code changed; preserve prior attempt separately'
            rows=old['cases']
    done={r['name'] for r in rows};new=0
    for name,pins in cases:
        if name in done:continue
        start=time.monotonic();solution=gst.solve(pins,witness=True,require_integral=True)
        independent=hanan_milp(pins,time_limit=30) if len(pins)<=6 else fst_milp(pins,time_limit=30)
        assert solution['length']==independent['length'],(name,solution['length'],independent)
        assert gst.length(pins[::-1])==solution['length']
        assert gst.length(pins*2)/2==solution['length']
        rows.append(dict(name=name,degree=len(pins),reference=solution['length'],independent=independent,seconds=time.monotonic()-start))
        state=dict(status='running',at_utc=utc_now(),binding=binding,reference_identity=reference_identity(),cases=rows)
        atomic_write_json(output,state)
        print(json.dumps(rows[-1]),flush=True)
        new+=1
        if a.max_new_cases is not None and new>=a.max_new_cases and len(rows)<len(cases):
            state['status']='paused';atomic_write_json(output,state);return
    rng=np.random.default_rng(4321);batch=rng.uniform(.1,.9,(6,10,2)).astype(np.float32)
    with ParallelGeoSteinerEvaluator(2) as parallel:np.testing.assert_array_equal(parallel.lengths(batch),gst.lengths(batch))
    try:gst.solve(np.array([[.1,.1],[.3333333333333333,.8]]),require_integral=True)
    except ValueError:pass
    else:raise AssertionError('nonintegral formal input was accepted')
    state=dict(status='pass',at_utc=utc_now(),binding=binding,reference_identity=reference_identity(),cases=rows,
               parallel_equivalence=True,nonintegral_formal_input_rejected=True)
    atomic_write_json(output,state)
    print(json.dumps(dict(status='pass',cases=len(rows))),flush=True)
if __name__=='__main__':main()
