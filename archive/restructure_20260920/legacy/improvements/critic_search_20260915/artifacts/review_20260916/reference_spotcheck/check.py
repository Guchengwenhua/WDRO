"""CPU-only exact-scale reference spot check, independent of existing caches."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path('/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915')
sys.path.insert(0,str(ROOT))
import bootstrap
from adv_v3 import common
from src import geosteiner as gs
import numpy as np

OUT=ROOT/'artifacts/review_20260916/reference_spotcheck'
EXP=ROOT/'artifacts';rng=np.random.default_rng(20260916)
samples=[];start=time.monotonic()
for task in [EXP/'fixed/validation_s101']+[EXP/'fixed'/f'test_s{s}' for s in (101,102,103)]:
    for path in sorted((task/'evaluation').glob('*_0000.npz')):
        with np.load(path) as a:
            row=int(rng.integers(len(a['centers'])));candidate=int(rng.integers(64))
            samples.append((path,row,candidate,a['support'][row,candidate],float(a['all_lstar'][row,candidate])))
for task in sorted((EXP/'training').glob('*_d30_s*')):
    for step in (1,64,128):
        path=task/'search'/f'u{step:06d}.npz'
        with np.load(path) as a:
            row=int(rng.integers(len(a['selected_pins'])))
            samples.append((path,row,int(a['selected_index'][row]),a['selected_pins'][row],float(a['selected_lstar'][row])))
evaluator=gs.GeoSteinerEvaluator();rows=[]
for index,(path,row,candidate,pins,expected) in enumerate(samples):
    values=[]
    for scale in (2**27,2**28):
        gs.SCALE=scale
        value=evaluator.solve(pins,require_integral=True)
        assert value==expected,(str(path),row,candidate,scale,expected,value)
        values.append(value)
    rows.append(dict(path=str(path),row=row,candidate=candidate,expected=expected,
        scale27=values[0],scale28=values[1],coordinate_sha256=hashlib.sha256(pins.tobytes()).hexdigest()))
    if index%50==0:print(json.dumps(dict(checked=index+1,total=len(samples))),flush=True)
payload=dict(status='pass',sample_count=len(samples),fresh_geosteiner_solves=2*len(samples),
    exact_equality_both_scales=True,scales=[2**27,2**28],seed=20260916,
    integral_inputs_optimal_status_connected_integer_embedding_checked=True,
    scope='one deterministic candidate per fixed seed/distribution/method cell, plus selected nets at training updates 1/64/128; no new Actor inference or training; this does not re-solve every reference',
    library_path=str(evaluator.library_path),library_sha256=common.sha256(evaluator.library_path),
    implementation_sha256=common.sha256(gs.__file__),elapsed_seconds=time.monotonic()-start,rows=rows)
(OUT/'result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
print(json.dumps({k:payload[k] for k in ('status','sample_count','fresh_geosteiner_solves','elapsed_seconds')}),flush=True)
