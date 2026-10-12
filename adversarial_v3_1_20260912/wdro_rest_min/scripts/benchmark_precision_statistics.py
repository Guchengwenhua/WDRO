"""Exercise full bootstrap size using validation-only inputs before test unlock."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import utc_now
from stage6.statistics import bootstrap_cell

def main():
    src=EXP/'logs/stage6_evaluation_smoke/d50_continuous/cells/uniform/predictions.npz'
    with np.load(src,allow_pickle=False) as data:base=data['regret']
    assert len(base)==2000
    values=np.tile(base,(3,5));start=time.monotonic()
    result=bootstrap_cell(values,samples=10000,seed=606)
    assert all(v.shape==(10000,3) and np.isfinite(v).all() for v in result.values())
    assert all(np.array_equal(v[:,0],v[:,1]) and np.array_equal(v[:,1],v[:,2]) for v in result.values())
    dst=EXP/'logs/stage6_pipeline/statistics_validation_benchmark.json'
    atomic_write_json(dst,dict(status='pass',at_utc=utc_now(),samples=10000,nets=10000,seeds=3,
        seconds=time.monotonic()-start,source=str(src),source_sha256=file_sha256(src),
        data_construction='repeat one validation vector only for compute and pairing check; not a model comparison',
        test_used=False,statistics_sha256=file_sha256(ROOT/'stage6/statistics.py')))
    print(json.dumps(dict(status='pass',benchmark=str(dst))))
if __name__=='__main__':main()
