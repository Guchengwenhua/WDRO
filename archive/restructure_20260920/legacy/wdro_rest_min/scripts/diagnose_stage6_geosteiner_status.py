"""Read-only single-net solver diagnosis; never changes frozen references."""
import ctypes as C
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.geosteiner import GeoSteinerEvaluator
from src.io_utils import atomic_write_json
from src.job_control import utc_now

OUT = ROOT.parent / 'experiments/logs/stage6_negative_regret'

def main():
    diagnosis = json.loads((OUT / 'diagnosis.json').read_text())
    batch = np.load(OUT / 'failed_batch.npz', allow_pickle=False)
    offset = diagnosis['bad'][0]['net_index'] - diagnosis['failed_batch_start']
    xy = batch['pins'][offset].astype(np.float64)
    lib = GeoSteinerEvaluator()._library
    dp, ip = C.POINTER(C.c_double), C.POINTER(C.c_int)
    lib.gst_rsmt.argtypes = [C.c_int, dp, dp, ip, dp, ip, ip, ip, C.c_void_p]
    lib.gst_rsmt.restype = C.c_int
    lib.gst_create_param.argtypes = [ip]
    lib.gst_create_param.restype = C.c_void_p
    lib.gst_free_param.argtypes = [C.c_void_p]
    lib.gst_set_int_param.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.gst_get_int_param.argtypes = [C.c_void_p, C.c_int, ip]
    lib.gst_get_dbl_param.argtypes = [C.c_void_p, C.c_int, dp]
    variants = [('default', 1., np.arange(40), {})]
    variants += [(f'power2_scale_{p}', float(2**p), np.arange(40), {}) for p in (10, 20, 27)]
    variants += [('reverse', 1., np.arange(40)[::-1], {}),
                 ('permutation_seed0', 1., np.random.default_rng(0).permutation(40), {}),
                 ('lp_scale', 1., np.arange(40), {1008: 1}),
                 ('eps_mult_1', 1., np.arange(40), {1001: 1})]
    records = []
    for name, scale, order, changes in variants:
        code = C.c_int(-99)
        param = lib.gst_create_param(C.byref(code))
        assert param and code.value == 0
        for key, value in changes.items():
            assert lib.gst_set_int_param(param, key, value) == 0
        defaults = {}
        for key in (1001, 1003, 1004, 1007, 1008, 1022):
            value = C.c_int()
            assert lib.gst_get_int_param(param, key, C.byref(value)) == 0
            defaults[str(key)] = value.value
        gap = C.c_double()
        assert lib.gst_get_dbl_param(param, 2003, C.byref(gap)) == 0
        points = np.ascontiguousarray(xy[order] * scale)
        ns, ne, status = C.c_int(), C.c_int(), C.c_int(-99)
        steiners = np.zeros((160, 2), dtype=np.float64)
        edges = np.zeros((240, 2), dtype=np.int32)
        length = C.c_double()
        t = time.perf_counter()
        rc = lib.gst_rsmt(40, points.ctypes.data_as(dp), C.byref(length), C.byref(ns),
                          steiners.ctypes.data_as(dp), C.byref(ne), edges.ctypes.data_as(ip),
                          C.byref(status), param)
        elapsed = time.perf_counter() - t
        lib.gst_free_param(param)
        nodes = np.concatenate((points, steiners[:ns.value])) / scale
        topology = edges[:ne.value]
        embedded = float(np.abs(nodes[topology[:, 0]] - nodes[topology[:, 1]]).sum())
        row = dict(variant=name, scale=scale, return_code=rc, solution_status=status.value,
                   length=length.value / scale, embedding_length=embedded, seconds=elapsed,
                   parameter_values=defaults, gap_target=gap.value,
                   number_steiners=ns.value, number_edges=ne.value,
                   pins=points.tolist(), steiners=(steiners[:ns.value] / scale).tolist(), edges=topology.tolist())
        records.append(row)
        atomic_write_json(OUT / 'solver_status.json', dict(at_utc=utc_now(), records=records,
                          complete=len(records) == len(variants), net_id=diagnosis['bad'][0]['net_id']))
        print(json.dumps({k:v for k,v in row.items() if k not in ('pins','steiners','edges')}), flush=True)

if __name__ == '__main__':
    main()
