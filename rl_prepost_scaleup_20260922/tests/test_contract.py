import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ops
from report import signflip,holm
import audit_utils as a
import numpy as np

class Contract(unittest.TestCase):
    def test_budget_and_dependency_graph(self):
        p=json.loads((ROOT/'protocol.json').read_text());ts=p['tasks'];known=set()
        self.assertEqual(len(ts),104)
        self.assertEqual(sum(t['end']-t['start'] for t in ts),638976)
        for t in ts:
            self.assertGreater(t['end'],t['start']);self.assertEqual(t['milestones'][-1],t['end'])
            for v in t['parents'].values():
                if v['source']=='new':self.assertIn(v['run'],known)
            known.add(t['id'])
        for seed in p['seeds']:
            for method in p['main_methods']:
                for pre in [2048,8192]:self.assertIn(f'post_pre{pre}_{method}_s{seed}',known)
    def test_primary_family_resolution(self):
        self.assertEqual(signflip([1]*8),1/256)
        self.assertEqual(signflip([-1]*8),1)
        self.assertEqual(signflip([0]*8),1)
        self.assertEqual(holm([1/256]*6),[6/256]*6)
    def test_corrupted_nominal_topology_rejected(self):
        arrays={'uniform__nominal_pins':np.array([[[.1,.1],[.5,.2],[.6,.8]]],np.float32),
                'uniform__nominal_topology':np.array([[0,1,0,1]]),'uniform__nominal_length':np.array([1.])}
        errors=a.audit_arrays(arrays,{'kind':'warmup'},dict(degree=3,distributions=['uniform']))
        self.assertTrue(any('recompute' in e for e in errors))
    def test_resource_pause_gates(self):
        p=json.loads((ROOT/'protocol.json').read_text());r=p['resources']
        snap=dict(free_bytes=60*2**30,mem_available_bytes=128*2**30,load=[1,1,1],
            gpus=[dict(index=0,uuid='0',free_mib=20000,util=100)])
        self.assertFalse(ops.admission(snap,[snap]*3,{},r)[1])
        snap['gpus'][0]['util']=10;snap['free_bytes']=24*2**30
        self.assertEqual(ops.admission(snap,[snap],{},r)[0],0)

if __name__=='__main__':unittest.main()
