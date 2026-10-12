import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scheduler
from confirmatory import holm,signflip_p

class ScaleupTests(unittest.TestCase):
    def test_actual_protocol_dependencies_and_eight_cards(self):
        p=json.loads((ROOT/'protocol.json').read_text());ts=scheduler.tasks(p)
        self.assertEqual(len(ts),80)
        self.assertEqual(len(scheduler.ready_tasks(ts,set(),{},{})),8)
        snap=dict(free_bytes=90*2**30,load=[1,1,1],mem_available_bytes=128*2**30,
                  gpus=[dict(index=i,uuid=str(i),free_mib=10000,util=20) for i in range(8)])
        cap,cards,_=scheduler.admission(snap,[snap],{},p['resources'])
        self.assertEqual((cap,len(cards)),(8,8))
        snap['gpus'][0]['util']=100
        self.assertNotIn(0,[g['index'] for g in scheduler.admission(snap,[snap]*3,{},p['resources'])[1]])
        snap['mem_available_bytes']=2**30
        self.assertEqual(scheduler.admission(snap,[snap],{},p['resources'])[0],0)
    def test_exact_directional_p_and_holm(self):
        self.assertEqual(signflip_p([1]*8),1/256)
        self.assertEqual(signflip_p([-1]*8),1)
        self.assertEqual(signflip_p([0]*8),1)
        self.assertEqual(holm([.005,.02]),[.01,.02])
        self.assertEqual(holm([1/256]*8),[8/256]*8)
    def test_parallel_training_input_regeneration(self):
        import report
        p=json.loads((ROOT/'protocol.json').read_text())
        p.update(seeds=[991,992],warmup_updates=1,fixed_updates=1,coev_updates=1,batch_size=4)
        parallel=report.regenerate_training_ids(p)
        self.assertEqual(len(parallel),24)
        self.assertEqual(parallel,report._regenerate_training_ids_serial(p))

if __name__=='__main__':unittest.main()
