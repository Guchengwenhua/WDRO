import unittest,sys,json,ast
from pathlib import Path
import numpy as np
from stage6.statistics import bootstrap_cell,ci,metrics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))

class Stage6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global choose_gpu,measured_peak
        from stage6_pipeline import choose_gpu,measured_peak
    def test_paired_bootstrap_constant_shift_has_zero_extra_uncertainty(self):
        values=np.arange(96,dtype=float).reshape(3,32)/32
        a=bootstrap_cell(values,samples=128,seed=7);b=bootstrap_cell(values+2,samples=128,seed=7)
        for k in a:np.testing.assert_allclose(b[k]-a[k],2,rtol=0,atol=1e-14)
    def test_macro_tail_keeps_each_seed_tail(self):
        values=np.array([[0]*19+[10],[10]+[0]*19,[2]*20],float)
        self.assertAlmostEqual(metrics(values)['p95'],(0.5+0.5+2)/3)
        self.assertNotAlmostEqual(metrics(values)['p95'],float(np.quantile(values.mean(axis=0),.95)))
    def test_bootstrap_batching_does_not_change_draws(self):
        values=np.arange(111,dtype=float).reshape(3,37)
        a=bootstrap_cell(values,samples=123,seed=9,chunk=7);b=bootstrap_cell(values,samples=123,seed=9,chunk=31)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_shared_gpu_process_is_not_automatic_exclusion(self):
        rows=[dict(index=0,eligible=True,minimum_free_memory_mib=12000),dict(index=1,eligible=True,minimum_free_memory_mib=6000)]
        self.assertEqual(choose_gpu(rows,{},[],{},5000,1),0)
        self.assertEqual(choose_gpu(rows,{'run':{'gpu':0}},[],{},5000,1),1)
        self.assertIsNone(choose_gpu(rows,{},[],{},13000,1))
    def test_dynamic_capacity_history_and_cooldown(self):
        rows=[dict(index=0,eligible=True,minimum_free_memory_mib=12000)]
        self.assertIsNone(choose_gpu(rows,{},[{'gpus':[{'index':0,'memory_free_mib':1000}]}],{},5000,1))
        self.assertIsNone(choose_gpu(rows,{},[],{0:5},5000,1))
    def test_stage5_and_stage6_update_loop_unchanged(self):
        def loop(path):
            tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_wdro')
            node=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='update')
            return ast.dump(node,include_attributes=False)
        self.assertEqual(loop(ROOT/'stage5/trainer.py'),loop(ROOT/'stage6/trainer.py'))
    def test_complete_frozen_scope_without_new_search(self):
        plan=json.loads((ROOT/'configs/stage6_plan_v1.json').read_text())
        self.assertEqual(len(plan['reused_models']),61);self.assertEqual(len(plan['tasks']),15)
        self.assertEqual({(t['degree'],t['seed']) for t in plan['tasks']},{(d,s) for d in [10,20,30,40,50] for s in [101,102,103]})
        chosen=json.loads((ROOT/'configs/stage5_selected_wdro_v1.json').read_text())
        self.assertIn(chosen['selected_rho_mode'],['zero','q50','q75'])
        self.assertFalse(chosen['test_or_ispd_used'])
        self.assertTrue(all(t['rho_mode']==chosen['selected_rho_mode'] for t in plan['tasks']))
        self.assertGreaterEqual(measured_peak(50),3602)
