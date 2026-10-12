import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from stage5 import configuration as c

class SelectionFirewallTests(unittest.TestCase):
    def test_nominal_guard_precedes_objective_and_tie_uses_declared_order(self):
        with tempfile.TemporaryDirectory() as td:
            exp=Path(td);tasks=[]
            for name,objective,nominal in [('zero',.01,.008),('q50',.009,.008),('q75',.001,.02)]:
                rid='test_'+name;tasks.append({'run_id':rid,'rho_mode':name})
                p=exp/'logs/stage5'/rid/'result.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'status':'complete','completed_updates':2000,'test_or_ispd_used':False,'nominal_limit':.01,
                    'best':{'feasible':nominal<=.01,'nominal_mean':nominal,'selection_objective':objective}}))
            with patch.object(c,'EXP',exp):
                r=c.select_rho(tasks);self.assertEqual(r['selected']['task']['rho_mode'],'q50')
                p=exp/'logs/stage5/test_zero/result.json';d=json.loads(p.read_text());d['best']['selection_objective']=.009;p.write_text(json.dumps(d))
                self.assertEqual(c.select_rho(tasks)['selected']['task']['rho_mode'],'zero')
                d['test_or_ispd_used']=True;p.write_text(json.dumps(d))
                with self.assertRaises(AssertionError):c.select_rho(tasks)
    def test_source_scope_and_formal_budget_are_fixed(self):
        plan=json.loads(c.PLAN.read_text())
        self.assertEqual((plan['degree'],plan['training_seed'],plan['policy_updates'],plan['batch_size']),(30,101,2000,256))
        self.assertEqual(plan['minimum_total_unique_runs'],9)
        self.assertEqual(plan['selection_source'],'synthetic_validation_only')
        self.assertFalse(plan['test_or_ispd_used'])

class ResourceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(c.ROOT/'scripts'))
        from stage5_pipeline import choose_gpu
        cls.choose=staticmethod(choose_gpu)
    def test_partially_occupied_gpu_is_eligible(self):
        rows=[{'index':4,'eligible':True,'minimum_free_memory_mib':12000}]
        self.assertEqual(self.choose(rows,{},[],{},4702,100),4)
    def test_recent_capacity_low_and_cooldown_override_free_snapshot(self):
        rows=[{'index':4,'eligible':True,'minimum_free_memory_mib':40000}]
        history=[{'gpus':[{'index':4,'memory_free_mib':4000}]}]
        self.assertIsNone(self.choose(rows,{},history,{},4702,100))
        self.assertIsNone(self.choose(rows,{},[],{4:101},4702,100))
    def test_spread_before_second_tenant_and_cap_at_two(self):
        rows=[{'index':4,'eligible':True,'minimum_free_memory_mib':40000},{'index':5,'eligible':True,'minimum_free_memory_mib':12000}]
        self.assertEqual(self.choose(rows,{'a':{'gpu':4}},[],{},4702,100),5)
        self.assertIsNone(self.choose(rows,{'a':{'gpu':4},'b':{'gpu':4},'c':{'gpu':5},'d':{'gpu':5}},[],{},4702,100))
    def test_saturated_gpu_is_excluded_even_with_memory(self):
        rows=[{'index':4,'eligible':False,'minimum_free_memory_mib':40000}]
        self.assertIsNone(self.choose(rows,{},[],{},4702,100))

if __name__=='__main__':unittest.main()
