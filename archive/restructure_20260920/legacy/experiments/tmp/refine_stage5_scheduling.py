from pathlib import Path
p=Path('/remote-home/Guilin/DRO/Codex/wdro_rest_min/scripts/stage5_pipeline.py');s=p.read_text()
insert='''def choose_gpu(rows,active,history,blocked,need,now):
    eligible=[]
    for row in rows:
        gpu=row['index'];count=sum(a['gpu']==gpu for a in active.values())
        minfree=min([row['minimum_free_memory_mib']]+[g['memory_free_mib'] for h in history for g in h['gpus'] if g['index']==gpu])
        if row['eligible'] and minfree>need and count<2 and now>blocked.get(gpu,0):eligible.append((count,-minfree,gpu))
    return min(eligible)[2] if eligible else None

'''
s=s.replace('def run_queue(tasks,lease,phase):',insert+'def run_queue(tasks,lease,phase):')
a=s.index("                    need=profile['maximum_reserved_mib']+1024+2048;eligible=[]")
b=s.index("                        cmd=['bash'",a)
s=s[:a]+'''                    need=profile['maximum_reserved_mib']+1024+2048
                    gpu=choose_gpu(snap['gpu_decisions'],active,history,blocked,need,time.time())
                    if gpu is not None:
                        t=pending[0];rid=t['run_id'];attempts[rid]=attempts.get(rid,0)+1
                        if attempts[rid]>10:raise RuntimeError('repeated recovery failures; inspect '+rid)
''' +s[b:]
p.write_text(s)
p=Path('/remote-home/Guilin/DRO/Codex/wdro_rest_min/tests/test_stage5_operations.py');s=p.read_text()
s=s.replace("if __name__=='__main__':unittest.main()",'''class ResourceGateTests(unittest.TestCase):
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

if __name__=='__main__':unittest.main()''')
p.write_text(s)
