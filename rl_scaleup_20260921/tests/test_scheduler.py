"""Scheduler tests: fake procfs/resources/artifacts; no worker or GPU launch."""
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('research_scheduler_under_test', ROOT/'scheduler.py')
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)


class TestScheduler(unittest.TestCase):
    def setUp(self):
        tmp = ROOT/'artifacts/tmp'
        tmp.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=tmp)
        self.root = Path(self.temp.name)
        self.protocol = json.loads((ROOT/'tests/legacy_protocol_fixture.json').read_text())
        self.expected = dict(protocol_sha256='protocol', code_sha256={'code':'digest'}, dependencies={'model':'digest'})
        self.resources = dict(max_workers=4, min_free_mib=6144, max_admission_util=85,
                              reserve_disk_gib=20, saturated_duty_fraction=.25)

    def tearDown(self):
        self.temp.cleanup()

    def snap(self, util=20, free=8000, disk=30):
        return dict(time=100, gpus=[dict(index=i,uuid=f'GPU-{i}',free_mib=free,util=util) for i in range(8)],
                    load=[1,1,1],free_bytes=disk*2**30)

    def fake_complete(self, task=None):
        task = task or dict(id='warmup_s201',kind='warmup',method='erm',seed=201,deps=[])
        out = self.root/'artifacts/runs'/task['id']
        out.mkdir(parents=True,exist_ok=True)
        identity = dict(pid=12345,start_ticks='999',boot_id='boot-a',device='cuda:0',physical_gpu_index=2,CUDA_VISIBLE_DEVICES='2')
        binding = dict(deepcopy(self.expected),parents={})
        for dep in task['deps']:
            parent = self.root/'artifacts/runs'/dep/'latest.pt'
            parent.parent.mkdir(parents=True,exist_ok=True)
            parent.write_bytes(b'parent checkpoint')
            binding['parents'][str(parent)] = s.sha256(parent)
        run = dict(run_id=task['id'],smoke=False,identity=identity,binding=binding)
        (out/'run.json').write_text(json.dumps(run))
        for name in ['latest.pt','progress.jsonl','actual_calls.jsonl']+(['results.npz'] if task['kind']!='warmup' else []):
            (out/name).write_bytes(b'test content')
        complete = dict(status='complete',run_id=task['id'],kind=task['kind'],method=task['method'],seed=task['seed'],
                        step=self.protocol[task['kind']+'_updates'],smoke=False,device='cuda:0',identity=identity,
                        peak_allocated_bytes=100,calls={'oracle_calls':0},binding=binding,
                        artifacts_sha256={p.name:s.sha256(p) for p in out.iterdir()})
        (out/'complete.json').write_text(json.dumps(complete))
        return task,out,complete

    def validate(self, task):
        return s.validate_complete(task,self.protocol,self.root,self.expected,identity_fn=lambda pid:None)

    def test_all_51_dependencies_and_bounded_retries(self):
        ts=s.tasks(self.protocol)
        self.assertEqual(len(ts),51)
        ready=s.ready_tasks(ts,set(),{}, {})
        self.assertEqual([t['kind'] for t in ready],['warmup']*3)
        done={'warmup_s201','fixed_ppo_s201'}
        ready=s.ready_tasks(ts,done,{'fixed_mappo_s201':{}},{'fixed_reinforce_s201':3,'fixed_bestof_ppo_s201':2})
        ids={t['id'] for t in ready}
        self.assertIn('coev_population_ppo_s201',ids)
        self.assertIn('fixed_bestof_ppo_s201',ids)
        self.assertNotIn('fixed_reinforce_s201',ids)
        self.assertNotIn('fixed_mappo_s201',ids)
        self.assertNotIn('coev_mappo_s201',ids)

    def test_resource_limits_and_one_worker_per_card(self):
        snap=self.snap()
        running={f't{i}':{'gpu':i} for i in range(3)}
        capacity,cards,_=s.admission(snap,[snap],running,self.resources)
        self.assertEqual(capacity,1)
        self.assertEqual([g['index'] for g in cards],[3,4,5,6,7])
        self.assertTrue(all(g['duty_fraction']==1 for g in cards))
        self.assertEqual(s.admission(self.snap(free=6143),[snap],{},self.resources)[1],[])
        self.assertEqual(s.admission(self.snap(disk=20),[snap],{},self.resources)[0],0)
        self.assertEqual(s.admission(self.snap(disk=22),[snap],{},self.resources)[0],2)
        hot=self.snap();hot['load'][0]=61
        self.assertEqual(s.admission(hot,[hot],{},self.resources)[0],0)
        oversized=dict(self.resources,max_workers=99)
        self.assertEqual(s.admission(snap,[snap],{},oversized)[0],8)

    def test_saturated_sharing_needs_three_samples_and_duty_cap(self):
        snap=self.snap(util=100)
        for count in (1,2):
            self.assertEqual(s.admission(snap,[snap]*count,{},self.resources)[1],[])
        cards=s.admission(snap,[snap]*3,{},self.resources)[1]
        self.assertEqual(len(cards),8)
        self.assertTrue(all(g['duty_fraction']==.25 for g in cards))
        high=dict(self.resources,saturated_duty_fraction=.8)
        self.assertTrue(all(g['duty_fraction']==.25 for g in s.admission(snap,[snap]*3,{},high)[1]))
        old=self.snap(util=50)
        self.assertTrue(all(g['duty_fraction']==1 for g in s.admission(snap,[old,old,snap],{},self.resources)[1]))

    def test_proc_identity_start_ticks_boot_and_zombie(self):
        proc=self.root/'proc'
        (proc/'42').mkdir(parents=True)
        (proc/'sys/kernel/random').mkdir(parents=True)
        (proc/'sys/kernel/random/boot_id').write_text('boot-a\n')
        fields=['S']+['0']*18+['9876']+['0']*10
        (proc/'42/stat').write_text('42 (name with ) paren) '+' '.join(fields))
        value=s.ident(42,proc)
        self.assertEqual(value,dict(pid=42,start_ticks='9876',boot_id='boot-a'))
        self.assertFalse(s.same_identity(value,dict(value,boot_id='boot-b')))
        self.assertFalse(s.same_identity(value,dict(value,start_ticks='9877')))
        fields[0]='Z';(proc/'42/stat').write_text('42 (zombie) '+' '.join(fields))
        self.assertIsNone(s.ident(42,proc))
        self.assertIsNone(s.ident(99,proc))

    def test_stale_lock_file_is_safe_held_lock_blocks(self):
        path=self.root/'run.lock';path.write_text('old stale metadata')
        self.assertFalse(s.lock_held(path))
        with path.open('r') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertTrue(s.lock_held(path))
        self.assertFalse(s.lock_held(path))
        path=self.root/'scheduler.lock';path.write_text(json.dumps(dict(pid=44,start_ticks='1',boot_id='old')))
        current=dict(pid=55,start_ticks='2',boot_id='new')
        with patch.object(s,'ident',side_effect=lambda pid:current if pid==s.os.getpid() else None):
            lock=s.acquire_scheduler_lock(path)
            self.assertTrue(s.lock_held(path));lock.close()
        self.assertEqual(json.loads(path.read_text())['recovered_stale_metadata']['pid'],44)

    def test_prior_live_workers_refused_reused_pid_allowed(self):
        task=dict(id='warmup_s201',kind='warmup',method='erm',seed=201,deps=[])
        ops=self.root/'artifacts/operations';ops.mkdir(parents=True)
        saved=dict(pid=44,start_ticks='1',boot_id='boot')
        (ops/'queue.json').write_text(json.dumps(dict(running={'a':dict(pid=44,identity=saved)},attempts={})))
        with self.assertRaisesRegex(RuntimeError,'Prior live worker'):
            s.ensure_no_prior_workers([task],self.root,identity_fn=lambda pid:saved,discover_fn=lambda root:[])
        s.ensure_no_prior_workers([task],self.root,identity_fn=lambda pid:dict(saved,start_ticks='2'),discover_fn=lambda root:[])
        with self.assertRaisesRegex(RuntimeError,'unrecorded live worker'):
            s.ensure_no_prior_workers([task],self.root,identity_fn=lambda pid:None,discover_fn=lambda root:[saved])

    def test_complete_requires_correct_metadata_hashes_and_cuda(self):
        task,out,complete=self.fake_complete()
        self.assertEqual(self.validate(task),(True,'validated'))
        mutations=[('device','cpu'),('step',2),('seed',202),('smoke',True),('method','wrong'),('peak_allocated_bytes',0)]
        for key,value in mutations:
            damaged=deepcopy(complete);damaged[key]=value
            (out/'complete.json').write_text(json.dumps(damaged))
            self.assertFalse(self.validate(task)[0],key)
        (out/'complete.json').write_text(json.dumps(complete))
        (out/'latest.pt').write_bytes(b'corrupt checkpoint')
        self.assertIn('Artifact hash mismatch',self.validate(task)[1])

    def test_complete_source_parent_and_process_validation(self):
        task=dict(id='fixed_ppo_s201',kind='fixed',method='ppo',seed=201,deps=['warmup_s201'])
        task,out,complete=self.fake_complete(task)
        self.assertTrue(self.validate(task)[0])
        mutated=deepcopy(complete);mutated['binding']['protocol_sha256']='old'
        (out/'complete.json').write_text(json.dumps(mutated))
        self.assertIn('protocol_sha256 mismatch',self.validate(task)[1])
        (out/'complete.json').write_text(json.dumps(complete))
        ok,reason=s.validate_complete(task,self.protocol,self.root,self.expected,identity_fn=lambda pid:complete['identity'])
        self.assertEqual((ok,reason),(False,'worker_still_live'))
        parent=self.root/'artifacts/runs/warmup_s201/latest.pt';parent.write_bytes(b'new parent')
        self.assertIn('Parent checkpoint binding mismatch',self.validate(task)[1])

    def test_source_freeze_detects_mutation(self):
        file=self.root/'source.py';file.write_text('original')
        path=self.root/'artifacts/operations/source_freeze.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(files={'source.py':s.sha256(file)},protocol_sha256='metadata')))
        self.assertTrue(s.verify_source_freeze(self.root))
        file.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'Source freeze mismatch'):
            s.verify_source_freeze(self.root)


if __name__=='__main__':
    unittest.main(verbosity=2)
