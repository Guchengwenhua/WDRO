"""Persistent postprocessing of an existing pipeline; never launches training."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/remote-home/Guilin/miniconda3/envs/py311/bin/python'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return {'path': str(path), 'sha256': sha(path)}


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+f'.partial.{os.getpid()}.{time.time_ns()}')
    with temp.open('w') as handle:
        handle.write(obj if isinstance(obj,str) else json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())
    temp.replace(path)


def process(record):
    pid=int(record['pid'])
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:
        return {'pid':pid,'active':False,'state':'exited'}
    same=str(record.get('process_start_ticks',fields[19]))==fields[19]
    return {'pid':pid,'process_start_ticks':fields[19],'state':fields[0],
            'active':same and fields[0] not in ('Z','X'),'identity_matches':same}


class Finalizer:
    def __init__(self, root=ROOT, poll_seconds=15):
        self.root=Path(root);self.exp=self.root/'experiments';self.folder=self.exp/'autofinish'
        self.poll_seconds=poll_seconds;self.identity=None

    def initialize(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        identity={'schema_version':1,'pipeline_launch':binding(self.exp/'pipeline/launch.json'),
            'protocol':binding(self.root/'protocol.json'),
            'sources':[binding(self.root/'diagnostics'/name) for name in
                       ('autofinish.py','interpret_results.py','delivery_audit.py')],
            'scope':'postprocess_existing_pipeline_only; no pipeline or research-code restart',
            'human_scientific_review_complete':False}
        path=self.folder/'identity.json'
        if path.exists():require(read(path)==identity,'Finalizer identity changed; preserve existing evidence and review before another launch')
        else:atomic(path,identity)
        self.identity=identity

    def verify_identity(self):
        for item in [self.identity['pipeline_launch'],self.identity['protocol']]+self.identity['sources']:
            require(sha(item['path'])==item['sha256'],f"Bound source/protocol/pipeline launch changed: {item['path']}")

    def state(self, phase, status='running', **details):
        value={'phase':phase,'status':status,'at':time.time(),'pid':os.getpid(),
               'human_scientific_review_complete':False,**details}
        atomic(self.folder/'state.json',value)
        atomic(self.folder/'heartbeat.json',value)

    def waiting_status(self, phase, extra=''):
        atomic(self.root/'DELIVERY_STATUS.md',
            '# 自动交付状态\n\n'
            f'当前状态：{phase}。{extra}\n\n'
            '后台只等待现有实验并生成机器汇总、独立交付审计；不启动或重启训练。\n\n'
            '人工中文科学解读尚未完成，现有研究结论草稿保留待复核。\n\n'
            '- [状态与心跳](experiments/autofinish/state.json)\n'
            '- [失败说明（如存在）](experiments/autofinish/failure.json)\n'
            '- [现有实验状态](STATUS.md)\n')

    def pipeline_ready(self):
        self.verify_identity()
        folder=self.exp/'pipeline';launch=read(folder/'launch.json')
        state=read(folder/'state.json') if (folder/'state.json').exists() else {}
        alive=process(launch)['active']
        require(state.get('status') not in ('failed','blocked'),f'Existing pipeline failed: {state}; no restart attempted')
        if not (folder/'complete.json').exists():
            require(alive,'Existing pipeline exited without successful completion; no restart attempted')
            return False,'等待现有 pipeline 完成 '+str(state.get('phase',''))
        require(read(folder/'complete.json').get('status')=='pass','Pipeline completion is not PASS')
        if alive:return False,'等待已完成的 pipeline 控制器退出'
        require((folder/'exit.json').exists() and read(folder/'exit.json').get('completed') is True,'Pipeline lacks successful exit evidence')
        manifest=self.root/'manifests/all_completion.json'
        require(manifest.exists(),'Pipeline passed without all_completion.json')
        done=read(manifest)
        require(done.get('status')=='pass' and done.get('all_experiments_complete') is True and done.get('phase')=='all','Final experiment audit is not complete PASS')
        require(done.get('phases')=={'fixed':{'tasks':30,'cells':1080},'joint':{'tasks':18,'cells':216}},'Final experiment grid mismatch')
        require(done.get('protocol_sha256')==self.identity['protocol']['sha256'],'Final experiment protocol mismatch')
        for phase in ('fixed','joint'):
            queue=self.exp/'queue'/phase
            require((queue/'complete.json').exists() and read(queue/'complete.json').get('status')=='pass',f'{phase} queue lacks completion')
            if process(read(queue/'launch.json'))['active']:return False,f'等待 {phase} queue 退出'
            require((queue/'controller_exit.json').exists(),f'{phase} queue lacks exit evidence')
        monitor=self.exp/'monitor'
        if (monitor/'launch.json').exists():
            if process(read(monitor/'launch.json'))['active']:return False,'等待监控器最后一次采样并退出'
            require((monitor/'exit.json').exists(),'Monitor lacks exit evidence')
        return True,'实验完成，进程退出，准备自动汇总'

    def output_manifest(self, step):
        return self.exp/('interpretation/all/extraction.json' if step=='interpretation' else 'diagnostics/delivery_audit.json')

    def output_valid(self, step):
        path=self.output_manifest(step)
        if not path.exists():return False
        document=read(path)
        if document.get('status')!='pass':return False
        completion=binding(self.root/'manifests/all_completion.json')
        if step=='interpretation':
            if document.get('phase')!='all' or not document.get('all_experiments_complete'):return False
            if document.get('script_sha256')!=sha(self.root/'diagnostics/interpret_results.py'):return False
            if not any(r.get('path')==completion['path'] and r.get('sha256')==completion['sha256'] for r in document.get('source_bindings',[])):return False
            return all(Path(r['path']).is_file() and sha(r['path'])==r['sha256'] for r in document.get('artifacts',[]))
        # The dedicated auditor has just recursively checked all scientific files.
        # Recovery checks its immutable report receipt, without repeating that scan.
        return document.get('full_audit_performed') is True and document.get('production_completion')==completion

    def step(self, name):
        require(name in ('interpretation','delivery'),'Only postprocessing steps are permitted')
        out=self.folder/name;out.mkdir(exist_ok=True)
        receipt=out/'complete.json';input_binding=binding(self.root/'manifests/all_completion.json')
        if receipt.exists():
            done=read(receipt)
            require(done.get('status')=='pass' and done.get('input')==input_binding,'Completed postprocessing input changed')
            item=done['output'];require(Path(item['path']).is_file() and sha(item['path'])==item['sha256'],'Completed postprocessing output changed')
            require(self.output_valid(name),'Completed postprocessing report is no longer valid')
            self.state(name,'complete',resumed_skip=True);return
        child_path=out/'launch.json'
        if child_path.exists():
            child=read(child_path)
            require(child['input']==input_binding,'Orphan postprocessing input changed')
            while process(child)['active']:
                self.verify_identity();self.state(name,child_pid=child['pid'],adopted=True)
                time.sleep(self.poll_seconds)
            require(self.output_valid(name),f'{name} process exited without valid output; prior attempt retained, no automatic rerun')
            exit_code=None
        else:
            script='interpret_results.py' if name=='interpretation' else 'delivery_audit.py'
            arguments=['--phase','all'] if name=='interpretation' else ['--mode','full']
            command=['bash',str(self.root/'run_env.sh'),PYTHON,'-u',str(self.root/'diagnostics'/script)]+arguments
            log=out/'process.log'
            with log.open('w') as handle:
                child_process=subprocess.Popen(command,cwd=self.root,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                current=process({'pid':child_process.pid})
                child={'pid':child_process.pid,'process_start_ticks':current.get('process_start_ticks'),
                       'at':time.time(),'command':command,'input':input_binding,'log':str(log)}
                # A process that already exited is never adopted by PID alone.
                if child['process_start_ticks'] is None:child['process_start_ticks']='already_exited'
                atomic(child_path,child)
                while child_process.poll() is None:
                    self.verify_identity();self.state(name,child_pid=child_process.pid)
                    time.sleep(self.poll_seconds)
                exit_code=child_process.returncode
            atomic(out/'exit.json',{'at':time.time(),'pid':child['pid'],'code':exit_code,'log':binding(log)})
            require(exit_code==0,f'{name} failed with code {exit_code}; see {log}; no research restart attempted')
            require(self.output_valid(name),f'{name} exited successfully but output verification failed')
        atomic(receipt,{'status':'pass','at':time.time(),'input':input_binding,'output':binding(self.output_manifest(name)),
                        'exit_code':exit_code,'adopted_after_parent_restart':exit_code is None,
                        'validation_scope':'report/source bindings; extraction CSV hashes; deep scientific hashes are checked once by delivery_audit'})
        self.state(name,'complete')

    def summary(self):
        location=self.exp/'interpretation/all'
        with (location/'fixed_pair_condition_summary.csv').open() as handle:fixed=list(csv.DictReader(handle))
        with (location/'joint_nominal_gate.csv').open() as handle:joint=list(csv.DictReader(handle))
        counts={}
        for row in fixed:
            key=(row['left'],row['right']);values=counts.setdefault(key,[0,0,0])
            for i,column in enumerate(('stable_positive_conditions','stable_negative_conditions','remaining_conditions')):values[i]+=int(row[column])
        lines=['# 自动结果汇总（待人工科学解读）','','这些计数由已通过审计的 CSV 自动汇总；没有进行人工结论复核，也没有新增统计检验。','',
               'A 完成 30 个任务、1080 个评价格；B 完成 18 个任务、216 行评价统计。','',
               '| A 的等搜索预算配对（左减右） | 稳定正向条件数 | 稳定负向条件数 | 其余 |','|---|---:|---:|---:|']
        for (left,right),values in counts.items():lines.append(f'| {left} − {right} | {values[0]} | {values[1]} | {values[2]} |')
        retained=sum(int(r['best_update'])==0 for r in joint)
        lines += ['',f'B 有 {retained}/18 个任务保留起始模型（best_update=0）；该结果保留在完整表中。','',
            'A 的正向只表示同父模型、degree、分布和 lambda 条件内的惩罚攻击分数更高；条件数量不能当作独立实验重复或全局胜率。训练费用、padding 和恢复额外调用另计。',
            'B 仅验证 degree30 的最小一步策略交替，并非完整多步 MARL。CI 条件于本次三个 seed 和固定中心，未校正多重比较。有限搜索与合成数据不构成连续 Wasserstein 证书或 ISPD 外推证明。','',
            '工程交付通过不等同于方法有效；三路线的最终中文科学结论仍待人工复核。','',
            '- [完整自动提取表](experiments/interpretation/all/extraction.json)',
            '- [独立交付审计](experiments/diagnostics/delivery_audit.json)',
            '- [研究数据报告](RESULTS.md)','']
        atomic(self.root/'AUTO_RESULTS.md','\n'.join(lines))

    def publish(self):
        self.summary()
        atomic(self.root/'DELIVERY_STATUS.md',
            '# 自动交付状态\n\n自动实验后处理与独立交付审计已完成。人工中文科学解读尚待下次复核。\n\n'
            '保留已有研究结论草稿；本状态不代表人工研究结论已标记 DONE。\n\n'
            '- [自动结果汇总（待人工解读）](AUTO_RESULTS.md)\n'
            '- [自动交付凭证](experiments/autofinish/complete.json)\n'
            '- [完整实验审计](manifests/all_completion.json)\n'
            '- [独立交付审计](experiments/diagnostics/delivery_audit.json)\n'
            '- [自动提取数据](experiments/interpretation/all/extraction.json)\n')
        files=[self.root/'diagnostics/autofinish.py',self.folder/'identity.json',self.output_manifest('interpretation'),
               self.output_manifest('delivery'),self.root/'AUTO_RESULTS.md',self.root/'DELIVERY_STATUS.md',
               self.root/'manifests/all_completion.json']
        complete={'status':'pass','automatic_delivery_complete':True,'human_scientific_review_complete':False,
                  'at':time.time(),'protocol':self.identity['protocol'],'pipeline_launch':self.identity['pipeline_launch'],
                  'scope':'Automatic evidence extraction and delivery audit only; human research interpretation pending',
                  'artifacts':[binding(p) for p in files]}
        atomic(self.folder/'complete.json',complete);self.state('all','complete',automatic_delivery_complete=True)

    def run(self):
        self.initialize()
        while True:
            ready,message=self.pipeline_ready()
            if ready:break
            self.state('waiting_pipeline',message=message)
            self.waiting_status(message)
            time.sleep(self.poll_seconds)
        self.waiting_status('正在生成自动结果和独立交付审计')
        self.step('interpretation');self.step('delivery');self.publish()


def self_test():
    class Tests(unittest.TestCase):
        def setUp(self):
            self.temp=tempfile.TemporaryDirectory(dir=ROOT/'experiments/diagnostics',prefix='autofinish_test_')
            self.root=Path(self.temp.name);self.exp=self.root/'experiments';self.calls=[]
            for name in ('autofinish.py','interpret_results.py','delivery_audit.py'):
                path=self.root/'diagnostics'/name;path.parent.mkdir(exist_ok=True);path.write_text(name)
            atomic(self.root/'protocol.json',{'frozen':True});atomic(self.exp/'pipeline/launch.json',{'pid':987654321,'process_start_ticks':'1'})
            atomic(self.exp/'pipeline/state.json',{'status':'running','phase':'fixed'})
            self.runner=Finalizer(self.root,.001);self.runner.initialize()
        def tearDown(self):self.temp.cleanup()
        def ready(self):
            atomic(self.exp/'pipeline/complete.json',{'status':'pass'});atomic(self.exp/'pipeline/exit.json',{'completed':True})
            atomic(self.root/'manifests/all_completion.json',{'status':'pass','phase':'all','all_experiments_complete':True,
                'protocol_sha256':sha(self.root/'protocol.json'),'phases':{'fixed':{'tasks':30,'cells':1080},'joint':{'tasks':18,'cells':216}}})
            for phase in ('fixed','joint'):
                folder=self.exp/'queue'/phase;atomic(folder/'launch.json',{'pid':987654322,'process_start_ticks':'1'});atomic(folder/'complete.json',{'status':'pass'});atomic(folder/'controller_exit.json',{})
        def test_failed_pipeline_never_launches(self):
            atomic(self.exp/'pipeline/state.json',{'status':'failed'})
            with patch('subprocess.Popen') as launch:
                with self.assertRaises(RuntimeError):self.runner.pipeline_ready()
                launch.assert_not_called()
        def test_dead_pipeline_never_restarts(self):
            with patch('subprocess.Popen') as launch:
                with self.assertRaises(RuntimeError):self.runner.pipeline_ready()
                launch.assert_not_called()
        def test_waiting_requires_process_identity(self):
            with patch(__name__+'.process',return_value={'active':True}),patch('subprocess.Popen') as launch:
                self.assertFalse(self.runner.pipeline_ready()[0]);launch.assert_not_called()
        def test_identity_drift_blocks(self):
            atomic(self.root/'protocol.json',{'changed':True})
            with self.assertRaises(RuntimeError):self.runner.verify_identity()
        def test_fake_children_and_valid_resume(self):
            self.ready();self.assertTrue(self.runner.pipeline_ready()[0]);runner=self.runner;calls=self.calls
            class Child:
                pid=987654323;returncode=0
                def poll(self):return 0
            def spawn(command,**kwargs):
                calls.append(command);self.assertTrue(kwargs['start_new_session'])
                self.assertNotIn('adv_v3.pipeline',' '.join(command));self.assertNotIn('adv_v3.queue',' '.join(command))
                step='interpretation' if any(str(x).endswith('interpret_results.py') for x in command) else 'delivery'
                obj={'status':'pass','phase':'all','all_experiments_complete':True,'script_sha256':sha(self.root/'diagnostics/interpret_results.py'),
                    'source_bindings':[binding(self.root/'manifests/all_completion.json')],'artifacts':[],
                    'full_audit_performed':True,'production_completion':binding(self.root/'manifests/all_completion.json')}
                atomic(runner.output_manifest(step),obj);return Child()
            with patch('subprocess.Popen',side_effect=spawn):
                runner.step('interpretation');runner.step('delivery');runner.step('interpretation');runner.step('delivery')
            self.assertEqual(len(calls),2)
        def test_failed_orphan_is_not_rerun(self):
            self.ready();folder=self.runner.folder/'interpretation';folder.mkdir()
            atomic(folder/'launch.json',{'pid':987654323,'process_start_ticks':'1','input':binding(self.root/'manifests/all_completion.json')})
            with patch('subprocess.Popen') as launch:
                with self.assertRaises(RuntimeError):self.runner.step('interpretation')
                launch.assert_not_called()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    evidence={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
              'real_pipeline_restarts':0,'GPU_used':False,'source':binding(Path(__file__)),'at':time.time()}
    atomic(ROOT/'experiments/diagnostics/autofinish_tests.json',evidence)
    return 0 if result.wasSuccessful() else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--poll-seconds',type=float,default=15)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:raise SystemExit(self_test())
    require(args.poll_seconds>=1,'poll-seconds must be >=1')
    runner=Finalizer(poll_seconds=args.poll_seconds);runner.folder.mkdir(parents=True,exist_ok=True)
    with (runner.folder/'autofinish.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own=process({'pid':os.getpid()});atomic(runner.folder/'launch.json',dict(own,at=time.time(),command=sys.argv,source=binding(Path(__file__))))
        try:runner.run()
        except BaseException as exc:
            if isinstance(exc,KeyboardInterrupt):raise
            failure={'status':'blocked','at':time.time(),'pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),
                     'research_code_modified':False,'pipeline_restart_attempted':False,'human_scientific_review_complete':False}
            atomic(runner.folder/f'failure_{time.time_ns()}.json',failure);atomic(runner.folder/'failure.json',failure)
            runner.state('blocked','failed',error=repr(exc));runner.waiting_status('自动收尾受阻',f'原因：{exc}。失败证据保留，未重启实验。')
            raise
        finally:
            atomic(runner.folder/'exit.json',{'pid':os.getpid(),'at':time.time(),'automatic_delivery_complete':(runner.folder/'complete.json').exists(),
                                            'human_scientific_review_complete':False})
            fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()
