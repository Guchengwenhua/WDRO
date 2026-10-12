"""Exercise final status gates without touching real running experiment state."""
import importlib.util,json,os,tempfile
from pathlib import Path
import closeout

def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))

def run_test():
    original=closeout.ROOT
    with tempfile.TemporaryDirectory(dir=original/'artifacts',prefix='closeout_test_') as d:
        project=Path(d)/'project';root=project/'study';exp=root/'artifacts'
        (root/'docs').mkdir(parents=True);(project/'summaries').mkdir(parents=True)
        (root/'docs/STATUS.md').write_text('DOING\n')
        (project/'summaries/STATUS.md').write_text('| tmp2 提出的 Critic 搜索七方向 | DOING | old |\n')
        closeout.ROOT=root;closeout.EXP=exp
        paths=[exp/'fixed'/f'test_s{s}' for s in (101,102,103)]+[exp/'fixed/validation_s101']
        paths += [exp/'training'/f'{v}_d30_s{s}' for s in (101,102,103) for v in ('refresh32_raw','refresh8_raw','refresh32_ema','refresh8_ema')]
        for p in paths:
            put(p/'complete.json',{});put(p/'exit.json',{'complete':True})
            put(p/'launch.json',{'pid':999999999,'start_ticks':'0'});(p/'task.lock').touch()
        put(exp/'audit/completion_audit.json',{'status':'pass','formal_tasks':16,'published_outputs':[]})
        put(exp/'operations/queue.json',{'running':[],'pending':['unfinished'],'failures':[],'completed':list(range(16))})
        put(exp/'operations/exit.json',{'pending':[],'failures':[]})
        put(exp/'operations/launch.json',{'pid':999999999,'start_ticks':'0'})
        put(exp/'freeze.json',{'files':[]});(exp/'operations/queue.lock').touch()
        try:closeout.main()
        except AssertionError:pass
        else:raise AssertionError('accepted unfinished queue')
        assert (root/'docs/STATUS.md').read_text()=='DOING\n'
        put(exp/'operations/queue.json',{'running':[],'pending':[],'failures':[],'completed':list(range(16))})
        put(exp/'operations/launch.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]})
        try:closeout.main()
        except AssertionError:pass
        else:raise AssertionError('accepted live scheduler')
        assert not (exp/'audit/final_delivery.json').exists()
        put(exp/'operations/launch.json',{'pid':999999999,'start_ticks':'0'})
        real_replace=closeout.os.replace;sequence=[]
        def record(src,dst):
            sequence.append(Path(dst));real_replace(src,dst)
        closeout.os.replace=record
        try:closeout.main()
        finally:closeout.os.replace=real_replace
        assert sequence[0]==exp/'audit/final_delivery.json'
        assert 'DONE' in (root/'docs/STATUS.md').read_text()
        assert all(closeout.sha(Path(a['path']))==a['sha256'] for a in json.loads((exp/'audit/final_delivery.json').read_text())['artifacts'])
    closeout.ROOT=original;closeout.EXP=original/'artifacts'
    print('PASS: incomplete/live process rejection, receipt-before-DONE, final hashes')

if __name__=='__main__':run_test()
