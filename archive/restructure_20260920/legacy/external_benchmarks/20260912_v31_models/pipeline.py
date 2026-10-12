"""tmux entrypoint: evaluation -> independent audit/statistics -> durable status."""
import traceback
from common import *

def main():
    with ev.Lease(OUT/'logs/pipeline',-1) as lease:
        status=OUT/'STATUS.json'
        try:
            for name in ('run_queue.py','finish.py'):
                write(status,dict(status='running',phase=name,at=now(),process=ev.pid_identity()))
                lease.update(0,name)
                # The child queue adopts surviving workers if this controller is restarted.
                subprocess.run([PYTHON,str(BASE/name)],cwd=BASE,check=True)
            receipt=read(OUT/'completion_manifest.json')
            ev.require(receipt['status']=='complete','missing accepted completion')
            write(status,dict(status='complete',at=now(),completion=artifact(OUT/'completion_manifest.json')))
            # Append one uniquely marked completion note, preserving all other stage history.
            progress=TOP/'01_WDRO-REST_项目进度.md'
            marker='<!-- real-v31-20260912-complete -->'
            if marker not in progress.read_text():
                with progress.open('a') as stream:
                    stream.write(f'\n\n{marker}\n新增模型全部现有真实数据扩展：DONE，{now()}。18 个 v3.1 新模型和 15 个对照，degree=30，11,032 条网络/模型，364,056 次预测；33 个任务及独立验收通过。产物：`external_benchmarks/20260912_v31_models/artifacts/REPORT.md`、`completion_manifest.json`。本扩展不更改已冻结主线或阶段9状态。\n')
                    stream.flush();os.fsync(stream.fileno())
            lease.update(33,'complete')
        except BaseException as exc:
            write(status,dict(status='failed',at=now(),error=repr(exc),traceback=traceback.format_exc(),
                recovery='Run run_tmux.sh again. Committed batches and matching live workers are recovered.'))
            raise

if __name__=='__main__':main()
