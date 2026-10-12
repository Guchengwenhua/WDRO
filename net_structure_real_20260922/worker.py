"""Per-job exclusion lock; all mutations occur inside the new experiment."""
from exp_common import *
import fcntl,subprocess
if __name__=='__main__':
    name=sys.argv[1];(ART/'locks').mkdir(exist_ok=True)
    with (ART/'locks'/f'{name}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(json.dumps(dict(pid=os.getpid(),starttime=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],time=utc())));lock.flush()
        result=subprocess.run([CFG['python'],*sys.argv[2:]],cwd=ROOT)
        sys.exit(result.returncode)
