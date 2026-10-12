"""Wait on the existing queue lock once, generate one comparison, then exit."""
import fcntl
from pathlib import Path
from summarize_comparison import main
ROOT=Path(__file__).resolve().parent
with (ROOT/'artifacts/queue.lock').open('a') as lock:
    print('Waiting for the active queue lock to be released',flush=True)
    fcntl.flock(lock,fcntl.LOCK_EX)
    print('Queue lock acquired; generating one comparison report',flush=True)
    main()
    print('Wrote COMPARISON.md and COMPARISON.json; exiting',flush=True)
