# Checkpoint bank exploratory continuation

All source changes are confined to this directory. Historical source is imported read-only. No engineering tests or smoke runs are part of this experiment.

Bank members are the validation-selected best checkpoint from each eligible formal d10 run (ERM, mixed ERM, group DRO, hard mining, WDRO; seeds 101/102/103). Ranking uses recorded worst OOD group mean relative regret with recorded nominal feasibility. Duplicate existing validation-manifest identifiers are conservatively represented once; this does not establish identical weights. This is not top-5 historical epochs from one run, since only best/latest survived. Pool paths, original ranking, sizes and mtimes freeze in inputs.json at first launch; test labels are never used for bank choice.

Five arms and two seeds are specified in protocol.json. Actor/critic/Adam/scheduler share the same parent state; training RNG differs by seed, not arm. SeedSequence produces fresh nominal training instances and common random/Sobol geometric candidates without content digests. Candidate family definitions and shortlist remain inherited. Thus this is a new exploratory protocol, not an exact reproduction of historical training streams. Teacher models stay eval and construction restores RNG. All greedy evaluation pads batches to 256. Original critic architecture remains in every arm; bank targets may be negative and are not clipped, which can challenge a nonnegative critic.

B=min teacher feasible length generally exceeds OPT. Therefore C/B-1 underestimates C/OPT-1 and may change adversarial rankings. Training reference substitution occurs in both search and sampled policy target. Cached original GeoSteiner references remain for full validation and frozen test evaluation. Historical teachers were oracle-assisted; no claim of wholly oracle-free learning is justified. Test labels are loaded only after training. Validation determines the best checkpoint and refreshed bank, without new solves. The offline candidate diagnostic compares gap, proxy bias, Spearman and argmax agreement and never feeds results back to training.

## Outputs and resume

artifacts is a symbolic link to /attached/remote-home1/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts. Each arm/seed saves config, train/validation JSONL, latest.pt every10updates, best.pt, cached offline oracle chunks, atomic NPZ per-instance results and completion metadata. Latest stores model, critic, optimizer, scheduler, RNG, lambda, bank metadata and step. Resume uses the identical command. Completed arms skip. SIGTERM requests a checkpoint after the current update. Evaluation files write atomically; offline exact-reference chunks survive interruption. RESULTS.md is regenerated from actual completed arms by summarize.py after each queued job; failures remain incomplete and have launch logs/queue exit codes.

## Launch and resume

Run `bash launch.sh` from this directory. It starts detached tmux session checkpoint-bank-20260930 on the explicit socket artifacts/bank.sock, with queue output in artifacts/queue.log. Only restart after the previous session has exited. The queue holds an exclusive nonblocking artifacts/queue.lock; duplicate queues are refused. Existing completed jobs skip and interrupted jobs resume their latest checkpoint.

Inspect session with `tmux -S /remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts/bank.sock list-sessions`. The launch wrapper invokes run_queue.sh, which sources run_env.sh before Python initialization. Default physical GPU is1, per-process CUDA memory fraction0.1, CPU math threads1. All caches and temporary paths are inside artifacts. Queue runs ten jobs sequentially. Do not start standalone jobs concurrently with the queue.

artifacts/queue_status.json records parent PID, start time, current job, child PID, finished jobs, failures and status. Each arm heartbeat.json records update progress, GPU, timestamp, observed current-segment mean wall time per update and estimated remaining training time every10updates; estimate includes observed periodic validation costs but excludes final test/offline diagnostics. A segment before its first validation may underestimate later wall time. Launch logs retain actual update times and failures.

No experiment has been run by the implementing subagent. Throughput and conclusions must come from actual launch logs/results.

## First-launch failure

The original local queue.py shadowed the Python standard-library queue module during dependency imports, causing each child to attempt the queue lock and exit. It was renamed to run_pipeline.py and run_queue.sh updated. Original failed launch logs and queue history are retained; no completed training results came from that first launch.
