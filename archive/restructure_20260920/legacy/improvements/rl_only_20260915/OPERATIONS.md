# Pure reinforcement-learning GPU experiment operations

All commands run from `/remote-home/Guilin/DRO/Codex/improvements/rl_only_20260915`.
All generated files, caches, checkpoints, and logs stay below this directory.

This branch trains from fresh Actor initialization using only the evaluated length of
its own sampled/greedy topology as reward. Runtime admission does not supply reference
tree costs, externally solved labels, or checkpoints from the paused earlier branch.
The six methods are `erm_critic`, `erm_rollout`, `wdro_random`, `wdro_es`, `wdro_critic`,
and `wdro_rl`; the first two full runs are `wdro_random_s101` and `erm_critic_s101`.

## Queue and admission

`queue.json` contains 6 CUDA smoke runs and 18 full runs (6 methods, seeds 101/102/103).
Each full run depends on its method's smoke acceptance. CPU diagnostics cannot satisfy
either CUDA gate. Existing valid completion artifacts skip work; all invocations pass
`--resume` and the trainer verifies checkpoint compatibility.
Before dispatch, the scheduler additionally requires the six-method CPU diagnostic
report and process summary, zero oracle and reference-solver calls, report/run binding hashes, and the PASS
restart audit. The protocol, theory documents, CPU report manifest, smoke process summary,
and restart audit are frozen with source hashes. These are engineering prerequisites;
formal results still require all CUDA tasks and the final independent report.
The passing environment test transcript is also frozen as launch provenance.

The default scheduler samples all 8 GPUs every 20 seconds. It starts at most one local
worker per GPU only after 3 consecutive samples have utilization at most 50% and at
least 8192 MiB free (6144 MiB provisional worker requirement plus 2048 MiB margin).
The worker checks the gate again immediately before starting Python training.
Free VRAM alone is insufficient when a GPU is persistently busy. No process belonging
to another user is signalled, stopped, or otherwise manipulated.

Up to 8 workers each use 3 distinct CPUs from the scheduler's inherited affinity
(31 CPUs were visible during development); BLAS/OpenMP thread limits match the CPU
allocation. The queue waits when disk free space drops below 20 GiB or GPU telemetry
is unavailable. Current resource facts are written at every sample.

Smoke runs retain memory measurements in trainer artifacts. The provisional admission
threshold is part of this initial configuration: if observed peaks exceed its worker
allowance, inspect and update the frozen protocol/queue before further admission.
Do not silently reduce batch size or training budget.
The trainer also sets a PyTorch allocator limit to 12.5% of its physical GPU's memory;
the 2048 MiB admission margin covers additional context/allocator overhead. A small
smoke batch does not establish the full-batch peak, so the cap also applies to full runs.

## Validate and start

```bash
bash run_env.sh -m unittest -v test_scheduler
bash run_env.sh scheduler.py --dry-run
tmux -S "$PWD/artifacts/operations/rl_only.sock" new-session -d -s rl_only_20260915 "cd '$PWD' && bash run_env.sh scheduler.py > artifacts/operations/scheduler.log 2>&1"
```

Create `artifacts/operations` before the tmux command if it does not exist. Start only
after `protocol.json`, `train.py`, and `report.py` are final and CPU checks pass. The
scheduler hashes root Python/shell sources, the queue, and `protocol.json` at launch;
changed source/config hashes prevent an accidental incompatible restart.
Admission pauses if the live scheduler detects any source/config change after launch.

## Inspect

```bash
tmux -S "$PWD/artifacts/operations/rl_only.sock" list-sessions
bash run_env.sh -c 'import json; from pathlib import Path; d=json.loads(Path("artifacts/operations/queue.json").read_text()); print(json.dumps({k:d[k] for k in ("status","at","pending","running","failures","eta")},indent=2))'
tail -n 30 artifacts/operations/scheduler.log
```

`resource_latest.json` and `resource_history.jsonl` record GPU memory, utilization,
visible compute processes, disk capacity, CPU affinity, and resolved paths.
Each attempt also has an immutable preflight snapshot, command, input hashes, GPU UUID,
PID/start ticks/boot ID, GPU lease, result record, and wrapper heartbeat. Trainer logs
are `<task_id>.log`; trainer heartbeats/checkpoints live in that task's output directory.

Queue `eta.capacity_wait_seconds` remains null: waiting for other users to release
compute cannot be predicted from spare VRAM. Completed full CUDA runs provide observed
training durations and a single-GPU active-time estimate. This estimate excludes
capacity waiting, variable sharing, smoke, and report time.
While a CUDA run is active, its heartbeat supplies per-update elapsed time and remaining
update estimates. An empty `nvidia-smi` process table is explicitly recorded as unknown
visibility and never interpreted as an idle GPU.

## Recovery and bounded retries

Inspect the tmux session and current process identities first. Do not launch a second
scheduler while `queue.lock` is held. A live scheduler is independent of an SSH client.
If it exited, rerun the same command through the same environment and manifest.

The scheduler holds an OS file lock; each detached wrapper independently owns its GPU
lease and writes a durable exit record. A restarted scheduler adopts wrappers whose
PID, process-start ticks, and boot ID match. If a wrapper was lost while its trainer is
still live, its recorded child identity preserves the GPU reservation and prevents a
duplicate launch. An explicit scheduler SIGTERM stops admission and preserves existing
workers; it does not stop training jobs. Reconnection needs no restart if tmux is live.

Failed tasks receive at most one retry (two total attempts), resuming `latest.pt` when
available. An immediate resource-gate deferral consumes no failure retry. Exhausted
smoke failures block dependent full runs. Source/config changes never silently resume.
Paused trainer output or a zero exit code without its valid `complete.json` is not
completion. GPU runs require `acceptance=accepted_gpu`; CUDA smoke requires `gpu_smoke`.
The scheduler verifies completion's method/seed/smoke bindings, current protocol and
source/dependency hashes, and SHA256 for all listed artifacts including both checkpoints,
results, and run manifest. The final report performs its own independent statistical audit.

After all 24 tasks pass, `report.py` runs automatically. Only successful report exit
plus `artifacts/report/complete.json` produces `operations/exit.json` with
`status=complete`. Report failure is explicit and a compatible scheduler restart retries
the report. Inspect exited workers, released locks, report hashes, and acceptance
statistics before describing the experiment as finished.
