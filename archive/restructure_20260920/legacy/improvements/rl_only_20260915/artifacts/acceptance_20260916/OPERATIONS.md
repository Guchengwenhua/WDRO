# Background final acceptance

This watcher performs CPU postprocessing only. It never starts, stops, migrates or
restarts any training process. The existing training queue remains responsible for
its last task and the automatic experiment report.

## Start after the partial audit passes

Run from `/remote-home/Guilin/DRO/Codex/improvements/rl_only_20260915` after
`artifacts/acceptance_20260916/audit_acceptance.py` is final:

```bash
bash run_env.sh artifacts/acceptance_20260916/watch_acceptance.py --check-only
tmux -S "$PWD/artifacts/operations/rl_only.sock" new-session -d -s rl_only_acceptance_20260916 "cd '$PWD' && bash run_env.sh artifacts/acceptance_20260916/watch_acceptance.py > artifacts/acceptance_20260916/watcher.log 2>&1"
```

The check-only command reads readiness without creating a watcher or launching an
audit. The persistent watcher checks at 20-second intervals (configurable up to
30 seconds) and writes `watcher_status.json` with identity and heartbeat.

## Completion and failure gates

Before auditing, all of the following must agree with the bound scheduler launch:

- `artifacts/operations/exit.json` has `status=complete`.
- `artifacts/report/complete.json` has `status=complete`, `acceptance=accepted_gpu`,
  and the current protocol hash.
- Every task is recorded complete; no task is running or pending.
- The exact scheduler PID/start ticks/boot ID has exited and its lock is released.

The watcher then runs this exact command with no visible GPU:

```bash
bash run_env.sh artifacts/acceptance_20260916/audit_acceptance.py --full
```

It requires exit code zero and `acceptance_final.json` reporting exactly
`status=PASS`. Other status values are rejected. It records that result's SHA256 in `watcher_exit.json`.
The audit log is `watcher_audit.log`. The independent full auditor owns all numerical,
checkpoint, final-report and completed-run acceptance checks.

Failed queue/report status, exhausted retries, changed launch/source hashes, stale
active scheduler heartbeat, or a dead scheduler without both completion markers
produce `watcher_exit.json` with `status=failed` and a concrete error. No repair or
training restart follows. A normal main report phase may take longer than a training
heartbeat interval; the watcher continues waiting while that scheduler remains alive.

## Identity, source binding and recovery

`watcher.lock` prevents concurrent watchers. The CPU audit inherits this lease, so
an abruptly terminated watcher cannot lead another watcher to duplicate an active
audit. `watcher_binding.json` records the exact main launch, protocol, frozen input
hashes, watcher/auditor/independent-statistics source hashes, and command.
`statistics_review.py` is required and bound alongside `audit_acceptance.py`; the full
auditor invokes it to independently recompute complete-run statistics. Do not edit these inputs after
launch. Review files and acceptance outputs are outside the frozen experiment code.

On reconnect, inspect the tmux session, `watcher_status.json`, exact recorded process
identity and lock before considering recovery. The watcher explicitly refuses to
follow an unexpected training-scheduler restart. A signal while waiting exits without
touching training; a signal during the audit lets that CPU audit finish.

Started on 2026-09-16 at 04:20 UTC in tmux session
`rl_only_acceptance_20260916`, using the original `rl_only.sock` socket.
The live PID/start identity, held lease, source bindings, and repeating waiting
heartbeat were verified after launch. See `handoff.json` and `watcher_status.json`.
Do not run the start command again while this watcher or its audit is active.

## Verification evidence

`test_watch_acceptance.py` contains seven isolated CPU tests. They cover ordinary
waiting, unexpected scheduler exit, failed postprocessing, a wrong report, source
drift, the scheduler finalization race, and source binding changes. The finalization
test also verifies that only `status=PASS` is accepted from the independent auditor.
The binding test checks that a changed or missing `statistics_review.py` is rejected.
`watcher_tests.log` preserves the test output; `watcher_tests.json` records the command,
exit code and SHA256 of the watcher and test sources used for that run.
