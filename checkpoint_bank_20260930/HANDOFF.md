# Operational handoff

Snapshot UTC: 2026-09-30T14:37:58.472974+00:00

Actual launch: CUDA_VISIBLE_DEVICES=5 bash launch.sh
Session: checkpoint-bank-20260930
Socket: artifacts/bank.sock
Queue log: artifacts/queue.log
Queue status: artifacts/queue_status.json
Worker logs: artifacts/launch_<arm>_s<seed>.log
Each run: artifacts/<arm>_s<seed>/ (latest.pt, heartbeat.json, train.jsonl, validation.jsonl, test arrays, offline reference chunks)

Status at snapshot:
```json
{
  "pid": 1335060,
  "start": 1790778899.1988158,
  "current_job": "k1_s101",
  "child_pid": 1335184,
  "finished": [
    "geo_s101"
  ],
  "failures": [],
  "status": "running"
}
```

Current worker heartbeat:
```json
{
  "step": 110,
  "steps": 200,
  "gpu": "5",
  "time": 1790779076.7913384,
  "segment_start_step": 1,
  "segment_seconds": 39.69338275492191,
  "mean_wall_seconds_per_update": 0.3608489341356538,
  "remaining_seconds_estimate": 32.47640407220884,
  "estimate_scope": "remaining training/periodic validation at observed segment wall rate; excludes final test and offline diagnostic"
}
```

All ten tasks are sequential: two seeds x GeoSteiner / K1 / K3 / K5 / refresh3. 200 updates each, training batch64, greedy scoring/evaluation batch256. Actual GPU5, memory fraction0.10, one CPU thread. The first measured Geo arm was around0.48 seconds/update; this is not yet a reliable whole-queue ETA because bank sizes add teacher inference and final evaluations. See fresh heartbeats for measured estimates.

After a stopped/exited session only, resume with CUDA_VISIBLE_DEVICES=5 bash launch.sh. Exclusive queue lock rejects duplicate queues. Completed arms skip; interrupted arms restore model/critic/optimizer/scheduler/RNG/lambda/bank/update from atomic latest.pt. Keep parameters and input metadata unchanged. No manual stop or restart is needed while running.

RESULTS.md is regenerated after each completed/failed arm. A queue exit is not proof of scientific completion: review failures and each complete.json. Original first-launch module-shadowing failure logs remain in place; repaired before training began.

Artifacts actual storage: /attached/remote-home1/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts, linked from this project.

## Latest handoff snapshot

UTC: 2026-09-30T14:42:12.407770+00:00

{
  "pid": 1335060,
  "start": 1790778899.1988158,
  "current_job": "refresh3_s101",
  "child_pid": 1335357,
  "finished": [
    "geo_s101",
    "k1_s101",
    "k3_s101",
    "k5_s101"
  ],
  "failures": [],
  "status": "running"
}

{
  "step": 30,
  "steps": 200,
  "gpu": "5",
  "time": 1790779329.555894,
  "segment_start_step": 1,
  "segment_seconds": 12.909251220524311,
  "mean_wall_seconds_per_update": 0.430308374017477,
  "remaining_seconds_estimate": 73.1524235829711,
  "estimate_scope": "remaining training/periodic validation at observed segment wall rate; excludes final test and offline diagnostic"
}

Completed successful jobs took about79-133 seconds each. Remaining queue is roughly10-15 minutes at this throughput; shared GPU load and bank size can change this.

A separate one-shot report session checkpoint-bank-report-20260930 is active on the same artifacts/bank.sock. It is blocked on the existing queue.lock, with log artifacts/comparison.log; after training exits it writes COMPARISON.md and COMPARISON.json once, then exits. These reports prioritize final update200 paired differences versus Geo; best checkpoint step0 ties are not evidence of equivalent training.
