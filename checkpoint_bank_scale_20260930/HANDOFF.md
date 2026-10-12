# Active experiment handoff

Launched 2026-09-30 15:20:39 UTC in tmux checkpoint-bank-scale, explicit socket /remote-home/Guilin/DRO/Codex/checkpoint_bank_scale_20260930/artifacts/scale.sock. Queue PID1336975; first child1336977. GPU5, memory fraction0.1, CPU1. Queue order: each seed101/102/103, arms geo,k3,lagged3, each2000updates/batch256. No d30 scheduled.

Last observed first job geo_s101 update35/2000. Actual train events update1 to35:1.292619seconds/update. Remaining1965training updates approximately2540seconds (42.3min), excluding future periodic validation and final full test. Initialization-inclusive heartbeat atstep30 estimated46.6min. Later-arm throughput is not yet measured; a reliable total9-job estimate is unavailable. This observation is not a promise of completion time.

Paths: artifacts/queue_status.json; artifacts/queue.log; artifacts/launch_geo_s101.log; artifacts/geo_s101/heartbeat.json; artifacts/geo_s101/logs/events.jsonl; artifacts/geo_s101/checkpoints/latest.pt. Last observed exact_train_nets35840, teacher_forward_nets0. No failures observed. Original trainer atomic checkpoint every100updates plus step0; includes added reference-bank state for resumed teachers.

Authorized small correctness comparison passed in equivalence.json (CPU8 same inputs/RNG, original search/loss versus Geo reference adapter, all compared arrays and model state after one update equal). Scientific algorithm mapping and unavoidable bookkeeping/cache/runtime differences are in README.md and protocol.json. This is original-loop new-baseline research; the earlier continued200step pilot is separate.

Inspect sessions: tmux -S /remote-home/Guilin/DRO/Codex/checkpoint_bank_scale_20260930/artifacts/scale.sock list-sessions. Only after the session has exited, restart via bash /remote-home/Guilin/DRO/Codex/checkpoint_bank_scale_20260930/launch.sh. Existing complete jobs skip; incomplete latest resumes. Failed job stops queue rather than silently proceeding. No constant polling helper was installed. Read actual logs after completion and report measured final-versus-best results with independent cached test references.
