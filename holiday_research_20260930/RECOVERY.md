# Recovery and handoff
Directory /remote-home/Guilin/DRO/Codex/holiday_research_20260930.
Session holiday-research-20260930; socket artifacts/holiday.sock; controller log artifacts/controller.log; atomic state artifacts/state.json; event history artifacts/events.jsonl.
Launch/resume command: bash /remote-home/Guilin/DRO/Codex/holiday_research_20260930/launch.sh
First inspect tmux and state. Do not relaunch an active session. A controller.lock prevents concurrent execution even if duplicate launch attempted.
SSH/laptop disconnect does not stop Server03 tmux. Server reboot ends tmux; no new system service/automatic boot registration is installed. After server recovers, run the same launch command. A persisted child PID plus kernel start token avoids duplicate jobs if controller alone restarts. Completed markers skip successful work; interrupted workers resume full training state or atomic prediction/reference shards.
State waiting_existing_scale is expected while the original queue owns its flock. Controller never kills or restarts that live queue. After lock release it only recovers incomplete transiently interrupted jobs, subject to limits; deterministic original errors are recorded and block dependent training.
Do not edit frozen_jobs.json or scientific job manifests to force resume after a protocol mismatch. Such mismatches stop rather than silently mixing experiments.
During own worker cutoff, controller sends SIGTERM to its directly launched child only and allows180seconds for atomic checkpoint exit. If it does not exit, controller records blocked_live_child and launches nothing else; no force-kill or unrelated process termination.
Individual stage scripts, checkpoints and completion markers are listed in the frozen manifest. Stages may be legitimately unrun if development gate, resources or remaining time are insufficient.

Controller recovery fix: graceful pause leaves future pending jobs pending; CPU gates/reports may proceed below artifact quota thresholds when each disk still has>=1GiB, so quota stops GPU growth without blocking small summaries. Ordinary GPU jobs retain full10/200GiB free-space floors. Original scale orphan child is also checked before incomplete-run recovery. A failed existing branch blocks only its corresponding week branch; the other branch may still proceed.
