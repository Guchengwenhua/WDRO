# Operational handoff

Snapshot UTC: 2026-09-30T14:37:58.472974+00:00

Both geometry experiments completed. Conditional diagnostics: 4096 model-instance evaluations, 322.02 seconds. Perturbation search: 32 centers x 8 candidates plus nominal, two frozen models, 288 offline exact references, 68.77 seconds. GPU4, batch256, float32, memory fraction0.10.

Scientific results: REPORT.md and ATTACK_REPORT.md. Raw statistics, per-instance shards, frozen inputs/selections and completion records: artifacts/ and artifacts/attack/.

Logs: artifacts/run.log and artifacts/attack/run.log. Completed tmux sessions normally exit: geom-uncertainty-20260930 (socket artifacts/geometry.sock), geom-attack-20260930 (socket artifacts/attack.sock). No restart is needed. If an incomplete run is ever resumed, keep identical parameters and use bash launch.sh or bash launch_attack.sh 4 after the previous session exits.

Historical test data and one checkpoint seed per method make these exploratory results; they do not establish training benefit or metric superiority across tasks.
