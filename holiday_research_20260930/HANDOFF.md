# Seven-day execution handoff
Server03 reachable through authorized normal SSH escalation. A separate seven-day controller is actually running, not merely planned.
Session: holiday-research-20260930
Private tmux socket: /remote-home/Guilin/DRO/Codex/holiday_research_20260930/artifacts/holiday.sock
Controller PID at deployment:1337683
Initial state:waiting_existing_scale, active child:null.
Dependency evidence: original scale queuePID1336975, childPID1336977, current geo_s101, no original failures. Controller waits original flock; successful old jobs are never restarted. Existing geometry task remains independent and is a second prerequisite.
First controller launch had a newline literal SyntaxError before any work; corrected and relaunched only this new controller. Original experiments untouched; failed startup log retained.
policy.json fixes cutoff, resource and retry limits. PLAN.json fixes staged acceptance. Branch jobs and numerical gates reside in week_bank_20260930 and week_geometry_20260930. The final deployment receipt lists frozen job IDs and counts.
Recovery instructions: RECOVERY.md. Laptop can disconnect; server tmux remains. Server reboot is different: no boot service created; after server returns the same launch command resumes checkpoints and deduplicates completed work.
Operational final inventory: ACCEPTANCE_INVENTORY.md and artifacts/acceptance_inventory.json. Scientific stage reports remain separate and identify incomplete branches.

Final deployment: frozen66 jobs (45bank,21geometry; includes gates/evaluations/reports, not66 independent training trials). Current controllerPID1338307, statuswaiting_existing_scale. Old idle controller1337683 was deliberately stopped only to load recovery-boundary fixes; its state/history preserved. Original scale PID1336975/child1336977 unchanged. All branches ready, no deployment blockers.
