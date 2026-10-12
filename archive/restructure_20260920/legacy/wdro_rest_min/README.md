# WDRO-REST minimal implementation

This directory contains the frozen protocol and all new implementation code. It does not overwrite the legacy REST worktree.

Operational and new-chat rules are defined in `../02_新对话执行提示词.md`. All task-generated artifacts, including temporary files and detached-job metadata, must be reachable through logical paths under `DRO/Codex`. GPU/process and resolved-filesystem disk availability must be checked afresh before every batch launch or resume; historical GPU allowlists are snapshots, not permanent allocations. Partially occupied GPUs should also be used when measured run peak memory plus a safety margin fits and compute is not persistently saturated. Long jobs should be resumable and run in a server-side `tmux` session when appropriate.

Run the phase-0 checks with the explicitly selected environment:

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python -m unittest discover -s Codex/wdro_rest_min/tests -v
/remote-home/Guilin/miniconda3/envs/py311/bin/python Codex/wdro_rest_min/scripts/run_preflight.py
```

`run_preflight.py` writes a timestamped manifest containing protocol, code, legacy-asset and checkpoint hashes plus the environment/hardware snapshot. Exit status is successful when development is safe. Add `--require-batch-ready` to enforce the 100 GiB disk gate before any batch run.

Host-local paths and GPU allocation live in `configs/runtime_migrated_server_v1.json`. They do not alter research protocol v1 or existing data hashes. On the migrated server, GPU jobs must use the runtime config's allowed device set and rerun preflight immediately before launch.

`Codex/experiments` is the stable logical artifact path. On the migrated server it resolves to `/attached/remote-home1/Guilin/DRO/Codex/experiments`; relative paths stored in manifests remain unchanged. The byte-for-byte verified pre-switch copy is retained at `Codex/experiments.local_backup_20260904T1127Z` for recovery.

The immutable source of truth is `configs/protocol_v1.json`; its write bits are removed after validation. Protocol changes require a new versioned file and synchronized updates to the project description, execution plan, paper checklist, and progress log.

Stage-1 data and audit commands:

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python Codex/wdro_rest_min/scripts/build_synthetic_data.py
/remote-home/Guilin/miniconda3/envs/py311/bin/python Codex/wdro_rest_min/scripts/build_lstar_cache.py --workers 8
/remote-home/Guilin/miniconda3/envs/py311/bin/python Codex/wdro_rest_min/scripts/build_ispd_protocol_a.py
/remote-home/Guilin/miniconda3/envs/py311/bin/python Codex/wdro_rest_min/scripts/verify_stage1.py
```

All generated NPZ/JSON datasets and caches are immutable. Builders verify existing NPZ artifacts rather than overwriting them. ISPD preparation stores only normalized inputs, inverse metadata, source identity, and invalid counts; model metrics remain locked.
