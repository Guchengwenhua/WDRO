# Source selection manifest

The original workspace at `/data2/hqm/gwsun` contained research snapshots,
backups, third-party baselines, environments, generated data, and experiment
outputs. This release keeps only the following paper-critical implementation:

- `mspd/graduate/models/actor_critic_judge.py` ->
  `models/selector_conductor.py`
- `mspd/graduate/models/self_attn.py` -> `models/attention.py`
- `mspd/graduate/models/utils.py` -> `models/layers.py`
- `mspd/graduate/models/evaluate.py` -> rewritten as `models/evaluate.py`
- `mspd/graduate/STT.cpp` -> `cpp/mspd.cpp`
- `mspd/graduate/STT_thread.cpp` and `.h` -> `cpp/optional/mspd_mt.*`
- `mspd/graduate/train.py` -> cleaned and parameterized as `train.py`

Engineering edits made during cleanup:

- renamed generic modules to match the paper's Selector-Conductor terminology;
- replaced absolute paths and fixed GPU indices with command-line parameters;
- removed checkpoint loading assumptions and all trained weights;
- added a portable source build for the MSPD Python extension;
- kept Boost multithreading optional instead of shipping a platform-specific
  precompiled `.so`;
- added validation, a minimal demo, tests, dependency metadata, and exclusions.

Intentionally omitted categories:

- `save_*`, `tb*`, `.pt`, event files, figures, results, and generated data;
- notebooks and visualization/analysis scripts;
- duplicate `graduate`, `clcopy`, `parametersCopy`, and backup trees;
- archives such as `mspd.tar.gz`, ZIP files, and repository bundles;
- Python/Conda environments, CUDA installers, IDE servers, and caches;
- BST-DME, Timing-RST, REST, SALT, GeoSteiner, and other comparison baselines.

Validation performed after assembly:

- compiled `cpp/mspd.cpp` into a fresh Python extension;
- matched its full 11-alpha output to the original `pdnew` extension on ten
  random routing instances;
- checked Actor/Critic forward shapes and finite MSPD objective values;
- completed one Actor-Critic optimization/evaluation/checkpoint cycle on CPU.
