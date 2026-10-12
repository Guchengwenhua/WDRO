# Public open-source chip physical-design inputs (2026-09-12)

## Selected author-maintained source

- Repository: [NVIDIA Research CircuitOps](https://github.com/NVlabs/CircuitOps).
- Pinned commit: `2b56a3b32157e6070df8111282ce22b68f77a519` (upstream commit date 2025-06-24).
- The [author README](https://github.com/NVlabs/CircuitOps/blob/2b56a3b32157e6070df8111282ce22b68f77a519/README.md) describes public sample design files, technology platforms, and pre-existing IR tables extracted with OpenROAD.
- The [IR inventory](https://github.com/NVlabs/CircuitOps/blob/2b56a3b32157e6070df8111282ce22b68f77a519/IRs/README.md) describes the provided technology/design combinations and post-filler instance counts.
- Original files: `sources/real_designs/CircuitOps/IRs/<technology>/<design>/{pin_properties.csv[.gz],net_properties.csv,net_pin_edge.csv,design_properties.csv}` and `designs/<technology>/<design>/6_final.def.gz`.
- Download inventory and exact SHA-256 hashes: `sources/real_designs/download_manifest.json`. Every downloaded blob was additionally checked against the pinned Git repository's blob SHA-1 and byte length. Gzip files are read directly; there is no archive extraction or execution of upstream scripts.
- License evidence: the [upstream LICENSE](https://github.com/NVlabs/CircuitOps/blob/2b56a3b32157e6070df8111282ce22b68f77a519/LICENSE), retained locally, uses Apache-2.0 with a separately included BSD-3-Clause notice for OpenROAD portions. The available Nangate45 platform LICENSE is also retained. These are the repository's notices, not a new claim of licensing all underlying third-party IP or manufacturing PDKs under one license.

The selected physical designs are open-source chip/IP research implementations. They are not evidence of a shipped commercial chip's exact layout, a foundry-qualified PDK, or a successful silicon tapeout. Nangate45 and ASAP7 physical research implementations in particular must not be presented as released production-chip layouts. NVDLA partitions and BlackParrot front/back-end designs are related family members, not independent chip families.

## Selected set

| Platform | Designs | Family treatment |
|---|---|---|
| Nangate45 | AES, Ariane133, BlackParrot back end (`bp_be`), BlackParrot front end (`bp_fe`), GCD, Ibex, JPEG, SweRV | `bp_be` and `bp_fe` share `blackparrot` family |
| ASAP7 | NVDLA partitions a, m, p; UART | All three NVDLA partitions share `nvdla` family |
| SKY130HD | RISC-V32I | Separate RISC-V32I implementation |

The choice was made from available physical data and extraction suitability before observing model evaluation results. GCD remains in the source manifest even if it has no net of the frozen model degrees; lack of an eligible degree is reported rather than filled by splitting or synthetic reconstruction.

## Strict geometry and connectivity adapter

`parsers/real_designs.py` provides `iter_design_records(descriptor)` with one record for every ordinary DEF `NETS` entry. Each record contains `net_id`, original `raw_degree`, `ir_degree`, `valid`, `reasons`, and either the full pin-coordinate array or `points=None`. `parse_design(descriptor)` is a convenience generator for complete nets only; consumers needing exclusion accounting must use `iter_design_records`.

The upstream [Python extractor](https://github.com/NVlabs/CircuitOps/blob/2b56a3b32157e6070df8111282ce22b68f77a519/scripts/python/openroad_helpers.py) uses `ITerm.getAvgXY()` for the global pin centers, so instance origin, orientation, and library pin geometry have already been handled by OpenDB. The adapter divides these DBU coordinates by the original DEF `UNITS DISTANCE MICRONS` value. This is a uniform units conversion. It does not normalize each axis separately, move pins, deduplicate pin coordinates, truncate nets, split large nets, or choose model degree. Frozen project normalization and exact degree selection are the downstream evaluation's responsibility.

**Upstream IR contains instance terminals but omits top-level I/O terminals.** The adapter reads the original final DEF connectivity, compares the complete instance pin set with both pin-properties and net-pin edges, and rejects the entire net when its DEF includes an absent top-level I/O. It never silently lowers the degree. Missing or extra instance terminals, absent net properties, duplicate/colliding identifiers, missing-coordinate `-1` sentinels, non-finite coordinates, and edge disagreements are likewise rejected or cause an explicit parsing error. All ordinary DEF nets must be accounted for; unexplained IR nets absent from the DEF cause an error. Power/ground `SPECIALNETS` are separate, not part of the ordinary signal-net test population; their declared count is retained in metadata. Clock nets in ordinary `NETS` remain in the population.

Detailed per-design counts and every rejected net are written in `sources/real_designs/audits/`. These distinguish raw degree distribution, fully resolved degree distribution, rejected degree distribution, and the target degrees 10/20/30/40/50. The stage-independent source population is not altered after downstream model scores are available.

## Verification evidence

- `sources/real_designs/parser_validation/validation.json`: deleting one pin from a real degree-17 GCD net leaves `raw_degree=17`, reports `ir_degree=16`, rejects the entire net, and preserves the total source net count. An upstream missing-coordinate sentinel also rejects the entire net.
- `sources/real_designs/parser_validation/def_lef_coordinate_crosscheck.json`: an independent reconstruction of all 1,279 GCD instance-terminal coordinates from the original DEF instance placement and Nangate45 LEF rectangle centers agrees with the upstream IR within one DBU. Both N and FS orientations are represented. This check validates the geometry convention on this source design; it is not a formal audit of every upstream design's physical implementation.
- `sources/real_designs/all_design_audits.json`: full original-DEF connectivity audit across the selected set.

## Other sources reviewed

- [TILOS MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement), including [author testcase provenance](https://tilos-ai-institute.github.io/MacroPlacement/Testcases/), exposes Ariane, MemPool, NVDLA, and BlackParrot physical-design research cases. A repository tree snapshot is retained. Many easily found DEF files contain only macro floorplans or clustered proxies, so they were not substituted for full instance-pin nets. The CircuitOps final DEF plus author-exported pin IR is the selected path for this batch.
- [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) includes placed/routed AES, Ibex and GCD regression DEFs. A repository tree snapshot is retained. They were not added as additional independent designs because CircuitOps already provides a broader, internally consistent final-DEF and pin-coordinate source.

This source expansion measures frozen REST model performance on complete selected terminal sets. It does not measure contest routing score, timing closure, congestion, manufacturability, obstacle-aware routing, layer assignment, or performance over an entire chip's full routing workload.

## Completed source inventory

All 13 designs passed full connectivity accounting. The 75 downloaded author files total 602,454,204 bytes. Across 690,475 ordinary DEF nets, 680,012 are fully resolved and 10,463 are explicitly rejected because the upstream IR omits their top-level I/O terminals. There are 3,341 complete nets at the frozen degrees; the degree counts are 10: 2,869, 20: 181, 30: 146, 40: 107, 50: 38. These are source-preparation counts, before downstream normalization or any geometry-validity exclusions.
