# RLBufferTree component pilot patches

Upstream: https://github.com/USTC-DIAG-LAB/RLBufferTree/tree/c4f921a45a46eaec07b1f4afe1b5a73ec5a38e31 . Original source and per-file hashes remain in `../sources/RLBufferTree/`.

1. Local package imports; remove unused turtle/matplotlib/scipy imports and plotting method. Do not run upstream training scripts or checkpoints.
2. Source index zero is fixed in the upstream actor. Remove its categorical log probability, because the source is not sampled.
3. Replace the deterministic phase output by 16 categorical phase bins. Add each phase action's log probability to the policy gradient. This changes the action parameterization, and supplies a true score-function gradient for the final phase. No pathwise differentiation through buffer insertion is claimed.
4. Replace `atan(sin/cos)` with `atan2(sin,cos)`, apply period modulo, skip complete harmonic cancellation, and exclude duplicate endpoint buffer nodes.
5. Continue each sine wave across corners and split segments; convert reversed segments to a common orientation before summing phases. The original helper reused the same phase after a split. Keep original BFS projection of the geometric graph into a rooted timing tree.
6. Fix a sink-node capacitance early return that omitted its children. Check sink connectivity and iterate actual node keys.
7. Use the paper-inspired objective `(max_delay_seconds * 1e18 + 0.25 * inserted_buffers) / 10`. Upstream training returns a different four-term objective. Keep upstream local positive-delay-gain insertion, which has no buffer-count penalty in its internal selection and is not the Shi-Li optimal DP stated in the paper.
8. Fixed sink capacitance `0.5e-15`, source index 0, and the upstream degree-dependent RC constants. A normalized-coordinate controlled test, not a calibrated industrial timing claim.

The continuous-to-categorical phase adaptation and local-buffer-solver discrepancy must be stated in any result. No original published speed, improvement, or optimal-buffering guarantee transfers automatically.
