# Adaptive Hub-Node Selection for Cost-Skew Tradeoff

This directory contains the compact source release for the DAC 2025 paper
“To Tackle Cost-Skew Tradeoff: An Adaptive Learning Approach for Hub Node
Selection.” 

## Included core components

| File | Role in the paper |
| --- | --- |
| `models/selector_conductor.py` | Selector, adaptive Conductor, and Critic networks |
| `models/attention.py` | Multi-head self-attention encoders |
| `models/layers.py` | Embedding, pointer, stopping, and attention-pooling layers |
| `models/evaluate.py` | Cost/skew objectives and Python interface to MSPD |
| `cpp/mspd.cpp` | MSPD/STT construction and cost/skew evaluation |
| `cpp/optional/mspd_mt.*` | Original Boost-based batch evaluator (not built by default) |
| `train.py` | Actor-Critic training from random initialization |
| `demo.py` | Minimal end-to-end smoke example; it does not represent trained quality |

Node 0 is the root. Input tensors have shape `[batch, degree, 2]`, normally
with coordinates in `[0, 1]`. The Selector returns at most `max_hubs` indices;
zeros after the adaptive stop decision are padding and are ignored by MSPD.

## Installation

Python 3.8 or newer and a C++14 compiler are required. The following builds
the small pybind11 MSPD extension directly from source:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the checks and the minimal example with:

```bash
python -m pytest tests
python demo.py
```

`pytest` is only needed for running the tests and can be installed separately.

## Training from scratch

No trained model is included. A short CPU sanity run is:

```bash
python train.py \
  --degree 10 \
  --max-hubs 5 \
  --objective 2 \
  --device cpu \
  --batch-size 4 \
  --eval-size 8 \
  --num-batches 2 \
  --log-interval 1
```

For real experiments, use a CUDA device and increase the batch count and
batch size. Objective presets are:

- `1`: 0.5 wirelength + 0.5 skew
- `2`: 0.25 wirelength + 0.75 skew
- `3`: 0.75 wirelength + 0.25 skew

A floating-point value in `[0, 1]` can instead be supplied as the wirelength
weight in Eq. (3) of the paper. Checkpoints produced by new runs are written
under `outputs/`; that directory is ignored by Git.

The default evaluator is deliberately single-process so the public source does
not depend on precompiled Python extensions or a particular Boost runtime. It
uses the same MSPD construction as the original batch evaluator, but training
is slower. This release favors portability and auditability over the original
experiment workspace's throughput optimizations. The original multithreaded
source is retained under `cpp/optional/` for readers who want to reproduce the
high-throughput evaluator; it requires Boost.Thread/Boost.Asio and a separate
pybind11 build step.



## Attribution and licensing note

`cpp/mspd.cpp` derives from the TILOS-AI-Institute Multi-Source Prim-Dijkstra
implementation. Its BSD 3-Clause license is preserved in
`third_party/MSPD-BSD-3-Clause.txt`.

