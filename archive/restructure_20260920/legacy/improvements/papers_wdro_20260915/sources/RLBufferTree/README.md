# Achieving Simultaneous Buffering and Steiner Tree Synthesis via Harmonic-Based Reinforcement Learning

Official implementation for the paper "Achieving Simultaneous Buffering and Steiner Tree Synthesis via Harmonic-Based Reinforcement Learning".

## Summary
This project extends an RL-based Rectilinear Steiner Tree (RST) framework to jointly synthesize Steiner topology and predict buffer candidates using a harmonic-function parameterization. The code implements an actor–critic model that outputs routing/topology decisions together with buffering information.

## Repository layout (key files)
- train.py — training script (uses models2 and utils2)
- test.py — evaluation / test script
- README.md — this file
- models2/
  - actor_critic.py — actor and critic architectures
  - self_attn.py — encoder / attention modules
  - utils.py — model helper utilities
- utils2/
  - rsmt_utils5.py — evaluation utilities (interfaces to Steiner evaluation)
  - log_utils.py — logging helpers
- data/, nets_data/, nets_data/* — example nets and datasets used for evaluation
- logs/ — TensorBoard logs (created by train.py)
- saved_model/sinModel/ — model checkpoints saved by training
- results/ — evaluation outputs

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- tqdm
- TensorBoard



## Quick start

1. Train (example):
```bash
python train.py --degree 10 --batch_size 256 --num_batches 10000 --learning_rate 5e-5
```
- Models and checkpoints saved under saved_model/sinModel/.
- TensorBoard logs written to logs/sinModel/<degree>/

2. Evaluate / test:
```bash
python test.py --degree 10
```

3. View training curves:
```bash
tensorboard --logdir=logs
```

## Notes
- train.py expects utilities in models2/ and utils2/. Evaluator implemented in utils2/rsmt_utils5.py performs batch evaluation of routing/buffering decisions.
- Saved checkpoints: saved_model/sinModel/<degree>.pt and saved_model/sinModel/<degree>b.pt
- Random seeds are set in train.py via the `--seed` argument for reproducibility.


## Acknowledgments
The implementation builds on the REST RSMT work:
Liu, J., Chen, G., & Young, E. F. (2021). REST: Constructing Rectilinear Steiner Minimum Tree via Reinforcement Learning. DAC 2021.
