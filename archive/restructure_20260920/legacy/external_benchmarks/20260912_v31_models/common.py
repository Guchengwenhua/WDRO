"""Paths and durable utilities for the separately authorized real-data extension."""
from pathlib import Path
import json
import os
import subprocess
import time
import evaluate as ev

BASE = Path(__file__).absolute().parent
OUT = BASE / 'artifacts'
TOP = ev.TOP
OLD = TOP / 'external_benchmarks/20260912_public'
V31 = TOP / 'revisions/adversarial_v3_1_20260912/Codex'
REV = ev.REVISION
PYTHON = '/remote-home/Guilin/miniconda3/envs/py311/bin/python'
ENV = ev.PACKAGE / 'scripts/stage3_env.sh'
NEW = ('random_top4', 'critic_calibrated4', 'rl_frozen4', 'rl_joint4', 'random64', 'es64')
CONTROLS = ('erm', 'mixed_erm', 'hard_mining', 'group_dro', 'wdro_rest')
METHODS = NEW + CONTROLS
SEEDS = (101, 102, 103)
read, write, artifact, checked, now = ev.read_json, ev.atomic_json, ev.artifact, ev.checked, ev.now

def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    return [dict(zip(('index','total_mib','free_mib','utilization'), map(int, row.split(','))))
            for row in text.strip().splitlines()]

def task_id(model):
    return f"{model['method']}_d{model['degree']}_s{model['seed']}"

def worker_command(model, gpu, output=None, input_path=None, reference_path=None, receipt=None):
    return ['bash', str(ENV), PYTHON, str(BASE/'evaluate.py'),
            '--method', model['method'], '--degree', str(model['degree']), '--seed', str(model['seed']),
            '--gpu', str(gpu), '--input', str(input_path or OUT/'inputs/degree_30.npz'),
            '--reference', str(reference_path or OUT/'inputs/reference_30.npz'),
            '--reference-manifest', str(receipt or OUT/'inputs/reference_30.json'),
            '--model-index', str(OUT/'models_frozen.json'),
            '--output', str(output or OUT/'results'/task_id(model))]
