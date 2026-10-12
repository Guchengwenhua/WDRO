"""Explicitly restore the layout recorded by the 2026-09-28 move journal."""
import runpy
from pathlib import Path
path=Path('/remote-home/Guilin/DRO/Codex/benchmarks/all_models_all_data_20260928/organize.py')
import sys
sys.path.insert(0,str(path.parent))
runpy.run_path(str(path))['restore']()
