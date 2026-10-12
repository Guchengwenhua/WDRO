"""Import frozen v3.1 implementation without modifying its source or artifacts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PARENT = Path('/remote-home/Guilin/DRO/Codex/adversarial_v3_1_20260912')
sys.path.insert(0, str(PARENT))
EXP = ROOT / 'artifacts'

