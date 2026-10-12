"""Import frozen v3.1 implementation without modifying its source or artifacts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PARENT = PROJECT / 'revisions/adversarial_v3_1_20260912/Codex'
sys.path.insert(0, str(PARENT))
EXP = ROOT / 'artifacts'

