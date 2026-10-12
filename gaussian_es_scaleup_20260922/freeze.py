"""Freeze only after executable engineering evidence and a fresh admission."""
import json
import platform
import subprocess
import sys
import time
import torch
from common import ROOT,ART,PROTOCOL,atomic_json,artifact,source_bindings,dual_settings
from resources import preflight

def main():
    path=ART/'manifests/source_freeze.json'
    if path.exists():raise RuntimeError('freeze already exists; use the existing protocol')
    unit=subprocess.run([sys.executable,'-m','unittest','-v','test_engine'],cwd=ROOT,capture_output=True,text=True)
    (ART/'tests/unit.log').write_text(unit.stdout+unit.stderr)
    if unit.returncode:raise RuntimeError('CPU tests failed')
    gate=json.loads((ART/'tests/gpu_gate.json').read_text())
    if gate['status']!='PASS':raise RuntimeError('GPU gate failed')
    # Reuse the independent result validator on actual engineering evaluation shards.
    import report
    engineering=json.loads((ART/'runs/engineering_eval/complete.json').read_text())
    saved=report.P;report.P=dict(saved,test_count=8,pressure_count=2)
    try:
        for kind in ('nominal','pressure'):report.load_eval(engineering['task'],kind)
    finally:report.P=saved
    atomic_json(ART/'tests/result_validator.json',dict(status='PASS',source=artifact(ART/'runs/engineering_eval/complete.json')))
    admission=preflight()
    if not admission['admitted']:raise RuntimeError('no GPU currently admissible; rerun freeze when available')
    freeze=dict(at=time.time(),sources=source_bindings(),protocol=PROTOCOL,
        python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        platform=platform.platform(),dual_by_degree={str(d):dual_settings(d) for d in PROTOCOL['degrees']},
        gpu_gate=artifact(ART/'tests/gpu_gate.json'),unit_test=artifact(ART/'tests/unit.log'),
        result_validator=artifact(ART/'tests/result_validator.json'),storage=artifact(ART/'manifests/storage.json'))
    atomic_json(path,freeze)
    print(json.dumps(dict(status='FROZEN',source_files=len(freeze['sources']),admitted=[g['index'] for g in admission['admitted']])))

if __name__=='__main__':main()
