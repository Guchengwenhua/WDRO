"""Freeze launch sources only after verified tests, restarts, GPU profiles and full fixture."""
import geometry as G
import engine as E
from common import ROOT,read,now

def main():
    out=ROOT/'artifacts/tests'; cpu=read(out/'final_cpu.json'); gpu=read(out/'final_gpu.json'); e2e=read(out/'e2e.json')
    if any(x['status']!='PASS' for x in (cpu,gpu,e2e)): raise RuntimeError('Missing engineering gate')
    if not all(cpu['checks'].values()) or not all(gpu['resume']['checks'].values()): raise RuntimeError('Restart mismatch')
    for name,h in e2e['source_sha256'].items():
        if G.sha256(ROOT/name)!=h or G.sha256(out/'e2e_fixture'/name)!=h: raise RuntimeError('Source changed since fixture: '+name)
    text=(out/'unit_final.log').read_text()
    if not text.rstrip().endswith('OK') or 'Ran 27 tests' not in text: raise RuntimeError('Unit tests failed or incomplete')
    if len(gpu['profiles'])!=4: raise RuntimeError('Missing shape profiles')
    for profile in gpu['profiles']:
        if profile['new_updates']!=32 or profile['ledger']['test_rows'] or profile['ledger']['oracle_calls']: raise RuntimeError('Invalid profile')
        if profile['peak_reserved_bytes']>6*2**30: raise RuntimeError('Admission budget below measured peak')
    evidence=[out/'unit_final.log',out/'final_cpu.json',out/'final_gpu.json',out/'e2e.json',out/'e2e_fixture/artifacts/report/complete.json']
    G.atomic_json(out/'preflight.json',dict(status='PASS',time=now(),scope='Engineering launch gate; formal results pending',
        unit_tests_passed=True,cpu_resume_exact=True,gpu_resume_exact=True,gpu_batch64_pilot_passed=True,
        all_oracle_calls_zero=True,end_to_end_pipeline_passed=True,protocol_sha256=G.sha256(ROOT/'protocol.json'),
        source_sha256={str(x.relative_to(ROOT)):G.sha256(x) for x in E.sources()},
        evidence_sha256={str(x.relative_to(ROOT)):G.sha256(x) for x in evidence},profiles=gpu['profiles']))
    print('PASS: sources and launch gates frozen')

if __name__=='__main__': main()
