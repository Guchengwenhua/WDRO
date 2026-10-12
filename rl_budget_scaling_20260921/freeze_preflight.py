"""Bind the passed engineering evidence to the exact launch sources."""
from pathlib import Path
import geometry as G
from common import ROOT,read,now

def main():
    t=ROOT/'artifacts/tests'
    cpu=read(t/'final_cpu/cpu_resume.json'); gpu=read(t/'final_gpu/gpu_resume_pilot.json'); e2e=read(t/'e2e_result.json')
    if any(x['status']!='PASS' for x in (cpu,gpu,e2e)): raise RuntimeError('Incomplete preflight')
    for source,h in e2e['source_sha256'].items():
        if G.sha256(ROOT/source)!=h: raise RuntimeError('End-to-end validated source changed')
        if G.sha256(t/'e2e_fixture'/source)!=h: raise RuntimeError('End-to-end fixture source differs')
    log=(t/'unit_final.log').read_text()
    if 'Ran 21 tests' not in log or not log.rstrip().endswith('OK'): raise RuntimeError('Unit suite did not pass')
    for r in cpu['rows']+gpu['rows']:
        if not all(r['checks'].values()): raise RuntimeError('Resume equality failed')
    for pilot in gpu['pilots']:
        if pilot['new_updates']!=32 or pilot['ledger']['test_rows'] or pilot['ledger']['oracle_calls']:
            raise RuntimeError('Pilot budget or isolation failure')
    evidence=[t/'unit_final.log',t/'final_cpu/cpu_resume.json',t/'final_gpu/gpu_resume_pilot.json',t/'e2e_result.json',
              t/'e2e_fixture/artifacts/report/complete.json']
    sources=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+list((ROOT/'vendor/models').glob('*.py'))+[ROOT/'protocol.json',ROOT/'docs/THEORY_PROTOCOL.md']
    G.atomic_json(t/'preflight.json',dict(status='PASS',time=now(),unit_tests_passed=True,cpu_resume_exact=True,
        gpu_resume_exact=True,gpu_batch64_pilot_passed=True,all_oracle_calls_zero=True,end_to_end_pipeline_passed=True,
        scope='Engineering launch gate only; no formal outcome established',
        protocol_sha256=G.sha256(ROOT/'protocol.json'),source_sha256={str(x.relative_to(ROOT)):G.sha256(x) for x in sources},
        evidence_sha256={str(x.relative_to(ROOT)):G.sha256(x) for x in evidence},pilots=gpu['pilots']))
    print('PASS: frozen engineering launch gate')

if __name__=='__main__': main()
