"""Generate the fixed dependency graph, no training or test inference."""
import geometry as G
from common import ROOT,read

def main():
    p=read(ROOT/'protocol.json'); tasks=[]; training=[]
    for arm in p['arms']:
        for seed in p['seeds']:
            key=f'{arm}_s{seed}'; training.append(key)
            out=f'artifacts/runs/{key}'
            tasks.append(dict(id=key,kind='training',args=['worker.py','--mode','train','--arm',arm,'--seed',str(seed),'--outdir',out],
                complete=out+'/complete.json',checkpoint=out+'/latest.pt',resume_args=['--resume'],
                depends_on=[f'M_s{seed}'] if arm in ('ME','MG','SG') else [],
                cpu_threads=3,required_free_mib=6144,verify_artifacts=True,
                completion_checks=dict(acceptance='training_complete'),binding_checks=dict(arm=arm,seed=seed)))
    for arm in p['arms']:
        for seed in p['seeds']:
            key=f'eval_{arm}_s{seed}'; out=f'artifacts/eval/{arm}_s{seed}'
            tasks.append(dict(id=key,kind='evaluation',args=['worker.py','--mode','eval','--arm',arm,'--seed',str(seed),'--outdir',out],
                complete=out+'/complete.json',resume_args=['--resume'],depends_on=training,
                cpu_threads=3,required_free_mib=6144,verify_artifacts=True,required_artifacts=['selection.json','run.json','best.npz'],
                completion_checks=dict(acceptance='evaluation_complete'),binding_checks=dict(arm=arm,seed=seed)))
    G.atomic_json(ROOT/'queue.json',dict(schema_version=1,tasks=tasks,
        cpu_diagnostic_gate=dict(report='artifacts/tests/preflight.json'),
        source_paths=['protocol.json','docs/THEORY_PROTOCOL.md','artifacts/data/manifest.json','artifacts/inputs/lineage.json',
                      *[str(x.relative_to(ROOT)) for x in (ROOT/'vendor/models').glob('*.py')]],
        postprocess=dict(args=['report.py'],complete='artifacts/report/complete.json',completion_checks=dict(status='complete'))))
    print('Frozen queue: 18 training tasks + 18 evaluation tasks + automatic report')

if __name__=='__main__': main()
