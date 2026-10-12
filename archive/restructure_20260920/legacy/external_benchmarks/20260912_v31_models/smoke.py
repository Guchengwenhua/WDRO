"""Real GPU checks of tail padding, resume, cross-device identity and statistics IO."""
import argparse
import numpy as np
from common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    directory=OUT/'smoke';directory.mkdir(exist_ok=True)
    if not args.verify:
        data=ev.load_npz(OUT/'inputs/degree_30.npz');ref=ev.load_npz(OUT/'inputs/reference_30.npz')
        # 257 exercises one full batch, one padded row, and a committed pause boundary.
        ids=np.r_[np.arange(256),len(data['pins'])-1]
        ev.atomic_npz(directory/'input.npz',{k:v[ids] for k,v in data.items()})
        ev.atomic_npz(directory/'reference.npz',{k:v[ids] for k,v in ref.items()})
        receipt=read(OUT/'inputs/reference_30.json')
        receipt.update(completed_nets=257,input=artifact(directory/'input.npz'),reference=artifact(directory/'reference.npz'))
        write(directory/'reference.json',receipt)
        models=read(OUT/'models_frozen.json')['models']
        new=next(m for m in models if m['method']=='random_top4' and m['seed']==101)
        control=next(m for m in models if m['method']=='erm' and m['seed']==101)
        commands=[]
        for name,model,gpu in [('continuous',new,2),('resumed',new,3),('control',control,4)]:
            command=worker_command(model,gpu,directory/name,directory/'input.npz',directory/'reference.npz',directory/'reference.json')
            if name=='resumed':command+=['--max-new-batches','1']
            log=(directory/f'{name}.log').open('w')
            commands.append((name,model,command,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT),log))
        for name,model,command,proc,log in commands:
            code=proc.wait();log.close();ev.require(code==0,f'smoke {name} failed; see {directory}/{name}.log')
        command=worker_command(new,2,directory/'resumed',directory/'input.npz',directory/'reference.npz',directory/'reference.json')
        with (directory/'resume.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        before=ev.sha256(directory/'resumed/calls.jsonl')
        with (directory/'idempotent.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        ev.require(before==ev.sha256(directory/'resumed/calls.jsonl'),'idempotence generated new calls')
        subprocess.run(['bash',str(ENV),PYTHON,str(BASE/'smoke.py'),'--verify'],check=True)
        return
    a=ev.load_npz(directory/'continuous/predictions.npz');b=ev.load_npz(directory/'resumed/predictions.npz')
    ev.require(a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a),'cross-GPU resume differs')
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    import sys
    sys.path.insert(0,str(ev.PACKAGE))
    import torch
    from src.rest import build_actor, RestTreeEvaluator
    from src.trainer import configure_determinism, load_trainer_config
    configure_determinism(load_trainer_config())
    model=next(m for m in read(OUT/'models_frozen.json')['models'] if m['method']=='random_top4' and m['seed']==101)
    actor=build_actor(30,torch.device('cuda:0'))
    actor.load_state_dict(torch.load(checked(model['best']),map_location='cpu',weights_only=False)['actor_state_dict']);actor.eval()
    sys.path.insert(0,str(V31))
    from adv_v3.common import Oracle
    # Invoke the original v3.1 scoring method while supplying existing exact references.
    ref_by_pins={p.tobytes():r for p,r in zip(a['pins'],a['lstar'])}
    class Cache:
        def lengths(self,pins,evaluator):return np.array([ref_by_pins[p.tobytes()] for p in pins]),None
    oracle=Oracle.__new__(Oracle);oracle.actor=actor;oracle.microbatch=256;oracle.ledger={}
    oracle.persist=lambda:None;oracle.rest=RestTreeEvaluator();oracle.cache=Cache();oracle.geosteiner=None
    output=oracle.evaluate(a['pins'])
    ev.require(np.array_equal(output['length'],a['rest_lengths']),'new worker differs from original v3.1 Oracle')
    ev.require(oracle.ledger['actor_forward_rows_completed']==512 and oracle.ledger['actor_padding_rows_completed']==255,'original scorer padding mismatch')
    # Bind and compare every legacy Actor source, not only its wrapper.
    equivalence=[]
    for source in sorted((REV.parent/'DRO/WDRO-REST/models').rglob('*.py')):
        other=V31.parent/'DRO/WDRO-REST/models'/source.relative_to(REV.parent/'DRO/WDRO-REST/models')
        ev.require(ev.sha256(source)==ev.sha256(other),'legacy Actor sources differ')
        equivalence.extend([artifact(source),artifact(other)])
    # Restore our common module for report imports after the v3.1 package path insertion.
    sys.path.insert(0,str(BASE))
    import finish
    import stats_helpers as st
    values={(m,s):a['regret']+(i*3+j)*1e-5 for i,m in enumerate(METHODS) for j,s in enumerate(SEEDS)}
    manifest={'coverage':[dict(design_id=d) for d in st.string_vector(a['design_ids'])[:1]]}
    rows,means,ci=finish.build_tables(manifest,a,values,[])
    for name,table in [('metrics',rows),('means',means),('ci',ci)]:
        st.write_csv(directory/f'{name}.csv',table);write(directory/f'{name}.json',table)
    # Verify recovery and audit logic on the real 257-net result, with only output root redirected.
    fixture=directory/'audit_fixture';(fixture/'results').mkdir(parents=True,exist_ok=True)
    link=fixture/'results'/task_id(model)
    if not link.exists():link.symlink_to(directory/'continuous',target_is_directory=True)
    finish.OUT=fixture
    audit=finish.verify_model(model)
    write(directory/'completion.json',dict(status='pass',at=now(),real_test_nets=257,
        checks=dict(tail_padding=True,original_v31_oracle_equal=True,cross_gpu_resume_exact=True,
                    idempotent_no_new_calls=True,independent_delivery_audit=True,statistics_csv_json=True),
        audit=audit,actor_source_equivalence=equivalence))
    print('PASS: GPU tail-padding, original v3.1 scorer, cross-GPU resume, idempotence, independent audit and statistics IO.')

if __name__=='__main__':main()
