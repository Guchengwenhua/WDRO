"""Freeze 18 validation-selected joint Actors, 15 controls and all eligible real nets."""
import hashlib
import numpy as np
from common import *

def main():
    import torch
    if (OUT/'input_manifest.json').exists():
        for item in read(OUT/'input_manifest.json')['dependencies']:
            checked(item)
        for item in read(OUT/'models_frozen.json')['models']:
            checked(item['best'])
        print('Already frozen; validated unchanged dependencies.')
        return
    accepted = read(V31/'manifests/all_completion.json')
    ev.require(accepted['status']=='pass' and accepted['phases']['joint']['tasks']==18, 'v3.1 not accepted')
    accepted_by_path = {str(Path(a['path']).resolve()): a for a in accepted['artifacts']}
    def accepted_artifact(path):
        rec = artifact(path)
        ev.require(accepted_by_path[rec['resolved_path']]['sha256']==rec['sha256'], f'not accepted: {path}')
        return rec
    models=[]
    for method in NEW:
        for seed in SEEDS:
            directory = V31/'experiments/joint'/f'{method}_d30_s{seed}'
            completion = read(directory/'complete.json')
            ev.require(completion['status']=='pass' and not completion['smoke'] and completion['updates']==512,
                       'joint training completion invalid')
            best=accepted_artifact(directory/'best.pt')
            cp=torch.load(checked(best),map_location='cpu',weights_only=False)
            ev.require(cp['update']==completion['best_update'], 'best update mismatch')
            models.append(dict(method=method,degree=30,seed=seed,kind='v31_joint',best=best,
                selected_update=cp['update'],training_completion=accepted_artifact(directory/'complete.json'),
                training_identity=accepted_artifact(directory/'identity.json')))
    parent=read(ev.PARENT_FROZEN)
    ev.require(parent['status']=='frozen' and not parent['test_used_for_selection'], 'unfrozen parent')
    for original in parent['models']:
        if original['degree']!=30: continue
        cp=torch.load(checked(original['best']),map_location='cpu',weights_only=False)
        models.append(dict(method=original['method'],degree=30,seed=original['seed'],
            kind='precision_v2_control',best=original['best'],selected_update=cp['update']))
    ev.require(len(models)==33 and len({task_id(m) for m in models})==33, 'model task coverage')
    # New and old Actors share the exact implementation and deterministic configuration.
    source_equivalence=[]
    for relative in ('src/rest.py','src/geosteiner.py','src/trainer.py','configs/stage2_trainer_v1.json'):
        a,b=artifact(REV/'wdro_rest_min'/relative),artifact(V31/'wdro_rest_min'/relative)
        ev.require(a['sha256']==b['sha256'], 'actor/reference implementation differs')
        source_equivalence.extend([a,b])
    write(OUT/'models_frozen.json', dict(status='frozen',at=now(),selection='synthetic_validation_only',
        test_used_for_selection=False,models=models,source_equivalence=source_equivalence,
        parent_index=artifact(ev.PARENT_FROZEN),v31_completion=artifact(V31/'manifests/all_completion.json')))

    public=read(OLD/'cache/input_manifest.json')
    stage7=read(ev.PACKAGE/'manifests/stage7_plan_v1.json')
    pg=next(g for g in public['groups'] if g['degree']==30)
    sg=next(g for g in stage7['groups'] if g['degree']==30)
    pr=read(OLD/'cache/references/degree_30.json')
    sr=read(REV/'experiments/cache/stage7/references/degree_30/result.json')
    ev.require(sr['status']=='complete' and pr['status']=='complete', 'incomplete references')
    ev.require(sr['binding']['input']['input']==sg['input'], 'stage7 reference input binding')
    ev.require(checked(pr['input'])==checked(pg['input']), 'public reference input binding')
    ev.require(sr['binding']['reference_identity']==pr['reference_identity'], 'reference identities differ')
    ev.require(sr['binding']['scales']==pr['scales']==[2**27,2**28], 'reference scales differ')
    sd,pd=ev.load_npz(checked(sg['input'])),ev.load_npz(checked(pg['input']))
    sref,pref=ev.load_npz(checked(sr['reference'])),ev.load_npz(checked(pr['reference']))
    ev.validate_inputs(sd,30);ev.validate_inputs(pd,30)
    ev.validate_reference(sref,sd);ev.validate_reference(pref,pd)
    def strings(a): return np.char.decode(a,'utf-8') if a.dtype.kind=='S' else a.astype(str)
    sd['dataset_ids']=np.array(['ispd2018_protocol_a' if d.startswith('ispd18_') else 'ispd2019_protocol_a'
                               for d in strings(sd['design_ids'])],dtype='S128')
    sd['family_ids']=sd['design_ids'].astype('S128')
    sd['content_hashes']=np.array([hashlib.sha256(p[np.lexsort((p[:,1],p[:,0]))].tobytes()).hexdigest()
                                   for p in sd['pins']],dtype='S64')
    # Recompute identical canonical content hashes in both sources for overlap auditing.
    pd['content_hashes']=np.array([hashlib.sha256(p[np.lexsort((p[:,1],p[:,0]))].tobytes()).hexdigest()
                                   for p in pd['pins']],dtype='S64')
    fields=('pins','net_ids','design_ids','dataset_ids','family_ids','source_index','scales','offsets','content_hashes')
    data={k:np.concatenate((sd[k],pd[k])) for k in fields}
    data['net_ids']=np.concatenate((np.array(['stage7::'+x for x in strings(sd['net_ids'])],dtype='S640'),
                                   np.array(['public::'+x for x in strings(pd['net_ids'])],dtype='S640')))
    data['source_partition']=np.array(['stage7']*len(sd['pins'])+['public']*len(pd['pins']),dtype='S16')
    # Conservatively connect known families and designs sharing exact normalized geometry.
    labels=strings(data['family_ids']); design=strings(data['design_ids']); parents={x:x for x in set(labels)}
    def root(x):
        while parents[x]!=x: x=parents[x]
        return x
    first={}; cross_design_duplicates=0
    for h,d,f in zip(data['content_hashes'],design,labels):
        if h in first:
            old_d,old_f=first[h]
            if d!=old_d:
                cross_design_duplicates+=1
                a,b=root(f),root(old_f);parents[max(a,b)]=min(a,b)
        else: first[h]=(d,f)
    data['original_family_ids']=data['family_ids'].copy()
    data['family_ids']=np.array([root(x) for x in labels],dtype='S256')
    ref=dict(net_ids=data['net_ids'],lstar=np.concatenate((sref['lstar'],pref['lstar'])),
             verification_lstar=np.concatenate((sref['verification_lstar'],pref['verification_lstar'])))
    ev.validate_inputs(data,30);ev.validate_reference(ref,data)
    ev.atomic_npz(OUT/'inputs/degree_30.npz',data,mode=0o444)
    ev.atomic_npz(OUT/'inputs/reference_30.npz',ref,mode=0o444)
    dependencies=[artifact(OLD/'cache/input_manifest.json'),artifact(ev.PACKAGE/'manifests/stage7_plan_v1.json'),
                  artifact(OLD/'cache/references/degree_30.json'),artifact(REV/'experiments/cache/stage7/references/degree_30/result.json'),
                  pg['input'],sg['input'],pr['reference'],sr['reference'],artifact(OUT/'models_frozen.json'),
                  artifact(BASE/'protocol.json'),artifact(V31/'manifests/all_completion.json')]+source_equivalence
    write(OUT/'inputs/reference_30.json',dict(status='complete',completed_nets=len(data['pins']),
        input=artifact(OUT/'inputs/degree_30.npz'),reference=artifact(OUT/'inputs/reference_30.npz'),
        reference_identity=pr['reference_identity'],scales=pr['scales'],fresh_geosteiner_calls=0,
        provenance='Unchanged, hash-verified two-scale exact references, concatenated in source order.',
        dependencies=dependencies))
    coverage=[]
    for row in stage7['counts']:
        if row['degree']!=30:continue
        coverage.append(dict(design_id=row['design_id'],dataset='ispd2018_protocol_a' if row['design_id'].startswith('ispd18_') else 'ispd2019_protocol_a',
            eligible_nets=row['valid_count'],parsed_nets=row['parsed_count'],invalid_nets=row['invalid_count'],source_missing=row['source_missing']))
    for row in public['designs']:
        group=next(g for g in row['groups'] if g['degree']==30)
        coverage.append(dict(design_id=row['design_id'],dataset=row['dataset'],eligible_nets=group['valid_count'],
            parsed_nets=group['parsed_count'],invalid_nets=group['invalid_count'],source_missing=False))
    ev.require(sum(c['eligible_nets'] for c in coverage)==len(data['pins']), 'coverage mismatch')
    ev.require(len(coverage)==49, 'unexpected current real-data inventory')
    for c in coverage:
        actual=int(np.sum(design==c['design_id']));ev.require(actual==c['eligible_nets'],'design count mismatch')
        c['status']='evaluated' if actual else 'no_legal_degree30_nets'
    write(OUT/'input_manifest.json',dict(status='frozen',at=now(),degree=30,count=len(data['pins']),
        all_designs=49,eligible_designs=sum(c['eligible_nets']>0 for c in coverage),coverage=coverage,
        expected_model_tasks=33,expected_predictions=33*len(data['pins']),new_model_predictions=18*len(data['pins']),
        input=artifact(OUT/'inputs/degree_30.npz'),reference=artifact(OUT/'inputs/reference_30.npz'),
        dependencies=dependencies,source_counts={'stage7':len(sd['pins']),'public':len(pd['pins'])},
        duplicate_geometry_rows=len(data['pins'])-len(np.unique(data['content_hashes'])),
        cross_design_duplicate_occurrences=cross_design_duplicates,statistical_family_count=len(set(strings(data['family_ids']))),
        cluster_policy='Known design families union exact shared normalized pin geometry; retain all original rows.',
        input_order='stage7 frozen order, then public frozen order',no_sampling=True,no_training=True))
    print(json.dumps({k:v for k,v in read(OUT/'input_manifest.json').items() if isinstance(v,(str,int,bool))}))

if __name__=='__main__':main()
