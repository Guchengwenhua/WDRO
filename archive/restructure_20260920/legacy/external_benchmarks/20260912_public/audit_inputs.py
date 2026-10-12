"""Independent input identities, coverage, geometry and prior-data overlap."""
from collections import Counter, defaultdict
import sys

from common import *
sys.path.insert(0,str(PACKAGE))
from src.geometry import pin_set_sha256

def main():
    with lease('input_audit'):
        plan=read(BASE/'cache/input_manifest.json')
        all_new={}
        counts=Counter()
        inputs=[]
        duplicates=defaultdict(list)
        for group in plan['groups']:
            item=group['input']
            assert sha(item['logical_path'])==item['sha256']
            data=load(item['logical_path'])
            n=len(data['pins'])
            assert n==group['count'] and data['pins'].shape==(n,group['degree'],2)
            assert len(set(data['net_ids'].tolist()))==n
            for i,p in enumerate(data['pins']):
                identity=data['net_ids'][i].decode()
                h=pin_set_sha256(p)
                assert h==data['content_hashes'][i].decode()
                assert identity not in all_new
                all_new[identity]=h
                duplicates[h].append(identity)
                counts[(data['design_ids'][i].decode(),group['degree'])]+=1
            inputs.append(item)
        for d in plan['designs']:
            for g in d['groups']:
                assert counts[d['design_id'],g['degree']]==g['valid_count']
                assert g['parsed_count']==g['valid_count']+g['invalid_count']
        assert len(all_new)==plan['total_nets']
        overlaps=[]
        prior_count=0
        prior_paths=sorted((REV/'experiments/cache/datasets/protocol_v1').glob('*/*.npz'))
        prior_paths+=sorted((REV/'experiments/cache/stage7/inputs').glob('*.npz'))
        prior=[]
        for p in prior_paths:
            data=load(p)
            if 'pins' not in data:
                continue
            prior_count+=len(data['pins'])
            for i,xy in enumerate(data['pins']):
                h=pin_set_sha256(xy)
                if h in duplicates:
                    overlaps.append(dict(prior=str(p),prior_index=i,content_hash=h,new_net_ids=duplicates[h]))
            prior.append(artifact(p))
        frozen=read(plan['model_index']['logical_path'])
        for m in frozen['models']:
            assert sha(m['best']['logical_path'])==m['best']['sha256']
        for rel,digest in frozen['code_files'].items():
            assert sha(REV/rel)==digest, rel
        # Cross-check against the independently implemented source parser audits.
        contests=read(BASE/'sources/contests/coordinate_audit.json')
        lookup={d['design_id']:d for d in contests['designs']}
        source_audits=[artifact(BASE/'sources/contests/coordinate_audit.json')]
        for design in plan['designs']:
            if design['design_id'] in lookup:
                independent=lookup[design['design_id']]
                assert design['total_source_nets']==independent['num_nets']
                assert design['degree_histogram']==independent['raw_degree_counts']
                for g in design['groups']:
                    assert g['parsed_count']==independent['raw_degree_counts'].get(str(g['degree']),0)
            else:
                path=Path(design['binding']['design']['audit_path'])
                independent=read(path)
                assert design['total_source_nets']==independent['raw_nets']
                assert design['degree_histogram']==independent['raw_degree_counts']
                source_audits.append(artifact(path))
            for dep in design['binding']['source_dependencies']:
                assert sha(dep['logical_path'])==dep['sha256']
        result=dict(status='pass',at=now(),input_manifest=artifact(BASE/'cache/input_manifest.json'),
            inputs=inputs,prior_inputs=prior,prior_nets_compared=prior_count,prior_geometry_overlaps=overlaps,
            unique_source_ids=len(all_new),normalized_geometry_duplicate_groups=sum(len(v)>1 for v in duplicates.values()),
            models_verified=len(frozen['models']),frozen_code_bindings_verified=len(frozen['code_files']),
            source_audits=source_audits,
            scope='All input IDs and normalized unordered pin hashes; all best model and frozen code hashes. Prior comparison covers synthetic validation/test and ISPD18/19; no new training or model selection.',
            implementation=artifact(__file__))
        write(BASE/'cache/input_audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','prior_inputs')}),flush=True)

if __name__=='__main__':
    main()
