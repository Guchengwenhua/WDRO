"""Extract every supported original-degree net before any model evaluation."""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import importlib
import multiprocessing
import sys

from common import *
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(BASE / 'parsers'))
from src.geometry import normalize_net, inverse_net, InvalidNetError, has_spacing_violation, pin_set_sha256

def prepare_design(design):
    identity = design['design_id']
    module, func = design['parser'].rsplit('.', 1)
    parser = getattr(importlib.import_module(module), func)
    source = Path(design['path'])
    if 'sha256' in design:
        assert sha(source) == design['sha256']
    dependencies=[]
    if source.suffix=='.json':
        dependencies=[artifact(value) for key,value in read(source).items() if key.endswith('_path') and isinstance(value,str) and Path(value).is_file()]
    binding = dict(design=design, parser=artifact(Path(importlib.import_module(module).__file__)), normalization=artifact(PACKAGE/'src/geometry.py'), extractor=artifact(__file__), source_dependencies=dependencies)
    out = BASE / 'cache/designs' / identity
    result_path = out/'manifest.json'
    if result_path.exists():
        old = read(result_path)
        assert old['binding'] == binding
        for g in old['groups']:
            if 'input' in g:
                assert sha(g['input']['logical_path']) == g['input']['sha256']
        return old
    counts = Counter()
    invalid = {d:Counter() for d in DEGREES}
    chunks = {d:[] for d in DEGREES}
    rejected = []
    identifiers = set()
    maximum_inverse_normalized_error = 0.0
    for source_index, record in enumerate(parser(source)):
        if isinstance(record, dict):
            net_name, raw = record['net_id'], record.get('points')
            d = record['raw_degree']
            parser_reasons = record.get('reasons', []) if not record.get('valid', True) else []
        else:
            net_name, raw = record
            d = len(raw)
            parser_reasons = []
        counts[d] += 1
        if d not in DEGREES:
            continue
        net_id = f'public:{identity}:{net_name}:d{d}'
        assert net_id not in identifiers, f'duplicate source net ID: {net_id}'
        identifiers.add(net_id)
        if parser_reasons or raw is None:
            reason = 'source_incomplete:' + ','.join(sorted(parser_reasons or ['missing_coordinates']))
            invalid[d][reason] += 1
            rejected.append(dict(net_id=net_id, source_index=source_index, degree=d, reason=reason))
            continue
        try:
            pins64, transform = normalize_net(raw, expected_degree=d)
            np.testing.assert_allclose(inverse_net(pins64, transform), raw, rtol=1e-12, atol=1e-7)
            pins = pins64.astype(np.float32)
            if has_spacing_violation(pins):
                raise InvalidNetError('minimum_spacing_violation_after_float32')
            assert np.array_equal(pins.astype(np.float64)*2**27, np.rint(pins.astype(np.float64)*2**27))
            maximum_inverse_normalized_error = max(maximum_inverse_normalized_error, float(np.max(np.abs(pins.astype(np.float64)-pins64))))
        except InvalidNetError as e:
            invalid[d][e.reason] += 1
            rejected.append(dict(net_id=net_id, source_index=source_index, degree=d, reason=e.reason))
            continue
        chunks[d].append((net_id, source_index, pins, transform.offset_xy, transform.scale, pin_set_sha256(pins)))
    groups=[]
    for d in DEGREES:
        items=chunks[d]
        g=dict(degree=d, parsed_count=counts[d], valid_count=len(items), invalid_count=sum(invalid[d].values()), invalid_reasons=dict(invalid[d]))
        assert g['parsed_count'] == g['valid_count'] + g['invalid_count']
        if items:
            arrays=dict(net_ids=np.asarray([x[0] for x in items],dtype='S512'),
                source_index=np.array([x[1] for x in items],dtype=np.int64), pins=np.stack([x[2] for x in items]),
                offsets=np.stack([x[3] for x in items]), scales=np.array([x[4] for x in items],dtype=np.float64),
                content_hashes=np.array([x[5] for x in items],dtype='S64'),
                design_ids=np.full(len(items),identity,dtype='S128'),
                dataset_ids=np.full(len(items),design['family'],dtype='S128'),
                family_ids=np.full(len(items),design.get('design_family',identity),dtype='S128'))
            assert all(len(x[0].encode()) <= 512 for x in items)
            file=out/f'degree_{d}.npz'
            npz(file,arrays)
            g['input']=artifact(file)
        groups.append(g)
    write(out/'rejected_nets.json',rejected)
    result=dict(status='pass', at=now(), design_id=identity, dataset=design['family'], binding=binding, groups=groups,
                total_source_nets=sum(counts.values()), degree_histogram=dict(sorted(counts.items())),
                unsupported_degree_nets=sum(n for d,n in counts.items() if d not in DEGREES),
                maximum_inverse_normalized_error=maximum_inverse_normalized_error, rejected=artifact(out/'rejected_nets.json'),
                model_calls=0)
    write(result_path,result)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-manifest',action='append',required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--extract-only',action='store_true')
    args=p.parse_args()
    with lease('prepare'):
        manifests=[read(p) for p in args.source_manifest]
        designs=[d for m in manifests for d in m['designs']]
        assert len({d['design_id'] for d in designs}) == len(designs)
        designs=sorted(designs,key=lambda d:d['design_id'])
        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for r in pool.map(prepare_design, designs):
                results.append(r)
                write(BASE/'logs/prepare/heartbeat.json',dict(at=now(), status='extracting',completed_designs=len(results),total_designs=len(designs)))
        if args.extract_only:
            print(json.dumps(dict(extracted_designs=len(results),valid_nets=sum(g['valid_count'] for r in results for g in r['groups']))),flush=True)
            return
        groups=[]
        all_ids=set()
        content=defaultdict(list)
        for d in DEGREES:
            parts=[]
            for r in results:
                g=next(g for g in r['groups'] if g['degree']==d)
                if 'input' in g:
                    parts.append(load(g['input']['logical_path']))
            if not parts:
                continue
            arrays={k:np.concatenate([a[k] for a in parts]) for k in parts[0]}
            ids=set(arrays['net_ids'].tolist())
            assert len(ids)==len(arrays['pins']) and not(ids & all_ids)
            all_ids.update(ids)
            for h,identity,design_id in zip(arrays['content_hashes'],arrays['net_ids'],arrays['design_ids']):
                content[h.decode()].append((identity.decode(),design_id.decode()))
            out=BASE/'cache/inputs'/f'degree_{d}.npz'
            npz(out,arrays)
            groups.append(dict(degree=d,count=len(arrays['pins']),input=artifact(out)))
        repeats={h:rows for h,rows in content.items() if len(rows)>1}
        write(BASE/'cache/duplicate_geometry.json',dict(policy='Retain distinct source nets; identical normalized geometry is disclosed, not an independent design claim.',groups=repeats))
        index=PACKAGE/'manifests/stage6_models_frozen_v1.json'
        frozen=read(index)
        assert frozen['status']=='frozen' and frozen['selection']=='synthetic_validation_only'
        for m in frozen['models']:
            assert sha(m['best']['logical_path'])==m['best']['sha256']
        result=dict(status='frozen',at=now(),extension='public_physical_design_zero_shot_v1',
            source_manifests=[artifact(p) for p in args.source_manifest], designs=results, groups=groups,
            design_count=len(results), total_nets=len(all_ids), valid_design_count=sum(any(g['valid_count'] for g in r['groups']) for r in results),
            total_source_nets=sum(r['total_source_nets'] for r in results),
            unsupported_degree_nets=sum(r['unsupported_degree_nets'] for r in results),
            invalid_nets=sum(g['invalid_count'] for r in results for g in r['groups']),
            duplicate_geometry_groups=len(repeats), duplicate_geometry=artifact(BASE/'cache/duplicate_geometry.json'),
            model_index=artifact(index), models_verified=len(frozen['models']),
            protocol=artifact(PACKAGE/'configs/protocol_v1.json'),data_contract=artifact(PACKAGE/'configs/data_contract_v1.json'),
            inference=dict(batch_size=256,T=1,mode='deterministic_greedy'),train_or_retune=False,
            sampling='all nets with original degree in 10,20,30,40,50 passing the frozen geometry contract; no random subsampling',
            expected_predictions=15*len(all_ids), expected_model_tasks=15*len(groups),
            implementation=artifact(__file__))
        write(BASE/'cache/input_manifest.json',result)
        write(BASE/'logs/prepare/heartbeat.json',dict(at=now(),status='complete',total_nets=len(all_ids),designs=len(results)))
        print(json.dumps({k:v for k,v in result.items() if k not in ('designs','groups')}),flush=True)

if __name__=='__main__':
    main()
