"""Full proposal/score/tree replay and transaction accounting for CEM-256."""
from scipy.optimize import linear_sum_assignment

from src.datasets import array_bundle_sha256
from src.metrics import relative_regret, summarize_regret
from stage8.common import *
from stage8.attack import restore_rounds, reduce_candidates, independent_tree_lengths, task_id


def reconcile_calls(directory, state):
    events = {}
    for line in (directory / 'calls.jsonl').read_text().splitlines():
        e = json.loads(line)
        phases = events.setdefault(e['id'], {})
        assert e['phase'] not in phases, 'duplicate transaction receipt'
        phases[e['phase']] = e
    completed = dict(actor=0, rest=0, reference=0, verification=0)
    upper_extra = dict(completed)
    for eid, e in events.items():
        if 'started' in e:
            for kind in ('actor', 'rest', 'reference'):
                key = kind + '_completed'
                if key in e:
                    assert e[key]['calls'] == e['started'][kind+'_planned']
                    completed[kind] += e[key]['calls']
                else:
                    upper_extra[kind] += e['started'][kind+'_planned']
        elif 'verification_started' in e:
            if 'verification_completed' in e:
                assert e['verification_completed']['calls'] == e['verification_started']['planned']
                completed['verification'] += e['verification_completed']['calls']
            else:
                upper_extra['verification'] += e['verification_started']['planned']
        else:
            raise AssertionError('orphan completed receipt')
    committed = dict(actor=0, rest=0, reference=0, verification=0)
    committed_ids = set()
    for b in state['batches']:
        for r in b['rounds']:
            e = events[r['attempt']]
            assert r['attempt'] not in committed_ids
            committed_ids.add(r['attempt'])
            for kind in ('actor', 'rest', 'reference'):
                assert e[kind+'_completed']['calls'] == r['calls']
                committed[kind] += r['calls']
        e = events[b['verification_attempt']]
        assert e['verification_completed']['calls'] == b['end']-b['start']
        committed['verification'] += b['end']-b['start']
    assert all(completed[k] >= committed[k] for k in committed)
    return dict(committed=committed, recorded_completed_attempts=completed,
        completed_replay_extra={k:completed[k]-committed[k] for k in committed},
        unfinished_possible_extra_upper_bound=upper_extra,
        actual_calls_lower_bound=completed,
        actual_calls_upper_bound={k:completed[k]+upper_extra[k] for k in completed},
        events=artifact(directory/'calls.jsonl'))


def audit_cell(directory):
    directory = Path(directory)
    plan = gate()
    result_path, cursor = directory/'result.json', directory/'cursor.json'
    result, state = read(result_path), read(cursor)
    binding = dict(result=artifact(result_path), state=artifact(cursor), plan=artifact(PLAN),
                   auditor_sha256=file_sha256(Path(__file__)))
    dest = directory/'audit.json'
    if dest.exists():
        r = read(dest)
        assert r['binding'] == binding and r['status'] == 'pass'
        checked(result['predictions'])
        return r
    with Lease(directory/'audit_job', 'stage8_audit_'+directory.name) as lease:
        assert result['status']=='complete' and state['binding']==result['binding'] and not state['current']
        assert state['binding']['plan'] == artifact(PLAN)
        assert state['binding']['model'] in plan['models'] and not result['model_selection']
        assert state['binding']['attack'] == plan['attack']
        checked(state['binding']['model']['best'])
        data = load_npz(checked(state['binding']['input']['input']))
        out = load_npz(checked(result['predictions']))
        count = result['completed_nets']
        assert state['complete_nets'] == count
        np.testing.assert_array_equal(out['net_ids'], data['net_ids'][:count])
        audit_cursor = directory/'audit_job/cursor.json'
        previous = read(audit_cursor) if audit_cursor.exists() else dict(binding=binding, verified_batches=[])
        assert previous['binding']==binding
        end = 0
        for index, b in enumerate(state['batches']):
            assert b['start']==end and b['end']>end and len(b['rounds'])==16
            start, end = b['start'], b['end']
            if index < len(previous['verified_batches']):
                assert previous['verified_batches'][index] == array_bundle_sha256({'record':np.frombuffer(
                    json.dumps(b,sort_keys=True).encode(),dtype=np.uint8)})
                checked(b['predictions'])
                continue
            centers, ids = data['pins'][start:end], data['net_ids'][start:end]
            _, _, _, records = restore_rounds(centers, b['rounds'], plan['attack'], ids, reconstruct=True)
            for r, a in zip(b['rounds'], records, strict=True):
                pins = a['pins'].reshape(-1, centers.shape[1], 2)
                assert r['calls']==len(pins)==16*(end-start) and np.isfinite(pins).all()
                assert pins.min()>=.1-5e-8 and pins.max()<=.9+5e-8
                assert np.all(np.abs((pins.max(axis=1)-pins.min(axis=1)).max(axis=1)-.8)<1e-7)
                np.testing.assert_array_equal(independent_tree_lengths(pins, a['topology'].reshape(len(pins),-1)),
                                              a['rest_lengths'].ravel())
                np.testing.assert_array_equal(relative_regret(a['rest_lengths'],a['lstar']),a['regret'])
                assert np.isfinite(a['regret']).all() and np.all(a['regret']>=-1e-12)
            selected = load_npz(checked(b['predictions']))
            rebuilt = reduce_candidates(centers, ids, records)
            for k in rebuilt:
                np.testing.assert_array_equal(selected[k], rebuilt[k])
            all_regret = np.concatenate([a['regret'] for a in records],axis=1)
            for i, best in enumerate(selected['selected_index']):
                assert selected['regret'][i] == max(all_regret[i])
                assert best == np.flatnonzero(all_regret[i]==all_regret[i].max())[0]
                distances=np.abs(centers[i,:,None,:].astype(np.float64)-selected['pins'][i,None,:,:]).sum(axis=2)
                rows, cols=linear_sum_assignment(distances)
                assert selected['cost'][i] == distances[rows,cols].mean()
            np.testing.assert_array_equal(selected['lstar'], selected['verification_lstar'])
            for k in selected:
                np.testing.assert_array_equal(selected[k],out[k][start:end])
            previous['verified_batches'].append(array_bundle_sha256({'record':np.frombuffer(
                json.dumps(b,sort_keys=True).encode(),dtype=np.uint8)}))
            atomic_write_json(audit_cursor,previous)
            lease.update(end,'auditing_all_candidates')
        assert end == count and len(out['net_ids'])==count
        assert result['summary']==summarize_regret(out['regret']) and result['cost']==summarize_regret(out['cost'])
        accounting = reconcile_calls(directory,state)
        assert accounting['committed']==dict(actor=count*256,rest=count*256,reference=count*256,verification=count)
        r = dict(status='pass',at_utc=utc_now(),binding=binding,centers=count,candidates=count*256,
            full_proposal_replay=True,all_candidate_integer_tree_lengths=True,all_candidate_connectivity=True,
            all_candidate_scores=True,per_center_argmax=True,independent_hungarian_cost=True,
            selected_two_scale_reference=True,equal_256_calls=True,model_selection=False,accounting=accounting)
        atomic_write_json(dest,r,mode=0o444)
        lease.update(count,'pass')
        return r


def audit():
    plan = gate()
    p,h = capture('Stage8.3 complete attack audit (CPU)',peak=0)
    with Lease(LOG/'audit','stage8_full_audit',str(p)) as lease:
        records=[]
        for model in plan['models']:
            for dist in ('uniform','mix_gauss','edge_bias','grid_cluster'):
                directory=LOG/'attacks'/task_id(model,dist)
                r=audit_cell(directory)
                assert r['centers']==2000 and r['candidates']==512000
                records.append(artifact(directory/'audit.json'))
                lease.update(len(records),'audit')
        assert len(records)==300
        atomic_write_json(LOG/'audit/result.json',dict(status='pass',at_utc=utc_now(),plan=artifact(PLAN),
            records=records,model_center_attacks=600000,candidate_scores=153600000),mode=0o444)
        progress('8.2 全部300格CEM-256及153600000候选完整审计通过','DONE',str(LOG/'audit/result.json'),
                 '汇总效率、paired CI、最终表图与结论边界，完成交付复核')
