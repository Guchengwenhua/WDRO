"""CPU-only synthetic-schema checks; fixture values are never experiment results."""
import csv
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import interpret_results as subject


def fixture_tables():
    t={name:[] for name in subject.SCHEMAS}
    for parent,degree,seed,dist,method in itertools.product(subject.PARENTS,subject.DEGREES,subject.SEEDS,subject.DISTRIBUTIONS,subject.FIXED_BUDGETS):
        t['fixed_cells'].append(dict(parent=parent,degree=degree,seed=seed,distribution=dist,searcher=method,
            budget=subject.FIXED_BUDGETS[method],price=seed/100,rho=.03,nets=256,
            mean_regret=.03,p95_regret=.05,mean_score=.02,mean_cost=.01,mean_delta_score=.01,
            improved_fraction=.7,real_score_requests=256*subject.FIXED_BUDGETS[method],
            ledger_delta={},geometry={},selected_legal_fraction=1.0))
    for parent,degree,seed in itertools.product(subject.PARENTS,subject.DEGREES,subject.SEEDS):
        base=dict(parent=parent,degree=degree,seed=seed,task=f'{parent}_d{degree}_s{seed}')
        t['fixed_efficiency'].append(dict(base,critic_label_requests=9216,rl_training_requests=32768,
            rl_validation_requests=2048,rl_baseline_requests=16384,search_requests=344064,
            extra_completed_requests=0,actual_ledger={},walltime_scope='synthetic fixture'))
        t['fixed_diagnostics'].append(dict(base,**{key:None if 'spearman' in key else 0 for key in subject.SCHEMAS['fixed_diagnostics'][1].split() if key not in base}))
    pair_values=dict(difference=.02,ci_low=.01,ci_high=.03,per_seed_differences=[.01,.02,.03],
        all_seeds_positive=True,all_seeds_negative=False,stable_positive=True,stable_negative=False,
        seed_count=3,centers=256,bootstrap_replicates=2000)
    for parent,degree,dist,(left,right),metric in itertools.product(subject.PARENTS,subject.DEGREES,subject.DISTRIBUTIONS,subject.FIXED_PAIRS,('score_mean','regret_mean','regret_p95','cost_mean')):
        t['fixed_paired'].append(dict(pair_values,parent=parent,degree=degree,distribution=dist,left=left,right=right,metric=metric))
    for method,seed,dist,stage in itertools.product(subject.JOINT_METHODS,subject.SEEDS,subject.DISTRIBUTIONS,('initial','final','attacked_final')):
        value={'initial':.03,'final':.02,'attacked_final':.07}[stage]
        t['joint_cells'].append(dict(method=method,degree=30,seed=seed,distribution=dist,stage=stage,
            best_update=64,nets=64 if stage=='attacked_final' else 512,mean_regret=value,p95_regret=2*value))
    for method,seed in itertools.product(subject.JOINT_METHODS,subject.SEEDS):
        t['joint_efficiency'].append(dict(task=f'{method}_d30_s{seed}',method=method,degree=30,seed=seed,
            best_update=64,initial_price=1,final_price=.5,rho=.03,actual_ledger={},
            training_search_requests=512*64*(64 if method in ('random64','es64') else 4),
            online_attacker_requests=16384 if method=='rl_joint4' else 0,
            sampled_main_actor_nets=32768,committed_training_wall_seconds=123,
            initial_validation=dict(nominal_mean=.03,ood_worst_group_mean=.05),
            best_validation=dict(nominal_mean=.031,ood_worst_group_mean=.04)))
        for stage in ('initial','final','attacked_final'):
            t['joint_worst_group'].append(dict(method=method,seed=seed,stage=stage,
                worst_group='edge_bias',worst_group_mean=.05,group_means=dict(edge_bias=.05)))
    for dist,metric in itertools.product(subject.DISTRIBUTIONS,('regret_mean','regret_p95')):
        comparisons=[('after_minus_before',method,'initial_erm') for method in subject.JOINT_METHODS]
        comparisons.extend((stage,left,right) for stage,(left,right) in itertools.product(('final','attacked_final'),subject.JOINT_PAIRS))
        for comparison,left,right in comparisons:
            t['joint_paired'].append(dict(pair_values,comparison=comparison,left=left,right=right,
                distribution=dist,metric=metric,stable_improvement=False))
    return t


def create_fixture(root):
    (root/'experiments/tables').mkdir(parents=True)
    (root/'manifests').mkdir()
    (root/'EXPERIMENT_PLAN.md').write_text('SYNTHETIC SCHEMA FIXTURE - NOT RESEARCH EVIDENCE')
    protocol=dict(degrees=list(subject.DEGREES),seeds=list(subject.SEEDS),parents=list(subject.PARENTS),
                  plan_sha256=subject.sha256(root/'EXPERIMENT_PLAN.md'))
    (root/'protocol.json').write_text(json.dumps(protocol))
    artifacts=[]
    for name,rows in fixture_tables().items():
        path=root/'experiments/tables'/f'{name}.csv'
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
        artifacts.append(dict(path=str(path),sha256=subject.sha256(path)))
    for phase in ('fixed','joint'):
        path=root/'experiments/tables'/f'{phase}_audit.json'
        path.write_text(json.dumps(dict(status='pass',phase=phase)))
        artifacts.append(dict(path=str(path),sha256=subject.sha256(path)))
    completion=dict(status='pass',phase='all',all_experiments_complete=True,provenance=dict(status='pass'),
        protocol_sha256=subject.sha256(root/'protocol.json'),
        phases=dict(fixed=dict(tasks=30,cells=1080),joint=dict(tasks=18,cells=216)),artifacts=artifacts)
    (root/'manifests/all_completion.json').write_text(json.dumps(completion))


class EvidenceConsumerTests(unittest.TestCase):
    def test_full_grid_and_cost_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_fixture(root)
            tables,bindings=subject.load_evidence(root,'all')
            fixed=subject.fixed_extract(tables);joint=subject.joint_extract(tables)
            self.assertEqual(len(fixed['fixed_pairs_by_seed']),3360)
            self.assertEqual(len(fixed['fixed_pair_condition_summary']),70)
            self.assertEqual(len(joint['joint_pairs_by_seed']),336)
            self.assertEqual(len(joint['joint_before_after_by_seed']),72)
            rl=next(r for r in joint['joint_training_cost'] if r['method']=='rl_joint4')
            self.assertEqual(rl['pretrained_attacker_requests'],34816)
            self.assertEqual(rl['online_attacker_requests'],16384)
            self.assertEqual(rl['committed_search_plus_online_requests'],147456)
            self.assertTrue(all(r['validation_gate_pass'] for r in joint['joint_nominal_gate']))
            output=subject.publish(root/'synthetic_output',dict(**fixed,**joint),bindings,'all')
            self.assertEqual(output['status'],'pass')
            self.assertEqual(len(output['artifacts']),13)

    def test_changed_csv_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_fixture(root)
            with (root/'experiments/tables/fixed_cells.csv').open('a') as handle:handle.write('\n')
            with self.assertRaisesRegex(subject.EvidenceError,'unbound or modified'):
                subject.load_evidence(root,'all')

    def test_missing_or_nonpassing_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(subject.EvidenceError):subject.load_evidence(root,'all')
            create_fixture(root)
            path=root/'manifests/all_completion.json';record=json.loads(path.read_text())
            record['status']='incomplete';path.write_text(json.dumps(record))
            with self.assertRaisesRegex(subject.EvidenceError,'matching PASS'):
                subject.load_evidence(root,'all')


if __name__=='__main__':
    unittest.main(verbosity=2)
