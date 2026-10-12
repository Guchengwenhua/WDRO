"""Independent hand-computed checks for aggregation and family bootstrap."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import summarize as stats


class StatisticsTests(unittest.TestCase):
    def test_related_pdk_variants_share_cluster_and_retain_estimand_weights(self):
        designs = np.array(['a_pdk1'] * 4 + ['a_pdk2'] + ['b'] * 2)
        families = np.array(['a'] * 5 + ['b'] * 2)
        single = np.array([1., 1., 1., 1., 3., 5., 5.])
        rows = stats.family_bootstrap(np.stack([single, single*2, single*3]), designs, families)
        table = {(r['metric'], r['training_seed']): r for r in rows}
        net = table['net_weighted_mean', 'mean']
        macro = table['design_macro_mean', 'mean']
        # Two possible resampled families give AA, AB/BA, BB. This direct
        # enumeration gives the limiting 2.5/97.5 percentile endpoints.
        self.assertAlmostEqual(net['mean_difference'], 34/7)
        self.assertAlmostEqual(net['ci_low'], 14/5)
        self.assertAlmostEqual(net['ci_high'], 10)
        self.assertAlmostEqual(macro['mean_difference'], 6)
        self.assertAlmostEqual(macro['ci_low'], 4)
        self.assertAlmostEqual(macro['ci_high'], 10)
        self.assertEqual(macro['family_count'], 2)
        self.assertEqual(macro['design_count'], 3)
        self.assertEqual(macro['count'], 7)
        self.assertAlmostEqual(table['net_weighted_mean', 101]['mean_difference'], 17/7)
        self.assertEqual(rows, stats.family_bootstrap(np.stack([single,single*2,single*3]), designs, families))

    def test_single_family_is_explicitly_degenerate(self):
        rows = stats.family_bootstrap(np.ones((3, 3)), np.array(['a', 'a', 'b']), np.array(['family']*3))
        self.assertTrue(all(r['interval_status'] == 'single_family_degenerate' for r in rows))
        self.assertTrue(all(r['ci_low'] == r['ci_high'] == r['mean_difference'] == 1 for r in rows))

    def test_seed_p95_before_averaging_and_empty_design_rows(self):
        groups = [dict(degree=d, parsed_count=4 if d == 10 else 0,
            valid_count=4 if d == 10 else 0, invalid_count=0) for d in stats.DEGREES]
        manifest = dict(designs=[dict(design_id='a', dataset='test', groups=groups,
            binding={'design': {'design_family': 'a'}}), dict(design_id='empty', dataset='test',
            groups=[dict(degree=d, parsed_count=1, valid_count=0, invalid_count=1) for d in stats.DEGREES],
            binding={'design': {'design_family': 'empty'}})])
        metadata = dict(design_ids=np.array(['a']*4), dataset_ids=np.array(['test']*4),
            family_ids=np.array(['a']*4), degree=np.array([10]*4))
        seed_values = [np.array([0.,0.,0.,100.]), np.array([100.,0.,0.,0.]), np.array([0.,100.,0.,0.])]
        values = {(method,seed): seed_values[i] for method in stats.METHODS for i,seed in enumerate(stats.SEEDS)}
        tables = stats.build_tables(manifest, metadata, values)
        summaries = tables['summary']
        self.assertEqual(len(summaries), 10)
        for row in summaries:
            self.assertAlmostEqual(row['p95'], 85)
            self.assertAlmostEqual(row['net_weighted_mean'], 25)
            self.assertEqual(row['count'], 4)
            self.assertEqual(row['design_count'], 1)
        averaged_per_net_tail = np.quantile(np.mean(seed_values, axis=0), .95)
        self.assertNotAlmostEqual(summaries[0]['p95'], averaged_per_net_tail)
        empty = [r for r in tables['per_design_degree_seed'] if r['design_id'] == 'empty']
        self.assertEqual(len(empty), 75)
        self.assertTrue(all(r['count'] == 0 and r['mean'] is None and r['invalid_count'] == 1 for r in empty))
        self.assertTrue(all(r['mean_difference'] == 0 for r in tables['paired_design_cluster_ci']))

    def test_worst_design_is_computed_separately_each_seed(self):
        first = stats.pooled_metrics(np.array([0.,4.]), np.array(['a','b']), np.array(['a','b']))
        second = stats.pooled_metrics(np.array([6.,0.]), np.array(['a','b']), np.array(['a','b']))
        self.assertEqual(first['worst_design_id'], 'b')
        self.assertEqual(second['worst_design_id'], 'a')
        self.assertEqual((first['worst_design_mean']+second['worst_design_mean'])/2, 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
