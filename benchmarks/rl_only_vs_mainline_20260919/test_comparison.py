"""Checks for cohort inference/statistics and the two real GPU smoke artifacts."""
import unittest
import numpy as np
from common import ROOT, read, checked, G, independent_lengths
from report import bootstrap


class ComparisonTests(unittest.TestCase):
    def test_cross_cohort_seeds_are_not_artificially_paired(self):
        # Identical seed-dependent outcomes: paired within-cohort differences
        # must vanish, whereas unrelated cohorts retain seed uncertainty.
        values=np.broadcast_to(np.arange(3)[None,:,None,None],(3,3,4,16)).copy()
        boot=bootstrap(values,['old','old','new'],repeats=200)
        np.testing.assert_array_equal(boot[:,0],boot[:,1])
        self.assertGreater(np.var(boot[:,0]-boot[:,2]),0)
        np.testing.assert_array_equal(boot,bootstrap(values,['old','old','new'],repeats=200))

    def test_gpu_smokes_topology_and_reuse(self):
        for model in ('mainline_group_dro_s101','rl_only_erm_rollout_s101'):
            folder=ROOT/'artifacts/smoke'/model
            c=read(folder/'complete.json')
            self.assertEqual(c['acceptance'],'gpu_smoke')
            self.assertEqual(c['actual_calls']['unresolved_calls'],0)
            for cell in c['cells']:
                datafile=ROOT/'artifacts/data'/f'{cell["key"]}.npz'
                with np.load(datafile) as d:
                    pins=d['pins'].reshape(-1,30,2)[:cell['count']]
                with np.load(checked(cell['path'],cell['sha256'])) as z:
                    np.testing.assert_allclose(independent_lengths(pins,z['topology']),z['length'],atol=1e-12,rtol=0)
                self.assertEqual(cell['binding']['data_sha256'],G.sha256(datafile))
            expected=256 if model.startswith('rl_only') else 0
            self.assertEqual(c['reused_rows'],expected)

if __name__=='__main__':unittest.main()
