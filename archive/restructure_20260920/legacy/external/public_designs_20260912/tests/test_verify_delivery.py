"""Independent delivery checks: mutation detection and aggregation semantics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

PATH=Path(__file__).resolve().parents[1]/'verify_delivery.py'
spec=importlib.util.spec_from_file_location('verify_delivery',PATH)
verifier=importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class DeliveryVerifierTests(unittest.TestCase):
    def test_final_json_serializes_nested_numpy_scalar_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'delivery_verification.json'
            verifier.atomic_json(path,dict(status='pass',counts={'statistics_rows':np.int64(3260)},
                nested=[{'ratio':np.float32(.5),'passed':np.bool_(True)}]))
            result=json.loads(path.read_text())
            self.assertEqual(result,dict(status='pass',counts={'statistics_rows':3260},nested=[{'ratio':.5,'passed':True}]))
            self.assertEqual(list(Path(directory).glob('*.partial')),[])

    def test_sha_mismatch_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source';path.write_text('original')
            audit=verifier.Audit(directory)
            with self.assertRaisesRegex(AssertionError,'SHA256 mismatch'):
                audit.check(dict(path=str(path),sha256='0'*64))

    def test_mutation_during_audit_cannot_use_cached_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source';path.write_text('original')
            binding=dict(path=str(path),sha256=verifier.digest(path))
            audit=verifier.Audit(directory);audit.check(binding)
            path.write_text('changed and larger')
            with self.assertRaisesRegex(AssertionError,'Artifact changed'):
                audit.check(binding)

    def test_net_weight_and_equal_design_weight_are_distinct(self):
        values=np.array([0.,0.,0.,1.])
        result=verifier.pooled(values,np.array(['a','a','a','b']),np.array(['fa','fa','fa','fb']))
        self.assertEqual(result['net_weighted_mean'],.25)
        self.assertEqual(result['design_macro_mean'],.5)
        self.assertEqual(result['worst_design_mean'],1.)
        self.assertEqual(result['worst_design_id'],'b')
        self.assertEqual(result['design_count'],2)
        self.assertAlmostEqual(result['p95'],.85)

    def test_seed_p95_must_be_computed_before_seed_mean(self):
        seeds=np.array([[0.,1.],[1.,0.],[0.,1.]])
        metric_mean=np.mean([verifier.basic(row)['p95'] for row in seeds])
        wrong=verifier.basic(seeds.mean(axis=0))['p95']
        self.assertAlmostEqual(metric_mean,.95)
        self.assertNotAlmostEqual(metric_mean,wrong)
        with self.assertRaisesRegex(AssertionError,'!='):
            verifier.compare_value(wrong,metric_mean,'p95')

    def test_empty_cells_are_explicit_null_metrics(self):
        expected=dict(count=0,mean=None,p95=None,minimum=None,maximum=None)
        self.assertEqual(verifier.basic(np.array([])),expected)
        verifier.compare_value('',None,'empty CSV')
        with self.assertRaises(AssertionError):
            verifier.compare_value('0',None,'fabricated zero')


if __name__=='__main__':
    unittest.main()
