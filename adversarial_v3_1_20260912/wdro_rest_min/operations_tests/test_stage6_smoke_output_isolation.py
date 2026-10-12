"""New equivalence smoke must never rewrite previously frozen evidence."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_stage6_smoke as six
import run_stage5_smoke as five
import precision_stage6_isolated_smoke as adapter


class OutputIsolationTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT.parent/'experiments/tmp/stage6_smoke_output_tests'
        parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.old=self.base/'stage5';self.old.mkdir()
        self.profile=self.old/'memory_profile.json'
        self.profile.write_text('{"frozen":true}\n')

    def test_new_profile_is_isolated_and_training_arguments_unchanged(self):
        observed=[]
        def fake_launch(*args):
            observed.append((five.LOG,args))
            five.LOG.mkdir(parents=True,exist_ok=True)
            (five.LOG/'memory_profile.json').write_text('{"new":true}\n')
            return dict(completed_updates=12)
        def original_work():
            return five.launch('selected_equivalence_continuous',dict(config='/frozen/config.json',rho_mode='q75'),12)
        with patch.object(six,'LOG',self.base/'stage6'),patch.object(five,'LOG',self.old),patch.object(five,'launch',fake_launch),patch.object(six,'main',original_work):
            self.assertEqual(adapter.main(),dict(completed_updates=12))
            self.assertEqual(five.LOG,self.old)
        self.assertEqual(self.profile.read_text(),'{"frozen":true}\n')
        self.assertEqual(observed,[(self.base/'stage6/stage5_selected_equivalence',
            ('selected_equivalence_continuous',dict(config='/frozen/config.json',rho_mode='q75'),12))])

    def test_exception_restores_helper_destination(self):
        with patch.object(six,'LOG',self.base/'stage6'),patch.object(five,'LOG',self.old),patch.object(six,'main',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                adapter.main()
            self.assertEqual(five.LOG,self.old)
        self.assertEqual(self.profile.read_text(),'{"frozen":true}\n')


if __name__=='__main__':unittest.main(verbosity=2)
