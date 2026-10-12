"""Fault injection and corruption rejection for the receipt audit supplement."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.candidates import ReferenceCache
from precision_reference_receipts import transaction_totals, reconcile_cem


def event(update, misses):
    return dict(event='train', update=update, method_detail=dict(
        reference_requests=8, reference_cache_misses=misses, reference_cache_hits=8-misses))


def fixture():
    values = [(2, 2, 2)] * 10 + [(2, 2, 0)] + [(2, 0, 0)] * 6 + [(2, 2, 2)] * 2
    rows = [(i + 1, *r) for i, r in enumerate(values)]
    events = [event(1, 8), event(2, 8), dict(event='resume', from_update=1), event(2, 0), event(3, 4)]
    return rows, events


class ReceiptTests(unittest.TestCase):
    def test_real_cache_interruption_then_retry(self):
        temp_root = ROOT.parent / 'experiments/tmp/receipt_fault_injection'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / 'cache.sqlite'
            pins = np.asarray([[[.1, .1], [.9, .9]], [[.1, .9], [.9, .1]]], dtype=np.float32)
            class FailedSolver:
                def lengths(self, _):
                    raise RuntimeError('injected interruption before reference transaction')
            class GoodSolver:
                def lengths(self, value):
                    return np.full(len(value), 1.6)
            cache = ReferenceCache(path)
            with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                cache.lengths(pins, FailedSolver())
            self.assertEqual(cache.db.execute('SELECT COUNT(*) FROM lstar').fetchone()[0], 0)
            cache.close()
            cache = ReferenceCache(path)
            lengths, misses = cache.lengths(pins, GoodSolver())
            self.assertEqual(misses, 2)
            np.testing.assert_array_equal(lengths, [1.6, 1.6])
            rows = cache.db.execute('SELECT * FROM calls ORDER BY id').fetchall()
            totals = transaction_totals(rows, 2)
            self.assertEqual(totals['uncommitted_receipt_ids'], [1])
            self.assertNotEqual(totals['misses_started'], totals['misses_completed'])
            cache.close()

    def test_all_receipts_align_across_rollback(self):
        rows, events = fixture()
        result = reconcile_cem(rows, events, 24, per_round=2)
        self.assertEqual(result['totals']['uncommitted_receipt_ids'], [11])
        self.assertEqual(result['raw_updates'], 4)
        self.assertEqual(result['interrupted_updates'][0]['interrupted_update'], 3)

    def test_partial_commit_rejected(self):
        rows, events = fixture(); rows[10] = (11, 2, 2, 1)
        with self.assertRaisesRegex(AssertionError, 'non-atomic'):
            reconcile_cem(rows, events, 25, per_round=2)

    def test_missing_reference_rejected(self):
        rows, events = fixture()
        with self.assertRaisesRegex(AssertionError, 'committed values'):
            reconcile_cem(rows, events, 23, per_round=2)

    def test_false_logged_misses_rejected(self):
        rows, events = fixture(); events[0] = event(1, 6)
        with self.assertRaisesRegex(AssertionError, 'raw cache-miss'):
            reconcile_cem(rows, events, 24, per_round=2)

    def test_unexplained_extra_receipt_rejected(self):
        rows, events = fixture(); rows.append((20, 2, 0, 0))
        with self.assertRaisesRegex(AssertionError, 'unexplained receipts'):
            reconcile_cem(rows, events, 24, per_round=2)

    def test_missing_resume_evidence_rejected(self):
        rows, events = fixture(); events = [e for e in events if e['event'] != 'resume']
        with self.assertRaises(AssertionError):
            reconcile_cem(rows, events, 24, per_round=2)

    def test_pending_transaction_cannot_back_logged_update(self):
        rows, events = fixture(); rows[0] = (1, 2, 2, 0)
        with self.assertRaisesRegex(AssertionError, 'unfinished references'):
            reconcile_cem(rows, events, 22, per_round=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
