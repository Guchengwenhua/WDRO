"""Read-only accounting for atomic ReferenceCache transactions and CEM resumes.

The cache commits its request before invoking the solver. A terminated solver
therefore leaves a receipt with completed=0; only the subsequent transaction
commits both reference values and the completed count. Never rewrite that row.
"""
from __future__ import annotations


def transaction_totals(rows, cached_count):
    assert rows, 'empty receipt ledger'
    assert [r[0] for r in rows] == list(range(1, len(rows) + 1)), 'receipt ids'
    for _, requested, started, completed in rows:
        assert all(type(v) is int for v in (requested, started, completed)), 'integer counts'
        assert 0 <= started <= requested and requested > 0, 'request bounds'
        assert completed in (0, started), 'non-atomic partial completion'
    totals = dict(receipts=len(rows), requested=sum(r[1] for r in rows),
                  misses_started=sum(r[2] for r in rows),
                  misses_completed=sum(r[3] for r in rows))
    assert totals['misses_completed'] == cached_count, 'committed values mismatch'
    totals['cached_values'] = cached_count
    totals['uncommitted_miss_requests'] = totals['misses_started'] - totals['misses_completed']
    totals['uncommitted_receipt_ids'] = [r[0] for r in rows if r[2] != r[3]]
    return totals


def reconcile_cem(rows, raw_events, cached_count, *, per_round=4096):
    """Align every receipt with a logged update or a bounded interrupted round.

    This deliberately rejects unexplained extra calls. An unlogged partial
    update is accepted only at an observed checkpoint-resume boundary, has at
    most four rounds, and ends in an uncommitted transaction. Completed raw
    updates must agree individually with both request and cache-miss ledgers.
    """
    totals = transaction_totals(rows, cached_count)
    assert all(r[1] == per_round for r in rows), 'CEM round request size'
    segments = [dict(from_update=0, events=[])]
    for event in raw_events:
        if event['event'] == 'resume':
            segments.append(dict(from_update=event['from_update'],
                                 resume_at_utc=event.get('at_utc'), events=[]))
        elif event['event'] == 'train':
            segments[-1]['events'].append(event)
    index = 0
    boundaries = []
    summaries = []
    raw_requested = raw_misses = raw_updates = 0
    for si, segment in enumerate(segments):
        events = segment['events']
        start = segment['from_update']
        assert events, 'empty training segment'
        assert [e['update'] for e in events] == list(range(start + 1, start + len(events) + 1)), 'raw update sequence'
        for event in events:
            block = rows[index:index + 4]
            assert len(block) == 4 and all(r[2] == r[3] for r in block), 'logged update used unfinished references'
            detail = event['method_detail']
            assert sum(r[1] for r in block) == detail['reference_requests'], 'raw request ledger'
            assert sum(r[3] for r in block) == detail['reference_cache_misses'], 'raw cache-miss ledger'
            assert detail['reference_requests'] - detail['reference_cache_misses'] == detail['reference_cache_hits'], 'raw hit ledger'
            raw_requested += detail['reference_requests']
            raw_misses += detail['reference_cache_misses']
            raw_updates += 1
            index += 4
        summaries.append(dict(from_update=start, first_update=events[0]['update'],
                              last_update=events[-1]['update'], raw_updates=len(events),
                              resume_at_utc=segment.get('resume_at_utc')))
        if si + 1 < len(segments):
            next_segment = segments[si + 1]
            assert next_segment['from_update'] <= events[-1]['update'], 'resume skips updates'
            pending = [j for j in range(index, min(index + 4, len(rows))) if rows[j][2] != rows[j][3]]
            if pending:
                assert len(pending) == 1, 'multiple interrupted rounds'
                end = pending[0] + 1
                block = rows[index:end]
                assert all(r[2] == r[3] for r in block[:-1]), 'earlier unfinished round'
                boundaries.append(dict(after_raw_update=events[-1]['update'],
                    interrupted_update=events[-1]['update'] + 1,
                    next_resume_from_update=next_segment['from_update'],
                    resume_at_utc=next_segment.get('resume_at_utc'),
                    receipts=[dict(zip(('id', 'requested', 'misses_started', 'misses_completed'), r)) for r in block]))
                index = end
    assert index == len(rows), 'unexplained receipts'
    accounted = [r for b in boundaries for r in b['receipts']]
    assert totals['requested'] == raw_requested + sum(r['requested'] for r in accounted), 'all requests accounted'
    assert totals['misses_completed'] == raw_misses + sum(r['misses_completed'] for r in accounted), 'all commits accounted'
    assert totals['uncommitted_receipt_ids'] == [b['receipts'][-1]['id'] for b in boundaries], 'all incomplete requests explained'
    return dict(status='pass', totals=totals, raw_updates=raw_updates,
                raw_reference_requests=raw_requested, raw_reference_misses=raw_misses,
                segments=summaries, interrupted_updates=boundaries,
                interpretation='started counts requests, not proved solver completions; uncommitted requests remain recorded')
