#!/usr/bin/env python3
"""Stage 8 only: independent attacks, efficiency, statistics and delivery."""
import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'attack', 'audit-cell', 'audit', 'smoke', 'freeze',
                                    'pipeline', 'launch', 'status', 'export', 'verify', 'supporting','launch-supporting'])
    p.add_argument('--model')
    p.add_argument('--distribution')
    p.add_argument('--gpu', type=int)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--max-new-rounds', type=int)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.action == 'prepare':
        from stage8.prepare import prepare
        r = prepare()
        print({'status': r['status'], 'groups': len(r['groups']), 'models': len(r['models'])}, flush=True)
    elif args.action == 'attack':
        from stage8.attack import attack, model_id
        from stage8.common import gate, read
        from datetime import datetime, timezone
        host = read(args.preflight)
        assert host['checks']['experiments_disk_meets_floor'] and args.gpu in host['eligible_gpu_indices']
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(host['captured_at_utc'])).total_seconds()
        assert 0 <= age < 180, 'stale launch admission; capture a new host-runtime'
        model = next(m for m in gate()['models'] if model_id(m) == args.model)
        r = attack(model, args.distribution, args.gpu, args.preflight, output=args.output,
                   limit=args.limit, max_new_rounds=args.max_new_rounds, workers=args.workers)
        print({'status': r['status'], 'completed_nets': r['completed_nets']}, flush=True)
    elif args.action in ('smoke', 'freeze', 'pipeline', 'launch', 'status','launch-supporting'):
        from stage8 import operations
        getattr(operations, args.action.replace('-','_'))()
    elif args.action in ('audit-cell', 'audit'):
        from stage8.audit import audit_cell, audit
        if args.action == 'audit-cell':
            print(audit_cell(args.output), flush=True)
        else:
            audit()
    else:
        from stage8 import delivery
        getattr(delivery, 'supporting_statistics' if args.action=='supporting' else args.action)()


if __name__ == '__main__':
    main()
