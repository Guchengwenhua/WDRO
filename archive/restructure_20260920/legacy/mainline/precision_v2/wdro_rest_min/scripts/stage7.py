#!/usr/bin/env python3
"""Stage 7 only: frozen ISPD protocol A evaluation and audited delivery."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'reference', 'evaluate', 'smoke', 'pipeline',
                                         'launch', 'freeze', 'audit', 'export', 'verify', 'status'])
    parser.add_argument('--degree', type=int)
    parser.add_argument('--model')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--preflight')
    parser.add_argument('--max-new-batches', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stage7.common import read, gate, LOG, PLAN, capture
    if args.action == 'prepare':
        from stage7.prepare import prepare
        p = prepare()
        print({'status': 'pass', 'plan': str(PLAN), 'models': len(p['models']), 'nets': sum(x['count'] for x in p['groups'])}, flush=True)
    elif args.action in ('reference', 'evaluate'):
        from stage7.evaluation import reference_task, evaluate, model_id
        assert args.preflight
        host = read(args.preflight)
        assert host['checks']['experiments_disk_meets_floor']
        if args.action == 'reference':
            result = reference_task(args.degree, args.preflight, args.max_new_batches)
        else:
            assert args.gpu in host['eligible_gpu_indices']
            model = next(m for m in gate()['models'] if model_id(m) == args.model)
            result = evaluate(model, args.preflight, args.gpu, output=args.output,
                              limit=args.limit, max_new_batches=args.max_new_batches)
        print({'status': result['status'], 'completed_nets': result['completed_nets']}, flush=True)
    elif args.action in ('smoke', 'pipeline', 'launch', 'freeze', 'status'):
        from stage7.operations import smoke, pipeline, launch, freeze, status
        {'smoke': smoke, 'pipeline': pipeline, 'launch': launch, 'freeze': freeze, 'status': status}[args.action]()
    else:
        from stage7.delivery import audit, export, verify
        {'audit': audit, 'export': export, 'verify': verify}[args.action]()


if __name__ == '__main__':
    main()
