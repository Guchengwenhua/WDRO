"""Summarize actual completed results; never infer unobserved metrics."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 lines=['# Checkpoint-bank exploratory results','','Degree 10; 200 continued updates; batch 64. Original GeoSteiner cached test references, 256 fixed instances per group. All values below are measured means; incomplete arms remain explicitly marked.','','| Arm | Seed | Status | Initial mean regret | Best mean regret | Change | Best worst OOD group | Bank gap | Proxy bias | f argmax agreement |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
 for seed in [101,102]:
  for arm in ['geo','k1','k3','k5','refresh3']:
   d=ROOT/'artifacts'/f'{arm}_s{seed}'
   if not (d/'complete.json').exists():lines.append(f'| {arm} | {seed} | incomplete | | | | | | | |');continue
   initial=json.loads((d/'initial_test.json').read_text());best=json.loads((d/'best_test.json').read_text());diag=json.loads((d/'offline_bank_diagnostic.json').read_text())
   a=np.mean([r['mean'] for r in initial['groups'].values()]);b=np.mean([r['mean'] for r in best['groups'].values()])
   lines.append(f"| {arm} | {seed} | complete | {a:.8f} | {b:.8f} | {b-a:+.8f} | {best['score']:.8f} | {diag['bank_gap_mean']:.8f} | {diag['proxy_bias_mean']:.8f} | {diag['argmax_agreement']:.4f} |")
 lines+=['','These are exploratory continuation results, not a replication of the 2000-update/batch-256 formal protocol. Historical teachers and cached validation labels are oracle-assisted. A negative change means lower independently referenced test error. Geo arm diagnostic uses K3 only for offline comparison. No claim of statistical significance is made from this table.']
 tmp=ROOT/'RESULTS.md.partial';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(ROOT/'RESULTS.md')
if __name__=='__main__':main()
