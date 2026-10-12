from week_common import *
def main():
 dev=read(ART/'gate.json');p=read(ART/'train/geometry_s101/segment_0007.json')
 evidence=p['completed_updates']>=7
 rate=p['wall_seconds_this_process']/7
 atomic(ART/'budget_gate.json',dict(allow=bool(dev.get('allow') and evidence),status='gate_evaluated',test_used=False,reason='development gate and seven official accelerated updates; no calendar cutoff',prefix_completed_updates=p['completed_updates'],projected_remaining_matrix_seconds=1493*rate,branch_allowance_seconds=None,estimate_scope='conservative serial work estimate; six GPUs reduce wall time subject to DAG'))
if __name__=='__main__':main()
