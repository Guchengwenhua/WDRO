"""Cost feasibility from the official 100-update prefix; never reads test results."""
from datetime import datetime,timezone
from week_common import *
def main():
 dev=read(ART/'gate.json');prefix=ART/'train/geometry_s101'
 marker=read(prefix/'segment_0100.json')
 events={}
 for line in (prefix/'logs/events.jsonl').read_text().splitlines():
  try:r=__import__('json').loads(line)
  except ValueError:continue
  if r.get('event')=='train' and 1<=r.get('update',0)<=100:events[r['update']]=r
 keys=sorted(events)
 evidence=len(keys)>=20 and marker['completed_updates']>=100
 policy=read(PROJECT/'holiday_research_20260930/policy.json')
 unlimited=policy.get('run_until_complete',False)
 now=datetime.now(timezone.utc)
 cutoff=None if unlimited else datetime.fromisoformat(policy['training_cutoff_utc'])
 remaining=None if unlimited else max(0.,(cutoff-now).total_seconds())
 allowance=None if unlimited else min(72*3600,.45*remaining)
 rate=None;projected=None
 if evidence:
  first=datetime.fromisoformat(events[keys[0]]['at_utc'].replace('Z','+00:00'))
  last=datetime.fromisoformat(events[keys[-1]]['at_utc'].replace('Z','+00:00'))
  rate=max((last-first).total_seconds()/(keys[-1]-keys[0]),marker['wall_seconds_this_process']/100)
  # H assigned same conservative bound as geometry; 6 matched H/G jobs less prefix100.
  projected=11900*rate*1.25+6*2000*2.*1.25+7200
 allow=bool(dev.get('allow') and evidence and projected is not None and (unlimited or projected<=allowance))
 reason='development evidence and measured official-prefix cost permit all three matched-seed blocks' if allow else 'do not start the full matrix: development gate denied, insufficient prefix evidence, or measured projection exceeds reserved branch time'
 atomic(ART/'budget_gate.json',dict(allow=allow,reason=reason,status='gate_evaluated',test_used=False,
  development_allow=dev.get('allow'),prefix_completed_updates=marker['completed_updates'],prefix_events=len(keys),
  measured_seconds_per_update_upper_estimate=rate,projected_remaining_matrix_seconds=projected,
  available_before_training_cutoff_seconds=remaining,branch_allowance_seconds=allowance,
  rule='user-authorized completion: retain dev and prefix evidence gates; measured cost is informational' if unlimited else 'all3seed full matched matrix must fit min(72h,45%remaining training window);1.25timing safety;H cost upper-bounded by geometry;original/random2s/update;2h final evaluation reserve',
  cutoff=cutoff.isoformat() if cutoff else None,run_until_complete=unlimited,
  runtime_override='User 2026-10-06: continue until complete; projection retained as ETA, not a calendar/budget stop' if unlimited else None,
  evaluated_at=now.isoformat()))
if __name__=='__main__':main()
