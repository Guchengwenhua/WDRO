"""Resolve nested checkpoint schemas from their producer code before inference."""
from common import *

def main():
    models=read(ART/'models.json');kept=[];excluded=read(ART/'excluded_checkpoints.json')
    for m in models:
        path=m['path']
        if any(token in path for token in ('/runs/cpu_','/runs/review_')):
            excluded.append(dict(path=path,reason='engineering or acceptance replay fixture, not a trained experiment'));continue
        key=None
        if m['state_key'] is None:
            if m['branch'] in ('rl_only_20260915','oracle_free_20260915') and '/runs/' in path and path.endswith('/latest.pt'):
                key='networks/actor'
            elif m['branch']=='critic_search_20260915' and '/training/' in path and path.endswith('/latest.pt'):
                key='current/actor_state_dict'
            elif m['branch']=='adversarial_v3_1_20260912' and '/joint/' in path and path.endswith('/latest.pt'):
                key='current/actor_state_dict'
        if key:
            m.update(family='rest',state_key=key,strip_prefix='',status='ready',width=128,depth=3,inner=512,
                     schema_source='producer checkpoint serialization; strict load occurs in formal evaluation')
            m.pop('reason',None)
        if m.get('state_key') is None and m['family']=='auxiliary':
            m['reason']='仅含训练攻击器/代理的检查点；不直接输出布线树。对应的受害 REST 或继续训练 REST 另列。'
        kept.append(m)
    atomic(ART/'models.json',kept);atomic(ART/'excluded_checkpoints.json',excluded)
    from collections import Counter
    print(json.dumps(dict(rows=len(kept),families=dict(Counter(m['family'] for m in kept)),status=dict(Counter(m['status'] for m in kept))),ensure_ascii=False))

if __name__=='__main__':main()
