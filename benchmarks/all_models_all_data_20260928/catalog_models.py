"""Inventory trained endpoints; use checkpoint metadata, never test performance."""
import collections
import re
import time
from pathlib import Path
import torch
from common import *

def excluded(path):
    s = str(path.relative_to(PROJECT))
    for term in ('smoke', 'diagnostic', '/tests/', 'e2e_', 'resume_audit', 'resume_before', 'allocator_probe', 'gpu_pilot', 'cpu_resume', 'cpu_noeval', 'engineering_', 'recovery', '/lineage/', '/artifacts/inputs/', '/runs/cpu_', '/runs/review_'):
        if term in s:
            return 'engineering fixture or copied lineage: ' + term
    if path.name in ('initial.pt', 'labels.pt', 'label_progress.pt', 'init.pt'):
        return 'initialization or labels, not a trained endpoint'
    if path.parent.name == 'checkpoints' or re.match(r'(actor_u|u\d|step_)', path.stem):
        return 'intermediate recovery snapshot; endpoint/best retained'
    return None

def scalar_metadata(cp):
    result = {}
    for source in (cp.get('run_identity'), cp.get('identity'), cp.get('task'), cp.get('binding'), cp.get('config')):
        if isinstance(source, dict):
            for k, v in source.items():
                if isinstance(v, (str, int, float, bool)) and 'sha' not in k and len(str(v)) < 300:
                    result[k] = v
    for k in ('step','update','batch_idx','epoch'):
        if isinstance(cp.get(k), (int, float)):
            result[k] = cp[k]
    return result

def main():
    torch.set_num_threads(1)
    candidates = read(ART/'checkpoint_candidates.json')
    models, excluded_rows = [], []
    cache = ART/'model_metadata'; cache.mkdir(exist_ok=True)
    for idx, rec in enumerate(candidates):
        p = Path(rec['path']); reason = excluded(p)
        if reason:
            excluded_rows.append(dict(path=str(p), reason=reason)); continue
        mid = f'M{idx:04d}'
        dest = cache/(mid+'.json')
        if dest.exists():
            cached=read(dest)
            if cached.get('catalog_version')==2 or cached.get('state_key') is not None:
                models.append(cached); continue
        try:
            try:
                cp = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
            except RuntimeError:
                cp = torch.load(p, map_location='cpu', weights_only=False)
            meta = scalar_metadata(cp)
            state_key = next((k for k in ('actor_state_dict','actor','adapter','model_state_dict','model','state_dict','generator') if isinstance(cp.get(k), dict)), None)
            state = cp.get(state_key, {})
            if not state_key:
                for container in ('networks','models','state','current'):
                    if isinstance(cp.get(container),dict):
                        for key in ('actor','actor_state_dict'):
                            if isinstance(cp[container].get(key),dict):
                                state_key=container+'/'+key;state=cp[container][key];break
                        if state_key:break
            keys = list(state)
            family, strip = 'auxiliary', ''
            if 'embedder.conv1d.weight' in state and 'q_lx.weight' in state:
                family = 'rest'
            elif state_key == 'adapter':
                family = cp.get('identity', {}).get('family', 'unknown_adapter')
                if family == 'rest':
                    strip = 'actor.'
            elif 'actor.embedder.conv1d.weight' in state:
                family, strip = 'rest', 'actor.'
            trained_degree = meta.get('degree')
            if trained_degree is None:
                match = re.search(r'(?:_d|degree_?|rsmt)(\d+)', str(p))
                trained_degree = int(match.group(1)) if match else (10 if family in ('timing','hub','buffer') or state_key == 'adapter' else 30)
            seed = meta.get('training_seed', meta.get('seed'))
            if seed is None:
                match = re.search(r'_s(\d+)', str(p)); seed = int(match.group(1)) if match else None
            row = dict(catalog_version=2,id=mid, path=str(p), real_path=str(p.resolve()), bytes=p.stat().st_size,
                       mtime_ns=p.stat().st_mtime_ns, family=family, state_key=state_key,
                       strip_prefix=strip, train_degree=trained_degree, seed=seed,
                       label=str(p.relative_to(PROJECT)).replace('/artifacts/runs/',' / ').replace('/experiments/checkpoints/',' / ').replace('/artifacts/',' / '),
                       branch=p.relative_to(PROJECT).parts[0], metadata=meta, state_keys=keys[:8], status='ready')
            if family == 'rest':
                st = extract_state(cp, row)
                row['width'] = int(st['embedder.conv1d.weight'].shape[0])
                layers = [int(m.group(1)) for k in st if (m:=re.search(r'encoder\.layer_stack\.(\d+)\.', k))]
                row['depth'] = max(layers)+1 if layers else 3
                row['inner'] = int(next(v.shape[0] for k,v in st.items() if k.endswith('pos_ffn.w_1.weight')))
            elif family == 'auxiliary':
                row['status'] = 'not_standalone_router'
                row['reason'] = '攻击器/代理/生成器权重不直接输出布线树；其训练出的 REST 另列。不可将其当成独立布线模型。'
            if 'fixed_' in p.parent.name or p.parent.name.startswith('attacker'):
                row['note'] = '固定受害 REST；该行的线长不衡量攻击器强度。'
            if 'legacy' in row['branch'] or 'adversarial_v3_2026' in row['branch']:
                row['note'] = '历史训练权重，使用本次统一推理重新评分；不复用旧参考误差。'
            atomic(dest, row); models.append(row)
            del cp, state
        except Exception as exc:
            row = dict(id=mid,path=str(p),label=str(p.relative_to(PROJECT)),branch=p.relative_to(PROJECT).parts[0],family='unresolved',status='catalog_error',error=repr(exc))
            atomic(dest,row); models.append(row)
        if idx % 50 == 0:
            print(json.dumps(dict(phase='catalog_models', candidate=idx,total=len(candidates),rows=len(models))),flush=True)
    atomic(ART/'models.json',models)
    atomic(ART/'excluded_checkpoints.json',excluded_rows)
    print(json.dumps(dict(rows=len(models), status=dict(collections.Counter(m['status'] for m in models)), families=dict(collections.Counter(m['family'] for m in models))),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
