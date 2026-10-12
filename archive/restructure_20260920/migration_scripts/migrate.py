#!/usr/bin/env python3
"""One-shot journalled layout migration. Does not traverse artifact symlinks."""
from pathlib import Path
import os, json, shutil, hashlib, re, time

ROOT = Path('/remote-home/Guilin/DRO/Codex')
AUDIT = ROOT / 'archive/restructure_20260920'
LEGACY = AUDIT / 'legacy'
SCRATCH = Path('/tmp/dro_restructure_20260920')
METHODS = {
    'revisions/reference_precision_v2/Codex': 'wdro_precision_v2',
    'revisions/adversarial_v3_20260912/Codex': 'adversarial_v3_20260912',
    'revisions/adversarial_v3_1_20260912/Codex': 'adversarial_v3_1_20260912',
    **{f'improvements/{x}': x for x in ['critic_search_20260915', 'oracle_free_20260915', 'rl_only_20260915', 'rl_research_20260915', 'papers_wdro_20260915', 'rl_only_pretraining_plan_20260919']},
    'improvements/rl_only_vs_mainline_20260919': 'benchmarks/rl_only_vs_mainline_20260919',
    'external_benchmarks/20260912_public': 'benchmarks/public_designs_20260912',
    'external_benchmarks/20260912_v31_models': 'benchmarks/v31_models_20260912',
}
MAP = dict(METHODS)
MAP.update({'wdro_rest_min':'wdro_legacy_v1/wdro_rest_min', 'experiments':'shared/experiments',
            'experiments.local_backup_20260904T1127Z':'archive/storage_backup_20260904',
            'project_tools':'tools', 'docs':'summaries', 'docs/mainline':'wdro_precision_v2/docs/stages',
            'docs/improvements/ADVERSARIAL_V3_1.md':'adversarial_v3_1_20260912/docs/OVERVIEW.md'})
for rev in ['reference_precision_v2','adversarial_v3_20260912','adversarial_v3_1_20260912']:
    MAP[f'revisions/{rev}/DRO']='shared/legacy_dro'
for old,new in METHODS.items():
    for p in (ROOT/old).glob('*.md'):
        if p.name != 'README.md': MAP[str(p.relative_to(ROOT))]=new+'/docs/'+p.name
root_docs={}
for p in ROOT.glob('*.md'):
    if p.name=='README.md': continue
    if p.name.startswith(('阶段6','阶段7')): dest='wdro_precision_v2/docs/history/'
    elif p.name.startswith(('阶段','迁移后','WDRO-REST_最简','论文同步','00_')): dest='wdro_legacy_v1/docs/history/'
    elif p.name.startswith(('04_','05_对抗')): dest='adversarial_v3_1_20260912/docs/history/'
    elif p.name.startswith('05_公开'): dest='benchmarks/public_designs_20260912/docs/history/'
    else: dest='summaries/history/'
    root_docs[p.name]=dest+p.name; MAP[p.name]=dest+p.name

def mapped(rel):
    rel=str(rel)
    for old,new in sorted(MAP.items(), key=lambda x:-len(x[0])):
        if rel==old or rel.startswith(old+'/'): return new+rel[len(old):]
    return rel

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def link(p,target):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.is_symlink():p.unlink()
    assert not p.exists(),p
    p.symlink_to(os.path.relpath(target,p.parent))

def prepare():
    assert not (AUDIT/'prepared.json').exists()
    AUDIT.mkdir(parents=True,exist_ok=True)
    write_json(AUDIT/'path_map.json',MAP)
    inventory=[]
    for base,ds,fs in os.walk(ROOT,followlinks=False):
        b=Path(base)
        ds[:]=[d for d in ds if b/d != AUDIT]
        for name in list(ds)+fs:
            p=b/name; st=p.lstat()
            row=dict(path=str(p.relative_to(ROOT)),mode=st.st_mode,size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev)
            if p.is_symlink():row['target']=os.readlink(p)
            elif p.is_dir():continue
            inventory.append(row)
    write_json(AUDIT/'inventory_before.json',inventory)
    heavy={'artifacts','manifests','experiments','sources','.cache','.deps','results','logs','cache','tables','figures','evidence','vendor_harrl'}
    frozen=[]
    def snapshot(src,dst):
        rel=str(src.relative_to(ROOT))
        if src.is_symlink():
            # Freeze document bytes, but use one canonical storage target for data.
            if src.suffix=='.md' and src.is_file():
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                frozen.append(dict(old_path=rel,snapshot=str(dst.relative_to(ROOT)),sha256=digest(dst)))
            else:
                resolved=src.resolve()
                target=ROOT/mapped(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else resolved
                link(dst,target)
        elif src.is_dir():
            if src.name=='__pycache__':return
            if rel=='archive':
                dst.mkdir(parents=True,exist_ok=True)
                for p in src.iterdir():
                    if p!=AUDIT:link(dst/p.name,p.resolve())
                return
            if src.name in heavy or (rel.startswith('revisions/') and src.name=='DRO'):
                link(dst,ROOT/mapped(rel));return
            dst.mkdir(parents=True,exist_ok=True)
            for p in src.iterdir():snapshot(p,dst/p.name)
        else:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            frozen.append(dict(old_path=rel,snapshot=str(dst.relative_to(ROOT)),sha256=digest(dst)))
    LEGACY.mkdir()
    for p in list(ROOT.iterdir()):
        snapshot(p,LEGACY/p.name)
    link(AUDIT/'DRO',ROOT.parent/'DRO')
    link(AUDIT/'Codex',LEGACY)
    write_json(AUDIT/'frozen_files.json',frozen)
    for p in SCRATCH.glob('*duplicates.json'):shutil.copy2(p,AUDIT/p.name)
    write_json(AUDIT/'prepared.json',dict(time=time.time(),inventory=len(inventory),frozen=len(frozen)))
    print('PREPARED',len(inventory),'inventory entries;',len(frozen),'frozen source/doc/config files',flush=True)

def apply():
    assert (AUDIT/'prepared.json').exists() and not (AUDIT/'applied.json').exists()
    journal=json.loads((AUDIT/'moves.json').read_text()) if (AUDIT/'moves.json').exists() else []
    def move(old,new):
        src=ROOT/old;dst=ROOT/new
        if not src.exists() and not src.is_symlink():
            assert dst.exists(),(src,dst)
            return
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            assert src.is_dir() and dst.is_dir() and not src.is_symlink(),(src,dst)
            for child in list(src.iterdir()):move(str(child.relative_to(ROOT)),str((dst/child.name).relative_to(ROOT)))
            src.rmdir()
        else:src.rename(dst)
        journal.append(dict(old=old,new=new));write_json(AUDIT/'moves.json',journal)
    # Materialize current reading copies; originals remain in the frozen snapshot/archive.
    for old,new in root_docs.items():
        if not (ROOT/old).exists():continue
        dst=ROOT/new;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/old,dst)
        (ROOT/old).unlink()
    for old,new in METHODS.items():move(old,new)
    for old,new in [('wdro_rest_min','wdro_legacy_v1/wdro_rest_min'),('project_tools','tools'),
                    ('experiments.local_backup_20260904T1127Z','archive/storage_backup_20260904')]:move(old,new)
    # Move project documentation into its owning method.
    for old,new in MAP.items():
        for a,b in METHODS.items():
            if old.startswith(a+'/') and old.endswith('.md'):
                current=b+old[len(a):]
                if current!=new and (ROOT/current).exists():move(current,new)
    move('docs/mainline','wdro_precision_v2/docs/stages')
    move('docs/improvements/ADVERSARIAL_V3_1.md','adversarial_v3_1_20260912/docs/OVERVIEW.md')
    (ROOT/'docs/improvements').rmdir();move('docs','summaries')
    move('experiments','shared/experiments')
    link(ROOT/'shared/legacy_dro',ROOT.parent/'DRO')
    # Duplicate removal is conditional on an immediate second hash comparison.
    removed=[]
    for rev in ['adversarial_v3_20260912','adversarial_v3_1_20260912']:
        evidence=json.loads((AUDIT/(rev+'_duplicates.json')).read_text())
        assert not evidence['different'] and not evidence['unique']
        for row in evidence['identical']:
            assert digest(Path(row['path']))==row['sha256']==digest(Path(row['source']))
        directory=ROOT/'revisions'/rev/'DRO'
        shutil.rmtree(directory);removed.extend(evidence['identical'])
    # Remaining former entry directories contain only obsolete navigation.
    for name in ['revisions','improvements','external_benchmarks','mainline','external']:
        src=ROOT/name
        if src.exists():move(name,'archive/restructure_20260920/old_navigation/'+name)
    # Old program and manifest paths see ORIGINAL source bytes through a frozen view.
    for name in ['revisions','improvements','external_benchmarks','wdro_rest_min','project_tools','mainline','external','docs']:
        link(ROOT/name,LEGACY/name)
    link(ROOT/'experiments',ROOT/'shared/experiments')
    # Keep only direct legacy final-document bindings at the root.
    keep=['01_WDRO-REST_项目进度.md','03_项目文件总索引.md','05_公开芯片外部测试扩展_报告.md',
          '阶段3_三基线与验收报告.md','阶段3_子步骤与运行说明.md','WDRO-REST_最简实验执行计划.md']
    for name in keep:link(ROOT/name,LEGACY/name)
    # Update internal artifact/category symlinks to canonical targets (never follow them).
    for base,ds,fs in os.walk(ROOT,followlinks=False):
        b=Path(base)
        ds[:]=[d for d in ds if b/d!=AUDIT]
        for name in ds+fs:
            p=b/name
            if not p.is_symlink() or p.parent==ROOT:continue
            target=os.readlink(p)
            old_abs=Path(os.path.normpath(str(p.parent/target))) if not os.path.isabs(target) else Path(target)
            if old_abs.is_relative_to(ROOT):
                canonical=ROOT/mapped(old_abs.relative_to(ROOT))
                if canonical!=old_abs:link(p,canonical)
    write_json(AUDIT/'duplicates_removed.json',dict(files=len(removed),bytes=sum(r['bytes'] for r in removed),records=removed))
    write_json(AUDIT/'applied.json',dict(time=time.time(),moves=len(journal),root_compatibility_docs=keep))
    print('MOVED',len(journal),'entries; removed',len(removed),'verified duplicate files',flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='apply':
        MAP=json.loads((AUDIT/'path_map.json').read_text())
        root_docs={a:b for a,b in MAP.items() if '/' not in a and a.endswith('.md')}
    {'prepare':prepare,'apply':apply}[sys.argv[1]]()
