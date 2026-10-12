#!/usr/bin/env python3
"""Read-only structural checks; writes a separate relocation receipt, never scientific receipts."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import subprocess
import stat
import time
from urllib.parse import unquote
from resolve_path import ROOT, AUDIT, current_path, historical_path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    errors=[];counts={};notices=[]
    frozen=json.loads((AUDIT/'frozen_files.json').read_text())
    for row in frozen:
        p=ROOT/row['snapshot']
        if not p.is_file() or sha(p)!=row['sha256']:errors.append(['frozen_source',row['old_path']])
    counts['frozen_originals']=len(frozen)
    inventory=json.loads((AUDIT/'inventory_before.json').read_text())
    rewritten=json.loads((AUDIT/'rewritten_files.json').read_text())
    for row in rewritten:
        p=ROOT/row['new']
        if not p.is_file() or sha(p)!=row['after_sha256']:errors.append(['maintained_source_changed',row['new']])
    mutable={row['old'] for row in rewritten}
    mutable.add('README.md')
    copies={row['old_path']:row for row in frozen}
    duplicate=json.loads((AUDIT/'duplicates_removed.json').read_text())
    duplicate_names={str(Path(row['path']).relative_to(ROOT)) for row in duplicate['records']}
    preserved=0;regenerable=0
    side_effects=json.loads((AUDIT/'test_fixture_changes.json').read_text()) if (AUDIT/'test_fixture_changes.json').exists() else []
    for row in inventory:
        old=row['path'];p=current_path(old)
        if 'target' in row:continue
        if stat.S_ISSOCK(row['mode']):
            if not p.exists() or not stat.S_ISSOCK(p.lstat().st_mode):errors.append(['missing_socket',old])
            else:preserved+=1
            continue
        if '__pycache__' in Path(old).parts and '/DRO/WDRO-REST/' in old:
            regenerable+=1;continue
        if old in duplicate_names:
            # Duplicates are checked against their original source below.
            continue
        if not p.is_file():
            if '__pycache__' in Path(old).parts and old.startswith('revisions/'):
                regenerable+=1;continue
            errors.append(['missing_original',old,str(p)]);continue
        st=p.stat()
        if (st.st_ino,st.st_dev,st.st_size,st.st_mtime_ns)==(row['inode'],row['device'],row['size'],row['mtime_ns']):
            preserved+=1;continue
        if old in copies:
            # Current code/doc paths may be edited; the frozen bytes are audited above.
            preserved+=1;continue
        # Storage deduplication can replace a file by a checked shared-data link.
        backup=AUDIT/'storage_backup_deduplication.json'
        known=json.loads(backup.read_text()) if backup.exists() else []
        item=next((x for x in known if x['backup']==str(p)),None)
        if item and sha(p)==item['sha256']:
            preserved+=1;continue
        fixture=next((x for x in side_effects if x['old_path']==old),None)
        if fixture and sha(p)==fixture['after_sha256']:
            notices.append(fixture);continue
        errors.append(['unrecorded_identity_change',old,str(p)])
    counts.update(preserved_file_identities=preserved,discarded_duplicate_bytecode=regenerable)
    for row in duplicate['records']:
        p=Path(row['source'])
        if not p.is_file() or sha(p)!=row['sha256']:errors.append(['duplicate_source',str(p)])
    counts['removed_verified_duplicates']=duplicate['files']
    owned=json.loads((AUDIT/'maintained_files.json').read_text())
    for name in ['tools/resolve_path.py','tools/verify_layout_20260920.py']:
        if name not in owned['code']:owned['code'].append(name)
    for name in owned['code']:
        p=ROOT/name
        if p.suffix=='.py':
            try:ast.parse(p.read_text(),filename=str(p))
            except SyntaxError as e:errors.append(['python_syntax',name,str(e)])
        elif p.suffix=='.sh':
            check=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
            if check.returncode:errors.append(['shell_syntax',name,check.stderr])
    counts['parsed_owned_code']=len(owned['code'])
    # Only project reading docs; immutable historical snapshots and vendor docs
    # deliberately retain their original relative-path semantics.
    document_paths=set(owned['docs'])|{'README.md','shared/README.md','tools/README.md','benchmarks/README.md'}
    for folder in ['summaries','wdro_precision_v2','wdro_legacy_v1','adversarial_v3_20260912',
                   'adversarial_v3_1_20260912','critic_search_20260915','oracle_free_20260915',
                   'rl_only_20260915','rl_research_20260915','papers_wdro_20260915',
                   'rl_only_pretraining_plan_20260919','benchmarks']:
        base=ROOT/folder
        if (base/'README.md').exists():document_paths.add(folder+'/README.md')
        for p in (base/'docs').rglob('*.md'):document_paths.add(str(p.relative_to(ROOT)))
        if folder=='summaries':
            for p in base.rglob('*.md'):document_paths.add(str(p.relative_to(ROOT)))
        if folder=='benchmarks':
            for sub in base.iterdir():
                if sub.is_dir():
                    for p in (sub/'docs').rglob('*.md'):document_paths.add(str(p.relative_to(ROOT)))
                    if (sub/'README.md').exists():document_paths.add(str((sub/'README.md').relative_to(ROOT)))
    broken=[];link_count=0
    for name in sorted(document_paths):
        p=ROOT/name
        if not p.exists():errors.append(['missing_reading_document',name]);continue
        text=p.read_text()
        for match in re.finditer(r'(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)',text):
            url=match.group(1).strip('<>').split('#',1)[0]
            if not url or re.match(r'[a-zA-Z]+:',url):continue
            url=unquote(url);url=re.sub(r':\d+$','',url);dest=p.parent/url;link_count+=1
            if not dest.exists():broken.append(dict(document=name,target=url))
    (AUDIT/'validation/document_links.json').write_text(json.dumps(dict(checked=link_count,broken=broken),ensure_ascii=False,indent=2))
    errors.extend(['broken_document_link',x] for x in broken)
    counts.update(reading_documents=len(document_paths),local_links=link_count)
    aliases=0
    for row in frozen:
        old=ROOT/row['old_path']
        # Removed root navigation names remain available through historical_path.
        if old.exists() and old.resolve().is_relative_to(AUDIT/'legacy'):
            aliases+=1
            if sha(old)!=row['sha256']:errors.append(['legacy_alias_bytes',row['old_path']])
    counts['legacy_alias_originals']=aliases
    before_links={x['path']:x['target'] for x in inventory if 'target' in x}
    links=[];broken=[]
    for base,dirs,files in os.walk(ROOT,followlinks=False):
        for name in dirs+files:
            p=Path(base)/name
            if not p.is_symlink():continue
            rel=str(p.relative_to(ROOT));target=os.readlink(p)
            links.append(dict(path=rel,target=target))
            if not p.exists():
                old=next((x for x in before_links if current_path(x)==p),None)
                # Zig stores a cache identifier as the symlink value; it is not
                # a filesystem destination. Preserve this pre-existing marker.
                if old and before_links[old]==target and '/.cache/zig_global/' in rel and p.name=='lld.id':
                    notices.append(dict(preexisting_zig_cache_identifier=rel,target=target))
                else:broken.append(dict(path=rel,target=target))
    (AUDIT/'validation/symlinks.json').write_text(json.dumps(dict(links=links,broken=broken),ensure_ascii=False,indent=2))
    errors.extend(['broken_symlink',x] for x in broken)
    counts['symlinks']=len(links)
    for name in ['wdro_precision_v2','wdro_legacy_v1','adversarial_v3_20260912','adversarial_v3_1_20260912',
                 'critic_search_20260915','oracle_free_20260915','rl_only_20260915','rl_research_20260915',
                 'papers_wdro_20260915','rl_only_pretraining_plan_20260919']:
        if not (ROOT/name).is_dir() or (ROOT/name).is_symlink():errors.append(['method_not_real_directory',name])
    tests=json.loads((AUDIT/'validation/tests.json').read_text())
    errors.extend(['failed_test',t['name']] for t in tests if t['returncode'])
    loaded=json.loads((AUDIT/'validation/frozen_model_load.json').read_text())
    if loaded['status']!='pass' or loaded['training_updates']!=0:errors.append(['frozen_model_load',loaded])
    isolation=(AUDIT/'validation/external_fixture_isolation.log').read_text()
    if not isolation.rstrip().endswith('OK'):errors.append(['external_fixture_isolation',isolation[-500:]])
    result=dict(status='pass' if not errors else 'fail',time=time.time(),counts=counts,errors=errors,
                tests=tests,notices=notices,scope='directory relocation, original byte preservation, CPU regressions; not scientific reacceptance')
    (AUDIT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],counts=counts,error_count=len(errors),first_errors=errors[:15]),ensure_ascii=False,indent=2))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
