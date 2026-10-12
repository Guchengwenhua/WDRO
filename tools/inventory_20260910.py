"""Metadata inventory and small completion-binding review; never runs experiments."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/remote-home/Guilin/DRO/Codex')
REV = Path('/remote-home/Guilin/DRO/Codex/wdro_precision_v2')
OUT = ROOT / 'experiments/logs/project_organization_20260910'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def group(path):
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith('experiments.local_backup_'):
        return 'storage_migration_backup'
    if rel.startswith('/remote-home/Guilin/DRO/Codex/wdro_precision_v2/'):
        tail = rel.split('/remote-home/Guilin/DRO/Codex/wdro_precision_v2/', 1)[1]
        return 'precision_v2/' + '/'.join(tail.split('/')[:2]) if '/' in tail else 'precision_v2/documents'
    if rel.startswith('experiments/'):
        return 'v1_or_shared/' + rel.split('/')[1]
    if rel.startswith('wdro_rest_min/'):
        return 'v1_or_operations/' + rel.split('/')[1]
    return 'project_documents_and_tools'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    authority_paths = [
        ROOT / 'wdro_rest_min/manifests/precision_v2_replay_completion.json',
        *[REV / 'wdro_rest_min/manifests' / n for n in (
            'stage2_completion_20260905.json', 'stage3_completion_v1.json',
            'stage4_completion_v1.json', 'stage5_completion_v1.json',
            'stage6_completion_v1.json', 'stage6_delivery_verification_v1.json',
            'stage7_completion_v1.json', 'stage7_delivery_verification_v1.json',
            'stage8_completion_v1.json', 'stage8_delivery_verification_v1.json',
            'stage8_delivery_review_v1.json', 'stage8_closeout_v1.json')],
    ]
    authorities = []
    for p in authority_paths:
        doc = json.loads(p.read_text())
        assert doc['status'] == 'pass', p
        authorities.append({'logical_path': str(p), 'resolved_path': str(p.resolve()),
                            'sha256': sha(p), 'status': doc['status'],
                            'reported_at_utc': doc.get('at_utc')})
    bindings = []
    closeout = json.loads(authority_paths[-1].read_text())

    def check_bindings(node):
        if isinstance(node, dict):
            if 'logical_path' in node and 'sha256' in node:
                p = Path(node['logical_path'])
                ok = p.is_file() and sha(p) == node['sha256']
                bindings.append({'logical_path': str(p), 'sha256_matches': ok})
                assert ok, p
            else:
                for v in node.values():
                    check_bindings(v)
        elif isinstance(node, list):
            for v in node:
                check_bindings(v)

    check_bindings(closeout)
    kinds = Counter()
    groups = defaultdict(lambda: {'files': 0, 'logical_bytes': 0, 'directories': 0, 'symlinks': 0, 'special': 0})
    links, errors = [], []
    excluded_real = str(OUT.resolve())
    seen_dirs = set()
    seen_files = set()
    unique_file_bytes = 0
    unique_files = 0
    part = OUT / 'all_files.csv.gz.partial'
    columns = ['logical_path', 'resolved_path', 'kind', 'group', 'size_bytes', 'mtime_ns', 'symlink_target', 'exists']
    with gzip.open(part, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        def walk(path, resolved_base=None):
            nonlocal unique_file_bytes, unique_files
            real = str(path.resolve()) if resolved_base is None else resolved_base
            if real == excluded_real:
                return
            try:
                st = path.stat()
                identity = (st.st_dev, st.st_ino)
                if identity in seen_dirs:
                    return
                seen_dirs.add(identity)
                with os.scandir(path) as scan:
                    entries = sorted(scan, key=lambda e: e.name)
                for e in entries:
                    p = Path(e.path)
                    s = e.stat(follow_symlinks=False)
                    if e.is_symlink():
                        kind = 'symlink'
                        resolved = str(p.resolve())
                        target = os.readlink(p)
                        exists = p.exists()
                        links.append({'logical_path': str(p), 'resolved_path': resolved, 'target': target, 'exists': exists})
                    else:
                        resolved = str(Path(real) / e.name)
                        target, exists = '', True
                        kind = 'directory' if e.is_dir(follow_symlinks=False) else ('file' if e.is_file(follow_symlinks=False) else 'special')
                    if resolved == excluded_real:
                        continue
                    g = group(p)
                    writer.writerow(dict(zip(columns, [str(p), resolved, kind, g, s.st_size, s.st_mtime_ns, target, exists])))
                    kinds[kind] += 1
                    key = {'file': 'files', 'directory': 'directories', 'symlink': 'symlinks', 'special': 'special'}[kind]
                    groups[g][key] += 1
                    if kind == 'file':
                        groups[g]['logical_bytes'] += s.st_size
                        file_id = (s.st_dev, s.st_ino)
                        if file_id not in seen_files:
                            seen_files.add(file_id)
                            unique_files += 1
                            unique_file_bytes += s.st_size
                    if kind == 'directory':
                        walk(p, resolved)
            except OSError as exc:
                errors.append({'path': str(path), 'error': repr(exc)})

        # Prefer the current revision's logical path for its large artifacts.
        # Other directory symlinks are recorded, not followed; their shared
        # project targets are covered by ROOT/experiments or ROOT itself.
        walk(REV / 'experiments')
        walk(ROOT / 'experiments')
        walk(ROOT)
    final = OUT / 'all_files.csv.gz'
    part.replace(final)
    count = 0
    with gzip.open(final, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            assert row['logical_path'].startswith(str(ROOT) + '/')
            count += 1
    assert count == sum(kinds.values())
    manifest = {
        'started_at_utc': start, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Codex physical files plus its two explicit experiments directory links; other symlinks recorded without recursive traversal',
        'status': 'pass' if not errors else 'incomplete',
        'inventory': {'logical_path': str(final), 'resolved_path': str(final.resolve()), 'sha256': sha(final), 'rows_verified': count},
        'counts': dict(kinds), 'unique_regular_file_inodes': unique_files,
        'unique_regular_file_logical_bytes': unique_file_bytes,
        'byte_semantics': 'st_size sum, not filesystem allocated blocks; unique totals deduplicate hard links',
        'groups': dict(sorted(groups.items())), 'symlinks': links, 'errors': errors,
        'excluded_output_directory': str(OUT),
        'authorities': authorities, 'stage8_direct_and_nested_closeout_bindings': bindings,
        'not_performed': ['full artifact rehash', 'model evaluation', 'training', 'numeric audit replay'],
    }
    (OUT / 'inventory_summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('status', 'counts', 'unique_regular_file_inodes', 'unique_regular_file_logical_bytes')}, ensure_ascii=False))
    print('authority_manifests', len(authorities), 'closeout_bindings', len(bindings), 'symlinks', len(links), 'errors', len(errors))
    print(final)


if __name__ == '__main__':
    main()
