#!/usr/bin/env python3
"""Resolve an old project path without changing frozen manifests or their hashes."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'archive/restructure_20260920'

def current_path(path):
    value = Path(path)
    if value.is_absolute():
        if not value.is_relative_to(ROOT):
            return value
        value = value.relative_to(ROOT)
    name = value.as_posix()
    mapping = json.loads((AUDIT / 'path_map.json').read_text())
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if name == old or name.startswith(old + '/'):
            return ROOT / (new + name[len(old):])
    return ROOT / name

def historical_path(path):
    value = Path(path)
    if value.is_absolute():
        if not value.is_relative_to(ROOT):
            return value
        value = value.relative_to(ROOT)
    snapshot = AUDIT / 'legacy' / value
    return snapshot if snapshot.exists() else ROOT / value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='Old absolute path or project-relative path')
    args = parser.parse_args()
    print(json.dumps({'current': str(current_path(args.path)),
                      'historical': str(historical_path(args.path))}, ensure_ascii=False, indent=2))
