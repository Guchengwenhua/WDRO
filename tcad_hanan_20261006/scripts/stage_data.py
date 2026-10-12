"""Copy existing research inputs; no solver, training, tests, or hash checks."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil

ROOT = Path('/remote-home/Guilin/DRO/Codex/tcad_hanan_20261006')
PUBLIC = Path('/remote-home/Guilin/DRO/Codex/benchmarks/public_designs_20260912/sources')
LEGACY = Path('/remote-home/Guilin/DRO/DRO')
rows = []
groups = []

def copy_group(name, base, files, relative_destination):
    records = []
    for source in sorted(files):
        rel = source.relative_to(base)
        target = ROOT / 'data' / relative_destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f'Refusing to replace an existing copy: {target}')
        temporary = target.with_name(target.name + '.copying')
        size = source.stat().st_size
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        record = dict(group=name, source=str(source), destination=str(target), source_bytes=size)
        rows.append(record)
        records.append(record)
    summary = dict(group=name, files=len(records), bytes=sum(r['source_bytes'] for r in records))
    groups.append(summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)

base = LEGACY / 'cugr-iccad-main/benchmark'
copy_group('ispd18_19_raw_lef_def', base,
           [p for p in base.glob('ispd*/*') if p.suffix.lower() in ('.lef', '.def')],
           'raw/ispd18_19')

base = PUBLIC / 'contests/ispd2008_gr'
copy_group('ispd2008_original_archives', base, base.glob('*.gr.gz'), 'raw/ispd2008')

base = PUBLIC / 'real_designs/CircuitOps'
copy_group('circuitops_source_snapshot', base,
           [p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts],
           'raw/CircuitOps')

base = LEGACY / 'benchmark'
copy_group('legacy_coordinate_inputs_reference_only', base,
           base.glob('ispd*.input/degree_*.txt'), 'legacy_coordinate_inputs')

base = PUBLIC
files = [base/'contests/designs_manifest.json',
         base/'real_designs/design_manifest.json',
         base/'real_designs/download_manifest.json']
files += list((base/'real_designs/design_descriptors').glob('*.json'))
copy_group('original_provenance_reference', base, files, 'source_records/original')

descriptor_dir = ROOT/'data/source_records/relocated_circuitops_descriptors'
descriptor_dir.mkdir(parents=True, exist_ok=True)
for source in sorted((PUBLIC/'real_designs/design_descriptors').glob('*.json')):
    record = json.loads(source.read_text())
    for key, value in list(record.items()):
        if key.endswith('_path') and isinstance(value, str) and '/CircuitOps/' in value:
            suffix = value.split('/CircuitOps/', 1)[1]
            record[key] = str(ROOT/'data/raw/CircuitOps'/suffix)
    record['tcad_note'] = 'Relocated source descriptor only; the old parser does not establish the driver needed by this experiment.'
    (descriptor_dir/source.name).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')

manifest = dict(
    created_at_utc=datetime.now(timezone.utc).isoformat(),
    purpose='TCAD Hanan future experiment input preparation',
    status='COPY_COMPLETED_NO_EXPERIMENT_RUN',
    logical_data_root=str(ROOT/'data'), physical_data_root=str((ROOT/'data').resolve()),
    groups=groups, total_files_copied=len(rows),
    total_source_bytes_copied=sum(r['source_bytes'] for r in rows), files=rows,
    notes=[
      'Copy operation completed; no per-file content hashes or post-copy integrity test was run.',
      'Raw inputs and original source documentation only; prior experiment scores are not imported.',
      'Legacy coordinate text is for reference and has not been accepted as a driver-aware dataset.',
      'ISPD metal5 variants are retained but must not be counted as independent design families.',
      'Eligible net counts, source labels, splits and all new measurements are pending.'
    ])
target = ROOT/'metadata/data_copy_receipt.json'
temporary = target.with_suffix('.json.tmp')
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
os.replace(temporary, target)
print(json.dumps({k:manifest[k] for k in ('status','total_files_copied','total_source_bytes_copied')}, ensure_ascii=False), flush=True)
