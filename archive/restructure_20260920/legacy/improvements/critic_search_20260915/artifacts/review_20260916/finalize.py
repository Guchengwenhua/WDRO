"""Publish the independent acceptance receipt; never rewrite original evidence."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915')
REVIEW = ROOT / 'artifacts/review_20260916'


def read(name):
    return json.loads((REVIEW / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = REVIEW / name
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    partial.replace(path)


integrity = read('integrity/integrity_report.json')
statistics = read('statistics/review.json')
isolation = read('integrity/expanded_data_isolation.json')
numerical = read('numerical_reaudit/result.json')
reference = read('reference_spotcheck/result.json')
protocol = read('protocol_review/completion.json')
assert all(x['status'] == 'pass' for x in
           [integrity, statistics, isolation, numerical, reference])
assert protocol['passed'] == protocol['checked'] == 876
assert protocol['blocking_data_failures'] == 0
assert len(integrity['queue']['completed']) == 16
assert not any(integrity['queue'][k] for k in ['pending', 'running', 'failures'])
assert all(x['count'] == 0 for x in isolation['intersections'])
controls = integrity['bound_control_sha256']
for name, expected in controls.items():
    assert sha(name) == expected, name
assert not any('critic_search_20260915/README.md' in json.dumps(x)
               for x in read('integrity/binding_checks.json'))

now = datetime.now(timezone.utc).isoformat()
tables = statistics['tables']
receipt = {
    'schema_version': 1,
    'accepted_at_utc': now,
    'status': 'pass_with_interpretation_qualification',
    'execution_integrity_and_numerical_acceptance': 'pass',
    'scientific_interpretation': 'accepted_with_explicit_provenance_correction',
    'formal_tasks': 16,
    'scope': 'tmp2 seven-direction mechanism study, degree30, three fixed parent models, four synthetic distributions',
    'integrity': {k: integrity[k] for k in [
        'canonical_files_hashed', 'manifest_binding_references',
        'mismatch_reference_count', 'scientific_mismatch_count',
        'committed_greedy', 'committed_sampled']},
    'independent_statistics': {
        'tables': len(tables),
        'rows': sum(x['rows'] for x in tables),
        'numeric_fields': sum(x['numeric_cells'] for x in tables),
        'exact_numeric_matches': sum(x['exactly_equal_numeric_cells'] for x in tables),
        'roundoff_only_fields': sum(x['roundoff_only_cells'] for x in tables),
        'maximum_absolute_difference': max(x['maximum_absolute_difference'] for x in tables),
        'bootstrap_replicates': statistics['bootstrap_replicates'],
        'primary_family_size': statistics['primary_family'],
        'resampling_unit': 'paired centers within distribution, shared indices across the three fixed models; no model-seed bootstrap',
    },
    'expanded_data_isolation': {
        'total_geometry_rows': sum(isolation['role_total_rows'].values()),
        'role_unique_nets': isolation['role_unique_nets'],
        'cross_role_intersections': isolation['intersections'],
        'all_fixed_search_supports_included': True,
    },
    'audit_only_reference_spotcheck': {
        'points': reference['sample_count'],
        'fresh_cpu_solves': reference['fresh_geosteiner_solves'],
        'exact_equality_both_scales': reference['exact_equality_both_scales'],
        'original_oracle_ledgers_unchanged': True,
    },
    'primary_contrasts': statistics['primary_contrasts'],
    'supported_directions': ['D3'],
    'required_interpretations': [
        'D3 has equal 64 true-score budget but twice the gradient rows; stronger fixed-model search does not prove Actor training benefit or total-compute efficiency.',
        'D4 and D5 selected the identical configuration and are not independent replications.',
        'Methods and experiment settings were frozen before execution; full nine-comparison analysis code was persistently bound before fixed test but after all D7 results.',
        'D7 inference remains exploratory. Huber was a planned experimental variant, but its secondary analysis status is not established as frozen before all results.',
        'Confidence intervals condition on three existing models; inconclusive intervals do not establish equivalence or a continuous Wasserstein certificate.',
        'Each direction implements a minimum controlled version; not every tmp2 subproposal was tested.',
    ],
    'lifecycle': 'all study workers, scheduler, autofinish and dedicated tmux exited; 18 locks released',
    'lifecycle_evidence_at_utc': integrity['at'],
    'no_gpu_inference_or_training_started_by_review': True,
    'original_scientific_artifacts_preserved': True,
    'navigation_change': 'Only the unbound branch README gains the acceptance link and interpretation correction notice.',
    'report': 'REPORT.md',
    'original_control_sha256_rechecked_at_publication': controls,
}
write('acceptance.json', receipt)

files = []
for path in sorted(REVIEW.rglob('*')):
    if not path.is_file() or path.name == 'evidence_manifest.json' or '__pycache__' in path.parts:
        continue
    if path.name.endswith('.partial'):
        raise RuntimeError(f'Unpublished partial evidence: {path}')
    files.append({'path': str(path.relative_to(REVIEW)), 'bytes': path.stat().st_size,
                  'sha256': sha(path)})
manifest = {'schema_version': 1, 'created_at_utc': now,
            'scope': 'All files in this independent review, excluding this manifest itself and interpreter caches',
            'files': files,
            'navigation': {'path': str(ROOT / 'README.md'), 'sha256': sha(ROOT / 'README.md')}}
write('evidence_manifest.json', manifest)

links = 0
for path in [REVIEW / 'REPORT.md', ROOT / 'README.md']:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text()):
        target = target.strip('<>').split('#')[0]
        if not target or '://' in target:
            continue
        assert (path.parent / target).exists(), (str(path), target)
        links += 1
for entry in manifest['files']:
    assert sha(REVIEW / entry['path']) == entry['sha256'], entry['path']
assert sha(manifest['navigation']['path']) == manifest['navigation']['sha256']
print(json.dumps({'status': receipt['status'], 'review_files_bound': len(files),
                  'local_links_checked': links, 'original_control_hashes_rechecked': len(controls),
                  'report': str(REVIEW / 'REPORT.md')}, ensure_ascii=False))
