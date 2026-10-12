"""Read atomic checkpoint bytes without changing running training."""
import hashlib
import io
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path('/remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min')
sys.path.insert(0, str(ROOT))
from src.stage2_audit_checks import optimizer_checks
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256

trainer = json.loads((ROOT / 'configs/stage2_trainer_v1.json').read_text())
records = []
for name in ('random_64', 'cem_64'):
    rid = 'stage5_' + name + '_d30_s101_u2000_v1'
    for kind in ('latest', 'best'):
        path = ROOT.parent / 'experiments/checkpoints/stage5' / rid / (kind + '.pt')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        identity = checkpoint['run_identity']
        checks = optimizer_checks(checkpoint, trainer)
        checks['frozen_run_identity'] = (identity['ablation_config']['config_id'] == name
            and identity['expected_updates'] == 2000 and identity['batch_size'] == 256
            and identity['training_seed'] == 101 and identity['degree'] == 30)
        checks['update_in_range'] = 0 < checkpoint['update'] <= 2000
        checks['corrected_reference_identity'] = identity['reference_solver']['reference_id'] == 'geosteiner_exact_binary_scale_v2'
        assert all(checks.values()), (str(path), checks)
        records.append(dict(logical_path=str(path), resolved_path=str(path.resolve()), sha256_at_read=digest,
            checkpoint_kind=kind, run_id=rid, update=checkpoint['update'], checks=checks))
        del data, checkpoint
output = Path(__file__).parent / ('live_checkpoint_review_' + str(time.time_ns()) + '.json')
atomic_write_json(output, dict(status='pass', at_utc=utc_now(), snapshots=records,
    checks=sum(len(row['checks']) for row in records), script=str(Path(__file__)), script_sha256=file_sha256(Path(__file__)),
    read_only=True, model_calls=0, note='Immutable bytes at read; live latest files may advance.'), mode=0o444)
print(json.dumps(dict(status='pass', checks=32, evidence=str(output), updates=[dict(run_id=row['run_id'],
    kind=row['checkpoint_kind'], update=row['update']) for row in records])), flush=True)
