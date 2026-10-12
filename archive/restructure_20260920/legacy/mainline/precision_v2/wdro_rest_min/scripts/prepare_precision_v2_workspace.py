"""Create an isolated replay workspace without modifying any v1 artifact."""
import json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256

NEW=CODEX.parent/'wdro_precision_v2'
REV=NEW

def main():
    if NEW.exists():
        raise RuntimeError('precision_v2 already exists; this one-time historical workspace creation must not overwrite the flat method directory')
    destination=NEW/'wdro_rest_min';records=[]
    for sub in ['src','stage4','stage5','stage6','scripts','tests','precision_v2']:
        for source in sorted((ROOT/sub).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix not in ('.py','.sh'):
                continue
            path=destination/source.relative_to(ROOT);text=source.read_text().replace(str(CODEX),str(NEW))
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert path.read_text()==text,str(path)
            else:path.write_text(text)
            records.append(dict(original=str(source),original_sha256=file_sha256(source),copy=str(path),copy_sha256=file_sha256(path),change='path_relocation' if source.read_text()!=text else 'none'))
    for name in ['protocol_v1.json','data_contract_v1.json','stage2_trainer_v1.json','stage3_baselines_v1.json','stage4_wdro_v1.json']:
        source=ROOT/'configs'/name;path=destination/'configs'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert file_sha256(path)==file_sha256(source)
        else:shutil.copy2(source,path)
        records.append(dict(original=str(source),copy=str(path),sha256=file_sha256(path),change='none'))
    path=destination/'src/geosteiner.py';source=ROOT/'precision_v2/geosteiner.py'
    # The only scientific computation implementation changed in the replay.
    if file_sha256(path)!=file_sha256(source):
        old=file_sha256(path);path.write_bytes(source.read_bytes())
        records.append(dict(copy=str(path),before_sha256=old,after_sha256=file_sha256(path),change='authorized_reference_precision_repair'))
    for p in ['checkpoints','logs','tables','figures','tmp','cache/runtime']:
        (CODEX/'experiments/precision_v2'/p).mkdir(parents=True,exist_ok=True)
    links={NEW/'experiments':CODEX/'experiments/precision_v2', REV/'DRO':CODEX.parent/'DRO',
           NEW/'experiments/cache/datasets':CODEX/'experiments/cache/datasets',
           NEW/'experiments/cache/ispd_protocol_a':CODEX/'experiments/cache/ispd_protocol_a'}
    for link,target in links.items():
        link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():assert link.resolve()==target.resolve()
        else:assert not link.exists();link.symlink_to(target,target_is_directory=True)
    (destination/'manifests').mkdir(parents=True,exist_ok=True)
    manifest=dict(status='prepared',at_utc=utc_now(),authorization='User explicitly approved GeoSteiner precision repair, affected early-stage review and necessary reruns on 2026-09-07.',
        parent_codex=str(CODEX),revision_codex=str(NEW),source_inventory=records,
        links=[dict(logical_path=str(p),target=str(t),resolved_path=str(p.resolve()),read_only_reuse=p.name in ('DRO','datasets','ispd_protocol_a')) for p,t in links.items()],
        frozen_research_configs_unchanged=True,old_models_or_references_copied=False,
        stages=['R1 independent reference correctness','R2 all 40 fixed reference caches and impact audit','R3 ERM/baseline trajectory replay and full audits',
                'R4 original WDRO diagnostic/validation-only ablations and selection replay','R5 complete 75-model freeze','R6 all synthetic test/statistics/delivery'],
        test_firewall='No new actor test predictions until corrected reference and training/selection gates pass. No added hyperparameters, distributions, seeds or budgets. Stop before stage7.')
    atomic_write_json(CODEX/'wdro_rest_min/manifests/precision_v2_workspace.json',manifest)
    print(json.dumps(dict(status='prepared',workspace=str(NEW),copied_files=len(records))))
if __name__=='__main__':main()
