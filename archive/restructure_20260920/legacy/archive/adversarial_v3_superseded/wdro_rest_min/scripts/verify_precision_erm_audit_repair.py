"""Exercise Adam's update-zero boundary and review the real selected checkpoint."""
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent / 'experiments'
sys.path.insert(0, str(ROOT))
from audit_stage2_formal import optimizer_checks
from run_precision_training_queue import source_gate
from src.checkpoints import load_checkpoint
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from src.trainer import load_trainer_config


def main():
    source_gate()
    trainer = load_trainer_config()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.Adam([parameter], lr=trainer['optimizer']['learning_rate'],
                           eps=trainer['optimizer']['epsilon'],
                           weight_decay=trainer['optimizer']['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)

    def checkpoint(update):
        return dict(update=update, optimizer_state_dict=copy.deepcopy(opt.state_dict()),
                    scheduler_state_dict=copy.deepcopy(sched.state_dict()),
                    rng_state=dict(python=None, numpy=None, torch_cpu=torch.get_rng_state(), torch_cuda=[]))

    zero = checkpoint(0)
    checks = {'fresh_adam_zero_state_is_empty': zero['optimizer_state_dict']['state'] == {},
              'fresh_adam_zero_passes': all(optimizer_checks(zero, trainer).values())}
    (parameter ** 2).sum().backward()
    opt.step()
    sched.step()
    one = checkpoint(1)
    checks['one_real_adam_step_passes'] = all(optimizer_checks(one, trainer).values())
    missing = copy.deepcopy(one)
    missing['optimizer_state_dict']['state'] = {}
    checks['missing_state_after_update_rejected'] = not optimizer_checks(missing, trainer)['optimizer_steps_match_checkpoint']
    wrong = copy.deepcopy(one)
    wrong['update'] = 2
    checks['wrong_positive_step_rejected'] = not optimizer_checks(wrong, trainer)['optimizer_steps_match_checkpoint']
    wrong['update'] = 0
    checks['populated_state_at_zero_rejected'] = not optimizer_checks(wrong, trainer)['optimizer_steps_match_checkpoint']
    missing_groups = copy.deepcopy(zero)
    missing_groups['optimizer_state_dict']['param_groups'] = []
    checks['missing_groups_at_zero_rejected'] = not optimizer_checks(missing_groups, trainer)['optimizer_steps_match_checkpoint']
    invalid = copy.deepcopy(one)
    next(iter(invalid['optimizer_state_dict']['state'].values()))['exp_avg'].fill_(float('nan'))
    checks['nan_moment_rejected'] = not optimizer_checks(invalid, trainer)['model_optimizer_rng_finite']

    rid = 'formal_erm_d40_s102_u2000'
    result_path = EXP / 'logs/stage2' / rid / 'result.json'
    result = json.loads(result_path.read_text())
    best = load_checkpoint(Path(result['best_checkpoint']), map_location='cpu')
    latest = load_checkpoint(Path(result['latest_checkpoint']), map_location='cpu')
    legacy = torch.load(result['initialization']['path'], map_location='cpu', weights_only=False)
    checks.update({'actual_best_' + k: v for k, v in optimizer_checks(best, trainer).items()})
    checks.update({'actual_latest_' + k: v for k, v in optimizer_checks(latest, trainer).items()})
    checks['actual_best_is_unmodified_dac21_actor'] = (
        best['actor_state_dict'].keys() == legacy['actor_state_dict'].keys()
        and all(torch.equal(v, legacy['actor_state_dict'][k]) for k, v in best['actor_state_dict'].items()))
    checks['actual_best_zero_latest_2000'] = best['update'] == result['best']['update'] == 0 and latest['update'] == 2000
    checks['checkpoints_hashes_unchanged'] = all(
        file_sha256(Path(result[k + '_checkpoint'])) == result[k + '_checkpoint_sha256'] for k in ('best', 'latest'))
    events = [json.loads(line) for line in Path(result['event_log']).read_text().splitlines()]
    validation = {e['update']: e['result']['selection']['worst_group_mean'] for e in events if e['event'] == 'validation'}
    checks['all_21_validation_points_select_zero'] = (
        sorted(validation) == list(range(0, 2001, 100))
        and min(validation, key=lambda u: (validation[u], u)) == 0
        and validation[0] == result['best']['selection_objective'])
    paths = [Path(__file__), ROOT / 'scripts/audit_stage2_formal.py', result_path,
             Path(result['best_checkpoint']), Path(result['latest_checkpoint'])]
    output = EXP / 'logs/precision_erm_audit_repair/verification.json'
    report = dict(at_utc=utc_now(), status='pass' if all(checks.values()) else 'fail', checks=checks,
                  research_protocol_override=False, training_source_gate=True,
                  artifacts=[dict(logical_path=str(p), resolved_path=str(p.resolve()), sha256=file_sha256(p)) for p in paths])
    atomic_write_json(output, report, mode=0o444)
    print(json.dumps(dict(status=report['status'], checks=len(checks), failed=[k for k, v in checks.items() if not v], output=str(output))))
    assert all(checks.values()), checks


if __name__ == '__main__':
    main()
