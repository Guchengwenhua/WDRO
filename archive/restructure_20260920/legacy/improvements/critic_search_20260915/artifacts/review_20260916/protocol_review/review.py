"""Read-only protocol/claim and documentation review; writes only beside this file."""
from pathlib import Path
import csv, datetime, hashlib, json, re, tarfile
from urllib.parse import unquote
import numpy as np

ROOT=Path('/remote-home/Guilin/DRO/Codex')
STUDY=ROOT/'improvements/critic_search_20260915'
EXP=STUDY/'artifacts'
OUT=EXP/'review_20260916/protocol_review'
DISTS=['uniform','mix_gauss','edge_bias','grid_cluster']
SEEDS=[101,102,103]
VARIANTS=['refresh32_raw','refresh8_raw','refresh32_ema','refresh8_ema']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rows(name):
    with (EXP/'tables'/name).open() as f:return list(csv.DictReader(f))
def excerpt(name,start,end):
    p=STUDY/name; ls=p.read_text().splitlines()
    return dict(path=str(p),sha256=sha(p),line_start=start,line_end=end,
                text='\n'.join(f'{i}: {ls[i-1]}' for i in range(start,end+1)))

def main():
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(check=name,passed=bool(ok),detail=detail))
    frozen=read(EXP/'freeze.json');automation=read(EXP/'automation_manifest.json')
    protocol=read(STUDY/'protocol.json');selection=read(EXP/'selection.json')
    fixed_table=rows('fixed_metrics.csv');task_table=rows('tasks.csv')
    for item in frozen['files']+automation['files']:
        check('frozen source '+Path(item['path']).name,sha(item['path'])==item['sha256'])
    timeline=[dict(event='method_protocol_freeze',at=frozen['frozen_at'],utc=utc(frozen['frozen_at']),path=str(EXP/'freeze.json')),
              dict(event='analysis_code_automation_binding',at=automation['at'],utc=utc(automation['at']),path=str(EXP/'automation_manifest.json')),
              dict(event='search_selection_freeze',at=selection['frozen_at'],utc=utc(selection['frozen_at']),path=str(EXP/'selection.json'))]
    schema=[];budgets=[];prices={};best_updates={}
    for phase,seeds,count in [('validation',[101],32),('test',SEEDS,64)]:
        for seed in seeds:
            p=EXP/'fixed'/f'{phase}_s{seed}';d=read(p/'complete.json');identity=read(p/'identity.json')
            check(f'{p.name} complete schema',d['status']=='pass' and d['phase']==phase and d['seed']==seed and d['count_per_distribution']==count and d['identity']==identity)
            schema.append(dict(task=str(p.relative_to(EXP)),keys=list(d),artifact_count=len(d['artifacts'])))
            if phase=='test':
                prices[seed]=identity['price']
                check(f'{p.name} selection binding',identity['selection_sha256']==sha(EXP/'selection.json'))
            methods=sorted({r['method'] for r in fixed_table if r['phase']==phase and int(r['seed'])==seed})
            check(f'{p.name} method count',len(methods)==(23 if phase=='validation' else 11),methods)
            ledger=read(p/'oracle/oracle_ledger.json');expected=count*4*len(methods)*64+(2560 if phase=='test' else 0)
            check(f'{p.name} greedy budget',ledger['actor_requested']==ledger['actor_completed']==expected)
            budgets.append(dict(task=str(p.relative_to(EXP)),expected_greedy=expected,actual_greedy=ledger['actor_requested']))
            for dist in DISTS:
                for offset in range(0,count,16):
                    for method in methods:
                        with np.load(p/'evaluation'/f'{dist}_{method}_{offset:04d}.npz') as z:
                            check(f'{p.name}/{dist}/{method}/{offset} price/budget',np.all(z['price']==identity['price']) and np.all(z['calls']==64))
                    if phase=='test':
                        with np.load(p/'evaluation'/f'{dist}_direction4_{offset:04d}.npz') as a,np.load(p/'evaluation'/f'{dist}_direction5_{offset:04d}.npz') as b:
                            check(f'{p.name}/{dist}/{offset} D4=D5 support',np.array_equal(a['support'],b['support']) and np.array_equal(a['score'],b['score']))
            if phase=='test':
                sd=read(p/'surrogates/summary.json')
                check(f'{p.name} surrogate labels',sd['total_unique_real_labels']==2560 and sd['logical_training_labels_per_variant']==1024 and sd['shared_validation_labels']==512 and sd['test_labels_used']==0)
                variants=sd['variants']; hashes={x['identity']['initial_model_sha256'] for x in variants.values()}
                check(f'{p.name} four common initial surrogates',len(variants)==4 and len(hashes)==1)
                for name,v in variants.items():
                    check(f'{p.name}/{name} epochs and selection',len(v['history'])==6 and v['best_epoch']==min(v['history'],key=lambda r:r['validation']['mse'])['epoch'])
            launch=read(p/'launch.json');timeline.append(dict(event=str(p.relative_to(EXP))+'_launch',at=launch['at'],utc=utc(launch['at']),path=str(p/'launch.json')))
    for seed in SEEDS:
        for variant in VARIANTS:
            p=EXP/'training'/f'{variant}_d30_s{seed}';d=read(p/'complete.json');identity=read(p/'identity.json');settings=identity['settings']
            check(p.name+' formal schema',d['status']=='pass' and not d['smoke'] and d['degree']==30 and d['updates']==128 and len(d['history'])==128 and d['identity']==identity)
            schema.append(dict(task=str(p.relative_to(EXP)),keys=list(d),artifact_count=len(d['artifacts'])))
            check(p.name+' declared budget',settings['batch_size']==16 and settings['true_search_budget_per_center']==4 and d['ledger']['actor_requested']==12544 and d['ledger']['sampled_actor_requested']==2048)
            interval=8 if variant.startswith('refresh8_') else 32
            check(p.name+' refresh timing',settings['critic_interval']==interval and sum(x['online'] is not None for x in d['history'])==128//interval)
            check(p.name+' fixed rho/initial price',d['initial_price']==prices[seed] and all(x['rho']==d['rho'] for x in d['history']))
            selected=0;best=d['initial_validation']['ood_worst_group_mean'];eligible=d['initial_validation']['nominal_mean']+.002
            for row in d['history']:
                v=row['validation']
                if v and v['nominal_mean']<=eligible and v['ood_worst_group_mean']<best:selected=row['update'];best=v['ood_worst_group_mean']
            check(p.name+' validation-only selected update',d['best_update']==selected)
            best_updates[p.name]=selected
            budgets.append(dict(task=str(p.relative_to(EXP)),expected_greedy=12544,actual_greedy=d['ledger']['actor_requested'],sampled=2048))
            exit_record=read(p/'exit.json');timeline.append(dict(event=str(p.relative_to(EXP))+'_exit',at=exit_record['at'],utc=utc(exit_record['at']),path=str(p/'exit.json')))
    # Recompute family winners from published validation means, independently of report.py.
    families={'direction3':[f'step_{a}_{t}' for t in [4,8,16] for a in ['0.0025','0.005','0.01']]+['decay16','adaptive16','adam16'],
              'direction4':['mixed_scales','structured_starts','wide126_short4','narrow21_long24'],
              'direction5':['trajectory21_8','narrow21_long24','trajectory_diverse'],
              'direction6':['differentiable_normalization','remove_rigid','adaptive16']}
    for family,names in families.items():
        means={n:np.mean([float(r['mean_score']) for r in fixed_table if r['phase']=='validation' and r['method']==n]) for n in names}
        check(family+' validation winner',max(names,key=lambda n:means[n])==selection['selected'][family],means)
    grad={}
    for method in ['baseline','direction3','direction4','direction5','direction6']:
        rs=[r for r in fixed_table if r['phase']=='test' and r['method']==method]
        grad[method]=dict(forward_rows_per_center=sorted({int(r['gradient_forward_rows'])/int(r['centers']) for r in rs}),backward_rows_per_center=sorted({int(r['gradient_backward_rows'])/int(r['centers']) for r in rs}))
    # Read-only archive and current navigation checks. Do not run legacy verify,
    # which would overwrite prior verification receipts.
    archive=ROOT/'archive/documentation_20260915';manifest=read(archive/'manifest.json');archive_checks=[]
    with tarfile.open(archive/'original_documents.tar.gz') as tar:
        for item in manifest['entries']:
            archive_checks.append(dict(original=item['original'],old_path_matches=sha(ROOT/item['original'])==item['sha256'],archive_matches=hashlib.sha256(tar.extractfile(item['original']).read()).hexdigest()==item['sha256']))
    check('57 original/archive source bytes',len(archive_checks)==57 and all(x['old_path_matches'] and x['archive_matches'] for x in archive_checks))
    documents=[ROOT/'README.md',ROOT/'02_新对话执行提示词.md',archive/'README.md',*sorted((ROOT/'docs').rglob('*.md')),*[ROOT/n/'README.md' for n in ['mainline','improvements','external','archive']]]
    links=[]
    for p in documents:
        for raw in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=unquote(raw.strip().strip('<>')).split('#',1)[0]
            if not target or re.match(r'^[a-zA-Z]+://',target):continue
            target=re.sub(r':\d+$','',target);dest=Path(target) if target.startswith('/') else p.parent/target
            links.append(dict(document=str(p),target=raw,exists=dest.exists()))
    check('current documentation links',all(x['exists'] for x in links),dict(count=len(links),missing=[x for x in links if not x['exists']]))
    past_links=read(archive/'historical_link_review.json')['links']
    check('original-path relative links',all(Path(x['resolved_from_original_path']).exists() for x in past_links),len(past_links))
    timeline.sort(key=lambda x:x['at'])
    for row in timeline:row['sha256']=sha(row['path'])
    prereg=dict(method_protocol_freeze_at=frozen['frozen_at'],analysis_code_binding_at=automation['at'],
                all_twelve_training_exits_before_analysis_code_binding=all(x['at']<automation['at'] for x in timeline if x['event'].startswith('training/') and x['event'].endswith('_exit')),
                all_fixed_test_launches_after_analysis_code_binding=all(x['at']>automation['at'] for x in timeline if x['event'].startswith('fixed/test_') and x['event'].endswith('_launch')),
                finding='Full nine-contrast primary/secondary designation is not established by the experiment-start freeze; report.py is persistently bound before fixed tests but after all D7 tasks. No evidence of result-driven selection is asserted; strict preregistration is not established.')
    evidence=[excerpt('protocol.json',9,18),excerpt('THEORY_01_02.md',59,65),excerpt('THEORY_01_02.md',73,115),excerpt('RESULTS.md',10,30),excerpt('RESULTS.md',39,42),excerpt('report.py',563,579)]
    write('timeline.json',dict(timeline=timeline,analysis=prereg,sources=[dict(path=str(p),sha256=sha(p)) for p in [EXP/'freeze.json',EXP/'automation_manifest.json',EXP/'selection.json']]))
    write('evidence_inputs.json',dict(freeze=frozen,automation_manifest=automation,
          task_exit_records={str(p.relative_to(EXP)):read(p) for p in sorted((EXP/'training').glob('*/exit.json'))},
          fixed_test_launch_records={str(p.relative_to(EXP)):read(p) for p in sorted((EXP/'fixed').glob('test_s*/launch.json'))}))
    write('text_evidence.json',evidence)
    write('documentation_readonly.json',dict(archive_checks=archive_checks,links=links,historical_link_count=len(past_links),original_verification_not_overwritten=True))
    write('task_schemas_and_budgets.json',dict(schemas=schema,budgets=budgets,initial_fixed_prices=prices,best_updates=best_updates,gradient_rows_per_center=grad))
    write('checks.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='pass_with_claim_correction_required' if all(x['passed'] for x in checks) else 'failure',checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],finding=prereg))
    print(json.dumps(dict(checks=len(checks),passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],initial_prices=prices,best_updates=best_updates,gradients=grad,documentation_links=len(links),prereg=prereg),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
