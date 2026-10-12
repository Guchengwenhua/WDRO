"""Authorized, reversible categorization after evaluation; preserve old aliases.

This is a move journal, not a verification platform. No contents or weights are
deleted or hashed. Frozen sources/protocols are not rewritten.
"""
import shutil
from common import *

METHODS=['wdro_precision_v2','wdro_legacy_v1','adversarial_v3_20260912','adversarial_v3_1_20260912',
 'critic_search_20260915','oracle_free_20260915','rl_only_20260915','rl_research_20260915',
 'papers_wdro_20260915','rl_only_pretraining_plan_20260919','rl_budget_scaling_20260921',
 'rl_scaleup_20260921','rl_prepost_scaleup_20260922','rl_expansion_20260922',
 'gaussian_es_scaleup_20260922','entropic_teacher_wdro_20260922','vae_wdro_real_20260921','net_structure_real_20260922']
JOURNAL=PROJECT/'archive/restructure_20260928/moves.json'

def organize(mode='compat'):
    archive=JOURNAL.parent;archive.mkdir(parents=True,exist_ok=True)
    state=read(JOURNAL) if JOURNAL.exists() else dict(status='moving',moves=[],links=[],rebased_links=[])
    if state['status']=='complete':return
    for entry in state['moves']:
        if entry['phase']!='planned':continue
        src,dst=Path(entry['source']),Path(entry['destination'])
        if dst.exists() or dst.is_symlink():
            for rel,old,_ in entry.get('relative_links',[]):
                p=dst/rel if rel else dst
                if p.is_symlink():p.unlink();p.symlink_to(old)
            if src.is_symlink():src.unlink()
            if not src.exists():dst.rename(src)
        entry['phase']='rolled_back_interrupted_move'
    atomic(JOURNAL,state)
    # Keep the root README's original bytes for an explicit future rollback.
    backup=archive/'README.before.md'
    if not backup.exists():shutil.copy2(PROJECT/'README.md',backup)
    moves=[(PROJECT/name,PROJECT/'methods'/name) for name in METHODS]
    moves += [(PROJECT/'summaries',PROJECT/'reports/summaries'),(PROJECT/'literature',PROJECT/'reports/literature'),
              (PROJECT/'rl_acceptance_20260926',PROJECT/'reports/acceptance/rl_acceptance_20260926')]
    for src,dst in moves:
        if src.is_symlink():continue
        if not src.exists():continue
        # A moved relative symlink still names its original logical destination.
        # Capture before the move and use absolute paths through compatibility aliases.
        links=[]
        for folder,dirs,files in os.walk(src,followlinks=False):
            for name in dirs+files:
                p=Path(folder)/name
                if p.is_symlink():
                    target=os.readlink(p)
                    if not os.path.isabs(target):links.append((p.relative_to(src),target,os.path.abspath(p.parent/target)))
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():raise RuntimeError(f'move target exists: {dst}')
        entry=dict(source=str(src),destination=str(dst),phase='planned',alias=mode=='compat',relative_links=[(str(r),o,n) for r,o,n in links])
        state['moves'].append(entry);atomic(JOURNAL,state)
        src.rename(dst)
        # Source aliases also retain all historical absolute paths in saved manifests.
        if mode=='compat':src.symlink_to(os.path.relpath(dst,src.parent),target_is_directory=True)
        for rel,old,new in links:
            p=dst/rel;p.unlink();p.symlink_to(new,target_is_directory=True)
            state['rebased_links'].append(dict(path=str(p),original=old))
        entry['phase']='moved';atomic(JOURNAL,state)
    def link(path,target):
        if path.exists() or path.is_symlink():return
        path.parent.mkdir(parents=True,exist_ok=True)
        path.symlink_to(os.path.relpath(target,path.parent),target_is_directory=True)
        state['links'].append(str(path));atomic(JOURNAL,state)
    # Namespace bridges preserve source-relative sibling imports without changing
    # frozen Python files. Full historical queue replay is not claimed by a move.
    for name in ('shared','benchmarks','tools','archive','reports','summaries','experiments','revisions','improvements','external','external_benchmarks','project_tools','mainline','wdro_rest_min'):
        if (PROJECT/name).exists():link(PROJECT/'methods'/name,PROJECT/name)
    link(PROJECT/'DRO',PROJECT/'shared/legacy_dro')
    for name in METHODS:
        link(PROJECT/'reports'/name,PROJECT/'methods'/name)
    for name in ('benchmarks','shared','tools','archive'):
        link(PROJECT/'reports'/name,PROJECT/name)
    # Ordinary files, including historical Markdown aliases, leave the root.
    history=PROJECT/'reports/history/root_documents';history.mkdir(parents=True,exist_ok=True)
    for p in list(PROJECT.iterdir()):
        if p.name=='README.md' or p.is_dir():continue
        if p.is_file() or p.is_symlink():
            dest=history/p.name
            if dest.exists() or dest.is_symlink():raise RuntimeError(f'root document destination exists: {dest}')
            entry=dict(source=str(p),destination=str(dest),phase='planned',alias=False,
                       relative_links=[('',os.readlink(p),os.path.abspath(p.parent/os.readlink(p)))] if p.is_symlink() else [])
            state['moves'].append(entry);atomic(JOURNAL,state)
            if p.is_symlink():
                old=os.readlink(p);absolute=os.path.abspath(p.parent/old)
                p.rename(dest);dest.unlink();dest.symlink_to(absolute)
                state['rebased_links'].append(dict(path=str(dest),original=old))
            else:p.rename(dest)
            entry['phase']='moved';atomic(JOURNAL,state)
    data=PROJECT/'data';data.mkdir(exist_ok=True)
    for name,target in {
        'cross_evaluation':ART/'data',
        'synthetic_protocol_v1':PROJECT/'methods/wdro_legacy_v1/experiments/cache/datasets',
        'real_ISPD2018_2019':PROJECT/'methods/wdro_precision_v2/experiments/cache/stage7/inputs',
        'real_ISPD2008_CircuitOps':PROJECT/'benchmarks/public_designs_20260912/cache/inputs',
        'downloaded_chip_sources':PROJECT/'benchmarks/public_designs_20260912/sources',
        'legacy_original_tests':PROJECT/'shared/legacy_dro/REST-main/test_set',
        'Timing_RST_original_tests':PROJECT/'methods/papers_wdro_20260915/sources/Timing-RST/data/test_data',
    }.items():link(data/name,target)
    (PROJECT/'methods/README.md').write_text('# 方法与实验\n\n'+''.join(f'- [{name}]({name}/)\n' for name in METHODS)+'\n每个实验的协议、源码和原始产物入口保留在同一目录。目录内的共享入口是兼容链接。\n')
    (data/'README.md').write_text('# 数据索引\n\n数据按来源分入口保存，链接指向同一份原始数据，不复制历史数据。\n\n'+''.join(f'- [{p.name}]({p.name}/)\n' for p in sorted(data.iterdir()) if p.name!='README.md')+'\n本次评估列定义见 ../benchmarks/all_models_all_data_20260928/artifacts/datasets.json。\n')
    (PROJECT/'README.md').write_text('''# DRO / Codex 研究项目索引

统一结果入口：**[全模型 × 全数据集唯一总表](reports/全模型全数据集总表_20260928.html)**。表格按模型权重和测试数据展开，包含测试用途、训练规模、原始指标和缺失原因。

- [methods/](methods/)：各方法的代码、协议、模型与实验产物入口。
- [benchmarks/](benchmarks/)：跨方法评估项目；[本次全数据扩展复测](benchmarks/all_models_all_data_20260928/README.md)。
- [reports/](reports/)：总表、[跨方法总结](reports/summaries/)、[既有验收记录](reports/acceptance/)、[文献资料](reports/literature/)和历史根目录文档。
- [data/](data/)：随机数据、规定分布、真实芯片原始来源与规范化数据的分类入口。
- [tools/](tools/)：共享工具、路径查询与目录恢复。
- [shared/](shared/)：原始项目依赖及附加盘存储入口。
- [archive/](archive/)：历史冻结内容、旧版本和历次迁移记录。

根目录普通文件仅保留本 README。旧方法名等目录链接是历史路径兼容入口，日常浏览请从上面的分类目录进入。方法源码、协议和权重未因整理改写。目录迁移记录位于 archive/restructure_20260928；历史冻结队列如依赖解析后的绝对路径，可以先用 tools/restore_layout_20260928.py 恢复迁移前布局再运行。
''')
    state['status']='complete';state['finished_at']=time.time();atomic(JOURNAL,state)

def restore():
    state=read(JOURNAL)
    for path in reversed(state['links']):
        p=Path(path)
        if p.is_symlink():p.unlink()
    for entry in reversed(state['rebased_links']):
        p=Path(entry['path'])
        if p.is_symlink():p.unlink();p.symlink_to(entry['original'])
    for entry in reversed(state['moves']):
        src,dst=Path(entry['source']),Path(entry['destination'])
        if entry['phase'] not in ('moved','planned'):continue
        if src.is_symlink():src.unlink()
        if not src.exists() and (dst.exists() or dst.is_symlink()):dst.rename(src)
    shutil.copy2(JOURNAL.parent/'README.before.md',PROJECT/'README.md')
    state['status']='restored';atomic(JOURNAL,state)

if __name__=='__main__':
    import sys
    restore() if '--restore' in sys.argv else organize()
