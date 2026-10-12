from pathlib import Path
import os,json,shutil
R=Path('/remote-home/Guilin/DRO/Codex');A=R/'archive/restructure_20260920'
methods=[
 ('wdro_precision_v2','WDRO 正式主线 precision_v2','ERM、Mixed ERM、Hard Mining、GroupDRO 和 WDRO 的统一冻结实验；当前数值主线。'),
 ('wdro_legacy_v1','WDRO 历史 v1','保留早期实现和阶段记录；旧参考精度结果用于溯源。'),
 ('adversarial_v3_20260912','对抗搜索 v3（历史）','ES、Critic、RL 的早期实现；已由 v3.1 修订。'),
 ('adversarial_v3_1_20260912','对抗搜索 v3.1','修正推理批形状后的 ES、Critic、RL 固定搜索及继续训练实验。'),
 ('critic_search_20260915','Critic 七方向研究','D1–D7 的代理搜索与继续训练研究；共用同一研究协议。'),
 ('oracle_free_20260915','Oracle-free 几何下界方法','使用几何下界构造目标，去除 GeoSteiner 依赖。'),
 ('rl_only_20260915','纯 RL 对抗训练','以 Actor 自身动作线长为目标，使用 indexed mean-L1 运输。'),
 ('rl_research_20260915','纯 RL 六方向研究','REINFORCE/PPO、best-of-K、序贯、多智能体及种群方向。'),
 ('papers_wdro_20260915','WDRO 三论文迁移','REST、Hub、Buffer/Timing 等论文任务的独立适配研究。'),
 ('rl_only_pretraining_plan_20260919','纯 RL 预训练后续方案','匹配预算的后续实验计划及既有结果分析；计划不等于已运行。'),
]
def write(rel,text):
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
rows='\n'.join(f'| [{label}]({name}/) | `{name}/` | {desc} |' for name,label,desc in methods)
write('README.md',f'''# DRO / Codex 研究项目

这里按独立方法版本和研究协议平级组织。每个方法目录实际存放自己的代码、README、`docs/` 和产物入口；改进方法不再嵌套在父方法内部。同一协议下共用代码的算法变体仍在该方法内维护，避免再次复制代码和模型。

先看 [方法与测评总览](summaries/方法与测评总览_20260919.md)、[研究状态](summaries/STATUS.md) 和 [全目录索引](summaries/DIRECTORY_INDEX.md)。

| 方法 / 版本 | 实际目录 | 范围 |
|---|---|---|
{rows}

| 公共分类 | 内容 |
|---|---|
| [summaries/](summaries/) | 跨方法总览、状态、理论、运维、重构说明 |
| [benchmarks/](benchmarks/) | 公开芯片、v3.1 外部测试、RL-only 与主线比较 |
| [shared/](shared/) | 原始 DRO 依赖和附加盘存储的共享入口 |
| [literature/](literature/) | 文献调研和组会资料 |
| [tools/](tools/) | 项目工具、路径查询和重构完整性验证 |
| [archive/](archive/) | 历史原件、迁移快照、审计清单和旧缓存备份 |

每个方法的项目说明、理论、进展、结果和运行说明集中在其 `docs/` 中；代码包的测试与配置仍就近存放。大体积模型和数据没有再复制。

`revisions`、`improvements`、`wdro_rest_min` 等旧入口以及根目录少数旧 Markdown 名称是历史兼容链接。它们读取迁移前的源码/文档快照和同一份实验产物，以保留旧文件的 SHA-256；当前维护请从上表的新目录进入。完整列表、验证和限制见 [重构说明](summaries/RESTRUCTURE_20260920.md)。
''')
for name,label,desc in methods:
 p=R/name/'README.md';existing=p.read_text() if p.exists() else ''
 prefix=f'''# {label}

{desc}

- [本方法文档与进展](docs/README.md)
- [全项目方法与测评总览](../summaries/方法与测评总览_20260919.md)
- [返回项目入口](../README.md)

此目录是重构后的实际维护目录。历史源码及路径绑定另存于 [迁移归档](../archive/restructure_20260920/README.md)。

'''
 if existing:prefix+='---\n\n'+existing
 write(name+'/README.md',prefix)
 docroot=R/name/'docs';docroot.mkdir(exist_ok=True)
 items=[]
 for q in sorted(docroot.rglob('*.md')):
  if q.name=='README.md':continue
  rel=q.relative_to(docroot);items.append(f'- [{rel}]({rel.as_posix()})')
 write(name+'/docs/README.md',f'# {label}：文档与进展\n\n[方法入口](../README.md)\n\n'+ '\n'.join(items)+'\n\n历史文件中的日期、阶段状态和资源快照按原实验时点理解；当前结论优先查看最终结果与验收材料。\n')

bench=[('public_designs_20260912','公开芯片外部测试'),('v31_models_20260912','v3.1 模型外部测试'),('rl_only_vs_mainline_20260919','RL-only 与主线的交叉比较')]
write('benchmarks/README.md','# 跨方法测评与比较\n\n'+ '\n'.join(f'- [{label}]({name}/README.md)' for name,label in bench)+'\n\n这些是测评工程，不是新的训练方法。各测评的配置、结果和适用边界独立保留。\n')
for name,label in bench:
 base=R/'benchmarks'/name;doc=base/'docs';doc.mkdir(exist_ok=True)
 items='\n'.join(f'- [{p.relative_to(doc)}]({p.relative_to(doc).as_posix()})' for p in sorted(doc.rglob('*.md')) if p.name!='README.md')
 write(f'benchmarks/{name}/docs/README.md',f'# {label}：文档\n\n{items}\n')
 p=base/'README.md';existing=p.read_text() if p.exists() else ''
 write(f'benchmarks/{name}/README.md',f'# {label}\n\n[说明、进展与结果](docs/README.md) · [测评总目录](../README.md)\n\n'+existing)

write('summaries/README.md','''# 跨方法汇总

| 内容 | 入口 |
|---|---|
| 所有方法及测评覆盖 | [方法与测评总览](方法与测评总览_20260919.md) |
| 研究状态 | [STATUS.md](STATUS.md) |
| 理论与方法关系 | [THEORY.md](THEORY.md) |
| 外部测试汇总 | [EXTERNAL.md](EXTERNAL.md) |
| 环境与恢复 | [OPERATIONS.md](OPERATIONS.md) |
| 当前目录及路径映射 | [DIRECTORY_INDEX.md](DIRECTORY_INDEX.md) |
| 本次重构、去重和验证 | [RESTRUCTURE_20260920.md](RESTRUCTURE_20260920.md) |
| 历史项目级进度 | [history/](history/) |

各方法的详细阶段、理论、运行说明和结果在各自的 `docs/`；汇总目录不再充当方法源码的父目录。
''')
write('shared/README.md','''# 公共依赖与存储

- [legacy_dro/](legacy_dro/)：链接原始 `DRO/DRO`，包括 REST、WDRO-REST、benchmark 等依赖。这里没有再复制整个原始工程。
- [experiments/](experiments/)：链接既有附加盘 `/attached/remote-home1/Guilin/DRO/Codex/experiments`。各方法的 `experiments/` 或 `artifacts/` 指向其中已有命名空间；其余方法的产物仍在本方法目录。

两份 adversarial revision 中的 WDRO-REST 副本经逐文件 SHA-256 比较一致后移除。验证清单位于 [duplicates_removed.json](../archive/restructure_20260920/duplicates_removed.json)。原始 `DRO` 内容没有被修改。
''')
write('summaries/DIRECTORY_INDEX.md','# 全目录索引\n\n## 方法\n\n'+ '\n'.join(f'- [{label}](../{name}/README.md)：[文档](../{name}/docs/README.md)' for name,label,desc in methods)+'''

## 分类资料

- [跨方法汇总](README.md)
- [测评与比较](../benchmarks/README.md)
- [公共依赖与存储](../shared/README.md)
- [文献](../literature/)
- [工具](../tools/README.md)
- [迁移审计](../archive/restructure_20260920/README.md)
- [旧路径到当前路径映射](../archive/restructure_20260920/path_map.json)

旧 `revisions/.../Codex` 和 `improvements/...` 的方法实体已迁出；现在仅留兼容视图。当前方法目录之间的继承关系通过代码依赖和文档表达。
''')
write('tools/README.md','''# 项目工具

- `verify_layout_20260920.py`：核对目录、链接、迁移前后文件身份、原始源码哈希及去重证据；输出独立重构验证结果。
- `resolve_path.py OLD_PATH`：查询旧路径对应的当前维护路径和原始历史读取路径。
- 其他带日期的脚本保留其研究用途；涉及历史冻结验收时，应区分当前维护源码与迁移前源码快照。

[重构记录](../archive/restructure_20260920/README.md) · [项目入口](../README.md)
''')
write('archive/restructure_20260920/README.md','''# 2026-09-20 目录重构审计

- `inventory_before.json`：迁移前 50,512 个普通文件/符号链接的 inode、大小、时间和链接目标（不跟随外部数据链接）。
- `frozen_files.json`、`legacy/`：迁移前维护源码、配置与文档的原字节及 SHA-256；实验产物只使用链接，不复制。
- `path_map.json`、`moves.json`：原路径、当前维护路径及实际搬迁日志。
- `rewritten_files.json`：路径修订文件的迁移前后哈希。
- `duplicates_removed.json`：两份完全相同的 WDRO-REST 依赖副本及原始来源的逐文件核对。
- `validation/`：CPU 回归测试、导入/路径检查与文档链接检查证据。
- `verification.json`：本次整理的最终完整性检查；它不替代原科研实验的验收。

旧代码入口指向 `legacy` 的原始源码，当前维护入口指向平级方法目录。冻结 JSON、模型和历史日志保持字节不变；路径搬迁记录单独保存，不改写历史 SHA-256 来制造通过。

严格续跑通常绑定源码、协议和路径身份；请使用原始快照核对历史运行，当前维护源码要通过原有绑定检查才能继续，不能跳过检查。此次没有重新训练或宣称新的科学验收。

恢复时先查 `path_map.json` 和 `moves.json`，从 `legacy` 取回原始源码/配置/文档；产物由对应当前目录取回，已去重依赖可从 `shared/legacy_dro` 恢复。不应把 `legacy` 的链接递归复制成另一套大型实验产物。
''')
write('summaries/ORGANIZATION.md','''# 目录组织说明

当前结构和验证见 [2026-09-20 重构说明](RESTRUCTURE_20260920.md)。

2026-09-15 的分类链接方案是历史记录，原件见 [旧说明](../archive/restructure_20260920/legacy/docs/ORGANIZATION.md)。本次已实际搬迁源码与资料，不再以多层 revision 目录作为当前维护入口。
''')
print('navigation generated')
