"""Assemble acceptance from completed independent checks; never mutate experiment inputs."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import statistics

ROOT = Path('/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922')
OUT = ROOT / 'artifacts/review_20260926'


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def link(label, path):
    return f'[{label}]({path})'


def main():
    audit = read('audit.json')
    replay = read('replay.json')
    recheck = read('final_recheck.json')
    primary = read('statistics_recomputed.json')
    assert audit['status'] == 'PASS_ARTIFACTS_STATISTICS_PENDING_GPU_REPLAY'
    assert replay['status'] == recheck['status'] == 'PASS'
    assert recheck['original_bindings_unchanged'] == audit['unique_verified_bindings']
    assert replay['models'] == audit['task_kinds']['train'] == 49
    assert not primary['primary_gain_supported']
    assert 'Ran 5 tests' in (OUT / 'unit.log').read_text()
    assert (OUT / 'unit.log').read_text().rstrip().endswith('OK')
    rows = list(csv.DictReader((OUT / 'training_audit.csv').open()))
    pre = [r for r in rows if r['phase'] == 'pre']
    post = [r for r in rows if r['phase'] == 'post']
    comparisons = list(csv.DictReader((OUT / 'comparisons.csv').open()))
    search = list(csv.DictReader((OUT / 'search_cells.csv').open()))
    metrics = list(csv.DictReader((OUT / 'metrics_recomputed.csv').open()))
    assert len(pre) == 12 and len(post) == 33 and len(metrics) == 264
    hours = {m: sum(float(r['train_hours']) for r in post if r['method'] == m)
             for m in ('erm', 'random', 'es')}
    diagnostics = dict(
        pretraining_best_steps={r['task']: int(r['best_step']) for r in pre},
        pretraining_final_step_selected=sum(r['best_step'] == r['updates'] for r in pre),
        posttraining_final_step_selected=sum(r['best_step'] == r['updates'] for r in post),
        training_worker_hours=sum(float(r['train_hours']) for r in rows),
        posttraining_worker_hours=hours,
        es_extra_posttraining_worker_time=hours['es'] / hours['random'] - 1,
        transport=[], budget_validation=[])
    for degree in (20, 30, 40):
        for method in ('random', 'es'):
            group = [r for r in post if int(r['degree']) == degree and r['method'] == method]
            fractions = [float(r['zero_price_fraction']) for r in group]
            diagnostics['transport'].append(dict(
                degree=degree, method=method,
                mean_cost_over_rho=statistics.mean(float(r['mean_cost']) / float(r['rho']) for r in group),
                zero_lambda_min=min(fractions), zero_lambda_max=max(fractions)))
    for budget in (64, 256):
        cell = {}
        for method in ('random', 'es'):
            group = [r for r in search if r['task'] == f'budget_{method}{budget}']
            assert len(group) == 4, (method, budget, [r['task'] for r in search])
            cell[method] = dict(gain=statistics.mean(float(r['gain']) for r in group),
                                seconds=sum(float(r['seconds']) for r in group))
        diagnostics['budget_validation'].append(dict(
            budget=budget, **cell, relative_gain=cell['es']['gain'] / cell['random']['gain'] - 1))
    write(OUT / 'diagnostics.json', diagnostics)

    degree_lines = []
    pressure_lines = []
    for degree in (20, 30, 40):
        group = [r for r in comparisons if int(r['degree']) == degree and r['suite'] == 'nominal' and r['metric'] == 'p95']
        erm = next(r for r in group if r['baseline'] == 'erm')
        random = next(r for r in group if r['baseline'] == 'random')
        degree_lines.append(f"| {degree} | {100*float(erm['baseline_value']):.6f}% | {100*float(random['baseline_value']):.6f}% | {100*float(random['es']):.6f}% | {100*float(random['relative_improvement']):+.3f}% | {random['improving_seeds']}/{random['seeds']} |")
        pressure = next(r for r in comparisons if int(r['degree']) == degree and r['suite'] == 'pressure' and r['metric'] == 'p95' and r['baseline'] == 'random')
        pressure_lines.append(f"| {degree} | {100*float(pressure['baseline_value']):.6f}% | {100*float(pressure['es']):.6f}% | {100*float(pressure['relative_improvement']):+.3f}% |")
    transport_lines = [f"| {r['degree']} | {r['method']} | {r['mean_cost_over_rho']:.3f} | {100*r['zero_lambda_min']:.2f}%–{100*r['zero_lambda_max']:.2f}% |" for r in diagnostics['transport']]
    search_lines = [f"| {r['budget']} | {r['random']['gain']:.8f} | {r['es']['gain']:.8f} | {100*r['relative_gain']:.2f}% | {r['random']['seconds']:.2f} / {r['es']['seconds']:.2f} |" for r in diagnostics['budget_validation']]
    pre_lines = []
    for degree in (20, 30, 40):
        group = sorted([r for r in pre if int(r['degree']) == degree], key=lambda r: int(r['seed']))
        text = '；'.join(f"{r['seed']}→{r['best_step']}" for r in group)
        pre_lines.append(f"| {degree} | {group[0]['updates']} | {text} |")

    report = f'''# Gaussian-ES 扩规模实验验收（2026-09-26）

**工程与数值验收通过；预注册的 ES64 稳定训练收益未建立。** 实验已于 2026-09-25 14:12:59 UTC 完成，97/97 项任务结束，从首个任务启动至正式完成耗时 {audit['total_wall_hours']:.2f} 小时。本次验收重新核对原始产物、独立计算统计量，并在服务器 GPU 上复算 checkpoint 输出，不以原自动报告作为验收依据。

主比较 degree30 中，ES64 的 OOD P95 相对 Random64 下降 **0.706%**，5 个种子中 4 个改善，但未达到预设的 ≥1% 降幅，单侧精确种子符号翻转检验 **p=0.1875**，也未达到 0.05。结论是当前协议下证据不足，不能据此断言两种方法等效，或 ES 不可能有效。

## 验收范围与协议执行

本次验收对象为 {link('gaussian_es_scaleup_20260922', ROOT)}。原 {link('protocol.json', ROOT/'protocol.json')}、{link('理论与实验方案', ROOT/'docs/理论与实验方案.md')}、训练代码、checkpoint、测试结果及历史交接文件均未修改。新文件仅写入本项目下的 `artifacts/review_20260926` 和本报告。

| 阶段 | 完成情况 | 实际规模 |
|---|---:|---|
| 扩展预训练 | 12/12 | degree30 的 5 个正式种子及独立 seed901：8192 updates、batch128；degree20/40 各 3 种子：4096 updates、batch128 |
| 外层调参 | 4/4 | 仅 seed901；1024 updates、batch64；学习率和 nominal fraction 的 2×2 组合 |
| 固定模型搜索与预算验证 | 15/15 | 9 个 ES 参数组合、Random/CEM 对照、独立预算集上的 ES/Random K64/K256 |
| 正式后训练 | 33/33 | ERM、Random64、ES64 配对；degree30 每种方法 5 种子×8192 updates，其余各 3 种子×4096；batch64 |
| 统一测试 | 33/33 | 每模型 4 分布×4096 常规测试点，另有 4 分布×128 独立 CEM64 压力中心 |

最终由 validation 选中 `mu=0.01`、`search_step=0.01`、`lr=1e-5`、`nominal_fraction=0.25`。选择过程已重新计算并与冻结清单一致。正式模型在统一测试开始前冻结；同 degree/seed 的三种后训练方法使用相同父 checkpoint、源中心序列、更新数和学习率。seed901 不进入正式五种子统计。

所有 49 个训练任务合计执行 **274,432 次更新**，对应 **22,282,240 次训练样本使用**；配对方法会重复使用同一源中心，这个数不能解释为独立、不重复的 net 数量。累计训练计时约 **{diagnostics['training_worker_hours']:.2f} worker-hours**，包含参考求解及共享资源等待，不能当作纯 GPU 计算时长。

## 独立证据核对

| 检查 | 结果及覆盖范围 |
|---|---|
| 来源与文件完整性 | 71 个冻结来源文件；5,137 项唯一文件绑定重新 SHA-256 核对，GPU 复算后再次确认全部未变 |
| 任务生命周期 | 97 个完成回执与任务、PID/start tick、exit、heartbeat 一致；旧进程身份均已退出；98 把任务/控制器锁均可释放；正式任务无 failure 回执 |
| checkpoint 与续训 | 49 组 latest/best 的步数、身份、数值和状态通过；历史连续 6 步与中断恢复至 6 步的 Actor、Critic、Adam、lambda、RNG 逐项相等 |
| 更新与调用预算 | 274,432 条更新记录无漏步/重步；逐步核对真实评分、采样更新、参考调用及 lambda 更新方程；请求/完成账本一致 |
| 配对与训练输入 | 所有后训练配对中心 hash 序列一致；另独立重建每训练任务首、中、末批，共 147 批 |
| 搜索合法性与选优 | 核对保存的 1,108 个训练审计中心及其完整候选 support；合法性、Hungarian-L1 成本及原始 penalized score argmax 通过 |
| 全部最终测试产物 | 4,752 个 NPZ 分片、557,568 条输出的完整性、模型绑定、坐标合法性、regret 恒等式与配对通过；其中普通测试 540,672 条、压力测试 16,896 条 |
| 汇总与隔离 | 从原始数组重算全部 264 行指标、801 条 validation 曲线；每 degree 的 6 个 held-out split 全量重建，45 组跨 split 比较无交集，split 内无重复 |
| GPU 重放 | 49 个 best checkpoint 的全部 validation：50,176 次预测；33 个正式模型另抽查 1,056 个测试输出，共 51,232 次预测，最大指标/线长误差均为 0 |
| 参考数值 | 重放参考值重新求解；1,056 个抽查测试点另用 GeoSteiner 2^27/2^28 双尺度复核，两尺度结果一致 |
| 统计与测试 | 独立重算全部 32 个种子符号组合、2000 次交叉 bootstrap，与原始统计一致；冻结的 5 项单元测试重跑通过 |

GPU 重放使用新鲜资源准入选出的 GPU7（RTX 6000 Ada），约 54 秒，峰值 reserved 显存 212 MiB；仅推理和参考复算，无新训练。验收程序复用冻结的数据生成、几何和模型基础实现，但没有调用原最终报告、模型选择或统计函数。独立性是验收逻辑和统计计算的独立实现，不代表另有独立人员审稿或完全独立的几何求解器。

## 预注册主指标

指标为 `relative_regret = L_REST/L_star - 1`，越低越好。先对每个种子、每个 OOD 分布计算 P95，再对三个 OOD 分布和种子平均；不是把所有测试点混合后的 pooled P95。

| degree30 主判据 | 实测 | 门槛 | 判定 |
|---|---:|---:|---|
| ES64 相对 Random64 的 OOD P95 降幅 | 0.706% | ≥1% | 未达到 |
| 单侧精确种子符号翻转 p | 0.1875 | ≤0.05 | 未达到 |
| 改善种子 | 4/5 | ≥4/5 | 通过 |
| uniform mean：ES−ERM | +0.0000391133 | ≤0.002 | 通过 |
| 独立 CEM64 四分布 mean：ES−Random | −0.0003458961 | ≤0 | 通过 |

Random64 与 ES64 的 OOD P95 分别为 **2.556474%**、**2.538421%**，差值为 **−0.018054 个百分点**。描述性交叉 bootstrap 95% 区间为 **[−0.072908, +0.033919] 个百分点**，跨过 0；显著性判定依据是预注册的种子检验，不将数万个 net 当作独立训练重复。

## 次要结果与搜索结果

普通 OOD 测试如下；正的相对降幅有利于 ES，负值表示变差。degree20/40 各只有 3 个种子，以下是描述性结果，不新增事后显著性结论。

| degree | ERM OOD P95 | Random64 OOD P95 | ES64 OOD P95 | ES 相对 Random 降幅 | 改善种子 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(degree_lines)}

独立 CEM64 压力测试的三个 OOD 分布平均 P95（与主判据的四分布 mean 不同）：

| degree | Random64 训练模型 | ES64 训练模型 | ES 相对 Random 降幅 |
|---|---:|---:|---:|
{chr(10).join(pressure_lines)}

degree20 在压力测试中反而变差，degree40 普通 OOD P95 略差，因此不能将 degree30 的方向性改善推广成跨规模稳定优势。

固定模型、独立 budget-validation 集上的搜索结果：

| 每中心真实评分预算 | Random 平均搜索 gain | ES 平均搜索 gain | ES 的 gain 相对增加 | Random / ES 实测秒数 |
|---|---:|---:|---:|---:|
{chr(10).join(search_lines)}

此处 gain 是 `max_Z[regret(Z)−lambda*c(X,Z)]−f(X)`。它度量攻击器找到更高原始目标值的能力，来自一个冻结 degree30 父模型，不能当作训练后 regret 降低 29% 或 58%，也不证明所有模型上都优于 CEM。

## 影响解释的训练诊断

**增加更新数不等于最终模型使用了训练末轮。** 12 个预训练任务只有 1 个选中最后一步，实际下游初始化步数如下（种子→best step）：

| degree | 完成更新数 | 选中的父模型 |
|---|---:|---|
{chr(10).join(pre_lines)}

后训练也只有 {diagnostics['posttraining_final_step_selected']}/33 个选中末步。这符合冻结的 validation 选模规则，但当前实验没有独立的预训练时长因果对照，不能宣称延长预训练本身带来收益。

**运输预算在多数更新中未形成持续的正对偶惩罚。** 下表平均 cost/rho 由每个种子的全程平均成本计算，lambda=0 比例给出种子间范围：

| degree | 方法 | 平均 cost/rho | 更新后 lambda=0 的步数占比 |
|---|---|---:|---:|
{chr(10).join(transport_lines)}

逐步更新方程已核对无误，lambda=0 本身不是实现错误。当 lambda 为 0 时，内层目标等于原始 regret，因此大量训练步骤更接近无运输惩罚的难例挖掘；这些记录不足以证明正对偶惩罚在训练中持续发挥作用。

**真实调用数配平不等于 GPU 工作量配平。** ES64 与 Random64 每个 batch64 都进行 4096 次真实候选评分，但当前 ES 分块补齐为 5632 个 actor forward rows，Random 为 4096。全部任务账本中真实评分 556,762,112 次、forward rows 664,331,264 次，padding 107,569,152 次。33 个正式后训练任务的累计计时：ERM {hours['erm']:.2f}、Random {hours['random']:.2f}、ES {hours['es']:.2f} worker-hours；ES 比 Random 高约 {100*diagnostics['es_extra_posttraining_worker_time']:.1f}%。共享服务器上的计时含争用和 CPU 求解，不能作为隔离硬件基准。

## 结论边界与归档

可接受的结论是：扩规模训练、调参、选模和统一评价按冻结协议完成，数值与产物可复核；高斯平滑在本次固定模型验证中提升了搜索 gain，但 ES64 相对同真实调用预算 Random64 的预注册训练收益未建立。

未覆盖的内容必须保留：没有逐一重建全部训练 net 并做完整 train-test 内容交集检查；已验证独立命名空间、全部 held-out split 隔离及 147 批训练重建。保存的候选 support 只覆盖 1,108 个中心，并非全部约 5.6 亿次候选评分。全部测试数组已做代数与几何核查，GPU 上重跑的是全部 best validation 和抽样测试，未重新求解全部最终测试。有限候选、有限分布和有限种子不能给出连续 Wasserstein 最坏情形证书，也不能给出普遍优越性结论。

后续研究应先检查早期选模、validation 方差和 rho/惩罚的实际作用，再设计预训练时长与等计算成本对照。任何进一步调参需另立协议并使用新的最终测试集；本次验收不修改现有选模、不追调已有测试、不追加训练。

验收过程中初版脚本误将 step0 checkpoint 的空 Adam state 判为错误；已更正为只要求 step>0 时存在已初始化的 Adam state，完整重跑通过。原异常保留在 `audit_failure.json`。它与历史磁盘迁移时人工中断产生的 `artifacts/tests/gpu_failure.json` 均不属于正式训练任务失败。`audit.json` 保留 CPU 阶段的 `PENDING_GPU_REPLAY` 名称，最终联合判定以本次 `acceptance.json` 为准。

产物入口：

- {link('联合验收清单 acceptance.json', OUT/'acceptance.json')}：最终工程/科学状态与本次审计文件 SHA-256。
- {link('CPU 产物审计 audit.json', OUT/'audit.json')}、{link('GPU 重放 replay.json', OUT/'replay.json')}、{link('最终复核 final_recheck.json', OUT/'final_recheck.json')}。
- {link('独立统计 statistics_recomputed.json', OUT/'statistics_recomputed.json')}、{link('逐模型分布指标 metrics_recomputed.csv', OUT/'metrics_recomputed.csv')}、{link('配对比较 comparisons.csv', OUT/'comparisons.csv')}。
- {link('训练诊断 training_audit.csv', OUT/'training_audit.csv')}、{link('补充诊断 diagnostics.json', OUT/'diagnostics.json')}、{link('搜索记录 search_cells.csv', OUT/'search_cells.csv')}。
- {link('CPU 验收代码 audit.py', OUT/'audit.py')}、{link('GPU 重放代码 replay.py', OUT/'replay.py')}、{link('单元测试输出 unit.log', OUT/'unit.log')}。

逻辑入口全部位于 `DRO/Codex`。`artifacts` 沿用原实验的 NVMe 存储链接，实际位置为 `{(ROOT/'artifacts').resolve()}`。原任务与本次 GPU 复算均已退出，不需要继续挂 tmux 或等待 GPU 训练。
'''
    report_path = ROOT / 'ACCEPTANCE_20260926.md'
    report_path.write_text(report)
    review_files = [p for p in OUT.iterdir() if p.is_file() and p.name != 'acceptance.json' and not p.name.endswith('.partial')]
    manifest = dict(
        accepted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        experiment='gaussian_es_scaleup_20260922_v1',
        engineering_status='PASS_ENGINEERING_NUMERICAL',
        scientific_status='PRIMARY_GAIN_NOT_ESTABLISHED',
        primary=primary,
        source_freeze_sha256=sha(ROOT/'artifacts/manifests/source_freeze.json'),
        experiment_input_bindings=5137,
        witnesses=['audit.json', 'replay.json', 'final_recheck.json', 'unit.log'],
        phase_status_note='CPU audit status is historical; GPU replay PASS completes combined acceptance.',
        resolved_reviewer_error='Initial auditor required nonempty Adam state for legal step0 best checkpoint; fixed and full audit rerun passed. Historical audit_failure.json retained.',
        original_training_modified=False,
        pending_work=False,
        scope_limits=audit['provenance_limits'],
        files=[dict(path=str(p), sha256=sha(p)) for p in sorted(review_files + [report_path])])
    write(OUT/'acceptance.json', manifest)
    for binding in manifest['files']:
        assert sha(Path(binding['path'])) == binding['sha256']
    print(json.dumps(dict(engineering=manifest['engineering_status'], scientific=manifest['scientific_status'],
                         report=str(report_path), bound_review_files=len(manifest['files'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
