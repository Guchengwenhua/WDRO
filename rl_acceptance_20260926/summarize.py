"""Produce an evidence-scoped Chinese acceptance report."""
import argparse,csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def main(final=False):
    old=read(ROOT/'old/acceptance.json');directory=ROOT/'final/new' if final else ROOT/'new';new=read(directory/'acceptance.json')
    migration=read(ROOT/'storage_migration.json');compatibility=read(ROOT/'storage_compatibility.json');guard=read(ROOT/'storage_guard_probe.json')
    status='PASS' if old.get('status')=='PASS' and new.get('status')=='PASS' else 'INCOMPLETE' if new.get('status')=='INCOMPLETE' else 'NEEDS_REVIEW'
    lines=['# RL 扩大实验验收', '',f'生成时间：{time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}。综合状态：**{status}**。','',
        '| 实验 | 审计任务 | 实验完整性 | 已完成产物检查 |','|---|---:|---|---|',
        f'| 20260921 首轮 | {old.get("tasks",0)}/80 | {old.get("status","PENDING")} | {old.get("status","PENDING")} |',
        f'| 20260922 W/T/A 扩展 | {new.get("audited_tasks",0)}/104 | {new.get("status","PENDING")} | {new.get("completed_subset_status","PENDING")} |','',
        '## 首轮结论','',
        '逐文件、调用预算、几何/拓扑/线长、输入隔离和检查点/优化器/RNG/λ递推重新检查；重新生成统计表，并核对原表的逐字节一致性。主检验额外用独立实现枚举符号翻转与Holm校正。','',
        '| 阶段 | 指标 | 方法/对照 | 相对改善 | Holm p | 改善seed | 结论 |','|---|---|---|---:|---:|---:|---|']
    for r in old.get('primary',[]):lines.append(f'| {r["kind"]} | {r["suite"]} | {r["method"]}/{r["reference"]} | {r["relative_improvement"]:.2%} | {r["p_holm"]:.5f} | {r["improving_seeds"]}/8 | {r["conclusion"]} |')
    lines+=['','正数表示改善。固定模型搜索gain更高与联合训练后布线策略更好是不同结论；未达到门槛不等于证明无效或等效。','',
        '## 扩展轮','',f'已审计新增updates：{new.get("added_updates_verified",0):,}/{new.get("expected_additional_updates",638976):,}。',
        f'输入隔离：独立重建训练输入 {new.get("training_unique_inputs",0):,}，评价输入 {new.get("heldout_unique_inputs",0):,}，交集 {new.get("overlap","PENDING")}。','',
        '下表只展示8个seed齐全的比较。未齐全的比较不删seed、不补均值；全实验完成前不发布六项主检验的最终优势判定。','',
        '| 比较 | 方法 | 可用seed | 相对改善 | 描述性95%区间（绝对改善） | 判定 |','|---|---|---:|---:|---|---|']
    for r in new.get('primary',[]):
        pct=f'{r["relative_improvement"]:.2%}' if 'relative_improvement' in r else '待齐全'
        ci=f'[{r["ci_low"]:.6f}, {r["ci_high"]:.6f}]' if 'ci_low' in r else '待齐全'
        lines.append(f'| {r["contrast"]} | {r["method"]} | {r["available_seeds"]}/8 | {pct} | {ci} | {r["conclusion"]} |')
    means=directory/'full_seed_group_means.csv'
    if means.exists():
        with means.open() as f:mr=[x for x in csv.DictReader(f) if x['suite']=='robust']
        lines+=['','各方法的完整8-seed鲁棒线长均值（越小越好）：','',
            '| W | 方法 | T=2048 | T=4096 | T=8192 | 后训练延长的相对改善 |','|---:|---|---:|---:|---:|---:|']
        for w in [2048,8192]:
            for method in ['erm','random','bestof_ppo','sequential_ppo']:
                values={int(x['posttraining']):float(x['mean']) for x in mr if int(x['pretraining'])==w and x['method']==method}
                if len(values)==3:
                    gain=(values[2048]-values[8192])/values[2048]
                    lines.append(f'| {w} | {method} | {values[2048]:.6f} | {values[4096]:.6f} | {values[8192]:.6f} | {gain:.2%} |')
    lines+=['','固定模型上的攻击器延长训练（λ=5，次要描述性比较）：','',
        '| 方法 | 可用seed | gain相对改善 | 描述性95%区间（gain绝对差） |','|---|---:|---:|---|']
    for r in new.get('attacker_secondary',[]):
        pct=f'{r["relative_improvement"]:.2%}' if 'relative_improvement' in r else '待齐全'
        ci=f'[{r["ci_low"]:.6f}, {r["ci_high"]:.6f}]' if 'ci_low' in r else '待齐全'
        lines.append(f'| {r["method"]} | {r["available_seeds"]}/8 | {pct} | {ci} |')
    if new.get('missing_tasks'):
        lines+=['','尚未完成：','']+['- '+n for n in new['missing_tasks']]
    lines+=['','## 存储、恢复与验收范围','',f'附加盘初次复制校验：{migration.get("status","PENDING")}。初次复制记录保存在storage_migration.json；最终采用混合存储，保留原DRO/Codex访问路径。']
    for m in migration.get('moves',[]):lines.append(f'- 初次复制 `{m["logical"]}` 至 `{m["resolved"]}`；{m["files"]}个文件，{m["bytes"]/2**30:.2f} GiB。')
    if compatibility:
        lines += ['',f'恢复原权重路径检查后的布局：{compatibility["status"]}。约{compatibility["restored_bytes"]/2**30:.3f} GiB必需检查点及小文件保留主盘，其余完成档案和大型评价数组保留附加盘；剩余17项任务的输出检查点也留在本实验根目录内。',
            f'原始权重检查实测：{guard.get("status","PENDING")}，加载{guard.get("required_parents_loaded_under_original_guard",0)}个必需父检查点，外部权重仍被拒绝。',
            '直接链接整个目录曾触发权重路径检查，相关任务在初始化阶段退出，未进入训练更新；这些退出记录保留，修复采用文件布局调整，没有放宽检查或修改冻结代码。']
    lines+=['','迁移未改变训练源码、协议、步数、seed、选择规则和GPU预算；使用原tmux队列续跑。验收脚本本身只读训练产物。',
        '实验结论限定为degree30、8个训练seed、自身布线长度和有限候选攻击；不是最优性差距或连续Wasserstein鲁棒性认证。',
        '首轮重算产物：`../rl_scaleup_20260921/artifacts/acceptance_20260926/`。本目录old/new记录本次首轮及扩展子集审计；final/new记录扩展完成后的自动终验。']
    errors=old.get('errors',[])+new.get('errors',[])
    if errors:lines+=['','## 未通过项目','']+['- '+x for x in errors]
    name='FINAL_REPORT.md' if final else 'INTERIM_REPORT.md'
    (ROOT/name).write_text('\n'.join(lines)+'\n')
    (ROOT/'STATUS.json').write_text(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status=status,report=name,
        old=old.get('status'),new=new.get('status'),new_tasks=new.get('audited_tasks'),scope='Full completion required for PASS'),ensure_ascii=False,indent=2))
    print(json.dumps(dict(status=status,report=str(ROOT/name))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args();main(args.final)
