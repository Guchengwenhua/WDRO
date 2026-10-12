"""Wait for successful inference/report, verify lifecycle, publish interpretation."""
import csv
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import scheduler as S

def main():
    ops=ROOT/'artifacts/operations'
    lease=(ops/'closeout.lock').open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    who=S.identity(__import__('os').getpid())
    original=S.read_json(ops/'launch.json');launch=original['launch_id'];scheduler_identity=original['identity']
    binding=S.sha256(Path(__file__))
    try:
        while True:
            assert S.sha256(Path(__file__))==binding
            q=S.read_json(ops/'queue.json');ex=S.read_json(ops/'exit.json')
            assert q['launch_id']==launch,'scheduler launch changed'
            if ex is not None:
                assert ex['launch_id']==launch and ex['status']=='complete',ex
                if not S.is_alive(scheduler_identity):break
            else:assert S.is_alive(scheduler_identity),'scheduler exited without successful completion'
            assert q['status'] not in ('failed','source_drift','failed_dependencies'),q['status']
            S.atomic(ops/'closeout_status.json',dict(status='waiting',at=time.time(),identity=who,completed=len(q['completed']),expected=33))
            time.sleep(20)
        assert not q['running'] and not q['pending'] and not q['failures']
        assert len(q['completed'])==33
        report=ROOT/'artifacts/report';c=S.read_json(report/'complete.json')
        assert c['status']=='complete' and c['acceptance']=='accepted_gpu' and c['task_count']==33
        assert c['independent_length_rows']==1995840
        assert c['manifest_sha256']==S.sha256(ROOT/'manifest.json')
        frozen=S.read_json(ops/'source_hashes.json')['hashes']
        assert frozen==c['source_hashes']
        for name,digest in frozen.items():assert S.sha256(ROOT/name)==digest,name
        for name,digest in c['artifacts'].items():assert S.sha256(report/name)==digest,name
        for name,digest in c['run_bindings'].items():assert S.sha256(name)==digest,name
        attempts=list(ops.glob('attempt_*.json'))
        for path in attempts:
            a=S.read_json(path)
            assert not S.is_alive(a['worker_identity'])
            h=S.read_json(ROOT/a['heartbeat_path'],{})
            if h.get('child_identity'):assert not S.is_alive(h['child_identity'])
        locks=[p for p in ops.glob('*.lock') if p.name!='closeout.lock']
        locks+=list((ROOT/'artifacts/runs').glob('*/run.lock'))+list((ROOT/'artifacts/smoke').glob('*/run.lock'))
        for path in locks:
            with path.open() as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
        def rows(name):
            with (report/name).open() as f:return list(csv.DictReader(f))
        summary=rows('summary.csv');comparisons=rows('comparisons.csv')
        index={(r['method'],r['suite']):r for r in summary}
        methods=list(dict.fromkeys(r['method'] for r in summary))
        now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        lines=['# 与 GroupDRO 等主线方法的比较结论','',f'完成与验收时间：{now}。**33/33 个模型统一评测通过，原任务进程已退出，锁释放。**','',
          '## 同一输入、同一线长口径的结果','',
          '各列分别对该套件的四分布与三个 seed 等权；越低越好，不跨列混合平均。', '',
          '| 模型 | 新方案测试 4,096 nets | 主线测试 40,000 nets | 新方案窗口攻击 |','|---|---:|---:|---:|']
        for m in methods:
            v=[float(index[m,s]['mean']) for s in ('rl_test','mainline_test','rl_attack32')]
            lines.append(f'| {m} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} |')
        lines+=['','## 两个纯 RL 主线相对已有 GroupDRO','',
            '跨训练体系的 seed 独立重采样，输入 nets 配对重采样；区间为探索性 95% bootstrap，未做多重比较校正。正差表示新模型线长更高。','',
            '| 套件 | 新模型 | 平均差 | 相对均值差 | 95% 区间 |','|---|---|---:|---:|---|']
        for r in comparisons:
            if r['method'] in ('rl_only:erm_rollout','rl_only:wdro_rl') and r['baseline']=='mainline:group_dro' and r['group']=='equal_weight_4_groups':
                lines.append(f'| {r["suite"]} | {r["method"]} | {float(r["difference"]):+.6f} | {float(r["relative_mean_difference_percent"]):+.2f}% | [{float(r["ci_low"]):+.6f}, {float(r["ci_high"]):+.6f}] |')
        lines+=['','## 如何解释','',
            '- 主线模型从 DAC21 预训练起步，随后每模型训练 512,000 个 sampled 样本；新方案从随机起步，每模型为 131,072 个。该数字不含候选/rollout 额外前向，也不含 DAC21 预训练。',
            '- Mixed ERM/GroupDRO 每批直接覆盖四种分布；新 RL-only 的训练中心来自 uniform。训练覆盖、归一化、学习率、反馈目标及选模也不同。',
            '- 因而可以比较这些冻结模型在相同输入上的解质量；不能把差异全部归因于是否使用 GeoSteiner，也不能据此判断同起点同预算下的 GroupDRO 与新算法优劣。',
            '- GroupDRO 的权重更新并非数学上必须依赖最优值：可用自身动作线长定义组损失。那是新的 RL-only GroupDRO 训练配置，本轮没有将旧模型冒充该新配置。',
            '- 测量的是 REST span 记账线长，未测精确最优误差。攻击是固定窗口候选最大值，不是严格 ρ 球内风险。', '',
            '## 验收证据','',
            f'- 独立重算 {c["independent_length_rows"]:,} 条保存拓扑线长；模型、数据、源码和逐格结果哈希、连通性、实际调用预算通过。',
            f'- 新完成推理 {c["new_inference_completed_real_rows"]:,} 行；复用既有结果 {c["reused_rows"]:,} 行。复用前逐模型/数据格的兼容性重放均通过。',
            f'- {len(attempts)} 个记录的 worker 尝试和原调度器均结束，{len(locks)} 个实验/调度/smoke 锁释放；本次无新训练、无参考求解器调用。','',
            '[完整比较报告](artifacts/report/REPORT.md) · [全部配对区间](artifacts/report/comparisons.csv) · [训练条件差异](TRAINING_DIFFERENCES.md) · [最终验收](artifacts/operations/acceptance.json)']
        (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        record=dict(status='PASS',at=time.time(),identity=who,task_count=33,independent_length_rows=1995840,
                    scheduler_exited=True,recorded_worker_attempts_exited=len(attempts),locks_released=len(locks),
                    script_sha256=binding,manifest_sha256=c['manifest_sha256'],report_complete_sha256=S.sha256(report/'complete.json'),
                    artifacts={str(p.relative_to(ROOT)):S.sha256(p) for p in (ROOT/'RESULTS.md',ROOT/'TRAINING_DIFFERENCES.md')},
                    new_training_updates=0,reference_solver_calls=0)
        S.atomic(ops/'acceptance.json',record)
        (ROOT/'STATUS.md').write_text(f'# 统一比较当前状态\n\n{now}：**33/33 个模型评测和独立验收完成。**\n\n[比较结论](RESULTS.md) · [完整表与解释](artifacts/report/REPORT.md) · [验收凭证](artifacts/operations/acceptance.json)\n\n两套测试集、共同窗口攻击、相同输入与 greedy 口径；未重训或调用参考求解器。旧模型的预训练、训练预算和数据覆盖不同，结论属于现有模型质量比较。\n')
        S.atomic(ops/'closeout_status.json',dict(status='complete',at=time.time(),identity=who,acceptance_sha256=S.sha256(ops/'acceptance.json')))
        print(json.dumps(record),flush=True)
    except BaseException as exc:
        S.atomic(ops/'closeout_status.json',dict(status='failed',at=time.time(),identity=who,error=repr(exc)))
        raise
    finally:lease.close()

if __name__=='__main__':main()
