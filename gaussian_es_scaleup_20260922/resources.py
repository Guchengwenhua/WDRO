"""Live admission, scoped to this experiment's workers only."""
import os
from pathlib import Path
import shutil
import subprocess
import time
from common import ART, PROTOCOL, atomic_json

def snapshot():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in output.strip().splitlines():
        index,uuid,name,total,used,free,util=[x.strip() for x in line.split(',')]
        gpus.append(dict(index=int(index),uuid=uuid,name=name,total=int(total),used=int(used),free=int(free),util=int(util)))
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    return dict(at=time.time(),gpus=gpus,load=os.getloadavg()[0],disk_free_gib=shutil.disk_usage(ART).free/2**30,
        mem_available_gib=mem['MemAvailable']/2**20,
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv'],text=True))

def admissible(samples,busy=()):
    cfg=PROTOCOL['resources'];last=samples[-1]
    if last['load']>=cfg['max_load'] or last['disk_free_gib']<cfg['min_disk_gib'] or last['mem_available_gib']<32:return []
    peak=0
    gate=ART/'tests/gpu_gate.json'
    if gate.exists():
        import json
        peak=json.loads(gate.read_text()).get('peak_reserved_mib',0)
    required=max(cfg['min_free_mib'],peak+cfg['memory_margin_mib'])
    available=[]
    for g in last['gpus']:
        values=[next(v for v in s['gpus'] if v['uuid']==g['uuid']) for s in samples]
        if g['index'] not in busy and min(v['free'] for v in values)>=required and sum(v['util'] for v in values)/len(values)<cfg['max_mean_util']:
            available.append(g)
    return sorted(available,key=lambda g:(g['util'],-g['free']))

def preflight():
    samples=[]
    for i in range(3):
        samples.append(snapshot())
        if i<2:time.sleep(1)
    record=dict(samples=samples,admitted=admissible(samples))
    atomic_json(ART/'manifests'/f'preflight_{time.time_ns()}.json',record)
    return record
