"""Produce exactly one interactive method-by-dataset results table."""
import html
import time
from common import *

def render():
    models=read(ART/'models.json');datasets=read(ART/'datasets.json')
    matrix=[];completed=0;cells_done=0
    for m in models:
        result=ART/'results'/m['id'];cells=read(result/'cells.json') if (result/'cells.json').exists() else {}
        completed+=int((result/'complete.json').exists());cells_done+=len(cells)
        row=[]
        for d in datasets:
            c=cells.get(d['id'])
            if c:
                v=[c['length']['mean'],c['length']['p95'],c['objective']['mean'],c['objective']['p95']]
                v += [c.get('regret_percent',{}).get('mean'),c.get('regret_percent',{}).get('p95')]
                row.append(v)
            else:row.append(None)
        matrix.append(row)
        if (result/'failure.json').exists():m['evaluation_error']=read(result/'failure.json')['error']
    ready=sum(m['status']=='ready' for m in models)
    payload=dict(models=models,datasets=datasets,matrix=matrix,completed=completed,ready=ready,cells_done=cells_done,updated=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>DRO 全模型 × 全数据集唯一总表</title>
<style>
*{box-sizing:border-box}body{font:14px system-ui,sans-serif;margin:20px;color:#172333;background:#f6f8fa}h1{font-size:23px;margin-bottom:8px}p{max-width:1300px;line-height:1.6}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}input,select,button{padding:8px;border:1px solid #bcc8d5;border-radius:5px;background:white}input{width:260px}#scroll{height:72vh;overflow:auto;border:1px solid #bac8d6;background:white}table{border-collapse:separate;border-spacing:0;font-size:12px}th,td{border-right:1px solid #dce3ec;border-bottom:1px solid #dce3ec;padding:8px;min-width:128px}thead th{position:sticky;top:0;background:#e6edf5;z-index:2;vertical-align:bottom;height:170px;max-width:150px;overflow-wrap:anywhere}th:first-child{left:0;z-index:3;min-width:340px;max-width:340px}tbody th{position:sticky;left:0;background:#edf2f7;text-align:left;z-index:1;overflow-wrap:anywhere}td{text-align:right;white-space:nowrap}small{display:block;color:#526174;font-weight:normal;margin-top:5px}.muted{background:#f1f3f5;color:#8793a0}.error{background:#ffe1dd;color:#842a26}#detail{white-space:pre-wrap;background:white;padding:16px;border:1px solid #ccd5df;max-height:320px;overflow:auto}#status{font-weight:600}.notice{color:#8a4c00}button{cursor:pointer}
</style>
<h1>所有训练模型 × 项目测试数据：唯一总表</h1><div id="status"></div>
<p>默认指标为平均归一化坐标线长 L，越小越好。P95 是样本线长的第 95 百分位。G% = 100×(L/L*−1)，仅在有可信参考的列显示。任务目标只在同一家族内比较，Hub/Timing/Buffer 使用 pin 0 作为约定根，并不代表真实芯片驱动引脚。颜色在同一数据列、同一模型家族内比较，绿低红高；不计算跨数据集总排名。点击单元格查看权重、样本数、训练规模、测试用途和全部指标。</p>
<p class="notice">这是对已有项目数据的扩展复测，不是全新盲测。含 A_train/A_cal/A_val 的真实数据对相关模型可能已见。历史规则列按记录的原生成器、参数、样本量重建；保留各数据原有坐标尺度。跨 pin 数推理只改变解码长度，不改变权重。辅助攻击器、代理和生成器本身不输出路由树，因此标为“非独立路由器”；其对应训练所得 REST 单独列出。待测或失败的格子不是零分。</p>
<div class="controls"><input id="mq" placeholder="筛选模型 / 方法 / seed / best"><input id="dq" placeholder="筛选测试数据 / 芯片设计 / 分布"><select id="family"><option value="rest">REST 路由模型</option><option value="all">所有模型与辅助权重</option><option value="timing">Timing</option><option value="hub">Hub</option><option value="buffer">Buffer</option><option value="auxiliary">辅助网络</option></select><select id="metric"><option value="0">平均线长 L ↓</option><option value="1">P95 线长 ↓</option><option value="4">平均参考误差 G% ↓</option><option value="5">P95 参考误差 G% ↓</option><option value="2">平均任务目标 ↓</option><option value="3">P95 任务目标 ↓</option></select><select id="degree"><option value="all">全部测试 pin 数</option></select><button id="prev">上一页</button><button id="next">下一页</button><span id="page"></span></div>
<div id="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<p>每页显示 20 个模型以保持浏览流畅，所有行和列均内嵌于本文件，筛选和翻页不会丢弃数据。该文件由正式结果持续更新；“模型完成”指其全部数据列已评估。</p><pre id="detail">点击任意数据格查看详情。</pre>
<script>const DATA=__DATA__;const $=s=>document.getElementById(s);let page=0;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('status').textContent=`模型 ${DATA.models.length} 行；测试数据 ${DATA.datasets.length} 列；可运行模型完成 ${DATA.completed}/${DATA.ready}；已产生 ${DATA.cells_done} 个数值格。更新时间：${DATA.updated}`;
for(const d of [...new Set(DATA.datasets.map(x=>x.degree))].sort((a,b)=>a-b))$('degree').add(new Option('d'+d,d));
function draw(){const fq=$('family').value,mq=$('mq').value.toLowerCase(),dq=$('dq').value.toLowerCase(),dg=$('degree').value,k=+$('metric').value;
 const ri=DATA.models.map((m,i)=>i).filter(i=>(fq==='all'||DATA.models[i].family===fq)&&DATA.models[i].label.toLowerCase().includes(mq));
 const ci=DATA.datasets.map((d,i)=>i).filter(i=>DATA.datasets[i].label.toLowerCase().includes(dq)&&(dg==='all'||DATA.datasets[i].degree===+dg));
 const pages=Math.max(1,Math.ceil(ri.length/20));page=Math.min(page,pages-1);$('page').textContent=`${page+1}/${pages} 页 · ${ri.length} 行 × ${ci.length} 列`;
 $('head').innerHTML='<tr><th>模型 / 方法 / 检查点<br><small>单个训练 seed、预算和检查点各自成行</small></th>'+ci.map(j=>{const d=DATA.datasets[j];return `<th title="${esc(d.source)}">${esc(d.id+' · '+d.label)}<small>n=${d.count} · d=${d.degree}<br>${esc(d.kind)}</small></th>`}).join('')+'</tr>';
 const ranges={};for(const j of ci){for(const i of ri){const m=DATA.models[i],v=DATA.matrix[i][j]?.[k];if(v==null)continue;const key=j+'/'+m.family;if(!ranges[key])ranges[key]=[v,v];ranges[key][0]=Math.min(v,ranges[key][0]);ranges[key][1]=Math.max(v,ranges[key][1]);}}
 $('body').innerHTML=ri.slice(page*20,page*20+20).map(i=>{const m=DATA.models[i];return `<tr><th>${esc(m.id+' · '+m.label)}<small>${esc(m.family)} · 训练 d${m.train_degree??'?'} · seed ${m.seed??'?'}${m.note?'<br>'+esc(m.note):''}</small></th>`+ci.map(j=>{const v=DATA.matrix[i][j]?.[k];let text,cls='',color='';if(v==null){text=m.status==='not_standalone_router'?'非独立路由器':m.status==='catalog_error'?'读取失败':m.evaluation_error?'评估失败':DATA.matrix[i][j]?'无参考':'待测';cls=(text.includes('失败')?'error':'muted')}else{const range=ranges[j+'/'+m.family],ratio=(v-range[0])/(range[1]-range[0]||1);color=`background:hsl(${125*(1-ratio)} 55% 88%)`;text=v.toFixed(k>=4?3:6)}return `<td class="${cls}" style="${color}" onclick="detail(${i},${j})">${text}${v!=null&&DATA.datasets[j].degree!==m.train_degree?'<small>跨规模</small>':''}</td>`}).join('')+'</tr>'}).join('');}
function detail(i,j){const m=DATA.models[i],d=DATA.datasets[j],v=DATA.matrix[i][j];$('detail').textContent=JSON.stringify({模型:m.label,权重:m.path,家族:m.family,训练pin数:m.train_degree,seed:m.seed,训练记录:m.metadata,数据:d.label,原始来源:d.source,样本数:d.count,测试pin数:d.degree,数据用途:d.kind,坐标范围:[d.minimum,d.maximum],根节点:m.family==='rest'?'无根':'约定 pin 0',指标:v?{平均线长:v[0],P95线长:v[1],平均任务目标:v[2],P95任务目标:v[3],平均参考误差百分数:v[4],P95参考误差百分数:v[5]}:null,说明:m.reason||m.note,错误:m.error||m.evaluation_error},null,2)}
for(const id of ['mq','dq','family','metric','degree'])$(id).oninput=()=>{page=0;draw()};$('prev').onclick=()=>{page=Math.max(0,page-1);draw()};$('next').onclick=()=>{page++;draw()};draw();</script></html>'''.replace('__DATA__',data)
    path=ART/'总表.html';tmp=path.with_suffix('.partial');tmp.write_text(page);os.replace(tmp,path)
    report=PROJECT/'reports';report.mkdir(exist_ok=True)
    link=report/'全模型全数据集总表_20260928.html'
    if not link.exists():link.symlink_to(path)
    return dict(models=len(models),datasets=len(datasets),completed=completed,ready=ready,cells=cells_done,table=str(link))

if __name__=='__main__':print(json.dumps(render(),ensure_ascii=False))
