"""Replot frozen precision_v2 results; no training or result selection.

Run with /remote-home/Guilin/miniconda3/envs/py311/bin/python make_figures.py
Only the Python standard library, NumPy and Matplotlib are required.
"""
from pathlib import Path
import csv
import hashlib
import json
import os

ROOT = Path('/remote-home/Guilin/DRO/Codex')
OUT = ROOT / 'experiments/figures/wdro_erm_ppt_20260920'
TABLES = ROOT / 'revisions/reference_precision_v2/Codex/experiments/tables'
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

METHODS = ['erm', 'mixed_erm', 'wdro_rest']
NAMES = {'erm': 'REST / ERM', 'mixed_erm': 'Mixed ERM', 'wdro_rest': 'WDRO-REST'}
COLORS = {'erm': '#64748B', 'mixed_erm': '#E3A23C', 'wdro_rest': '#087F8C'}
MARKERS = {'erm': 'o', 'mixed_erm': 's', 'wdro_rest': 'D'}
DISTS = ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster', 'ispd']
LABELS = ['Uniform', 'Gaussian mixture', 'Edge bias', 'Grid cluster', 'ISPD']
DEGREES = [10, 20, 30, 40, 50]
SOURCES = ['stage6_per_degree.csv', 'stage6_per_seed.csv', 'stage6_main.csv',
           'stage7/per_degree.csv', 'stage7/main.csv']

def read(name):
    with (TABLES / name).open() as f:
        return list(csv.DictReader(f))

per = read(SOURCES[0])
seeds = read(SOURCES[1])
main = read(SOURCES[2])
ispd = read(SOURCES[3])
ispd_main = read(SOURCES[4])
values, totals, exported = {}, {}, []
for dist in DISTS:
    for method in METHODS:
        for degree in DEGREES:
            if dist == 'ispd':
                rows = [r for r in ispd if r['method'] == method and int(r['degree']) == degree]
                assert len(rows) == 3 and len({r['seed'] for r in rows}) == 3
                mean = np.mean([float(r['net_weighted_mean']) for r in rows])
                p95 = np.mean([float(r['p95']) for r in rows])
                count = int(rows[0]['count'])
            else:
                rows = [r for r in per if r['method'] == method and r['distribution'] == dist and int(r['degree']) == degree]
                assert len(rows) == 1
                mean, p95 = (float(rows[0][k]) for k in ('mean', 'p95'))
                count = int(rows[0]['nets_per_seed'])
                seed_rows = [r for r in seeds if r['method'] == method and r['distribution'] == dist and int(r['degree']) == degree]
                assert len(seed_rows) == 3
                for key, estimate in [('mean', mean), ('p95', p95)]:
                    assert np.isclose(estimate, np.mean([float(r[key]) for r in seed_rows]), atol=1e-12, rtol=0)
            values[dist, method, degree] = {'mean': mean, 'p95': p95}
            exported.append(dict(dataset=dist, method=method, degree=degree, seeds=3,
                                 unique_nets=count, mean_regret_pct=mean*100, p95_regret_pct=p95*100))
        if dist == 'ispd':
            r = next(r for r in ispd_main if r['method'] == method)
            totals[dist, method] = dict(mean=float(r['net_weighted_mean']), p95=float(r['p95']))
            for seed in ['101', '102', '103']:
                rr = [r for r in ispd if r['method'] == method and r['seed'] == seed]
                assert sum(int(r['count']) for r in rr) == 22498
            weighted = sum(next(e['unique_nets'] for e in exported if e['dataset']==dist and e['method']==method and e['degree']==d) * values[dist,method,d]['mean'] for d in DEGREES)/22498
            assert np.isclose(weighted, totals[dist,method]['mean'], atol=1e-12, rtol=0)
        else:
            r = next(r for r in main if r['method'] == method and r['distribution'] == dist)
            totals[dist, method] = {k:float(r[k]) for k in ['mean','p95']}
            for k in ['mean','p95']:
                assert np.isclose(totals[dist,method][k], np.mean([values[dist,method,d][k] for d in DEGREES]), atol=1e-12, rtol=0)

def gain(base, candidate):
    return 100 * (base - candidate) / base

for row in exported:
    base = values[row['dataset'], 'erm', row['degree']]
    for key in ['mean', 'p95']:
        row[f'{key}_reduction_vs_erm_pct'] = gain(base[key], row[f'{key}_regret_pct']/100)

summary=[]
for dist in DISTS:
    for method in METHODS:
        for metric in ['mean','p95']:
            baseline = totals[dist,'erm'][metric]
            val = totals[dist,method][metric]
            summary.append(dict(dataset=dist,method=method,metric=metric,
                                regret_pct=val*100, reduction_vs_erm_pct=gain(baseline,val),
                                reduction_vs_erm_percentage_points=(baseline-val)*100))
for name, rows in [('plot_data_per_degree.csv',exported),('plot_data_summary.csv',summary)]:
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':17,
                     'axes.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':False,'axes.spines.bottom':False,
                     'text.color':'#172B3A','axes.labelcolor':'#334155',
                     'xtick.color':'#475569','ytick.color':'#475569',
                     'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E7EDF1',linewidth=1)
    ax.tick_params(length=0,pad=8)

def header(fig, kicker, title, subtitle):
    fig.text(.065,.955,kicker,fontsize=12,color=COLORS['wdro_rest'],weight='bold')
    fig.text(.065,.893,title,fontsize=27,weight='bold')
    fig.text(.065,.846,subtitle,fontsize=13,color='#64748B')

def footer(fig, text):
    fig.text(.065,.069,text,fontsize=10.5,color='#64748B',linespacing=1.5)
    fig.text(.935,.025,'WDRO-REST  |  precision_v2',ha='right',fontsize=9,color='#94A3B8')

all_gain=[r['mean_reduction_vs_erm_pct'] for r in exported if r['method']!='erm']
gain_lims=(np.floor(min(0,min(all_gain))/2)*2-2,np.ceil(max(all_gain)/2)*2+2)
all_summary=[r['reduction_vs_erm_pct'] for r in summary if r['method']!='erm']
summary_lims=(np.floor(min(0,min(all_summary)))-1,np.ceil(max(all_summary))+1)

with PdfPages(OUT/'all_figures.pdf') as pdf:
    def save(fig,name):
        fig.savefig(OUT/f'{name}.png',dpi=220)
        fig.savefig(OUT/f'{name}.pdf')
        fig.savefig(OUT/f'{name}.svg')
        pdf.savefig(fig)
        plt.close(fig)

    fig=plt.figure(figsize=(16,9))
    header(fig,'RESULTS AT A GLANCE','WDRO reduces mean regret by 1.43–2.84% versus ERM',
           'All four synthetic distributions and the ISPD test set | 3 training seeds | frozen model evaluation')
    axes=fig.subplots(1,2)
    fig.subplots_adjust(left=.075,right=.96,bottom=.23,top=.73,wspace=.19)
    xx=np.arange(5)
    for ax,metric,title in zip(axes,['mean','p95'],['Mean regret reduction','P95 regret reduction']):
        for idx,method in enumerate(METHODS[1:]):
            yy=[gain(totals[d,'erm'][metric],totals[d,method][metric]) for d in DISTS]
            bars=ax.bar(xx+(idx-.5)*.34,yy,width=.30,color=COLORS[method],label=NAMES[method],zorder=3)
            if method=='wdro_rest':
                for b,y in zip(bars,yy):
                    ax.annotate(f'{y:.2f}%',(b.get_x()+b.get_width()/2,y),xytext=(0,7 if y>=0 else -8),
                                textcoords='offset points',ha='center',va='bottom' if y>=0 else 'top',
                                fontsize=12,weight='bold',color=COLORS[method])
        style(ax); ax.axhline(0,color='#64748B',linewidth=1)
        ax.set_title(title,loc='left',pad=20,weight='bold')
        ax.set_xticks(xx,['Uniform','Gaussian\nmixture','Edge\nbias','Grid\ncluster','ISPD'])
        ax.set_ylim(*summary_lims)
        ax.set_ylabel('Relative reduction vs. ERM (%)  ↑ better')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.23),ncol=2,frameon=False,fontsize=12)
    footer(fig,'Positive = lower regret than ERM; negative = higher regret. Reduction = (ERM − method) / ERM × 100.\n'
           'Synthetic: equal degree/seed weights; P95 computed within each degree/seed, then averaged. ISPD: pooled nets per seed, then seed average.\n'
           'Mixed ERM is stronger on Gaussian mixture and edge bias; these results do not show WDRO superiority over every baseline.')
    save(fig,'00_overview')

    for idx,(dist,label) in enumerate(zip(DISTS,LABELS),1):
        fig=plt.figure(figsize=(16,9))
        improvement=gain(totals[dist,'erm']['mean'],totals[dist,'wdro_rest']['mean'])
        header(fig,f'{idx:02d} / DATASET COMPARISON',f'{label}: {improvement:.2f}% lower mean regret with WDRO',
               'REST / ERM, Mixed ERM and WDRO-REST | 3-seed averages | degree = number of pins')
        axes=fig.subplots(1,2)
        fig.subplots_adjust(left=.08,right=.955,bottom=.24,top=.72,wspace=.22)
        ax=axes[0]
        for method in METHODS:
            yy=[values[dist,method,d]['mean']*100 for d in DEGREES]
            ax.plot(DEGREES,yy,color=COLORS[method],marker=MARKERS[method],markersize=8,
                    linewidth=3 if method=='wdro_rest' else 2,label=NAMES[method],
                    linestyle='--' if method=='mixed_erm' else '-',zorder=4 if method=='wdro_rest' else 3)
        style(ax); ax.set_title('Mean relative regret',loc='left',pad=20,weight='bold')
        ax.set_ylim(0,2.9); ax.set_xticks(DEGREES); ax.set_xlabel('Degree')
        ax.set_ylabel('Mean relative regret (%)  ↓ better')
        ax.legend(loc='upper left',bbox_to_anchor=(-.015,1.23),ncol=3,frameon=False,fontsize=11,
                  handlelength=1.7,columnspacing=1.3)
        ax=axes[1]; xx=np.arange(5)
        for j,method in enumerate(METHODS[1:]):
            yy=[gain(values[dist,'erm',d]['mean'],values[dist,method,d]['mean']) for d in DEGREES]
            bars=ax.bar(xx+(j-.5)*.34,yy,width=.30,color=COLORS[method],zorder=3)
            if method=='wdro_rest':
                for b,y in zip(bars,yy):
                    label_value=f'{y:+.3f}%' if abs(y)<.01 else f'{y:+.2f}%'
                    ax.annotate(label_value,(b.get_x()+b.get_width()/2,y),xytext=(0,7 if y>=0 else -8),
                                textcoords='offset points',ha='center',va='bottom' if y>=0 else 'top',
                                fontsize=11,weight='bold',color=COLORS[method])
        style(ax); ax.axhline(0,color='#64748B',linewidth=1)
        ax.set_ylim(*gain_lims); ax.set_xticks(xx,DEGREES); ax.set_xlabel('Degree')
        ax.set_title('Improvement relative to ERM',loc='left',pad=20,weight='bold')
        ax.set_ylabel('Mean regret reduction (%)  ↑ better')
        baseline=totals[dist,'erm']['mean']*100; ours=totals[dist,'wdro_rest']['mean']*100
        fig.text(.08,.165,f'Aggregate mean regret:  ERM {baseline:.4f}%  →  WDRO {ours:.4f}%'
                 f'     |     decrease: {baseline-ours:.4f} percentage points',fontsize=14,weight='bold')
        notes=('ISPD: 22,498 nets from 20 designs; no ISPD training. Aggregate uses net weights; degree panels average 3 seeds.'
               if dist=='ispd' else '10,000 shared test nets per degree; all 5 degrees and all 3 seeds included. Aggregate uses equal degree/seed weights.')
        footer(fig,notes+'\nRegret = (REST length − reference length) / reference length. Bars: (ERM − method) / ERM × 100; positive = better.\n'
               'Fixed axis ranges across dataset figures; point estimates only. Source: precision_v2 stage 6 / stage 7 frozen result tables.')
        save(fig,f'{idx:02d}_{dist}')

manifest={'source_root':str(TABLES),'output_logical_path':str(OUT),'output_resolved_path':str(OUT.resolve()),
          'sources':{n:hashlib.sha256((TABLES/n).read_bytes()).hexdigest() for n in SOURCES},
          'checks':{'all_75_dataset_method_degree_cells_present':len(exported)==75,
                    'synthetic_per_seed_and_macro_recomputed':True,'ispd_net_weighted_mean_recomputed':True},
          'figure_count':6,'method_order':METHODS,'degree_order':DEGREES,
          'dataset_order':DISTS,'dataset_gain_axis_limits':gain_lims}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
print('Summary WDRO vs ERM:')
for r in summary:
    if r['method']=='wdro_rest': print(r)
