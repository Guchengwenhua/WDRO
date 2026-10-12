from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 def load(name):
  with (OUT/name).open() as f:return list(csv.DictReader(f))
 methods=CFG['methods'];labels=['Real ERM','RAW WDRO','VAE shell','Feature gate','Graph smooth','Set denoiser','Graph denoiser','Tile random','Graph + tile'];natural=load('aggregate.csv');attacks=load('attack_per_design.csv');geometry=load('candidate_geometry.csv');color=['#7e8794','#334e68','#9163a3','#b38340','#40948b','#367ec3','#d06250','#9b9860','#6467ae'];yy=np.arange(len(methods))
 fig,axes=plt.subplots(1,4,figsize=(16,5.2),gridspec_kw={'width_ratios':[1,1,1,1]})
 for ax,split,title in [(axes[0],'B_test','Held-out ISPD designs'),(axes[1],'C_external','External designs')]:
  rr={r['method']:r for r in natural if r['budget']=='2048' and r['role']=='val_best' and r['split']==split};means=[float(rr[m]['mean_pct']) for m in methods];sd=[float(rr[m]['seed_sd_mean_pp']) for m in methods]
  for i,(v,e) in enumerate(zip(means,sd)):ax.errorbar(v,i,xerr=e,fmt='o',color=color[i],capsize=3,ms=6)
  ax.axvline(float(rr['raw_wdro']['mean_pct']),color='#334e68',linestyle='--',alpha=.4);ax.set(title=title,xlabel='Mean true regret (%)')
 for i,m in enumerate(methods):
  r=[float(r['mean'])*100 for r in attacks if r['method']==m and r['generator']=='301' and r['design']=='ALL'];axes[2].errorbar(np.mean(r),i,xerr=np.std(r,ddof=1),fmt='o',color=color[i],capsize=3,ms=6)
 axes[2].set(title='Independent RAW-K32 stress',xlabel='Mean worst regret (%)')
 gg={r['pool']:r for r in geometry}
 for i,m in enumerate(methods):
  key=m+'_g301' if m in ['set_dae','graph_dae','graph_tile'] else m;cost=0 if m=='erm_a' else float(gg[key]['mean_non_nominal_cost']);axes[3].scatter(cost,i,c=color[i],s=40)
 axes[3].axvline(read(ART/'rho.json')['rho'],color='gray',ls=':',label='training rho');axes[3].set(title='Candidate displacement',xlabel='Mean Hungarian L1 cost');axes[3].legend(fontsize=8)
 for i,ax in enumerate(axes):
  ax.set_yticks(yy,labels if i==0 else ['']*len(yy));ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Acceptance audit: 2048-update budget, validation-selected checkpoints',fontsize=13)
 fig.text(.5,.015,'Generator seed 301; means across 3 REST seeds. Error bars are seed standard deviations, not confidence intervals. Lower regret is better.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/'acceptance_comparison.png',dpi=180);fig.savefig(OUT/'acceptance_comparison.pdf');plt.close(fig)
if __name__=='__main__':main()
