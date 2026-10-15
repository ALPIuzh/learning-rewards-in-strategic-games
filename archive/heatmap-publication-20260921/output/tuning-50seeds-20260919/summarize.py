from pathlib import Path
import json
import numpy as np
from scipy.stats import t,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
WS=[0,25,50,100,200];LS=[.001,.003,.01]
SEEDS=np.arange(2026094000,2026094050)

def summarize():
 with np.load(ROOT/'figure-data.npz') as d:
  assert np.array_equal(d['seeds'],SEEDS) and np.array_equal(d['warmup_budgets'],WS)
  assert np.array_equal(d['multipliers'],LS) and float(d['mle_ridge'])==1e-6
  values=d['transfer_suboptimality'].mean(axis=-1)
 assert values.shape==(5,3,50) and np.isfinite(values).all()
 cells=[]
 for y,w in enumerate(WS):
  for z,l in enumerate(LS):
   x=values[y,z];mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));half=t.ppf(.975,49)*se
   cells.append(dict(warmup=w,multiplier=l,mean=mean,se=se,ci95=[mean-half,mean+half],per_seed=x.tolist(),first_half_mean=float(x[:25].mean()),second_half_mean=float(x[25:].mean()),median=float(np.median(x)),max_seed=float(x.max())))
 ranking=sorted(cells,key=lambda c:(c['mean'],c['warmup'],c['multiplier']))
 best=ranking[0];warm=next(c for c in ranking if c['warmup'])
 ref=next(c for c in cells if c['warmup']==0 and c['multiplier']==warm['multiplier'])
 diff=np.array(warm['per_seed'])-np.array(ref['per_seed']);se=float(diff.std(ddof=1)/np.sqrt(50));h=float(t.ppf(.975,49)*se)
 halves=[[c[key] for c in cells] for key in ('first_half_mean','second_half_mean')]
 summary=dict(seeds=SEEDS.tolist(),ridge=1e-6,samples=1600,tasks=dict(seed=9400,count=100,candidate_pool=2000),cells=cells,ranking=[{'warmup':c['warmup'],'multiplier':c['multiplier'],'mean':c['mean']} for c in ranking],best_positive_warmup=dict(warmup=warm['warmup'],multiplier=warm['multiplier']),matched_warmup_difference=dict(mean=float(diff.mean()),ci95=[float(diff.mean())-h,float(diff.mean())+h]),split_half_rank_correlation=float(spearmanr(*halves).statistic),split_half_winners=[{'warmup':cells[int(np.argmin(v))]['warmup'],'multiplier':cells[int(np.argmin(v))]['multiplier']} for v in halves])
 (ROOT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
 return summary

def plot(summary,key,name,label,title):
 values=np.array([c[key]*1e4 for c in summary['cells']]).reshape(5,3)
 vmax=float(np.ceil(values.max()*2)/2)
 fig,ax=plt.subplots(figsize=(4.8,4.1))
 fig.subplots_adjust(left=.15,right=.85,bottom=.26,top=.87)
 cmap=plt.get_cmap('Blues');im=ax.imshow(values,origin='lower',aspect='auto',cmap=cmap,vmin=0,vmax=vmax)
 for y in range(5):
  for x in range(3):
   v=values[y,x];rgb=cmap(im.norm(v))[:3];color='white' if np.dot(rgb,[.2126,.7152,.0722])<.45 else '#14202B'
   ax.text(x,y,f'{v:.2f}',ha='center',va='center',color=color,fontsize=10)
 ax.set(xticks=np.arange(3),xticklabels=[f'{l:g}' for l in LS],yticks=np.arange(5),yticklabels=WS,xlabel=r'Information multiplier $\lambda/\eta^2$',ylabel='Warm-up observations',title=title)
 ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
 ax.grid(which='minor',color='white',linewidth=.65);ax.tick_params(which='both',length=0,pad=6)
 for s in ax.spines.values():s.set_visible(False)
 cb=fig.colorbar(im,ax=ax,fraction=.043,pad=.035);cb.set_label(label);cb.outline.set_visible(False)
 fig.text(.5,.015,'50 seeds · 1,600 observations · 100 transfer tasks\nMLE ridge $10^{-6}$',ha='center',va='bottom',fontsize=8.5)
 for ext in ('png','pdf','svg'):fig.savefig(ROOT/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)

if __name__=='__main__':
 plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'pdf.fonttype':42,'svg.hashsalt':'targeted-50-seed-tuning'})
 r=summarize()
 plot(r,'mean','heatmap','Mean transfer SubOpt ($10^{-4}$)','Lobéké: tuning refinement')
 plot(r,'se','uncertainty','Standard error ($10^{-4}$)','Uncertainty across learning seeds')
 print(json.dumps({k:r[k] for k in ('ranking','matched_warmup_difference','split_half_rank_correlation','split_half_winners')},indent=2))
