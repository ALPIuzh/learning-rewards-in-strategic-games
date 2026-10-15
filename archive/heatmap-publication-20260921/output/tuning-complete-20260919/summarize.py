from pathlib import Path
import json,re,hashlib
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'tuning-refinement-20260919'

def interval(x):
 x=np.asarray(x);mean=float(x.mean());half=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
 return {'mean':mean,'lower':mean-half,'upper':mean+half,'n':len(x)}

def pair(name):
 m=re.fullmatch(r'active_(no_warm|warm\d+)_lambda(.+)',name)
 return (0 if m[1]=='no_warm' else int(m[1][4:]),float(m[2].replace('p','.')))

def main():
 historical=np.load('reproducibility/data/lobeke/ridge_zero_tuning/data.npz')
 seeds=historical['seeds'];assert np.array_equal(seeds,np.arange(94000,94008))
 cells={}
 for i,(w,l) in enumerate(zip(historical['candidate_warmup_budgets'],historical['candidate_lambda_information_multipliers'])):
  values=historical['per_seed_downstream_suboptimality'][i]
  cells[int(w),float(l)]={'source':'reused ridge-zero replay','ridge':0.,'per_seed':values.tolist(),**interval(values)}
 exact={}
 for d in sorted((PREVIOUS/'evaluation').iterdir())+sorted((ROOT/'evaluation').iterdir()):
  if not d.is_dir():continue
  files=[d/f'{seed}.npz' for seed in seeds]
  if not all(p.exists() for p in files):continue
  values=np.array([np.load(p)['subopt'].mean() for p in files]);key=pair(d.name)
  assert key not in exact,key
  exact[key]={'source':'new ridge-1e-6 training and strict evaluation','ridge':1e-6,'per_seed':values.tolist(),**interval(values)}
 assert len(exact)==24, len(exact)
 cells.update(exact)
 assert len(cells)==35,len(cells)
 best=min(exact,key=lambda p:(exact[p]['mean'],p[0],p[1]))
 warm_best=min((p for p in exact if p[0]>0),key=lambda p:(exact[p]['mean'],p[0],p[1]))
 refs=list(dict.fromkeys([(0,.03),(200,.003),(200,.01),(50,.003),warm_best]))
 if (0,warm_best[1]) in exact and (0,warm_best[1]) not in refs:refs.append((0,warm_best[1]))
 comparisons=[]
 for ref in refs:
  if ref==best:continue
  stat=interval(np.array(exact[best]['per_seed'])-np.array(exact[ref]['per_seed']))
  comparisons.append({'candidate':best,'reference':ref,**stat})
 rank=sorted(exact,key=lambda p:exact[p]['mean'])
 result={'status':'exploratory selection; no independent confirmation','seeds':seeds.tolist(),'tasks':{'seed':9400,'count':100,'candidate_pool':2000},'samples':1600,'provisional_best':best,'provisional_best_positive_warmup':warm_best,'exact_ranking':[{'warmup':p[0],'multiplier':p[1],**exact[p]} for p in rank],'paired_selection_comparisons':comparisons,'cells':[{'warmup':w,'multiplier':l,**s} for (w,l),s in sorted(cells.items())]}
 result['best_warm_vs_previous_warm']=interval(np.array(exact[warm_best]['per_seed'])-np.array(exact[200,.003]['per_seed']))
 if (0,warm_best[1]) in exact:
  result['best_warm_vs_matched_no_warm']=interval(np.array(exact[warm_best]['per_seed'])-np.array(exact[0,warm_best[1]]['per_seed']))
 (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.hashsalt':'completed-tuning-grid'})
 ws=[0,25,50,100,200];ls=[.001,.003,.01,.03,.1,.3,1]
 grid=np.full((len(ws),len(ls)),np.nan)
 for (w,l),s in cells.items():grid[ws.index(w),ls.index(l)]=s['mean']*1e4
 fig,ax=plt.subplots(figsize=(7.8,4.1),layout='constrained')
 cmap=plt.get_cmap('Blues_r').copy();cmap.set_bad('#EEEEEE')
 im=ax.imshow(grid,origin='lower',aspect='auto',cmap=cmap,vmin=0,vmax=max(4,float(np.nanmax(grid))))
 for yi,w in enumerate(ws):
  for xi,l in enumerate(ls):
   if (w,l) not in cells:ax.text(xi,yi,'—',ha='center',va='center',color='.6');continue
   v=grid[yi,xi];is_new=(w,l) in exact
   luminance=np.dot(cmap(im.norm(v))[:3],[.2126,.7152,.0722])
   ax.text(xi,yi,f'{v:.2f}'+('' if is_new else ' R'),ha='center',va='center',color='white' if luminance<.45 else '#111111',fontsize=9)
 for p in [(0,.03),(200,.003)]:ax.add_patch(Rectangle((ls.index(p[1])-.46,ws.index(p[0])-.46),.92,.92,fill=False,edgecolor='#D55E00',lw=1.7))
 ax.add_patch(Rectangle((ls.index(best[1])-.40,ws.index(best[0])-.40),.80,.80,fill=False,edgecolor='black',lw=2.4))
 ax.set(xticks=range(len(ls)),xticklabels=[f'{x:g}' for x in ls],yticks=range(len(ws)),yticklabels=ws,xlabel='Information multiplier λ / η²',ylabel='Warm-up observations',title='Expanded tuning: 1,600 observations, 8 seeds, 100 tasks')
 fig.colorbar(im,ax=ax,label='Mean transfer SubOpt × 10⁴')
 fig.supxlabel('R: reused ridge-zero result; other cells: new MLE ridge 10⁻⁶.\nOrange: previously retained settings. Black: lowest new-grid mean (selection only).',fontsize=8)
 for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'tuning-grid.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.4,1.2+.55*len(comparisons)),layout='constrained')
 for i,c in enumerate(comparisons):
  ax.errorbar(c['mean']*1e4,i,xerr=np.array([[c['mean']-c['lower']],[c['upper']-c['mean']]])*1e4,fmt='o',color='#1769AA',capsize=3)
 ax.axvline(0,color='.5',ls='--',lw=.8)
 ax.set(yticks=range(len(comparisons)),yticklabels=[f"vs warm-up {c['reference'][0]}, multiplier {c['reference'][1]:g}" for c in comparisons],xlabel='Paired mean difference × 10⁴ (negative favors selected cell)',title=f'Selection uncertainty: warm-up {best[0]}, multiplier {best[1]:g}')
 ax.invert_yaxis();fig.supxlabel('Unadjusted 95% paired seed intervals on the tuning data; not an independent test.',fontsize=8)
 for ext in ['png','pdf']:fig.savefig(ROOT/f'tuning-comparisons.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 print(json.dumps({'best':best,'best_warm':warm_best,'means':[(p,exact[p]['mean']) for p in rank]},indent=2))
if __name__=='__main__':main()
