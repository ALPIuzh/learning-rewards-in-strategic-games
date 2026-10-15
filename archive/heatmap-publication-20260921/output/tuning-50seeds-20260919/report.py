from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());v=json.loads((ROOT/'validation.json').read_text())
with np.load(ROOT/'figure-data.npz') as d:values=d['transfer_suboptimality'].mean(-1)
comparisons=[]
for j in (0,2):
 x=values[0,j]-values[0,1];se=x.std(ddof=1)/np.sqrt(50);half=t.ppf(.975,49)*se
 comparisons.append(dict(multiplier=[.001,.003,.01][j],reference=.003,mean_difference=float(x.mean()),ci95=[float(x.mean()-half),float(x.mean()+half)],added18_difference=float(x[32:].mean())))
(ROOT/'no-warm-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
spot=json.loads((ROOT/'numerical-spotcheck.json').read_text())
rows=['| Warm-up | Multiplier | 32-seed mean | Added 18 mean | 50-seed mean | SE at 50 |','|---:|---:|---:|---:|---:|---:|']
for i,c in enumerate(r['cells']):
 x=values.reshape(15,50)[i]
 rows.append(f"| {c['warmup']} | {c['multiplier']:g} | {x[:32].mean()*1e4:.2f} | {x[32:].mean()*1e4:.2f} | {x.mean()*1e4:.2f} | {c['se']*1e4:.2f} |")
e=json.loads((ROOT/'full-execution.json').read_text());best=r['ranking'][0];diff=r['matched_warmup_difference']
text=f'''# 50-seed refinement of the left three heatmap columns

All 15 selected cells now have 50 seeds: the previous 32 were reused exactly and 18 fresh seeds were added. The remaining 20 cells of the full grid were not rerun and remain at 32 seeds. This figure deliberately shows only the uniform 50-seed region.

The smallest observed mean is warm-up {best['warmup']}, information multiplier {best['multiplier']:g}: {best['mean']:.7f}. This is a ranking of sample means, not a claim of a unique optimum.

## Unchanged design

Warm-up budgets: 0,25,50,100,200. Information multipliers lambda / eta squared: .001,.003,.01. MLE ridge 1e-6, 1600 observations, and the same 100 transfer tasks (seed 9400, candidate pool 2000). Existing strict task optima were reused exactly. New training seeds are 2026094032–2026094049, in batches of eight, eight and two for every cell. The merged dataset has seeds 2026094000–2026094049. Evaluation uses 64 starts and tolerance 1e-12.

The 15-cell region was chosen after inspecting the earlier results; the extension to 50 seeds was fixed before examining the added runs. No seed or cell was removed based on its performance. The original 32-seed dataset and figures, the manuscript, and paper defaults were not modified.

## Comparison

All values below are in units of 10^-4. SE is over 50 learning seeds after averaging 100 fixed tasks. The task suite is held fixed; these errors do not represent uncertainty over new task suites. The added 18 column is descriptive, not a new tuning selection rule.

{chr(10).join(rows)}

The best positive warm-up compared with no warm-up at its matched multiplier has mean difference {diff['mean']:.8f} and unadjusted paired 95% interval [{diff['ci95'][0]:.8f}, {diff['ci95'][1]:.8f}]. The comparison is selected on these data, so it is not independent confirmation. The first versus second 25-seed rank correlation is {r['split_half_rank_correlation']:.3f}; its interpretation is descriptive, since the region was selected using earlier scores.

## Validation and runtime

Added {v['seed_runs']} training runs and {v['task_evaluations']} task evaluations. All passed finite-value, policy-simplex, coefficient-radius, MLE-residual and provenance checks. The {v['protected_files_unchanged']} protected prior files are unchanged. The merged first 32 seeds are bit-identical to the prior compact data. Execution took {e['seconds']/60:.1f} minutes with eight single-threaded workers. Rechecking the highest-loss new case in each cell with 256 optimizer starts changed losses by at most {spot['max_abs_loss_change']:.3g}. The new heatmaps and statistics were regenerated in an isolated folder using only the compact data and plotting script, with byte-identical PNG and JSON outputs.

## Regeneration

Copy `figure-data.npz` and `summarize.py` together; NumPy, SciPy and Matplotlib suffice to regenerate the figures without experiment code or training:

```sh
python summarize.py
```

From the repository root, raw-result verification and collection are available through `validate.py` and `collect.py` in this folder. Raw new training trajectories and evaluations are preserved under `training/` and `evaluation/`. `results.json` includes per-seed scores and descriptive intervals. `heatmap.pdf` and `uncertainty.pdf` are vector figures, with PNG and SVG versions alongside them.
'''
(ROOT/'REPORT.md').write_text(text)
(ROOT/'caption.tex').write_text('Mean transfer suboptimality on Lobéké after 1,600 observations, as a function of the warm-up budget and the information multiplier $\\lambda/\\eta^2$. Each cell averages 50 learning seeds over the same 100 transfer tasks; lower values are better. All runs use MLE ridge $10^{-6}$.\n')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'best':best,'minutes':e['seconds']/60,'files':len(manifest)}))
