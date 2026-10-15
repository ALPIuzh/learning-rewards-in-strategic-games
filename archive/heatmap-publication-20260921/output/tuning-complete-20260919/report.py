from pathlib import Path
import hashlib,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());rank=r['exact_ranking'];best=rank[0];warm=next(c for c in rank if c['warmup'])
matched=r.get('best_warm_vs_matched_no_warm')
matched_text='' if matched is None else f"At the same multiplier, the best positive warm-up differs from no warm-up by {matched['mean']:.8f}, with an unadjusted paired 95% interval [{matched['lower']:.8f}, {matched['upper']:.8f}]. This does not establish a warm-up benefit."
rows=['| Warm-up | Multiplier | Mean SubOpt | Seed SE |','|---:|---:|---:|---:|']
for c in rank:
 rows.append(f"| {c['warmup']} | {c['multiplier']:g} | {c['mean']:.7f} | {np.std(c['per_seed'],ddof=1)/np.sqrt(8):.7f} |")
training=[json.loads((ROOT/f'{tag}-execution.json').read_text())['seconds'] for tag in ('first-two','remaining-six')]
evaluations=[json.loads(p.read_text()) for p in ROOT.glob('evaluation-wave-*.json')]
assert sum(e['new_evaluations'] for e in evaluations)==9600
validation=json.loads((ROOT/'validation.json').read_text())
text=f'''# Completed Lobéké tuning grid

All 12 formerly blank cells have been computed. The plot now contains all 35 combinations of warm-up budgets 0, 25, 50, 100, 200 and information multipliers .001, .003, .01, .03, .1, .3, 1.

The lowest mean among the 24 ridge-1e-6 cells is warm-up {best['warmup']}, multiplier {best['multiplier']:g}, mean transfer SubOpt {best['mean']:.7f}. The best positive warm-up is {warm['warmup']}, multiplier {warm['multiplier']:g}, mean {warm['mean']:.7f}. These are descriptive rankings on the tuning set, not independent confirmation or proof of a warm-up benefit.

{matched_text}

## Data and selection

Only the 12 missing configurations were trained: 96 learning seed runs and 9,600 task evaluations. Each uses 1,600 observations and the same eight seeds (94000–94007) in the same two-plus-six batches as the previous refinement. The 100 downstream tasks (seed 9400, candidate pool 2000), game and strict task optima are copied byte-for-byte from that refinement. Deployment uses 64 starts and tolerance 1e-12.

The 12 previously recomputed ridge-1e-6 cells are reused without modification. Eleven historical ridge-zero cells remain marked R and are excluded from selection under ridge 1e-6. Thus the display is complete, but it is not a homogeneous 35-cell ridge-1e-6 experiment. The earlier refinement and all original paper data are preserved. The information multiplier lambda / eta squared affects policy selection; it is distinct from MLE ridge.

No cells were discarded based on pilot scores, and no main-paper result or selected configuration was overwritten. Extending the grid after viewing earlier results makes this exploratory tuning on reused seeds and tasks.

## Ranking of ridge-1e-6 cells

{chr(10).join(rows)}

The comparison plot gives paired Student-t 95% intervals over eight learning seeds. They are unadjusted and evaluated on the data used for selection. They do not account for uncertainty over fresh task sets or for choosing a winner among many candidates. Different information multipliers also prevent isolating a causal warm-up effect from a protocol comparison.

## Verification and cost

All 96 additional runs passed finite-value, simplex, coefficient-radius and evaluation-provenance checks. Hash verification confirms {validation['original_files_unchanged']} protected files are unchanged. Saved fitted rewards match the training endpoints exactly. Each output stores its training source hash and task-model hash.

The two training processes took {training[0]/60:.1f} and {training[1]/60:.1f} minutes, largely concurrently. Evaluation took {sum(e['seconds'] for e in evaluations)/60:.1f} minutes summed across waves, overlapping training. Completed results and existing task optima were reused.

## Files and regeneration

`results.json` contains the 35 cell values, per-seed scores and paired comparisons. `training/` and `evaluation/` contain the 12 additional configurations. The previous 12 ridge-1e-6 configurations remain in `../tuning-refinement-20260919/`. `PLAN.md` records the fixed candidate list, runtime estimate and pilot check.

From the repository root:

```sh
PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache .venv/bin/python output/tuning-complete-20260919/summarize.py
```
'''
(ROOT/'REPORT.md').write_text(text)
shutil.copy2('reproducibility/data/lobeke/ridge_zero_tuning/source_config.yaml',ROOT/'source_config.yaml')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'best':[best['warmup'],best['multiplier']],'best_warm':[warm['warmup'],warm['multiplier']],'files':len(manifest),'training_minutes':[v/60 for v in training]}))
