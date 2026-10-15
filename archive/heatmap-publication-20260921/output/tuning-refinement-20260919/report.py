from pathlib import Path
import json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text())
x=r['exact_ranking'];best=x[0];warm=next(c for c in x if c['warmup']>0)
rows=['| Warm-up | Information multiplier | Mean transfer SubOpt | Seed SE |','|---:|---:|---:|---:|']
for c in x:
 se=np.std(c['per_seed'],ddof=1)/np.sqrt(8)
 rows.append(f"| {c['warmup']} | {c['multiplier']:g} | {c['mean']:.7f} | {se:.7f} |")
training=sum(json.loads((ROOT/f'{tag}-execution.json').read_text())['seconds'] for tag in ['pilot','remaining-six','new-first-two','control'])
evaluations=[json.loads(p.read_text()) for p in ROOT.glob('evaluation-*.json')]
eval_seconds=sum(e['seconds'] for e in evaluations)
assert sum(e['new_evaluations'] for e in evaluations)==9600
text=f'''# Targeted Lobéké tuning refinement

The lowest observed mean among the 12 new ridge-1e-6 cells is warm-up {best['warmup']}, information multiplier {best['multiplier']:g}: {best['mean']:.7f}. The best positive warm-up is {warm['warmup']} at multiplier {warm['multiplier']:g}: {warm['mean']:.7f}. The latter is an 18% lower mean than the rerun 200-observation setting at the same multiplier, but its paired 95% interval for the difference is [-0.0002707, +0.0001819]. These are tuning-set rankings, not independently validated improvements.

## What was run

Eight new grid locations and four reruns of historical anchors, each with eight learning seeds (94000–94007), 1,600 total follower observations and the original 100 downstream tasks (seed 9400, candidate pool 2000). New fits use MLE ridge 1e-6. The information multiplier is lambda / eta squared, a separate policy-selection parameter.

The expanded display contains 23 observed cells: 12 newly evaluated and 11 reused ridge-zero references marked R. Twelve gray locations were not run. Reused cells are excluded from selection under ridge 1e-6. This is an adaptive extension of the earlier tuning, not a fresh independent grid. No main-paper experiment or historical data file was replaced.

New warm-up budgets 25 and 100 were crossed with multipliers .003, .01, .03; additional cells (50,.001) and (100,.1) extend the low end and fill a gap. The four repeated anchors are (0,.03), (50,.003), (200,.003), (200,.01). Candidate choices were fixed before completing the eight-seed evaluations; no candidates were discarded using the two-seed pilot scores.

## Results

{chr(10).join(rows)}

The companion comparison plot uses paired differences across the eight learning seeds on the same fixed tasks, with unadjusted Student-t 95% intervals. Selection favors low observed means, so these intervals are descriptive and must not be read as confirmatory significance tests. Uncertainty over independently sampled task sets is not included. A comparison with different information multipliers does not isolate the effect of warm-up.

## Numerical and provenance checks

All 96 new training seed runs and 9,600 task evaluations passed finite-value, policy-simplex, coefficient-radius, data-link and source-hash checks. The original protected files are byte-identical. Deployment uses 64 optimizer starts and tolerance 1e-12; task optima are computed once and reused. This is numerical optimization, not a certificate of global optimality.

Fixed-observation refits are not sufficient to establish unchanged adaptive trajectories. A two-seed no-warm-up control found 564 of 3,200 follower actions differ between current ridge-zero and ridge-1e-6 runs. The current ridge-zero run also differs from historical ridge-zero training (602 actions), so historical/new differences cannot be attributed solely to ridge. The control used a two-seed batch versus the historical eight-seed batch. Full evaluations compare fresh candidates and anchors using the same two-plus-six seed batching. These trajectory differences do not by themselves establish a difference in expected performance.

## Cost and files

Training, including the ridge-zero control, took {training/60:.1f} minutes; strict evaluation took {eval_seconds/60:.1f} minutes summed across sequential evaluation invocations, overlapping training. Completed evaluations were reused, not repeated.

`results.json` contains all per-seed means, provenance labels and paired comparisons. `training/` preserves trajectories; `evaluation/` preserves every task result. `model.npz` and `oracles.npz` preserve the task suite and numerical optima. `PLAN.md` records the staged execution decision. The unused pilot task matrix is retained under `provenance/`; it was corrected before any evaluation and did not affect training.

From the repository root, regenerate the plots without retraining:

```sh
PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache .venv/bin/python output/tuning-refinement-20260919/summarize.py
```
'''
(ROOT/'REPORT.md').write_text(text)
shutil.copy2('reproducibility/data/lobeke/ridge_zero_tuning/source_config.yaml',ROOT/'provenance/source_config.yaml')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'training_minutes':training/60,'evaluation_minutes':eval_seconds/60,'files':len(manifest)}))
