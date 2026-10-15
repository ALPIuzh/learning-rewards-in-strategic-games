from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());v=json.loads((ROOT/'validation.json').read_text())
spot=json.loads((ROOT/'numerical-spotcheck.json').read_text())
rank=r['ranking'];best=rank[0];matched=r['matched_warmup_difference']
rows=['| Warm-up | Multiplier | Mean × 10⁴ | SE × 10⁴ | 95% interval × 10⁴ |','|---:|---:|---:|---:|---:|']
for c in sorted(r['cells'],key=lambda c:c['mean']):
 rows.append(f"| {c['warmup']} | {c['multiplier']:g} | {c['mean']*1e4:.3f} | {c['se']*1e4:.3f} | [{c['ci95'][0]*1e4:.3f}, {c['ci95'][1]*1e4:.3f}] |")
jobs=[json.loads(p.read_text()) for p in (ROOT/'jobs').glob('*/*.json')]
execution=json.loads((ROOT/'full-execution.json').read_text())
text=f'''# Uniform 32-seed Lobéké tuning heatmap

All 35 cells use the same estimator, numerical settings, 32 fresh learning seeds and 100 evaluation tasks. The lowest observed mean is warm-up {best['warmup']}, information multiplier {best['multiplier']:g}: {best['mean']:.7f}. This is a tuning-set ranking, not confirmation that this setting is uniquely optimal.

## Design

Warm-up observations: 0, 25, 50, 100, 200. Information multipliers lambda / eta squared: .001, .003, .01, .03, .1, .3, 1. MLE ridge is 1e-6 in every cell and is distinct from the policy-selection information multiplier. Each run uses 1,600 total observations and known inverse temperature from the unchanged source configuration.

Learning seeds are 2026094000–2026094031, in four batches of eight, with the same batching across every cell. None of these training seeds entered the previous eight-seed grid. The fixed 100 normalized transfer tasks (seed 9400, candidate pool 2000) and numerical reference optima are reused exactly. Every fitted reward is evaluated on every task with 64 optimizer starts and tolerance 1e-12. This gives 1,120 training runs and 112,000 task evaluations.

The grid and 32-seed budget were frozen before reading fresh scores. The two pilot batches contribute to the final data. No cell or seed was discarded based on performance. Historical ridge-zero cells are not mixed into these figures. Earlier data and figures remain unchanged. The manuscript and paper defaults were not modified.

## Precision and interpretation

The main heatmap shows actual cell means, without smoothing, interpolation or score clipping. The companion heatmap shows standard errors over the 32 independent learning seeds. The 100 transfer tasks are held fixed; these errors do not represent uncertainty over newly sampled task suites. Table intervals use Student-t with 31 degrees of freedom and are unadjusted for selecting among 35 configurations.

The first-versus-second 16-seed rank correlation is {r['split_half_rank_correlation']:.3f}. The half-sample winners are {r['split_half_winners']}. This low correlation indicates that the fine ranking remains unstable. This is a descriptive stability diagnostic, not an additional selection rule.

The best positive warm-up is {r['best_positive_warmup']}. Compared with no warm-up at the same information multiplier, its mean difference is {matched['mean']:.8f}, with an unadjusted paired 95% interval [{matched['ci95'][0]:.8f}, {matched['ci95'][1]:.8f}]. This comparison was selected using these data and must not be presented as independent confirmation.

More seeds reduce sampling uncertainty; they do not imply that small differences in heatmap shading are meaningful. A reliable claim that the selected setting improves over alternatives needs fresh confirmation data. A clean figure should not hide that limitation.

## Cell statistics

{chr(10).join(rows)}

## Validation and cost

All {v['seed_runs']} training endpoints and {v['task_evaluations']} evaluations passed finite-value, policy-simplex, coefficient-radius, MLE-residual and source-link checks. Maximum MLE residual across saved diagnostics is {v['max_mle_residual']:.3g}; maximum simplex error is {v['max_simplex_error']:.3g}. The {v['protected_files_unchanged']} protected prior files are byte-identical. Task data and reference optima match the earlier files byte-for-byte. Rechecking the highest-loss case in every cell with 256 starts changed losses by at most {spot['max_abs_loss_change']:.3g}, fitted objectives by {spot['max_abs_fitted_objective_change']:.3g}, and reference optima by {spot['max_abs_oracle_change']:.3g}.

The full scheduler took {execution['seconds']/60:.1f} minutes using {execution['workers']} single-threaded worker processes, reusing the completed pilots. Per-batch timings, source hashes, software commit and evaluation hashes are recorded under `jobs/`. Raw training trajectories and per-task evaluation scores are retained under `training/` and `evaluation/`.

## Reproduce the figures

From the repository root, without retraining:

```sh
PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache .venv/bin/python output/tuning-32seeds-20260919/validate.py
PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache .venv/bin/python output/tuning-32seeds-20260919/summarize.py
```

`figure-data.npz` contains every task score in a compact file. Copy it with `summarize.py` to regenerate both figures using only NumPy, SciPy and Matplotlib. This was verified in an isolated folder with a fresh Matplotlib cache: both PNG files and the statistics JSON were byte-identical. `results.json` contains every per-seed score. `heatmap.pdf` is the vector main figure; `uncertainty.pdf` is its companion. PNG and SVG versions are also included. `caption.tex` provides a short factual caption without a claim of optimality.
'''
(ROOT/'REPORT.md').write_text(text)
(ROOT/'caption.tex').write_text('Mean transfer suboptimality on Lobéké after 1,600 observations, as a function of the warm-up budget and the information multiplier $\\lambda/\\eta^2$. Each cell averages 32 learning seeds over the same 100 transfer tasks; lower values are better. All runs use MLE ridge $10^{-6}$.\n')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'best':best,'elapsed_minutes':execution['seconds']/60,'files':len(manifest)}))
