"""Export the verified distributed evaluation through existing figure schemas."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from experiments.guard.paper.figures import export_source_convergence as source, export_final as final, export_behavioral_transfer as behavioral, transfer_assets as assets
from experiments.guard.paper.figures.corrected_traces import load_figure_run_traces, DEFAULT_CORRECTION_DIR
from experiments.guard.paper.figures.data import METHOD_SLUGS, SeedTrace
from experiments.guard.paper.figures.task_sets import StructuredTaskSet
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.execution.artifacts import sha256_file

root=Path(__file__).resolve().parent
assert json.loads((root/'final-checks.json').read_text())['passed']
data=dict(np.load(root/'compact.npz',allow_pickle=False))
paths=tuple(sorted(Path('artifacts/runs').glob('lobeke-final-20260921-*/*')))
traces=load_figure_run_traces(paths)
config=LeaderOptimizationConfig(PolicyOptimizationConfig(**json.loads(str(data['evaluation_policy']))))
for key,slug in METHOD_SLUGS.items():
    assert np.array_equal(np.stack([t.omega_history for t in traces[key]]),data[f'source_{slug}_omegas'])
    traces[key]=[replace(t,metrics={**t.metrics,'stackelberg_suboptimality':data[f'source_{slug}_suboptimality'][0,i], 'fisher_transfer_error':data[f'source_{slug}_fisher_error'][0,i]}) for i,t in enumerate(traces[key])]

def cached(suite,**kwargs):
    slug=kwargs['slug']; ts=kwargs['traces']
    assert np.array_equal(kwargs['transfer_tasks'],data[suite+'_tasks'])
    assert np.array_equal(np.stack([t.omega_history for t in ts]),data[f'{suite}_{slug}_omegas'])
    assert np.array_equal(ts[0].checkpoints,data[f'{suite}_{slug}_checkpoints'])
    return data[f'{suite}_{slug}_suboptimality'],data[f'{suite}_{slug}_fisher_error']

source.load_figure_run_traces=lambda _:traces
source.export(list(paths),root/'exports/source')
final.load_figure_run_traces=lambda _:traces
final.LeaderOptimizationConfig=lambda _:config
final.TRANSFER_EVALUATION_TASKS=StructuredTaskSet('fresh_primary',10,202609700)
final._cached_transfer_metrics=lambda **kwargs:cached('primary',**kwargs)
final.export(paths,root/'exports/diagnostics')
behavioral_traces={key:[SeedTrace(seed=t.seed,checkpoints=data[f'behavioral_{slug}_checkpoints'],metrics={},omega_history=data[f'behavioral_{slug}_omegas'][i]) for i,t in enumerate(traces[key])] for key,slug in METHOD_SLUGS.items()}
behavioral.load_figure_run_traces=lambda _:behavioral_traces
behavioral.LeaderOptimizationConfig=lambda _:config
behavioral._cached_transfer_metrics=lambda **kwargs:cached('behavioral',**kwargs)
behavioral.export(run_dirs=paths,output_dir=root/'exports/behavioral',cache_dir=root/'evaluations',task_set=StructuredTaskSet('fresh_behavioral',180,202609701,candidate_pool=180))

def reward_from_saved(runtime,reward,*,coverage_indicator,gsg,target_nodes,method_estimates,**kwargs):
    matches=[(suite,i) for suite in ('source','primary') for i,r in enumerate(data[suite+'_tasks']) if np.array_equal(r,reward)]
    assert len(matches)==1,matches
    suite,i=matches[0]
    grid=lambda policy:assets._coverage_grid(gsg,target_nodes,coverage_indicator.T@policy)
    grids={}; summaries={}
    for key in assets.METHODS:
        slug=METHOD_SLUGS[key]
        assert np.array_equal(method_estimates[key],data[f'{suite}_{slug}_omegas'][:,-1])
        grids[key]=grid(data[f'{suite}_{slug}_final_policies'][i].mean(axis=0))
        summaries[key]=assets._suboptimality_summary(data[f'{suite}_{slug}_suboptimality'][i,:,-1])
    return grid(data[f'{suite}_active_warm_oracle_policies'][i]),grids,summaries

assets._evaluate_reward=reward_from_saved
assets.LeaderOptimizationConfig=lambda _:config
assets.generate(paths,output_dir=root/'exports/downstream',correction_dir=DEFAULT_CORRECTION_DIR,task_set=final.TRANSFER_EVALUATION_TASKS)
for name in ('source','diagnostics','behavioral','downstream'):
    path=root/'exports'/name/('summary.json' if name=='downstream' else 'provenance.json')
    meta=json.loads(path.read_text())
    meta['generation_command']='PYTHONPATH=. .venv/bin/python output/final-20260921/export.py'
    meta['distributed_evaluation']={'script_sha256':sha256_file(root/'evaluate.py'),'export_sha256':sha256_file(Path(__file__)),'checks_sha256':sha256_file(root/'final-checks.json'),'compact_sha256':sha256_file(root/'compact.npz'),'optimizer':json.loads(str(data['evaluation_policy']))}
    if name=='diagnostics':
        meta['transfer_cache']={'source':'verified distributed task/protocol results','checks_sha256':sha256_file(root/'final-checks.json')}
        meta['transfer_task_policy'].update(tuning_task_seed=9400,disjoint_from_tuning=True)
    if name=='behavioral':
        meta['cache']={'source':'verified distributed task/protocol results'}
        meta['task_policy']['disjoint_seeds']={'hyperparameter_tuning':9400,'ten_task_final_diagnostic':202609700}
    path.write_text(json.dumps(meta,indent=2)+'\n')
print('All four export groups completed',flush=True)
