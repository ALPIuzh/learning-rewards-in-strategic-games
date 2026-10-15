"""Frozen final-evaluation tasks; independent Slurm workers, one output per task/protocol."""
import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.corrected_traces import load_figure_run_traces
from experiments.guard.paper.figures.data import METHOD_SLUGS, SeedTrace, load_runtime
from experiments.guard.paper.figures.export_behavioral_transfer import _at_checkpoints, _source_records
from experiments.guard.paper.figures.export_final import _subsampled_traces
from experiments.guard.paper.figures.task_sets import StructuredTaskSet, normalized_structured_tasks
from mairl.core.math import cumulative_fisher_information_batch
from mairl.core.problem import compile_problem
from mairl.inference.mle import solve_mle_batch
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz

ROOT = Path(__file__).resolve().parent

def prepare():
    paths = tuple(sorted(Path('artifacts/runs').glob('lobeke-final-20260921-*/*')))
    assert len(paths) == 9
    records, composition = _source_records(paths)
    traces = load_figure_run_traces(paths)
    assert set(traces) == set(METHOD_SLUGS)
    runtime = load_runtime(paths[0])
    cfg = instantiate(OmegaConf.load(paths[0] / 'resolved_config.yaml').numerics)
    data = dict(features=runtime.problem.features, eta=runtime.problem.eta, radius=runtime.problem.radius,
                true_omega=runtime.true_omega, seeds=np.arange(2026097000,2026097050),
                evaluation_policy=json.dumps(asdict(replace(cfg.evaluation_policy, random_starts=64, tolerance=1e-12))))
    suites = {'primary': StructuredTaskSet('fresh_primary',10,202609700),
              'behavioral': StructuredTaskSet('fresh_behavioral',180,202609701,candidate_pool=180)}
    for suite, spec in suites.items():
        data[suite + '_tasks'] = normalized_structured_tasks(runtime,spec)
    data['source_tasks'] = np.asarray(runtime.metadata['defender_reward'])[None]
    requests = []
    residuals = []
    raw_histories = {}
    refitted = []
    for path in paths:
        for result in path.glob('jobs/*/result.npz'):
            job=json.loads(result.with_name('summary.json').read_text())['job']
            with np.load(result, allow_pickle=False) as saved:
                raw_histories[(job['method'],job['warmup_enabled'],job['seed'])] = {k:saved[k] for k in ('trace_sample_counts','omega_history','policies','follower_actions') if k in saved}
                for key in saved.files:
                    value = saved[key]
                    if np.issubdtype(value.dtype,np.number): assert np.isfinite(value).all(),(str(result),key)
                if 'policies' in saved:
                    policy=saved['policies'];assert policy.min() >= -1e-10
                    assert np.max(np.abs(policy.sum(-1)-1)) < 1e-9
                if 'diagnostic__mle_residual' in saved and saved['diagnostic__mle_residual'].size:
                    residuals.append(float(saved['diagnostic__mle_residual'].max()))
    for key, slug in METHOD_SLUGS.items():
        original = traces[key]
        assert [t.seed for t in original] == data['seeds'].tolist()
        requested=np.array([200,400,800,1200,1600])
        raw=[raw_histories[(*key,t.seed)] for t in original]
        histories=[]
        for checkpoint in requested:
            if all(checkpoint in r['trace_sample_counts'] for r in raw):
                histories.append(np.stack([r['omega_history'][np.flatnonzero(r['trace_sample_counts']==checkpoint)[0]] for r in raw]))
            else:
                assert key[0] in ('fixed_random','random_each_round'), key
                initial=np.stack([r['omega_history'][np.searchsorted(r['trace_sample_counts'],checkpoint,side='right')-1] for r in raw])
                fits=solve_mle_batch(runtime.problem,np.stack([r['policies'][:checkpoint] for r in raw]),
                                    np.stack([r['follower_actions'][:checkpoint] for r in raw]),cfg.mle,initial=initial)
                histories.append(np.stack([fit.omega for fit in fits]))
                refitted.append(dict(method=slug,checkpoint=int(checkpoint),max_mle_residual=max(float(f.projected_gradient_residual) for f in fits)))
        dense=np.stack(histories,axis=1)
        behavioral=[SeedTrace(seed=t.seed,checkpoints=requested,metrics={},omega_history=dense[i]) for i,t in enumerate(original)]
        for suite in ('source','primary','behavioral'):
            selected = original if suite=='source' else _subsampled_traces(original) if suite=='primary' else behavioral
            data[f'{suite}_{slug}_omegas'] = np.stack([t.omega_history for t in selected])
            data[f'{suite}_{slug}_checkpoints'] = selected[0].checkpoints
            for task in range(len(data[suite+'_tasks'])):
                requests.append(dict(suite=suite,slug=slug,task=task,task_seed=suites[suite].seed if suite!='source' else 202609699))
    assert not (ROOT/'inputs.npz').exists()
    atomic_write_npz(ROOT/'inputs.npz',data)
    atomic_write_json(ROOT/'requests.json',requests)
    atomic_write_json(ROOT/'validation.json',dict(runs=records,composition=composition,validated_seed_runs=450,
                      max_mle_residual=max(residuals),checkpoint_refits=refitted,input_sha256=hashlib.sha256((ROOT/'inputs.npz').read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),task_sets={k:asdict(v) for k,v in suites.items()}))
    print(f'Validated 450 seed runs; prepared {len(requests)} task/protocol evaluations',flush=True)

def worker(index, workers, limit):
    with np.load(ROOT/'inputs.npz',allow_pickle=False) as saved: data=dict(saved)
    requests=json.loads((ROOT/'requests.json').read_text())
    problem=compile_problem(data['features'],eta=float(data['eta']),radius=float(data['radius']))
    config=LeaderOptimizationConfig(PolicyOptimizationConfig(**json.loads(str(data['evaluation_policy']))))
    assigned=list(range(index,len(requests),workers)); started=time.monotonic()
    if limit: assigned=assigned[:limit]
    for done, ri in enumerate(assigned,1):
        r=requests[ri];suite,slug,task=r['suite'],r['slug'],r['task'];out=ROOT/'evaluations'/f'{ri:04d}.npz'
        if out.exists():continue
        reward=data[suite+'_tasks'][task];omegas=data[f'{suite}_{slug}_omegas'];shape=omegas.shape[:2]
        oracle=optimize_leader_policy(problem,data['true_omega'],reward,config,seed=r['task_seed']+task)
        estimates=omegas.reshape(-1,omegas.shape[-1]);policies=[];values=[];residual=[]
        for flat,omega in enumerate(estimates):
            si,ci=divmod(flat,shape[1]);seed=r['task_seed']*1000000+task*100000+int(data['seeds'][si])*100+ci
            fitted=optimize_leader_policy(problem,omega,reward,config,seed=seed)
            policies.append(fitted.policy);values.append(fitted.value)
        policies=np.asarray(policies)
        deployed=np.array([leader_value_and_gradient(problem,data['true_omega'],reward,p)[0] for p in policies])
        # A discovered policy also supplies a valid start for the true-reward oracle.
        if np.max(deployed)>oracle.value+1e-9:
            raise ValueError(f'True oracle missed a better policy in request {ri}')
        errors=data['true_omega']-estimates
        true_policies=np.broadcast_to(oracle.policy,policies.shape)
        a=cumulative_fisher_information_batch(problem,true_policies[:,None,:],estimates)
        b=cumulative_fisher_information_batch(problem,policies[:,None,:],estimates)
        qa=np.einsum('nd,nde,ne->n',errors,a,errors,optimize=True)
        qb=np.einsum('nd,nde,ne->n',errors,b,errors,optimize=True)
        assert min(qa.min(),qb.min()) >= -1e-10
        arrays=dict(suboptimality=np.maximum(0,oracle.value-deployed).reshape(shape),
                    fisher_error=np.sqrt(np.maximum(0,np.maximum(qa,qb))).reshape(shape),
                    policies=policies.reshape(*shape,policies.shape[-1]), fitted_values=np.asarray(values).reshape(shape),
                    deployed_values=deployed.reshape(shape),oracle_policy=oracle.policy,oracle_value=oracle.value)
        assert all(np.isfinite(a).all() for a in arrays.values())
        atomic_write_npz(out,arrays)
        atomic_write_json(ROOT/'progress'/f'{index:02d}.json',dict(completed=done,total=len(assigned),
                          request=ri,suite=suite,method=slug,elapsed_seconds=time.monotonic()-started))
        print(f'worker {index}: {done}/{len(assigned)} task/protocols; {suite}/{slug}/task={task}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker']);parser.add_argument('--index',type=int,default=0);parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:worker(args.index,args.workers,args.limit)
