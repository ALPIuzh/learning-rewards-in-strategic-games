"""Check strict deployments and assemble the frozen final-evaluation outputs."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import t
from mairl.core.problem import compile_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy, leader_value_and_gradient
from mairl_exp.execution.artifacts import atomic_write_npz, atomic_write_json
ROOT=Path(__file__).resolve().parent

def main():
    with np.load(ROOT/'inputs.npz') as f: inputs=dict(f)
    requests=json.loads((ROOT/'requests.json').read_text())
    problem=compile_problem(inputs['features'],eta=float(inputs['eta']),radius=float(inputs['radius']))
    config=LeaderOptimizationConfig(PolicyOptimizationConfig(**{**json.loads(str(inputs['evaluation_policy'])),'random_starts':256}))
    groups={};records=[];maximum_simplex_error=0.;deployments=0
    for i,r in enumerate(requests):
        p=ROOT/'evaluations'/f'{i:04d}.npz'
        with np.load(p) as f: d=dict(f)
        shape=inputs[f"{r['suite']}_{r['slug']}_omegas"].shape[:2]
        assert d['suboptimality'].shape==shape
        assert all(np.isfinite(a).all() for a in d.values())
        assert d['policies'].min()>=-1e-10
        error=float(np.max(np.abs(d['policies'].sum(-1)-1)));maximum_simplex_error=max(maximum_simplex_error,error)
        assert error<1e-9
        np.testing.assert_allclose(d['suboptimality'],np.maximum(0,d['oracle_value']-d['deployed_values']),rtol=0,atol=1e-14)
        groups.setdefault((r['suite'],r['slug']),[]).append((i,r,d))
        records.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        deployments+=d['suboptimality'].size
    checks=[]
    for (suite,slug),items in groups.items():
        # Inspect the highest loss anywhere on the curve and the first saved deployment.
        worst=max(items,key=lambda item:float(item[2]['suboptimality'].max()))
        chosen=[(items[0],(0,0)),(worst,np.unravel_index(worst[2]['suboptimality'].argmax(),worst[2]['suboptimality'].shape))]
        for (ri,r,d),(si,ci) in chosen:
            reward=inputs[suite+'_tasks'][r['task']];omega=inputs[f'{suite}_{slug}_omegas'][si,ci]
            seed=r['task_seed']*1000000+r['task']*100000+int(inputs['seeds'][si])*100+int(ci)
            fit=optimize_leader_policy(problem,omega,reward,config,seed=seed)
            oracle=optimize_leader_policy(problem,inputs['true_omega'],reward,config,seed=r['task_seed']+r['task'])
            deployed=leader_value_and_gradient(problem,inputs['true_omega'],reward,fit.policy)[0]
            change=abs(max(0,oracle.value-deployed)-float(d['suboptimality'][si,ci]))
            checks.append(dict(request=ri,suite=suite,method=slug,seed=int(inputs['seeds'][si]),checkpoint=int(inputs[f'{suite}_{slug}_checkpoints'][ci]),loss_change=change,fitted_objective_improvement=float(fit.value-d['fitted_values'][si,ci]),oracle_improvement=float(oracle.value-d['oracle_value'])))
        print('checked 256-start extremes:',suite,slug,flush=True)
    max_change=max(c['loss_change'] for c in checks)
    report=dict(evaluation_files=len(records),deployments=deployments,max_simplex_error=maximum_simplex_error,
                checks_256_starts=checks,max_loss_change=max_change,passed=max_change<1e-7,files=records,
                inputs_sha256=hashlib.sha256((ROOT/'inputs.npz').read_bytes()).hexdigest())
    atomic_write_json(ROOT/'final-checks.json',report)
    assert report['passed'],f'256-start check needs review: {max_change}'
    compact=dict(inputs);summary={}
    for (suite,slug),items in groups.items():
        items.sort(key=lambda item:item[1]['task'])
        for metric in ('suboptimality','fisher_error'):
            compact[f'{suite}_{slug}_{metric}']=np.stack([d[metric] for _,_,d in items])
        compact[f'{suite}_{slug}_oracle_policies']=np.stack([d['oracle_policy'] for _,_,d in items])
        compact[f'{suite}_{slug}_oracle_values']=np.array([float(d['oracle_value']) for _,_,d in items])
        if suite in ('primary','source'):
            compact[f'{suite}_{slug}_final_policies']=np.stack([d['policies'][:,-1] for _,_,d in items])
        values=compact[f'{suite}_{slug}_suboptimality'][:,:,-1].mean(axis=0)
        mean=float(values.mean());half=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
        summary.setdefault(suite,{})[slug]=dict(mean=mean,ci95=[mean-half,mean+half],per_seed=values.tolist())
    for suite,methods in list(summary.items()):
        a=np.array(methods['active_warm']['per_seed']);b=np.array(methods['active_no_warm']['per_seed']);delta=a-b
        half=float(t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)))
        methods['paired_warm_minus_no_warm']=dict(mean=float(delta.mean()),ci95=[float(delta.mean()-half),float(delta.mean()+half)])
    atomic_write_npz(ROOT/'compact.npz',compact)
    atomic_write_json(ROOT/'summary.json',summary)
    print('PASS:',deployments,'deployments;',max_change,'maximum checked loss change',flush=True)
if __name__=='__main__':main()
