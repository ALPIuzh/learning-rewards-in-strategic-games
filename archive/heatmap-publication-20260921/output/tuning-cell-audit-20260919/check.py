from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
from mairl.core.problem import compile_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig,optimize_leader_policy,leader_value_and_gradient
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'tuning-refinement-20260919'
NAME='active_warm25_lambda0p003'
def initialize():
 global MODEL,PROBLEM,CONFIG,ORACLES
 MODEL=dict(np.load(SOURCE/'model.npz'))
 PROBLEM=compile_problem(MODEL['features'],eta=float(MODEL['eta']),radius=float(MODEL['radius']))
 CONFIG=LeaderOptimizationConfig(replace(PolicyOptimizationConfig(**json.loads(str(MODEL['evaluation_policy']))),random_starts=256,tolerance=1e-12))
 ORACLES=np.load(SOURCE/'oracles.npz')['values']
def check(job):
 seed,i=job;d=np.load(SOURCE/'evaluation'/NAME/f'{seed}.npz');reward=MODEL['tasks'][i]
 fitted=optimize_leader_policy(PROBLEM,d['omega'],reward,CONFIG,seed=2026091900+seed*100+i)
 oracle=optimize_leader_policy(PROBLEM,MODEL['true_omega'],reward,CONFIG,seed=2026091900+i)
 true_value=leader_value_and_gradient(PROBLEM,MODEL['true_omega'],reward,fitted.policy)[0]
 return dict(seed=seed,task=i,old_loss=float(d['subopt'][i]),new_loss=float(oracle.value-true_value),fitted_objective_change=float(fitted.value-d['fitted_values'][i]),oracle_change=float(oracle.value-ORACLES[i]))
if __name__=='__main__':
 start=time.monotonic();jobs=[]
 for seed,n in ((94005,10),(94000,5),(94001,5),(94002,5)):
  loss=np.load(SOURCE/'evaluation'/NAME/f'{seed}.npz')['subopt']
  jobs.extend((seed,int(i)) for i in np.argsort(loss)[-n:])
 with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:results=list(pool.map(check,jobs))
 out={'seconds':time.monotonic()-start,'random_starts':256,'rows':results}
 (ROOT/'evaluation-check.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'seconds':out['seconds'],'tasks_checked':len(results),'max_abs_loss_change':max(abs(r['new_loss']-r['old_loss']) for r in results),'max_abs_fitted_objective_change':max(abs(r['fitted_objective_change']) for r in results),'max_abs_oracle_change':max(abs(r['oracle_change']) for r in results)},indent=2))
