from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
from mairl.core.problem import compile_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig,optimize_leader_policy,leader_value_and_gradient
ROOT=Path(__file__).resolve().parent

def initialize():
 global MODEL,PROBLEM,CONFIG,ORACLES
 MODEL=dict(np.load(ROOT/'model.npz'));PROBLEM=compile_problem(MODEL['features'],eta=float(MODEL['eta']),radius=float(MODEL['radius']))
 CONFIG=LeaderOptimizationConfig(replace(PolicyOptimizationConfig(**json.loads(str(MODEL['evaluation_policy']))),random_starts=256,tolerance=1e-12))
 ORACLES=np.load(ROOT/'oracles.npz')['values']
def check(job):
 name,seed,i=job;d=np.load(ROOT/'evaluation'/name/f'{seed}.npz');reward=MODEL['tasks'][i]
 fitted=optimize_leader_policy(PROBLEM,d['omega'],reward,CONFIG,seed=2026091900+seed*100+i)
 oracle=optimize_leader_policy(PROBLEM,MODEL['true_omega'],reward,CONFIG,seed=2026091900+i)
 value=leader_value_and_gradient(PROBLEM,MODEL['true_omega'],reward,fitted.policy)[0]
 return dict(protocol=name,seed=seed,task=i,old_loss=float(d['subopt'][i]),new_loss=float(oracle.value-value),fitted_objective_change=float(fitted.value-d['fitted_values'][i]),oracle_change=float(oracle.value-ORACLES[i]))
if __name__=='__main__':
 start=time.monotonic();jobs=[]
 # Check the highest-loss seed/task combination in every cell.
 for folder in sorted((ROOT/'evaluation').iterdir()):
  files=sorted(folder.glob('*.npz'));assert len(files)==18
  scores=np.stack([np.load(p)['subopt'] for p in files]);s,i=np.unravel_index(np.argmax(scores),scores.shape)
  jobs.append((folder.name,int(files[s].stem),int(i)))
 assert len(jobs)==15
 with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:rows=list(pool.map(check,jobs))
 r=dict(seconds=time.monotonic()-start,tasks_checked=len(rows),random_starts=256,max_abs_loss_change=max(abs(x['new_loss']-x['old_loss']) for x in rows),max_abs_fitted_objective_change=max(abs(x['fitted_objective_change']) for x in rows),max_abs_oracle_change=max(abs(x['oracle_change']) for x in rows),rows=rows)
 (ROOT/'numerical-spotcheck.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
