"""Deterministic scalar and batched leader optimization timing and output snapshots."""
from dataclasses import asdict
import json,pickle,sys
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
from mairl.core.problem import compile_problem
from mairl.planning.leader import LeaderOptimizationConfig,optimize_leader_policy,optimize_leader_policy_batch
from mairl.exploration.simplex import PolicyOptimizationConfig
root=Path('artifacts/efficiency-round2');report={};outputs=[]
for n,b,d in ((3,3,2),(23,23,6)):
 rng=np.random.default_rng(n)
 phi=rng.normal(size=(n,b,d));phi/=np.maximum(1.,1.1*np.linalg.norm(phi,axis=2,keepdims=True))
 problem=compile_problem(phi,eta=1.25,radius=2.)
 omegas=rng.normal(size=(4,d))*.2;rewards=rng.normal(size=(n,b))
 config=LeaderOptimizationConfig(PolicyOptimizationConfig(random_starts=4,batch_refine_starts=2))
 for mode in ('scalar','batch'):
  def run():
   if mode=='scalar': return (optimize_leader_policy(problem,omegas[0],rewards,config,seed=91),)
   return optimize_leader_policy_batch(problem,omegas,rewards,config,(91,92,93,94))
  run();durations=[]
  for _ in range(7):
   start=perf_counter();result=run();durations.append(perf_counter()-start)
  report[f'{n}-{mode}']={'median_seconds':median(durations),'samples':durations}
  outputs.extend(asdict(r) for r in result)
(root/f'{sys.argv[1]}-leader.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/f'{sys.argv[1]}-leader.pkl').open('wb') as f:pickle.dump(outputs,f)
print(report)
