from dataclasses import asdict, is_dataclass, replace
import pickle
import sys
import numpy as np
from mairl.algorithm import ActiveIRL, ActiveIRLConfig
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import compile_problem
from mairl.exploration.design import WarmupConfig
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.exploration.policy import select_information_policy_batch
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy_batch
from mairl.inference.mle import MLEConfig, solve_mle_batch

def freeze(x):
    if is_dataclass(x): return freeze(asdict(x))
    if isinstance(x, dict): return {k: freeze(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return tuple(freeze(v) for v in x)
    if isinstance(x, np.ndarray): return (x.dtype.str, x.shape, x.tobytes())
    return x

out=[]
for n in (3, 7, 513):
    rng=np.random.default_rng(71+n)
    problem=compile_problem(rng.normal(size=(n,3,2))*0.15, eta=1.2, radius=2.)
    omegas=rng.normal(size=(2,2))*0.3
    matrices=np.broadcast_to(np.eye(2),(2,2,2)).copy()
    reward=rng.normal(size=(n,3))
    for starts, iterations in ((0,0),(2,3)):
        config=PolicyOptimizationConfig(random_starts=starts,batch_prefilter_iterations=iterations)
        out.append(freeze(select_information_policy_batch(problem,omegas,matrices,config,tuple(np.random.default_rng(s) for s in (13,19)))))
        out.append(freeze(optimize_leader_policy_batch(problem,omegas,reward,LeaderOptimizationConfig(config),(13,19))))
    if n==513: continue
    policies=rng.dirichlet(np.ones(n),size=(2,20))
    actions=rng.integers(0,3,size=(2,20))
    for solver in ('newton','slsqp'):
        out.append(freeze(solve_mle_batch(problem,policies,actions,MLEConfig(ridge=.1,solver=solver))))
    for enabled in (False,True):
        config=ActiveIRLConfig(horizon=4,lambda_information=1.44,lambda_warm=.01,confidence_delta=.05,mle=MLEConfig(ridge=.1),warmup=WarmupConfig(budget=20,design_tolerance=.2,enabled=enabled),policy=PolicyOptimizationConfig(random_starts=2),seed=9)
        diagnostics=[]
        out.append(freeze(ActiveIRL(problem,config).run(SoftmaxFollowerOracle(problem,omegas[0]),on_round=diagnostics.append)))
        out.append(freeze(diagnostics))
with open(sys.argv[1], 'wb') as f: pickle.dump(out,f)
print(f'{len(out)} result groups saved')
