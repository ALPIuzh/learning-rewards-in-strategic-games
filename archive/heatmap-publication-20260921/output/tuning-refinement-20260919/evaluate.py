from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,time
import numpy as np
from mairl.core.problem import compile_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig,optimize_leader_policy,leader_value_and_gradient
from mairl_exp.execution.artifacts import atomic_write_npz,atomic_write_json
ROOT=Path(__file__).resolve().parent

def initialize():
 global DATA,PROBLEM,CONFIG,MODEL_HASH,ORACLES
 DATA=dict(np.load(ROOT/'model.npz'));MODEL_HASH=hashlib.sha256((ROOT/'model.npz').read_bytes()).hexdigest()
 PROBLEM=compile_problem(DATA['features'],eta=float(DATA['eta']),radius=float(DATA['radius']))
 CONFIG=LeaderOptimizationConfig(replace(PolicyOptimizationConfig(**json.loads(str(DATA['evaluation_policy']))),random_starts=64,tolerance=1e-12))
 ORACLES=np.load(ROOT/'oracles.npz')['values'] if (ROOT/'oracles.npz').exists() else None

def oracle(i):
 x=optimize_leader_policy(PROBLEM,DATA['true_omega'],DATA['tasks'][i],CONFIG,seed=9400+i)
 return x.value,x.policy

def solve(job):
 name,seed,omega,source_hash=job;path=ROOT/'evaluation'/name/f'{seed}.npz'
 if path.exists():
  d=np.load(path);assert np.array_equal(d['omega'],omega) and str(d['model_hash'])==MODEL_HASH
  return name,seed,True
 values=[];policies=[];fitted=[]
 for i,reward in enumerate(DATA['tasks']):
  x=optimize_leader_policy(PROBLEM,omega,reward,CONFIG,seed=9400*1000000+i*100000+seed*100)
  fitted.append(x.value);policies.append(x.policy)
  values.append(leader_value_and_gradient(PROBLEM,DATA['true_omega'],reward,x.policy)[0])
 loss=ORACLES-np.array(values)
 assert np.isfinite(loss).all() and loss.min()>=-1e-9
 atomic_write_npz(path,dict(subopt=np.maximum(loss,0),deployed_values=np.array(values),fitted_values=np.array(fitted),policies=np.array(policies),omega=omega,seed=np.array(seed),model_hash=np.array(MODEL_HASH),source_hash=np.array(source_hash)))
 return name,seed,False

def jobs():
 result=[]
 for group in ('pilot','remaining-six','new-first-two'):
  for p in sorted((ROOT/'training'/group).glob('*.npz')):
   d=np.load(p);assert float(d['mle_ridge'])==1e-6
   digest=hashlib.sha256(p.read_bytes()).hexdigest()
   for i,seed in enumerate(d['seeds']):result.append((p.stem,int(seed),d['omega_history'][i,-1],digest))
 assert len({(n,s) for n,s,_,_ in result})==len(result)
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--tag',default='complete');args=parser.parse_args();start=time.monotonic();initialize()
 if ORACLES is None:
  with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:r=list(pool.map(oracle,range(100)))
  atomic_write_npz(ROOT/'oracles.npz',dict(values=np.array([v for v,p in r]),policies=np.array([p for v,p in r]),model_hash=np.array(MODEL_HASH)))
 initialize();selected=jobs();new=0
 with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
  for i,(name,seed,reused) in enumerate(pool.map(solve,selected),1):
   new+=not reused
   print(f'{i}/{len(selected)} {name} seed={seed} '+('reused' if reused else 'evaluated'),flush=True)
 atomic_write_json(ROOT/f'evaluation-{args.tag}.json',dict(seconds=time.monotonic()-start,seed_runs=len(selected),new_evaluations=new*100,workers=args.workers,model_hash=MODEL_HASH))
