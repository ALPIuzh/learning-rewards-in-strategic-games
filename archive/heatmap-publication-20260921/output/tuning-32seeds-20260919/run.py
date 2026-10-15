from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
import argparse,hashlib,json,subprocess,time
import numpy as np
from experiments.guard.paper.run_sensitivity import PilotDesign,_protocols_from_pairs,_run_protocols
from experiments.guard.paper.figures.task_sets import StructuredTaskSet
from mairl_exp.execution.artifacts import atomic_write_json
import evaluation
ROOT=Path(__file__).resolve().parent
PAIRS=tuple((w,l) for w in (0,25,50,100,200) for l in (.001,.003,.01,.03,.1,.3,1))
START=2026094000

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def initialize():evaluation.initialize()
def execute(job):
 w,l,batch=job
 protocol=next(p for p in _protocols_from_pairs(((w,l),)) if p.method=='active')
 seeds=tuple(range(START+8*batch,START+8*(batch+1)))
 record=ROOT/'jobs'/protocol.slug/f'batch-{batch:02d}.json'
 training=ROOT/'training'/f'batch-{batch:02d}'/f'{protocol.slug}.npz'
 source=ROOT/'source_config.yaml'
 if record.exists():
  r=json.loads(record.read_text())
  assert r['training_hash']==digest(training) and r['source_hash']==digest(source)
  assert r['script_hash']==digest(Path(__file__)) and r['evaluation_script_hash']==digest(ROOT/'evaluation.py')
  for seed,h in r['evaluation_hashes'].items():assert digest(ROOT/'evaluation'/protocol.slug/f'{seed}.npz')==h
  return protocol.slug,batch,r['seconds'],True
 record.parent.mkdir(parents=True,exist_ok=True)
 started=time.monotonic()
 design=PilotDesign(seeds,StructuredTaskSet('uniform_tuning',100,9400,2000),(protocol,),(w,),(l,),1e-6)
 with record.with_suffix('.log').open('a') as log,redirect_stdout(log),redirect_stderr(log):
  t=time.monotonic();_run_protocols(source,training.parent,1600,design);train_seconds=time.monotonic()-t
  with np.load(training) as d:
   assert np.array_equal(d['seeds'],seeds) and float(d['mle_ridge'])==1e-6
   assert np.isfinite(d['omega_history']).all() and np.isfinite(d['policies']).all()
   simplex=float(np.max(np.abs(d['policies'].sum(-1)-1)))
   norm=float(np.linalg.norm(d['omega_history'],axis=-1).max())
   assert simplex<1e-10 and d['policies'].min()>=-1e-12 and norm<2+1e-8
   residual=float(np.max(d['diagnostic__mle_residual']))
   assert np.isfinite(residual) and residual<1e-5
   endpoints=d['omega_history'][:,-1].copy()
  h=digest(training);t=time.monotonic()
  for seed,omega in zip(seeds,endpoints):evaluation.solve((protocol.slug,seed,omega,h))
  eval_seconds=time.monotonic()-t
 r=dict(warmup=w,multiplier=l,batch=batch,seeds=seeds,ridge=1e-6,seconds=time.monotonic()-started,training_seconds=train_seconds,evaluation_seconds=eval_seconds,max_simplex_error=simplex,max_omega_norm=norm,max_mle_residual=residual,training_hash=h,source_hash=digest(source),script_hash=digest(Path(__file__)),evaluation_script_hash=digest(ROOT/'evaluation.py'),model_hash=digest(ROOT/'model.npz'),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),evaluation_hashes={str(seed):digest(ROOT/'evaluation'/protocol.slug/f'{seed}.npz') for seed in seeds})
 atomic_write_json(record,r)
 return protocol.slug,batch,r['seconds'],False

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','full']);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
 jobs=[(25,.003,0),(0,.001,0)] if args.stage=='pilot' else [(w,l,b) for b in range(4) for w,l in PAIRS]
 start=time.monotonic()
 with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
  pending={pool.submit(execute,j):j for j in jobs}
  for i,future in enumerate(as_completed(pending),1):
   name,batch,seconds,reused=future.result()
   print(f'{i}/{len(jobs)} {name} batch={batch} {seconds:.1f}s '+('reused' if reused else 'complete'),flush=True)
 atomic_write_json(ROOT/f'{args.stage}-execution.json',dict(seconds=time.monotonic()-start,workers=args.workers,batches=len(jobs),script_hash=digest(Path(__file__))))
