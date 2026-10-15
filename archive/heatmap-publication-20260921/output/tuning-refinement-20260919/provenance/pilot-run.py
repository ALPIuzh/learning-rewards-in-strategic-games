from pathlib import Path
import argparse, json, time, hashlib, subprocess
from dataclasses import replace, asdict
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from experiments.guard.paper.run_sensitivity import PilotDesign, _protocols_from_pairs, _run_protocols
from experiments.guard.paper.figures.task_sets import StructuredTaskSet, normalized_structured_tasks
ROOT=Path(__file__).resolve().parent
SOURCE=Path('reproducibility/data/lobeke/ridge_zero_tuning/source_config.yaml')
PILOT=((0,.03),(50,.003),(200,.003),(200,.01),(25,.01),(100,.01),(50,.001))
PAIRS=tuple(dict.fromkeys((*PILOT,*((w,l) for w in (25,100) for l in (.003,.01,.03)),(100,.1))))
def run(pairs,seeds,tag):
 protocols=tuple(p for p in _protocols_from_pairs(pairs) if p.method=='active')
 design=PilotDesign(seeds,StructuredTaskSet('tuning_refinement',100,9400,1000),protocols,tuple(sorted({p[0] for p in pairs})),tuple(sorted({p[1] for p in pairs})),1e-6)
 start=time.monotonic()
 runtime,numerics,_=_run_protocols(SOURCE,ROOT/'training'/tag,1600,design)
 result={'seconds':time.monotonic()-start,'pairs':pairs,'seeds':seeds,'ridge':1e-6,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 (ROOT/f'{tag}-execution.json').write_text(json.dumps(result,indent=2)+'\n')
 if not (ROOT/'model.npz').exists():
  np.savez_compressed(ROOT/'model.npz',features=runtime.problem.features,eta=runtime.problem.eta,radius=runtime.problem.radius,true_omega=runtime.true_omega,tasks=normalized_structured_tasks(runtime,design.tasks),evaluation_policy=json.dumps(asdict(numerics.evaluation_policy)))
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','remaining']);a=parser.parse_args()
 if a.stage=='pilot':run(PILOT,(94000,94001),'pilot')
 else:
  run(PAIRS,tuple(range(94002,94008)),'remaining-six')
  new=tuple(p for p in PAIRS if p not in PILOT)
  run(new,(94000,94001),'new-first-two')
