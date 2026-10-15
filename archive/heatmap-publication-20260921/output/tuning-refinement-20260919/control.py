from pathlib import Path
import time,json
from dataclasses import asdict,replace
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from experiments.guard.paper.run_sensitivity import PilotDesign,_protocols_from_pairs,_run_protocols
from experiments.guard.paper.figures.task_sets import StructuredTaskSet,normalized_structured_tasks
ROOT=Path(__file__).resolve().parent;SOURCE=Path('reproducibility/data/lobeke/ridge_zero_tuning/source_config.yaml')
config=OmegaConf.load(SOURCE);numerics=instantiate(config.numerics);game=instantiate(config.application).build(numerics)
tasks=StructuredTaskSet('active_expert_warmup_audit',100,9400,2000)
np.savez_compressed(ROOT/'model.npz',features=game.problem.features,eta=game.problem.eta,radius=game.problem.radius,true_omega=game.true_omega,tasks=normalized_structured_tasks(game,tasks),evaluation_policy=json.dumps(asdict(numerics.evaluation_policy)))
protocols=tuple(p for p in _protocols_from_pairs(((0,.03),)) if p.method=='active')
design=PilotDesign((94000,94001),tasks,protocols,(0,),(.03,),0.)
start=time.monotonic();_run_protocols(SOURCE,ROOT/'training/ridge-zero-control',1600,design)
(ROOT/'control-execution.json').write_text(json.dumps({'seconds':time.monotonic()-start},indent=2)+'\n')
