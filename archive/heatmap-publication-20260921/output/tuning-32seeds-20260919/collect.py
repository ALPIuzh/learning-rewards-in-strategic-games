from pathlib import Path
import hashlib,json,sys
import numpy as np,scipy,matplotlib
ROOT=Path(__file__).resolve().parent
WS=[0,25,50,100,200];LS=[.001,.003,.01,.03,.1,.3,1];SEEDS=np.arange(2026094000,2026094032)
model_hash=hashlib.sha256((ROOT/'model.npz').read_bytes()).hexdigest()
seen=set();groups={}
for record in sorted((ROOT/'jobs').glob('*/*.json')):
 r=json.loads(record.read_text());key=(r['warmup'],r['multiplier'],r['batch'])
 assert key not in seen and r['model_hash']==model_hash;seen.add(key)
 groups.setdefault(key[:2],[]).append((record.parent.name,r))
assert len(seen)==140
scores=np.empty((5,7,32,100))
for y,w in enumerate(WS):
 for x,l in enumerate(LS):
  records=groups[w,l];assert len(records)==4
  folder=records[0][0];assert all(name==folder for name,_ in records)
  for k,seed in enumerate(SEEDS):
   with np.load(ROOT/'evaluation'/folder/f'{seed}.npz') as d:
    assert int(d['seed'])==seed and str(d['model_hash'])==model_hash
    assert d['subopt'].shape==(100,) and np.isfinite(d['subopt']).all()
    scores[y,x,k]=d['subopt']
path=ROOT/'figure-data.npz'
if path.exists():
 with np.load(path) as d:assert np.array_equal(d['transfer_suboptimality'],scores)
else:
 np.savez_compressed(path,transfer_suboptimality=scores,seeds=SEEDS,warmup_budgets=WS,multipliers=LS,mle_ridge=1e-6,total_observations=1600,task_seed=9400,task_candidate_pool=2000,model_hash=np.array(model_hash))
provenance=dict(data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_sha256=model_hash,source_config_sha256=hashlib.sha256((ROOT/'source_config.yaml').read_bytes()).hexdigest(),python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,axis_order=['warmup','multiplier','seed','task'],shape=list(scores.shape),seeds='32 fresh learning seeds; fixed before scores were read',all_cells_mle_ridge=1e-6)
(ROOT/'figure-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Collected',scores.shape,'task scores; file size',path.stat().st_size)
