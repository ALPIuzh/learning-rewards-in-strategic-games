from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
protected=json.loads((ROOT/'protected-source-hashes.json').read_text())
for path,digest in protected.items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
model_hash=hashlib.sha256((ROOT/'model.npz').read_bytes()).hexdigest()
assert str(np.load(ROOT/'oracles.npz')['model_hash'])==model_hash
checks=[];count=0
for group in ('pilot','remaining-six','new-first-two'):
 for p in sorted((ROOT/'training'/group).glob('*.npz')):
  d=np.load(p);assert float(d['mle_ridge'])==1e-6
  for key in ('policies','omega_history'):
   assert np.isfinite(d[key]).all(),(p,key)
  simplex=float(np.max(np.abs(d['policies'].sum(axis=-1)-1)))
  assert simplex<1e-10 and d['policies'].min()>=-1e-12
  norms=np.linalg.norm(d['omega_history'],axis=-1)
  assert norms.max()<2+1e-8
  source_hash=hashlib.sha256(p.read_bytes()).hexdigest()
  for i,seed in enumerate(d['seeds']):
   e=np.load(ROOT/'evaluation'/p.stem/f'{seed}.npz')
   assert str(e['source_hash'])==source_hash and str(e['model_hash'])==model_hash
   assert np.array_equal(e['omega'],d['omega_history'][i,-1])
   assert e['subopt'].shape==(100,) and np.isfinite(e['subopt']).all() and e['subopt'].min()>=0
   count+=1
  checks.append({'file':str(p.relative_to(ROOT)),'seeds':len(d['seeds']),'max_simplex_error':simplex,'max_omega_norm':float(norms.max())})
assert count==96,count
result={'original_files_unchanged':len(protected),'new_seed_runs':count,'task_evaluations':count*100,'training_checks':checks}
(ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='training_checks'}))
