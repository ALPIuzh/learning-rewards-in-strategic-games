from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=json.loads((ROOT/'protected-source-hashes.json').read_text())
for path,h in protected.items():assert digest(Path(path))==h,path
for name in ('model.npz','oracles.npz'):
 assert (ROOT/name).read_bytes()==(ROOT.parent/'tuning-refinement-20260919'/name).read_bytes()
model_hash=digest(ROOT/'model.npz');source_hash=digest(ROOT/'source_config.yaml')
seen=set();max_residual=0.;max_simplex=0.;max_radius=0.
for p in sorted((ROOT/'jobs').glob('*/*.json')):
 r=json.loads(p.read_text());name=p.parent.name;batch=r['batch']
 assert r['source_hash']==source_hash and r['model_hash']==model_hash
 assert r['script_hash']==digest(ROOT/'run.py') and r['evaluation_script_hash']==digest(ROOT/'evaluation.py')
 training=ROOT/'training'/f'batch-{batch:02d}'/f'{name}.npz';h=digest(training)
 assert h==r['training_hash']
 with np.load(training) as d:
  assert np.array_equal(d['seeds'],np.arange(2026094032+8*batch,min(2026094040+8*batch,2026094050)))
  assert float(d['mle_ridge'])==1e-6
  assert np.isfinite(d['omega_history']).all() and np.isfinite(d['policies']).all()
  simplex=float(np.max(np.abs(d['policies'].sum(-1)-1)));radius=float(np.linalg.norm(d['omega_history'],axis=-1).max())
  residual=float(np.max(d['diagnostic__mle_residual']))
  assert simplex<1e-10 and d['policies'].min()>=-1e-12 and radius<2+1e-8 and residual<1e-5
  max_residual=max(max_residual,residual);max_simplex=max(max_simplex,simplex);max_radius=max(max_radius,radius)
  for i,seed in enumerate(d['seeds']):
   path=ROOT/'evaluation'/name/f'{seed}.npz';assert digest(path)==r['evaluation_hashes'][str(seed)]
   with np.load(path) as e:
    assert str(e['source_hash'])==h and str(e['model_hash'])==model_hash
    assert np.array_equal(e['omega'],d['omega_history'][i,-1])
    assert e['subopt'].shape==(100,) and np.isfinite(e['subopt']).all() and e['subopt'].min()>=0
   key=(r['warmup'],r['multiplier'],int(seed));assert key not in seen;seen.add(key)
expected={(w,l,s) for w in (0,25,50,100,200) for l in (.001,.003,.01) for s in range(2026094032,2026094050)}
assert seen==expected,(len(seen),len(expected))
result=dict(seed_runs=len(seen),task_evaluations=len(seen)*100,protected_files_unchanged=len(protected),max_mle_residual=max_residual,max_simplex_error=max_simplex,max_omega_norm=max_radius)
(ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
