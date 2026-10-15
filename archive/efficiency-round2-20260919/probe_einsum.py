"""Test contractions with no BLAS path, without changing array layouts or expressions."""
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
rng=np.random.default_rng(71)
report=[]
for shape in ((3,3,2),(32,23,10),(400,7,6),(1600,23,10)):
 n,b,d=shape
 cases={
 'nb,nbd->nd':(rng.random((n,b)),rng.normal(size=(n,b,d))),
 'snbd,sd->snb':(rng.normal(size=(3,n,b,d)),rng.normal(size=(3,d))),
 'snd,sd->sn':(rng.normal(size=(3,n,d)),rng.normal(size=(3,d))),
 'snb,snbd->snd':(rng.random((3,n,b)),rng.normal(size=(3,n,b,d))),
 'sd,sd->s':(rng.normal(size=(3,d)),rng.normal(size=(3,d))),
 }
 for expression,args in cases.items():
  # Include strided views, as used by batched solver gathers/prefixes.
  for layout in ('C','F','strided'):
   if layout=='F': arrays=tuple(np.asfortranarray(a) for a in args)
   elif layout=='strided': arrays=tuple(a[...,::-1] for a in args)
   else: arrays=args
   results={};times={}
   for optimize in (True,False):
    measurements=[]
    for _ in range(3):
     start=perf_counter()
     for _ in range(50): result=np.einsum(expression,*arrays,optimize=optimize)
     measurements.append((perf_counter()-start)/50)
    times[str(optimize)]=median(measurements);results[optimize]=result
   report.append({'shape':shape,'expression':expression,'layout':layout,'exact':results[True].tobytes()==results[False].tobytes(),'max_difference':float(np.max(np.abs(results[True]-results[False]),initial=0)),'seconds':times})
Path('artifacts/efficiency-round2/einsum-probe.json').write_text(json.dumps(report,indent=2)+'\n')
for expr in cases:
 rows=[r for r in report if r['expression']==expr]
 print(expr, 'exact',sum(r['exact'] for r in rows),'/',len(rows),'median speedup',median(r['seconds']['True']/r['seconds']['False'] for r in rows))
