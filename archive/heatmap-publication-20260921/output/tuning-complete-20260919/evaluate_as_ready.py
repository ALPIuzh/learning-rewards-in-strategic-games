from pathlib import Path
import subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
start=time.monotonic();iteration=0
while time.monotonic()-start<3600:
 jobs=[]
 for group in ('first-two','remaining-six'):
  for p in (ROOT/'training'/group).glob('*.npz'):
   with np.load(p) as d:
    jobs.extend((p.stem,int(s)) for s in d['seeds'])
 pending=sum(not (ROOT/'evaluation'/name/f'{seed}.npz').exists() for name,seed in jobs)
 finished=all((ROOT/f'{tag}-execution.json').exists() for tag in ('first-two','remaining-six'))
 if pending>=8 or (finished and pending):
  iteration+=1;tag=f'wave-{iteration:02d}';workers=4 if finished else 2
  with (ROOT/f'{tag}.log').open('x') as log:
   subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--workers',str(workers),'--tag',tag],check=True,stdout=log,stderr=subprocess.STDOUT)
  print(f'{tag}: evaluated {pending} new seed runs; {len(jobs)} training runs ready',flush=True)
 elif finished:
  assert len(jobs)==96
  print('All 96 seed runs evaluated.',flush=True)
  break
 else:
  time.sleep(10)
else:
 raise TimeoutError('Evaluation watcher exceeded one hour; inspect training progress.')
