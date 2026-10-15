"""Compare jobs by their specifications, ignoring only timing/provenance metadata."""
import json
import sys
from pathlib import Path
import numpy as np
root=Path('artifacts/efficiency-round2')
phases = tuple(sys.argv[1:]) or ('before', 'after')
assert len(phases) == 2
report={}
names = tuple(path.name.removeprefix('efficiency-') for path in (root/phases[0]).iterdir())
for name in names:
    runs=[next((root/phase/f'efficiency-{name}').iterdir()) for phase in phases]
    manifests=[json.loads((run/'manifest.json').read_text()) for run in runs]
    assert manifests[0]['expected_jobs']==manifests[1]['expected_jobs']
    assert all(m['status']=='complete' for m in manifests)
    arrays_checked=0; mismatches=[]; max_difference=0.; summary_mismatches=[]
    for job in manifests[0]['expected_jobs']:
        paths=[run/'jobs'/job['job_id'] for run in runs]
        with np.load(paths[0]/'result.npz',allow_pickle=False) as a, np.load(paths[1]/'result.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                arrays_checked+=1
                x,y=a[key],b[key]
                if x.shape!=y.shape or x.dtype!=y.dtype or x.tobytes()!=y.tobytes():
                    delta=float(np.max(np.abs(x-y),initial=0))
                    max_difference=max(max_difference,delta)
                    mismatches.append({'job':job['label'],'array':key,'max_absolute_difference':delta})
        summaries=[json.loads((path/'summary.json').read_text()) for path in paths]
        for key in summaries[0]:
            if 'seconds' not in key and summaries[0][key]!=summaries[1][key]:
                summary_mismatches.append({'job':job['label'],'key':key})
    report[name]={'jobs':len(manifests[0]['expected_jobs']),'arrays':arrays_checked,'mismatches':mismatches,'max_absolute_difference':max_difference,'summary_mismatches':summary_mismatches}
(root/f'comparison-{phases[0]}-{phases[1]}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert not any(r['mismatches'] or r['summary_mismatches'] for r in report.values())
