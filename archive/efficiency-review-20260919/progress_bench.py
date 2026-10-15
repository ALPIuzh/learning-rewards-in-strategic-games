"""Measure durable progress writes with representative pending job counts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from statistics import median
from time import perf_counter
from mairl_exp.execution.artifacts import RunStore, atomic_write_json, job_identity
from mairl_exp.execution.jobs import StudyJob
import sys
report={}
for count in (18,450):
    with TemporaryDirectory(dir='artifacts/efficiency-review') as tmp:
        store=RunStore.__new__(RunStore)
        store.path=Path(tmp)
        store.jobs_path=store.path/'jobs'
        store._started=perf_counter()
        jobs=tuple(StudyJob('active',False,seed) for seed in range(count))
        expected=[{'job_id':job_identity(j),'label':j.label} for j in jobs]
        atomic_write_json(store.path/'manifest.json',{'expected_jobs':expected})
        for job in jobs:
            atomic_write_json(store._job_path(job)/'status.json',{'status':'running','updated_at':'baseline'})
        store._refresh_manifest(status='running',current_job=jobs[-1].label)
        durations=[]
        for repeat in range(5):
            start=perf_counter()
            for job in jobs[:8]: store.update_job_progress(job,repeat+1,100)
            durations.append(perf_counter()-start)
        report[str(count)]={'median_seconds_per_8_updates':median(durations),'samples':durations}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(report)
