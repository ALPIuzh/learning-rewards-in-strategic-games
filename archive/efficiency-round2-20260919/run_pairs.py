"""Run short, matched CLI experiments; phase selects only the output directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

phase = sys.argv[1]
root = Path('artifacts/efficiency-round2').resolve()
scenarios = {
    'toy': ('runs/toy', 64, 24, []),
    'sparse': ('runs/sparse_information/comparison', 48, 16, []),
    'lobeke': ('runs/lobeke/comparison', 32, 0, ['experiment.warmup_modes=[false]']),
    'lobeke_long': ('runs/lobeke/comparison', 128, 32, ['experiment.seed_count=2', 'experiment.methods=[active,fixed_random,random_each_round]']),
}
if len(sys.argv) == 2:
    scenarios.pop('lobeke_long')
if len(sys.argv) > 2:
    scenarios = {name: scenarios[name] for name in sys.argv[2:]}
report = {}
for name, (config, samples, warmup, extra) in scenarios.items():
    command = [sys.executable, '-m', 'cProfile', '-o', str(root / f'{phase}-{name}.prof'),
        '-m', 'experiments.run', '--config-name', config, f'experiment.name=efficiency-{name}',
        'experiment.seed_count=3', f'experiment.total_samples={samples}',
        f'experiment.warmup_budget={warmup}', 'experiment.evaluation_interval=8',
        f'run.root={root / phase}', 'run.require_clean_git=false', 'run.resume=false', *extra]
    if os.environ.get('MAIRL_BENCH_PROFILE') == '0':
        command = [sys.executable, *command[5:]]
    start = perf_counter()
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, 'MPLCONFIGDIR': str(root/'mpl'), 'XDG_CACHE_HOME': str(root/'cache')}, timeout=180)
    (root/f'{phase}-{name}.log').write_text(result.stdout)
    report[name] = {'command': command, 'seconds': perf_counter()-start, 'returncode': result.returncode}
    print(name, report[name]['seconds'], result.returncode, flush=True)
    if result.returncode: raise RuntimeError(result.stdout[-5000:])
(root/f'{phase}-runs.json').write_text(json.dumps(report, indent=2)+'\n')
