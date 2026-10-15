"""Independent selected-ridge confirmation; selection precedes endpoint inspection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from validate import PROTOCOLS, SOURCE

from experiments.guard.paper.figures.task_sets import StructuredTaskSet
from experiments.guard.paper.run_sensitivity import PilotDesign, _run_protocols

RECORD = Path(__file__).parent
ROOT = Path("artifacts/scientific-validation-20260919/tuned-baseline")
SEEDS = tuple(range(2026093000, 2026093050))
SELECTED = tuple(
    p
    for p in PROTOCOLS
    if p.slug in ("active_no_warm_lambda0p03", "active_warm200_lambda0p003", "fixed_expert_matched_ridge")
)
TASKS = StructuredTaskSet("independent-selected-ridge", 50, 202609300, 1000)


def train():
    assert len(SELECTED) == 3
    for start in range(0, 50, 10):
        design = PilotDesign(SEEDS[start : start + 10], TASKS, SELECTED, (0, 200), (0.003, 0.03), 0.0)
        _run_protocols(SOURCE, ROOT / f"batch-{start:02d}", 1600, design)
        print(f"completed training batch {start}", flush=True)


def select():
    paths = [
        Path("artifacts/scientific-validation-20260919/strict-evaluation/202609200") / f"task-{i:03d}.npz"
        for i in range(100)
    ]
    rows = [dict(np.load(path)) for path in paths]
    names = list(rows[0]["protocols"])
    assert all(
        np.array_equal(row["protocols"], names) and np.array_equal(row["seeds"], rows[0]["seeds"]) for row in rows
    )
    values = np.stack([row["subopt"] for row in rows])
    assert values.shape == (100, 15, 50) and np.isfinite(values).all() and values.min() >= 0
    means = values.mean(axis=0).mean(axis=1)
    candidates = {0.0: float(means[names.index("fixed_expert_matched_ridge")])}
    candidates.update(
        {ridge: float(means[names.index(f"fixed_expert_ridge{ridge:g}")]) for ridge in (0.01, 0.1, 1.0, 10.0)}
    )
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    chosen = min(candidates, key=lambda ridge: (candidates[ridge], ridge))
    output = RECORD / "tuned-baseline-selection.json"
    assert not output.exists(), "selection is already frozen"
    output.write_text(json.dumps(dict(ridge=chosen, selection_means=candidates, source_files=sources), indent=2) + "\n")
    print(f"froze selected fixed-expert ridge {chosen:g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "select"))
    args = parser.parse_args()
    start = time.monotonic()
    {"train": train, "select": select}[args.command]()
    (RECORD / f"tuned-baseline-{args.command}-execution.json").write_text(
        json.dumps(
            dict(
                seconds=time.monotonic() - start, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ),
            indent=2,
        )
        + "\n"
    )
