#!/bin/sh
set -eu
for repeat in 1 2 3; do
    if [ "$repeat" = 2 ]; then
        PYTHONPATH=src .venv/bin/python artifacts/efficiency-review/run_pairs.py "after-$repeat"
        PYTHONPATH=/private/tmp/mairl-efficiency-baseline/src .venv/bin/python artifacts/efficiency-review/run_pairs.py "before-$repeat"
    else
        PYTHONPATH=/private/tmp/mairl-efficiency-baseline/src .venv/bin/python artifacts/efficiency-review/run_pairs.py "before-$repeat"
        PYTHONPATH=src .venv/bin/python artifacts/efficiency-review/run_pairs.py "after-$repeat"
    fi
done
