#!/bin/sh
# Tiny end-to-end runs; full paper experiments are deliberately not run here.
set -eu
cd "$(dirname "$0")"
smoke_root="$PWD/artifacts/smoke"
mkdir -p "$smoke_root"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$smoke_root/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$smoke_root/cache}"

run() {
    name=$1
    config=$2
    shift 2
    echo "Checking $name"
    uv run python -m experiments.run --config-name "$config" "experiment.name=smoke-$name" \
        "run.root=$smoke_root/runs" run.require_clean_git=false "$@" \
        > "$smoke_root/$name-run.log" 2>&1 || {
        cat "$smoke_root/$name-run.log" >&2
        return 1
    }
}

plot() {
    run_dir=$(sed -n 's/^run_dir=//p' "$smoke_root/$1-run.log")
    uv run python -m experiments.plot "run_dir=$run_dir" require_clean_git=false
}

run toy runs/toy \
    experiment.seed_count=2 experiment.evaluation_interval=4 \
    experiment.total_samples=32 experiment.warmup_budget=24
plot toy > "$smoke_root/toy-plot.log" 2>&1 || { cat "$smoke_root/toy-plot.log" >&2; exit 1; }

run sparse runs/sparse_information/comparison \
    experiment.seed_count=2 experiment.evaluation_interval=4 \
    experiment.total_samples=8 experiment.warmup_budget=0 'experiment.warmup_modes=[false]'
plot sparse > "$smoke_root/sparse-plot.log" 2>&1 || { cat "$smoke_root/sparse-plot.log" >&2; exit 1; }

run sparse_action_count runs/sparse_information/action_count \
    experiment.seed_count=2 experiment.evaluation_interval=4 \
    experiment.total_samples=8 'experiment.leader_action_counts=[2,4,8,16,32]'

run zero_sum runs/zero_sum/comparison \
    experiment.seed_count=2 experiment.evaluation_interval=4 \
    experiment.total_samples=8 experiment.warmup_budget=0

run guard runs/lobeke/comparison \
    experiment.seed_count=2 experiment.evaluation_interval=4 \
    experiment.total_samples=8 experiment.warmup_budget=0 'experiment.warmup_modes=[false]'

run lower_bound runs/minimax \
    'experiment.dimensions=[33]' 'experiment.samples_per_dimension=[4]' \
    experiment.replicates_per_coordinate=1

echo "Checking ridge-zero sensitivity"
uv run python -m experiments.guard.paper.run_sensitivity \
    --source-config reproducibility/data/lobeke/sensitivity_32seeds/source_config.yaml \
    --output-dir "$smoke_root/ridge-zero" --total-samples 200 \
    --seed-start 99000 --seed-count 2 --task-seed 9900 --task-count 2 \
    --task-candidate-pool 20 --mle-ridge 0 --candidate-pairs 0:0.03 \
    > "$smoke_root/ridge-zero-run.log" 2>&1 || {
    cat "$smoke_root/ridge-zero-run.log" >&2
    exit 1
}

echo "All smoke checks passed; results and logs: $smoke_root"
