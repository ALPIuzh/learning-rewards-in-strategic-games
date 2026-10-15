#!/bin/bash
set -eu
cd /home/antber/mairl-iclr
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/home/antber/mairl-iclr/.matplotlib-cache
methods=(active active fixed_random fixed_random random_each_round random_each_round fixed_expert_mle zero_sum_ols follower_ols)
warmups=(false true false true false true false false false)
method=${methods[$SLURM_ARRAY_TASK_ID]}
warmup=${warmups[$SLURM_ARRAY_TASK_ID]}
hostname
.venv/bin/python -u -m experiments.run \
 --config-name runs/lobeke/comparison \
 experiment.name=lobeke-final-20260921-$method-$warmup \
 experiment.seed_start=2026097000 experiment.seed_count=50 \
 experiment.total_samples=1600 experiment.evaluation_interval=40 \
 experiment.warmup_budget=50 \
 experiment.no_warmup_lambda_information_multiplier=0.1 \
 experiment.warmup_lambda_information_multiplier=0.03 \
 "experiment.methods=[$method]" "experiment.warmup_modes=[$warmup]" \
 numerics.mle.ridge=1e-6 experiment.fixed_expert_mle_ridge=1e-6 run.seed_batch_size=50
