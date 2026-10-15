SHELL := /bin/sh

PDF_DIR := output/pdf
RUN_DIR ?=

.PHONY: help install guard test lint format check toy paper smoke plots list generate check-figures preview

help:
	@echo "MAIRL development commands"
	@echo ""
	@echo "  make install   Install locked runtime, development, and GUARD dependencies"
	@echo "  make guard     Fetch the pinned public GUARD dependency"
	@echo "  make test      Run the behavioral test suite"
	@echo "  make lint      Check Python formatting and static lint rules"
	@echo "  make format    Apply Ruff's formatter and safe lint fixes"
	@echo "  make check     Run lint, tests, and figure checks"
	@echo "  make toy       Run/resume the small reference experiment"
	@echo "  make smoke     Run bounded data-generation and plotting checks"
	@echo "  make plots     Regenerate all registered paper plots and the preview PDF"
	@echo "  make generate  Regenerate the TikZ figures without compiling a PDF"
	@echo "  make check-figures  Check that tracked figures are up to date"
	@echo "  make list      List registered figures"
	@echo "  make paper     Show full GUARD experiment and export commands"

install:
	uv sync --locked --extra dev --extra experiments --extra guard

guard:
	@set -eu; \
	guard=experiments/guard/GUARD; \
	revision=58964e76f37bdfc32560308a14942677b27148f3; \
	if [ -e "$$guard/.git" ] && [ "$$(git -C "$$guard" rev-parse HEAD)" = "$$revision" ]; then \
	    :; \
	elif git ls-files --stage -- "$$guard" 2>/dev/null | grep -q '^160000 '; then \
	    git submodule update --init --recursive -- "$$guard"; \
	elif [ ! -d "$$guard" ]; then \
	    git clone --no-checkout https://github.com/CoffeeAndConvexity/GUARD.git "$$guard"; \
	    git -C "$$guard" checkout --detach "$$revision"; \
	fi; \
	actual=$$(git -C "$$guard" rev-parse HEAD); \
	if [ "$$actual" != "$$revision" ]; then \
	    echo "GUARD is at $$actual; expected $$revision. Existing checkout left untouched." >&2; \
	    exit 1; \
	fi; \
	if [ -n "$$(git -C "$$guard" status --porcelain)" ]; then \
	    echo "GUARD has local changes. Existing checkout left untouched." >&2; \
	    exit 1; \
	fi;

test:
	uv run pytest

lint:
	uv run ruff format --check src experiments tests reproducibility
	uv run ruff check src experiments tests

format:
	uv run ruff check --fix src experiments tests
	uv run ruff format src experiments tests reproducibility

check: lint test check-figures

toy:
	uv run python -m experiments.run

smoke:
	sh smoke_experiments.sh

list:
	uv run python -m reproducibility.figures list

generate:
	uv run python -m reproducibility.figures generate

check-figures:
	uv run python -m reproducibility.figures check

preview: generate
	mkdir -p "$(PDF_DIR)"
	latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
		-outdir="$(PDF_DIR)" reproducibility/tex/preview.tex

plots: preview
	$(MAKE) check-figures

.PHONY: experiment joint-discovery joint-expansion protected-selection protected-confirmation warm-selection no-warm-selection warm-confirmation no-warm-confirmation lambda-tuning lambda-tuning-preflight require-run main-figures

paper:
	@echo "Lobeke paper reproduction targets"
	@echo ""
	@echo "  make experiment — Run/resume the 450-job flagship experiment."
	@echo "  make joint-discovery — Run/resume the coarse warm-up/lambda interaction grid."
	@echo "  make joint-expansion — Expand the coarse grid around its upper warm-up/lambda boundaries."
	@echo "  make protected-selection — Run fresh-seed warm-up/lambda and no-warm-up/lambda selections."
	@echo "  make protected-confirmation — Confirm the two frozen selection winners without reselection."
	@echo "  make lambda-tuning — Run/resume protected lambda-only tuning on fresh seed splits."
	@echo "  make lambda-tuning-preflight — Exercise every lambda/mode at the full horizon on four fresh seeds."
	@echo "  make main-figures RUN_DIR=/path/to/completed/run — Export policy assets and TikZ figures from completed runs."

experiment:
	uv run python -m experiments.run --config-name runs/lobeke/comparison

joint-discovery:
	uv run python -m experiments.run --config-name runs/lobeke/joint_discovery

joint-expansion:
	uv run python -m experiments.run --config-name runs/lobeke/joint_expansion

protected-selection: warm-selection no-warm-selection

warm-selection:
	uv run python -m experiments.run --config-name runs/lobeke/warm_selection

no-warm-selection:
	uv run python -m experiments.run --config-name runs/lobeke/no_warm_selection

protected-confirmation: warm-confirmation no-warm-confirmation

warm-confirmation:
	uv run python -m experiments.run --config-name runs/lobeke/warm_confirmation

no-warm-confirmation:
	uv run python -m experiments.run --config-name runs/lobeke/no_warm_confirmation

lambda-tuning:
	uv run python -m experiments.run --config-name runs/lobeke/lambda_tuning

lambda-tuning-preflight:
	uv run python -m experiments.run --config-name runs/lobeke/lambda_tuning \
		experiment.name=guard-lobeke-lambda-preflight \
		experiment.selection_seed_start=30000 experiment.selection_seed_count=2 \
		experiment.confirmation_seed_start=31000 experiment.confirmation_seed_count=2 \
		experiment.minimum_seeds_per_split=2 experiment.reject_upper_boundary=false

require-run:
	@test -n "$(RUN_DIR)" || \
		(echo "RUN_DIR is required; pass a completed artifacts/runs/guard-lobeke-paper/<id>" >&2; exit 2)
	@test -f "$(RUN_DIR)/manifest.json" || \
		(echo "RUN_DIR does not contain manifest.json: $(RUN_DIR)" >&2; exit 2)

main-figures: require-run
	uv run python -m experiments.guard.paper.figures.transfer_assets --run-dir "$(RUN_DIR)"
	uv run python -m experiments.guard.paper.figures.tikz --run-dir "$(RUN_DIR)"
