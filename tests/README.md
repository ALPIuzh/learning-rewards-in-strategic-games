Run the full suite with `make test`, or one group with `uv run pytest tests/core`.

- `core/`: mathematical identities, inference, policy optimization, and the active algorithm.
- `experiments/`: study execution, baselines, seed handling, tuning, and application adapters.
- `reproducibility/`: paper data, exports, and figure regeneration.
- `integration/`: command-line entry points, package dependencies, and fresh-process plotting.

Shared fixtures live in `conftest.py`; experiment settings live in `experiments/conftest.py`.
`make check` also runs lint and verifies the 10 tracked figure fragments.
