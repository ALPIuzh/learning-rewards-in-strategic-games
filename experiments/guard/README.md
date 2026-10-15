# GUARD security games

The adapter loads security games from [GUARD](https://github.com/CoffeeAndConvexity/GUARD).
Leader actions are patrol schedules; follower actions are attack targets.

Run `make guard` from the repository root to install the pinned checkout, then:

```bash
uv run python -m experiments.run --config-name runs/guard --cfg
uv run python -m experiments.run --config-name runs/guard
```

`GuardApplication` supports three feature maps: `tabular` (one coefficient per
payoff), `target_coverage` (covered and uncovered rewards per target), and
`behavioral` (coverage, target value, and their product). The behavioral map
requires schedule-form games with target utilities; compilation checks that
it reconstructs the follower payoffs.

The [Lobeke paper experiment](paper/README.md) builds its game from the upstream
tracking data using `geographic/map_lobeke.py`.
