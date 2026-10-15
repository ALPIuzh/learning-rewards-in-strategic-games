# Lobeke environment schematic data

`data.npz` contains only the three 5-by-5 arrays shown in the appendix:
ecological target value, source-task Stackelberg patrol coverage, and the
induced poacher attack distribution. `provenance.json` pins the upstream GUARD
revision, behavioral model, optimization settings, array summaries, and data
checksum.

The GUARD-dependent producer is separate from the paper renderer:

```bash
uv run --extra experiments --extra guard python -m \
  experiments.guard.paper.figures.export_environment \
  --output-dir /tmp/lobeke-environment
```

Compare the regenerated export with this directory before replacing the frozen
paper data. The public GUARD submodule is read but never modified.
