# Research audits

These scripts preserve the reasoning and corrections behind the paper results.
They are not part of the installable library. Run them as modules from the
repository root, for example `uv run python -m investigations.active_expert_warmup.run_protocol_pilot --help`.

- `active_expert_warmup/`: warm-up, ridge, and fixed-expert investigations. The
  paper's ridge-zero sensitivity and confirmation targets still use its runner.
- `behavioral_transfer/`: transfer geometry and diagnostic experiments. Several
  helpers have regression tests.
- `warmup_baselines/`: OLS protocol correction and mechanism audits. The
  `pre_correction_publication/` snapshot is historical evidence, not a source of
  current paper figures. `FINDINGS.md` records the interpretation and caveats.

Historical summaries and compact correction data are retained for auditability.
Many diagnostic commands require the original raw runs under `artifacts/`, which
are not bundled with the source. The canonical publication entry point is
`reproducibility/figures.toml`; rendering those frozen exports requires no raw runs.
