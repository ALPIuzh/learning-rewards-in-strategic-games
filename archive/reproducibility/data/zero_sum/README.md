# Archived zero-sum synthetic evidence

These protected derived data are retained for reproducibility and implementation
validation, but are intentionally not registered in the publication figure
manifest. The Lobeke experiment already provides the paper's zero-sum comparison,
so presenting this additional toy as a full benchmark would be redundant.

- `comparison.json` contains the protected 100-seed source and downstream summaries.
- `directional_transfer.json` and `directional_transfer.npz` contain the
  target-dependent Fisher diagnostic and its per-seed arrays.

The generating experiment code remains in `experiments/zero_sum_comparison/`.
