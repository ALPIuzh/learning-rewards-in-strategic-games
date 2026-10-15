# Paper figures

The registry in `figures.toml` contains the nine empirical figures in the paper
and an alternate layout of the downstream comparison. `make plots` regenerates
all ten fragments and compiles them into `output/pdf/preview.pdf`.

| Paper figure | Registered output | Included data |
|---|---|---|
| 1: source convergence | `m1_source_convergence` | Lobeke source trajectories |
| 2: downstream policies | `m2_downstream_transfer_compact` | Lobeke downstream assets |
| 4: environment | `i1_environment` | GUARD environment |
| 5: diagnostics | `i1_source_diagnostics` | Lobeke final evaluation |
| 6: downstream comparison | `i1_downstream` | Lobeke downstream assets |
| 7: transfer difficulty | `i1_behavioral_transfer_gap` | 180-task behavioral transfer |
| 8: hyperparameters | `i1_hyperparameters` | 32-seed sensitivity, MLE ridge $10^{-6}$ |
| 9: sparse information | `i2_sparse_information` | Sparse-information sweep |
| 10: lower bound | `i3_lower_bound` | Minimax hard-instance study |

`m2_downstream_transfer` is the alternate layout of Figure 2. Paper Figure 3 is
pseudocode. The structured sensitivity and warm-up factorial studies, including
their data and scripts, are preserved under `archive/`.
