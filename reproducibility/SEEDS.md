# Protected seed registry

This file prevents accidental reuse of inspected outcomes during paper tuning.
Seed ranges are paired across candidates within a study and disjoint between
selection, confirmation, and final evaluation. Reusing the same numeric seed in
an unrelated environment would not create statistical dependence, but distinct
ranges make the audit unambiguous.

| Study | Selection | Confirmation | Final evaluation | Status |
|---|---:|---:|---:|---|
| Lobeke | `10000..10063` | `20000..20063` | `30000..30049` | selection, confirmation, and final complete |
| Zero-sum toy | `40000..40063` | `50000..50063` | `60000..60099` | selection, confirmation, and final complete |
| Sparse information | `70000..70063` | `71000..71063` | `72000..72099` | final complete |
| Structured toy warm-up | `80000..80063` | `81000..81063` | `82000..82099` | archived; seeds remain reserved |
| Structured toy temperature | --- | --- | `83000..83099` | archived; seeds remain reserved |
| Lower-bound instance | `90000..90063` | `91000..91063` | `92000..92063` | final complete |

The previous Lobeke run on `100..149`, and the previous zero-sum run on
`30000..30099`, have already been inspected. They remain useful pilot evidence
but are not final-evaluation samples after the new hyperparameter searches.

Downstream rewards consume no additional follower samples. Lobeke
hyperparameter selection uses the fixed ten-task set generated with task seed
`9100`; the appendix transfer diagnostics use a separate fixed ten-task set
at seed `9200`. This is a fixed deterministic diagnostic suite, not a sampled
task distribution; reported uncertainty is conditional on these tasks.
Main-text maps use their own fixed illustrative tasks.

The 32-seed sensitivity heatmap uses learning seeds `2026094000..2026094031`
and 100 fixed tasks (task seed `9400`), with MLE ridge `1e-6`. The archived
50-seed refinement extends this range through `2026094049`; all these outcomes
have been inspected and must not be reused as fresh confirmation.
