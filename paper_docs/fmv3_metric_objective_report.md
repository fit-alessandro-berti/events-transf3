# Adaptable metric-objective experiment report

## Decision

FM-v3 now supports explicit metric-target objectives for both tasks, with an
initial-gradient-calibrated `equilibrated` default. The five matched 20-epoch
experiments confirm that the profiles materially change what the model learns:

- `accuracy` improves target accuracy over the matched equilibrated retrain.
- `mae` gives the strongest target MAE and broadest target regression gain.
- `r2` strongly improves source R2 and improves target R2/RMSE, with the
  expected MAE/classification trade-off on target logs.
- `balanced_accuracy` improves its source metric but does not transfer that
  gain to the fixed target screen.

None of the from-scratch candidates replaces the promoted epoch-44 endpoint.
The base configuration remains `equilibrated`, while deployment-specific
profiles are available when one metric is the explicit priority.

## Implementation

Classification profiles operate over a complete episode, including alignment
of query-specific prototype columns into original label-ID space:

| Profile | Differentiable target |
|---|---|
| `accuracy` | `1 - mean(p_true)` |
| `balanced_accuracy` | One minus mean per-supported-class soft recall |
| `macro_f1` | One minus macro soft-F1 from soft TP/FP/FN |
| `nll` | Label-smoothed NLL normalized by uniform entropy |
| `brier` | Half multiclass Brier score |
| `equilibrated` | Gradient-calibrated blend of all five |
| `custom` / `legacy` | User weights / historical smoothed cross entropy |

Regression profiles compute all modern metrics in raw hours, including for
the base square-root head via differentiable conversion back to hours:

| Profile | Differentiable target |
|---|---|
| `mae` | Scale-normalized raw-hour MAE |
| `rmse` | Scale-normalized raw-hour RMSE |
| `r2` | `log1p(SSE/SST)` with a normalized variance floor |
| `equilibrated` | Gradient-calibrated nine-metric blend |
| `custom` / `legacy` | User mixture / historical native-space Huber |

Resolved profiles and weights are persisted in training summaries. Structured
diagnostics include every raw/weighted component, per-component gradient
attribution, raw-hour R2, overfitting signals for every adaptable metric, and
metric-gradient imbalance checks.

## Matched protocol

All five runs use the same seed 42, selected architecture, exact 10% held-out
source-case manifest, 50/50 task mix, learning-rate schedule, module-specific
LR multipliers, and 20 epochs x 300 episodes. The validation-manifest SHA-256
is identical in every checkpoint directory:

`23c793d5260f5e9e28b372937b57cdffe0b7057dc7b0815497ab8b09aa5e1b8e`

Only one task profile changes in each extreme run. Checkpoint selection uses
source validation only: the named metric for the extreme task and the
equilibrated score for the other task. The five target logs are opened only
after an epoch is fixed.

## Source-case results

The source-selected results are:

| Run | Epoch | Accuracy | Balanced accuracy | Macro-F1 | NLL | Brier | MAE h | RMSE h | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equilibrated | 16 | 0.337759 | 0.332518 | 0.291138 | 2.05649 | 0.772767 | 869.289 | 1176.28 | 0.405398 |
| accuracy | 20 | 0.335699 | 0.325366 | 0.282277 | 2.11662 | 0.788995 | 868.919 | 1180.47 | 0.402264 |
| balanced accuracy | 19 | 0.346111 | 0.339970 | 0.296208 | 2.06041 | 0.776577 | 874.931 | 1186.35 | 0.400853 |
| MAE | 14 | 0.346969 | 0.341683 | 0.298716 | 2.07585 | 0.775702 | 883.411 | 1177.61 | 0.376684 |
| R2 | 19 | **0.359846** | **0.353954** | **0.310940** | **2.02765** | **0.764912** | **862.111** | **1121.34** | **0.425207** |

The R2 run is the strongest source result across every reported metric. Its
best individual R2 is 0.427075 at epoch 15; epoch 19 is selected because it
also preserves the other task. The final analyzer reports no sustained
objective-, invariant-, or decision-metric overfitting. Full-MAE, by contrast,
shows regression invariant-metric overfitting after epoch 11.

## Fixed target confirmation

Each source-selected checkpoint is evaluated on the unchanged 96-row screen
over billing, helpdesk, receipt, roadtraffic100traces, and sepsis. Absolute
means are:

| Run | Balanced accuracy | Accuracy | Macro-F1 | MAE h | RMSE h | R2 |
|---|---:|---:|---:|---:|---:|---:|
| equilibrated e16 | 0.437430 | 0.699246 | 0.408430 | 1048.69 | 1644.03 | -0.312695 |
| accuracy e20 | 0.437283 | **0.702591** | 0.408368 | 1058.18 | 1652.66 | -0.337834 |
| balanced-accuracy e19 | 0.434776 | 0.685710 | 0.406087 | 1034.18 | 1627.29 | -0.277168 |
| MAE e14 | 0.434161 | 0.693994 | 0.405070 | **1013.50** | **1607.32** | **-0.251366** |
| R2 e19 | 0.433021 | 0.692464 | 0.403639 | 1058.39 | 1634.43 | -0.287175 |
| promoted e44 | **0.443837** | **0.708084** | **0.413911** | **1010.55** | **1591.74** | **-0.210744** |

Paired candidate-minus-equilibrated deltas make the specialization clearer:

| Profile | Accuracy | Balanced accuracy | MAE h | RMSE h | R2 |
|---|---:|---:|---:|---:|---:|
| accuracy | **+0.003345** | -0.000147 | +9.490 | +8.629 | -0.025139 |
| balanced accuracy | -0.013536 | -0.002654 | -14.515 | -16.738 | +0.035528 |
| MAE | -0.005252 | -0.003268 | **-35.192** | **-36.711** | **+0.061329** |
| R2 | -0.006782 | -0.004409 | +9.702 | -9.595 | **+0.025520** |

Full-accuracy therefore transfers in its intended direction. Full-MAE wins 38
of 48 paired MAE rows against equilibrated and unexpectedly gives the largest
target R2 improvement too. Pure R2 wins 29 of 48 target RMSE/R2 rows against
equilibrated, but its source-side all-metric dominance does not transfer to
MAE or classification. Balanced-accuracy's source improvement reverses on the
target screen.

Relative to promoted epoch 44, the closest regression candidate is full-MAE,
but it is still 2.95 h worse on MAE, 15.58 h worse on RMSE, 0.04062 worse on
R2, and lower on every classification decision metric. The promoted endpoint
therefore remains unchanged.

## Stability and validation

- All five runs complete 20 x 300 episodes with finite losses.
- Equilibrated calibration reduces initial component-gradient imbalance from
  5.94x to 1.11x for classification and from 23.60x to 1.17x for regression.
- R2's final epoch applies 300/300 updates with no overflow or non-finite step.
- A single 1/300 late AMP skip is classified as low-severity transient rather
  than persistent; persistent severity requires a 1% last-epoch or mean rate.
- Historical YAML roots explicitly retain legacy semantics, while the base
  square-root head can use every modern raw-hour metric profile.

## Reproducibility artifacts

- Source comparison:
  `evaluation_results/training_metric_objectives/matched_comparison`
- Per-run training analyses:
  `evaluation_results/training_metric_objectives/{equilibrated,accuracy,balanced_accuracy,mae,r2}`
- Fixed target screens:
  `evaluation_results/training_metric_objectives/target_screens`
- Paired target comparisons:
  `evaluation_results/training_metric_objectives/target_comparisons`

Key SHA-256 hashes:

| Artifact | SHA-256 |
|---|---|
| matched comparison JSON | `9b1885e75deafb2e0330b619cbae3078baa4a384dbe397c167d56c6296103d48` |
| equilibrated target CSV | `e0705b5b4f1dbe482e1d1be4d7cf82ea37de9e4f4647a13907a9402d34ef33df` |
| accuracy target CSV | `7a1eb895c256a07c3b95e389fc61fceaddbaadb496c7975c17f5e7ba269dcc1f` |
| balanced-accuracy target CSV | `0166befb55554f47dce950488505a9d2ef7a738ceb5c6ea401485640ec0fecd2` |
| MAE target CSV | `c5ee91083ed180f2082fcb9a43e1add886ab6a809fdc0fbad98553826bc31d94` |
| R2 target CSV | `744199ca09a2eeef047403d8c9226bc0c024e0e51407627a0cdebf1444df8acc` |

