# Matched metric-objective comparison

Normalization baseline: `equilibrated` epoch 1. Lower profile/equilibrated scores are better.

| Run | Classification | Regression | Profile epoch | Equilibrated epoch | Accuracy | Balanced accuracy | Macro-F1 | NLL | Brier | MAE h | RMSE h | R2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equilibrated | equilibrated | equilibrated | 16 | 16 | 0.337759 | 0.332518 | 0.291138 | 2.05649 | 0.772767 | 869.289 | 1176.28 | 0.405398 |
| accuracy | accuracy | equilibrated | 20 | 18 | 0.335699 | 0.325366 | 0.282277 | 2.11662 | 0.788995 | 868.919 | 1180.47 | 0.402264 |
| balanced_accuracy | balanced_accuracy | equilibrated | 19 | 18 | 0.346111 | 0.33997 | 0.296208 | 2.06041 | 0.776577 | 874.931 | 1186.35 | 0.400853 |
| mae | equilibrated | mae | 14 | 14 | 0.346969 | 0.341683 | 0.298716 | 2.07585 | 0.775702 | 883.411 | 1177.61 | 0.376684 |

Profile epoch selection uses the named metric for an extreme task and the equilibrated score for the other task; target data is not used.
