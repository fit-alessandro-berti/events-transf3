# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `clip5_e14`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.43829 | -0.00554765 | 16/8/24 |
| accuracy | max | 0.708084 | 0.704115 | -0.00396925 | 16/8/24 |
| macro_f1 | max | 0.413911 | 0.411885 | -0.00202552 | 23/1/24 |
| nll | min | 3.02119 | 3.03304 | +0.0118546 | 18/0/30 |
| multiclass_brier | min | 0.452898 | 0.454967 | +0.00206884 | 21/0/27 |
| ece_10 | min | 0.121629 | 0.122695 | +0.00106609 | 15/0/33 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1073.25 | +62.7033 | 15/0/33 |
| rmse_hours | min | 1591.74 | 1665.62 | +73.8797 | 26/0/22 |
| median_absolute_error_hours | min | 630.614 | 679.95 | +49.3369 | 12/0/36 |
| normalized_mae | min | 0.872341 | 0.944612 | +0.0722713 | 15/0/33 |
| mae_skill_vs_median | max | -0.178455 | -0.257955 | -0.0795008 | 15/0/33 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0673224 | -0.0316914 | 26/0/22 |
