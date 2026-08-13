# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `baseline_e16`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.439371 | -0.00446586 | 15/7/26 |
| accuracy | max | 0.708084 | 0.70222 | -0.00586381 | 12/11/25 |
| macro_f1 | max | 0.413911 | 0.41234 | -0.00157025 | 19/2/27 |
| nll | min | 3.02119 | 3.04972 | +0.0285329 | 16/0/32 |
| multiclass_brier | min | 0.452898 | 0.45981 | +0.00691194 | 14/0/34 |
| ece_10 | min | 0.121629 | 0.123013 | +0.00138396 | 24/0/24 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1065.39 | +54.8434 | 13/0/35 |
| rmse_hours | min | 1591.74 | 1656.17 | +64.4281 | 22/0/26 |
| median_absolute_error_hours | min | 630.614 | 663.315 | +32.7016 | 12/0/36 |
| normalized_mae | min | 0.872341 | 0.944673 | +0.072332 | 13/0/35 |
| mae_skill_vs_median | max | -0.178455 | -0.257863 | -0.0794084 | 13/0/35 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0679785 | -0.0323475 | 22/0/26 |
