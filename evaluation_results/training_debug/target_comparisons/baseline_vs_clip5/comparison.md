# Paired FM-v3 result comparison

Reference: `baseline_e16`. Candidate: `clip5_e14`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.439371 | 0.43829 | -0.0010818 | 14/14/20 |
| accuracy | max | 0.70222 | 0.704115 | +0.00189455 | 17/16/15 |
| macro_f1 | max | 0.41234 | 0.411885 | -0.000455271 | 21/7/20 |
| nll | min | 3.04972 | 3.03304 | -0.0166784 | 32/0/16 |
| multiclass_brier | min | 0.45981 | 0.454967 | -0.0048431 | 31/0/17 |
| ece_10 | min | 0.123013 | 0.122695 | -0.000317872 | 17/0/31 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1065.39 | 1073.25 | +7.85983 | 22/0/26 |
| rmse_hours | min | 1656.17 | 1665.62 | +9.45154 | 23/0/25 |
| median_absolute_error_hours | min | 663.315 | 679.95 | +16.6353 | 20/0/28 |
| normalized_mae | min | 0.944673 | 0.944612 | -6.07009e-05 | 22/0/26 |
| mae_skill_vs_median | max | -0.257863 | -0.257955 | -9.24277e-05 | 22/0/26 |
| rmse_skill_vs_median | max | -0.0679785 | -0.0673224 | +0.000656107 | 23/0/25 |
