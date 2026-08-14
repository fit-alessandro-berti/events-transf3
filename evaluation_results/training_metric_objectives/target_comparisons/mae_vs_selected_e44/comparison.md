# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `mae`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.434161 | -0.00967586 | 8/6/34 |
| accuracy | max | 0.708084 | 0.693994 | -0.0140895 | 7/7/34 |
| macro_f1 | max | 0.413911 | 0.40507 | -0.00884113 | 12/2/34 |
| nll | min | 3.02119 | 3.09964 | +0.0784492 | 4/0/44 |
| multiclass_brier | min | 0.452898 | 0.474261 | +0.0213624 | 3/0/45 |
| ece_10 | min | 0.121629 | 0.133443 | +0.011814 | 19/0/29 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00406516 | -0.00384615 | -0.00482821 |
| helpdesk | -0.00609768 | -0.0124138 | +0.00152587 |
| receipt | -0.0170606 | -0.0245902 | -0.0172752 |
| roadtraffic100traces | -0.00804924 | -0.00847458 | -0.00806322 |
| sepsis | -0.0127813 | -0.02 | -0.0154094 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1013.5 | +2.95022 | 30/0/18 |
| rmse_hours | min | 1591.74 | 1607.32 | +15.58 | 22/0/26 |
| median_absolute_error_hours | min | 630.614 | 647.564 | +16.9506 | 30/0/18 |
| normalized_mae | min | 0.872341 | 0.896862 | +0.0245212 | 30/0/18 |
| mae_skill_vs_median | max | -0.178455 | -0.199372 | -0.020917 | 30/0/18 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0460659 | -0.0104348 | 22/0/26 |
| d2_absolute_error | max | -0.178455 | -0.199372 | -0.020917 | 30/0/18 |
| r2 | max | -0.210744 | -0.251366 | -0.0406216 | 22/0/26 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +94.7096 | +154.886 | -0.201342 |
| helpdesk | -0.00563201 | -0.00816688 | +0.0499069 |
| receipt | +3.70297 | +3.58958 | -0.0880011 |
| roadtraffic100traces | -95.3919 | -103.259 | +0.0536856 |
| sepsis | -7.93242 | -1.07681 | +0.00150347 |
