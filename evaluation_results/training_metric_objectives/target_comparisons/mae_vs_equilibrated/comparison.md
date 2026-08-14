# Paired FM-v3 result comparison

Reference: `equilibrated_e16`. Candidate: `mae`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.43743 | 0.434161 | -0.00326843 | 13/5/30 |
| accuracy | max | 0.699246 | 0.693994 | -0.00525167 | 14/7/27 |
| macro_f1 | max | 0.40843 | 0.40507 | -0.00336048 | 16/4/28 |
| nll | min | 3.08854 | 3.09964 | +0.0110929 | 28/0/20 |
| multiclass_brier | min | 0.468698 | 0.474261 | +0.00556281 | 22/0/26 |
| ece_10 | min | 0.124678 | 0.133443 | +0.00876466 | 20/0/28 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.0012448 | +0.00144231 | -0.00152576 |
| helpdesk | -0.00537363 | -0.000689655 | -0.00336232 |
| receipt | -0.00417703 | -0.0233607 | -0.00608705 |
| roadtraffic100traces | -0.0011048 | +0 | -0.00163383 |
| sepsis | -0.00400917 | -0.0026 | -0.00384809 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1048.69 | 1013.5 | -35.1925 | 38/0/10 |
| rmse_hours | min | 1644.03 | 1607.32 | -36.7111 | 28/0/20 |
| median_absolute_error_hours | min | 674.461 | 647.564 | -26.8965 | 31/0/17 |
| normalized_mae | min | 0.922772 | 0.896862 | -0.0259098 | 38/0/10 |
| mae_skill_vs_median | max | -0.232636 | -0.199372 | +0.0332646 | 38/0/10 |
| rmse_skill_vs_median | max | -0.0634843 | -0.0460659 | +0.0174184 | 28/0/20 |
| d2_absolute_error | max | -0.232636 | -0.199372 | +0.0332646 | 38/0/10 |
| r2 | max | -0.312695 | -0.251366 | +0.0613295 | 28/0/20 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | -113.782 | -114.701 | +0.227137 |
| helpdesk | -0.00416739 | -0.00292891 | +0.0189724 |
| receipt | -0.209653 | -1.02291 | +0.0178746 |
| roadtraffic100traces | -48.8221 | -77.6731 | +0.0406958 |
| sepsis | -15.8705 | +1.65173 | -0.00215918 |
