# Paired FM-v3 result comparison

Reference: `head_focused_e17`. Candidate: `mae`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.442306 | 0.434161 | -0.00814443 | 5/7/36 |
| accuracy | max | 0.706749 | 0.693994 | -0.0127543 | 6/7/35 |
| macro_f1 | max | 0.412097 | 0.40507 | -0.00702782 | 10/5/33 |
| nll | min | 3.03786 | 3.09964 | +0.0617773 | 7/0/41 |
| multiclass_brier | min | 0.456688 | 0.474261 | +0.0175732 | 9/0/39 |
| ece_10 | min | 0.123076 | 0.133443 | +0.0103667 | 19/0/29 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00149744 | -0.00576923 | -0.00105872 |
| helpdesk | -0.00800733 | -0.00551724 | -0.00293247 |
| receipt | -0.0189474 | -0.0393443 | -0.0156739 |
| roadtraffic100traces | -0.00457702 | -0.00423729 | -0.00468697 |
| sepsis | -0.00697953 | -0.0072 | -0.0103189 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1017.41 | 1013.5 | -3.90988 | 30/0/18 |
| rmse_hours | min | 1619.54 | 1607.32 | -12.2237 | 26/0/22 |
| median_absolute_error_hours | min | 631.613 | 647.564 | +15.9512 | 23/0/25 |
| normalized_mae | min | 0.891932 | 0.896862 | +0.00493045 | 30/0/18 |
| mae_skill_vs_median | max | -0.196028 | -0.199372 | -0.00334359 | 30/0/18 |
| rmse_skill_vs_median | max | -0.0515293 | -0.0460659 | +0.00546342 | 26/0/22 |
| d2_absolute_error | max | -0.196028 | -0.199372 | -0.00334359 | 30/0/18 |
| r2 | max | -0.260866 | -0.251366 | +0.00950007 | 26/0/22 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | -9.18967 | -36.8583 | +0.0544601 |
| helpdesk | -0.00164021 | -0.00293707 | +0.0179319 |
| receipt | +2.07829 | +1.15025 | -0.0416481 |
| roadtraffic100traces | -9.42207 | -23.4819 | +0.0116156 |
| sepsis | -4.11676 | -4.17737 | +0.00556394 |
