# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `clip10_e6`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.435424 | -0.00841345 | 11/10/27 |
| accuracy | max | 0.708084 | 0.693272 | -0.0148118 | 7/18/23 |
| macro_f1 | max | 0.413911 | 0.405308 | -0.00860297 | 16/3/29 |
| nll | min | 3.02119 | 3.05155 | +0.0303634 | 15/0/33 |
| multiclass_brier | min | 0.452898 | 0.471669 | +0.0187704 | 13/0/35 |
| ece_10 | min | 0.121629 | 0.133465 | +0.011836 | 20/0/28 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.0108406 | -0.00576923 | -0.0055695 |
| helpdesk | -0.00537118 | -0.0137931 | -0.00272995 |
| receipt | -0.0111244 | -0.0393443 | -0.0219088 |
| roadtraffic100traces | -0.00347222 | -0.00423729 | -0.00337625 |
| sepsis | -0.0102706 | -0.0088 | -0.00838507 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1076.19 | +65.6429 | 15/0/33 |
| rmse_hours | min | 1591.74 | 1660.96 | +69.2225 | 24/0/24 |
| median_absolute_error_hours | min | 630.614 | 678.463 | +47.8496 | 11/0/37 |
| normalized_mae | min | 0.872341 | 0.9597 | +0.0873593 | 15/0/33 |
| mae_skill_vs_median | max | -0.178455 | -0.271785 | -0.0933303 | 15/0/33 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0650527 | -0.0294217 | 24/0/24 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours |
|---|---:|---:|
| billing | +292.717 | +336.172 |
| helpdesk | -0.00404144 | -0.00719018 |
| receipt | +8.04885 | +1.1739 |
| roadtraffic100traces | -23.3339 | -5.6522 |
| sepsis | +32.9919 | -0.549269 |
