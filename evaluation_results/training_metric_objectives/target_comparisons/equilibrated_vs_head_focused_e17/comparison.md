# Paired FM-v3 result comparison

Reference: `head_focused_e17`. Candidate: `equilibrated`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.442306 | 0.43743 | -0.004876 | 8/12/28 |
| accuracy | max | 0.706749 | 0.699246 | -0.00750262 | 8/14/26 |
| macro_f1 | max | 0.412097 | 0.40843 | -0.00366734 | 13/8/27 |
| nll | min | 3.03786 | 3.08854 | +0.0506844 | 9/0/39 |
| multiclass_brier | min | 0.456688 | 0.468698 | +0.0120104 | 10/0/38 |
| ece_10 | min | 0.123076 | 0.124678 | +0.00160203 | 20/0/28 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00025264 | -0.00721154 | +0.000467041 |
| helpdesk | -0.0026337 | -0.00482759 | +0.000429854 |
| receipt | -0.0147703 | -0.0159836 | -0.00958686 |
| roadtraffic100traces | -0.00347222 | -0.00423729 | -0.00305314 |
| sepsis | -0.00297035 | -0.0046 | -0.00647078 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1017.41 | 1048.69 | +31.2826 | 19/0/29 |
| rmse_hours | min | 1619.54 | 1644.03 | +24.4874 | 28/0/20 |
| median_absolute_error_hours | min | 631.613 | 674.461 | +42.8477 | 16/0/32 |
| normalized_mae | min | 0.891932 | 0.922772 | +0.0308403 | 19/0/29 |
| mae_skill_vs_median | max | -0.196028 | -0.232636 | -0.0366082 | 19/0/29 |
| rmse_skill_vs_median | max | -0.0515293 | -0.0634843 | -0.0119549 | 28/0/20 |
| d2_absolute_error | max | -0.196028 | -0.232636 | -0.0366082 | 19/0/29 |
| r2 | max | -0.260866 | -0.312695 | -0.0518294 | 28/0/20 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +104.592 | +77.8423 | -0.172677 |
| helpdesk | +0.00252718 | -8.16068e-06 | -0.00104053 |
| receipt | +2.28794 | +2.17316 | -0.0595227 |
| roadtraffic100traces | +39.4 | +54.1911 | -0.0290802 |
| sepsis | +11.7538 | -5.8291 | +0.00772313 |
