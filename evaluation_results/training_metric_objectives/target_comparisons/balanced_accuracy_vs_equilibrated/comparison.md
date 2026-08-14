# Paired FM-v3 result comparison

Reference: `equilibrated_e16`. Candidate: `balanced_accuracy`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.43743 | 0.434776 | -0.00265401 | 18/7/23 |
| accuracy | max | 0.699246 | 0.68571 | -0.0135361 | 9/11/28 |
| macro_f1 | max | 0.40843 | 0.406087 | -0.00234326 | 19/4/25 |
| nll | min | 3.08854 | 3.11168 | +0.0231405 | 18/0/30 |
| multiclass_brier | min | 0.468698 | 0.482057 | +0.0133586 | 8/0/40 |
| ece_10 | min | 0.124678 | 0.131711 | +0.00703232 | 23/0/25 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | +0.00282443 | +1.11022e-17 | +0.00284677 |
| helpdesk | -0.00752015 | -0.0296552 | -0.000149331 |
| receipt | -0.00814195 | -0.029918 | -0.00941926 |
| roadtraffic100traces | -0.0011048 | +1.38778e-17 | -0.00226207 |
| sepsis | +0.00098224 | -0.0054 | -0.00271617 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1048.69 | 1034.18 | -14.5155 | 21/0/27 |
| rmse_hours | min | 1644.03 | 1627.29 | -16.7385 | 25/0/23 |
| median_absolute_error_hours | min | 674.461 | 654.132 | -20.3287 | 23/0/25 |
| normalized_mae | min | 0.922772 | 0.911817 | -0.0109548 | 21/0/27 |
| mae_skill_vs_median | max | -0.232636 | -0.21981 | +0.0128265 | 21/0/27 |
| rmse_skill_vs_median | max | -0.0634843 | -0.0552446 | +0.00823969 | 25/0/23 |
| d2_absolute_error | max | -0.232636 | -0.21981 | +0.0128265 | 21/0/27 |
| r2 | max | -0.312695 | -0.277168 | +0.0355279 | 25/0/23 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | -75.1264 | -86.3272 | +0.174032 |
| helpdesk | -3.07601e-05 | -0.000292844 | +0.00197588 |
| receipt | +0.312595 | +0.0914615 | -0.00264275 |
| roadtraffic100traces | +6.59824 | +6.94874 | -0.00300457 |
| sepsis | -0.139069 | +0.332372 | -0.000427963 |
