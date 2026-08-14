# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `balanced_accuracy`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.434776 | -0.00906144 | 10/7/31 |
| accuracy | max | 0.708084 | 0.68571 | -0.022374 | 2/9/37 |
| macro_f1 | max | 0.413911 | 0.406087 | -0.00782392 | 15/2/31 |
| nll | min | 3.02119 | 3.11168 | +0.0904967 | 5/0/43 |
| multiclass_brier | min | 0.452898 | 0.482057 | +0.0291583 | 3/0/45 |
| ece_10 | min | 0.121629 | 0.131711 | +0.0100817 | 21/0/27 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | +4.0724e-06 | -0.00528846 | -0.000455685 |
| helpdesk | -0.0082442 | -0.0413793 | +0.00473887 |
| receipt | -0.0210255 | -0.0311475 | -0.0206074 |
| roadtraffic100traces | -0.00804924 | -0.00847458 | -0.00869146 |
| sepsis | -0.00778987 | -0.0228 | -0.0142774 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1034.18 | +23.6272 | 21/0/27 |
| rmse_hours | min | 1591.74 | 1627.29 | +35.5526 | 22/0/26 |
| median_absolute_error_hours | min | 630.614 | 654.132 | +23.5185 | 21/0/27 |
| normalized_mae | min | 0.872341 | 0.911817 | +0.0394763 | 21/0/27 |
| mae_skill_vs_median | max | -0.178455 | -0.21981 | -0.0413551 | 21/0/27 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0552446 | -0.0196135 | 22/0/26 |
| d2_absolute_error | max | -0.178455 | -0.21981 | -0.0413551 | 21/0/27 |
| r2 | max | -0.210744 | -0.277168 | -0.0664232 | 22/0/26 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +133.365 | +183.26 | -0.254446 |
| helpdesk | -0.00149538 | -0.00553082 | +0.0329103 |
| receipt | +4.22522 | +4.70395 | -0.108518 |
| roadtraffic100traces | -39.9716 | -18.637 | +0.0099853 |
| sepsis | +7.79903 | -2.39617 | +0.00323469 |
