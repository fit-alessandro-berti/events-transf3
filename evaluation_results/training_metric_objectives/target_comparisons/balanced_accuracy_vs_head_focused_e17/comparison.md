# Paired FM-v3 result comparison

Reference: `head_focused_e17`. Candidate: `balanced_accuracy`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.442306 | 0.434776 | -0.00753001 | 12/9/27 |
| accuracy | max | 0.706749 | 0.68571 | -0.0210387 | 6/9/33 |
| macro_f1 | max | 0.412097 | 0.406087 | -0.0060106 | 15/5/28 |
| nll | min | 3.03786 | 3.11168 | +0.0738249 | 6/0/42 |
| multiclass_brier | min | 0.456688 | 0.482057 | +0.025369 | 10/0/38 |
| ece_10 | min | 0.123076 | 0.131711 | +0.00863435 | 23/0/25 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | +0.00257179 | -0.00721154 | +0.00331381 |
| helpdesk | -0.0101538 | -0.0344828 | +0.000280523 |
| receipt | -0.0229123 | -0.0459016 | -0.0190061 |
| roadtraffic100traces | -0.00457702 | -0.00423729 | -0.00531521 |
| sepsis | -0.00198811 | -0.01 | -0.00918694 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1017.41 | 1034.18 | +16.7671 | 19/0/29 |
| rmse_hours | min | 1619.54 | 1627.29 | +7.74888 | 30/0/18 |
| median_absolute_error_hours | min | 631.613 | 654.132 | +22.5191 | 16/0/32 |
| normalized_mae | min | 0.891932 | 0.911817 | +0.0198855 | 19/0/29 |
| mae_skill_vs_median | max | -0.196028 | -0.21981 | -0.0237817 | 19/0/29 |
| rmse_skill_vs_median | max | -0.0515293 | -0.0552446 | -0.00371525 | 30/0/18 |
| d2_absolute_error | max | -0.196028 | -0.21981 | -0.0237817 | 19/0/29 |
| r2 | max | -0.260866 | -0.277168 | -0.0163016 | 30/0/18 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +29.4658 | -8.48488 | +0.00135525 |
| helpdesk | +0.00249642 | -0.000301004 | +0.000935349 |
| receipt | +2.60053 | +2.26463 | -0.0621655 |
| roadtraffic100traces | +45.9983 | +61.1399 | -0.0320847 |
| sepsis | +11.6147 | -5.49672 | +0.00729517 |
