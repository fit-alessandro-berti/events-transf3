# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `equilibrated`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.43743 | -0.00640743 | 11/9/28 |
| accuracy | max | 0.708084 | 0.699246 | -0.00883787 | 9/11/28 |
| macro_f1 | max | 0.413911 | 0.40843 | -0.00548066 | 17/3/28 |
| nll | min | 3.02119 | 3.08854 | +0.0673563 | 11/0/37 |
| multiclass_brier | min | 0.452898 | 0.468698 | +0.0157996 | 13/0/35 |
| ece_10 | min | 0.121629 | 0.124678 | +0.00304937 | 22/0/26 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00282036 | -0.00528846 | -0.00330245 |
| helpdesk | -0.000724054 | -0.0117241 | +0.0048882 |
| receipt | -0.0128836 | -0.00122951 | -0.0111881 |
| roadtraffic100traces | -0.00694444 | -0.00847458 | -0.00642939 |
| sepsis | -0.00877211 | -0.0174 | -0.0115613 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1048.69 | +38.1427 | 21/0/27 |
| rmse_hours | min | 1591.74 | 1644.03 | +52.2911 | 20/0/28 |
| median_absolute_error_hours | min | 630.614 | 674.461 | +43.8471 | 20/0/28 |
| normalized_mae | min | 0.872341 | 0.922772 | +0.0504311 | 21/0/27 |
| mae_skill_vs_median | max | -0.178455 | -0.232636 | -0.0541816 | 21/0/27 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0634843 | -0.0278532 | 20/0/28 |
| d2_absolute_error | max | -0.178455 | -0.232636 | -0.0541816 | 21/0/27 |
| r2 | max | -0.210744 | -0.312695 | -0.101951 | 20/0/28 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +208.492 | +269.587 | -0.428479 |
| helpdesk | -0.00146462 | -0.00523797 | +0.0309345 |
| receipt | +3.91262 | +4.61249 | -0.105876 |
| roadtraffic100traces | -46.5698 | -25.5857 | +0.0129899 |
| sepsis | +7.9381 | -2.72854 | +0.00366265 |
