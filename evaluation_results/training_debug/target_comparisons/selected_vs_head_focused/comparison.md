# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `head_focused_e17`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.442306 | -0.00153143 | 19/12/17 |
| accuracy | max | 0.708084 | 0.706749 | -0.00133525 | 15/14/19 |
| macro_f1 | max | 0.413911 | 0.412097 | -0.00181331 | 25/3/20 |
| nll | min | 3.02119 | 3.03786 | +0.0166718 | 23/0/25 |
| multiclass_brier | min | 0.452898 | 0.456688 | +0.00378926 | 19/0/29 |
| ece_10 | min | 0.121629 | 0.123076 | +0.00144733 | 21/0/27 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00256772 | +0.00192308 | -0.0037695 |
| helpdesk | +0.00190965 | -0.00689655 | +0.00445834 |
| receipt | +0.00188676 | +0.0147541 | -0.00160125 |
| roadtraffic100traces | -0.00347222 | -0.00423729 | -0.00337625 |
| sepsis | -0.00580176 | -0.0128 | -0.0050905 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1017.41 | +6.8601 | 24/0/24 |
| rmse_hours | min | 1591.74 | 1619.54 | +27.8037 | 18/0/30 |
| median_absolute_error_hours | min | 630.614 | 631.613 | +0.999397 | 29/0/19 |
| normalized_mae | min | 0.872341 | 0.891932 | +0.0195908 | 24/0/24 |
| mae_skill_vs_median | max | -0.178455 | -0.196028 | -0.0175734 | 24/0/24 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0515293 | -0.0158983 | 18/0/30 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours |
|---|---:|---:|
| billing | +103.899 | +191.745 |
| helpdesk | -0.0039918 | -0.00522981 |
| receipt | +1.62468 | +2.43933 |
| roadtraffic100traces | -85.9698 | -79.7769 |
| sepsis | -3.81566 | +3.10056 |
