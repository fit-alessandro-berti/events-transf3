# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `r2_e19`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.433021 | -0.0108164 | 11/6/31 |
| accuracy | max | 0.708084 | 0.692464 | -0.0156195 | 7/10/31 |
| macro_f1 | max | 0.413911 | 0.403639 | -0.0102719 | 16/2/30 |
| nll | min | 3.02119 | 3.09734 | +0.0761479 | 9/0/39 |
| multiclass_brier | min | 0.452898 | 0.471884 | +0.0189859 | 9/0/39 |
| ece_10 | min | 0.121629 | 0.125852 | +0.00422299 | 22/0/26 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.0061009 | -0.00721154 | -0.00647241 |
| helpdesk | -0.00241026 | -0.0144828 | +0.00393143 |
| receipt | -0.0155534 | -0.0110656 | -0.0143282 |
| roadtraffic100traces | -0.0277778 | -0.0360169 | -0.0301735 |
| sepsis | -0.00563212 | -0.0134 | -0.00829712 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1058.39 | +47.8448 | 16/0/32 |
| rmse_hours | min | 1591.74 | 1634.43 | +42.6961 | 29/0/19 |
| median_absolute_error_hours | min | 630.614 | 675.666 | +45.0523 | 15/0/33 |
| normalized_mae | min | 0.872341 | 0.947744 | +0.0754028 | 16/0/32 |
| mae_skill_vs_median | max | -0.178455 | -0.260192 | -0.0817374 | 16/0/32 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0553361 | -0.0197051 | 29/0/19 |
| d2_absolute_error | max | -0.178455 | -0.260192 | -0.0817374 | 16/0/32 |
| r2 | max | -0.210744 | -0.287175 | -0.0764309 | 29/0/19 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +207.62 | +207.168 | -0.318761 |
| helpdesk | +0.000583956 | -0.00585089 | +0.0361113 |
| receipt | +8.35746 | +3.39927 | -0.0956906 |
| roadtraffic100traces | -21.9927 | +9.79386 | -0.00788175 |
| sepsis | +31.2707 | -13.4556 | +0.0177774 |
