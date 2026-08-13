# Paired FM-v3 result comparison

Reference: `baseline_e16`. Candidate: `head_focused_e17`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.439371 | 0.442306 | +0.00293443 | 27/8/13 |
| accuracy | max | 0.70222 | 0.706749 | +0.00452855 | 21/12/15 |
| macro_f1 | max | 0.41234 | 0.412097 | -0.000243068 | 25/6/17 |
| nll | min | 3.04972 | 3.03786 | -0.0118611 | 30/0/18 |
| multiclass_brier | min | 0.45981 | 0.456688 | -0.00312268 | 31/0/17 |
| ece_10 | min | 0.123013 | 0.123076 | +6.33705e-05 | 21/0/27 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | +0.00173077 | +0.00384615 | -0.00593116 |
| helpdesk | +0.0125287 | +0.00965517 | +0.0142302 |
| receipt | -0.00446571 | +0.0102459 | -0.0140685 |
| roadtraffic100traces | +0.00323548 | +0.00423729 | +0.00431277 |
| sepsis | +0.00170314 | -0.0054 | +0.00115251 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1065.39 | 1017.41 | -47.9833 | 31/0/17 |
| rmse_hours | min | 1656.17 | 1619.54 | -36.6244 | 20/0/28 |
| median_absolute_error_hours | min | 663.315 | 631.613 | -31.7022 | 35/0/13 |
| normalized_mae | min | 0.944673 | 0.891932 | -0.0527412 | 31/0/17 |
| mae_skill_vs_median | max | -0.257863 | -0.196028 | +0.061835 | 31/0/17 |
| rmse_skill_vs_median | max | -0.0679785 | -0.0515293 | +0.0164492 | 20/0/28 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours |
|---|---:|---:|
| billing | -204.893 | -184.637 |
| helpdesk | -0.00491292 | -0.00449932 |
| receipt | -2.66672 | +3.15781 |
| roadtraffic100traces | +5.28574 | +4.37104 |
| sepsis | -26.9839 | +2.18999 |
