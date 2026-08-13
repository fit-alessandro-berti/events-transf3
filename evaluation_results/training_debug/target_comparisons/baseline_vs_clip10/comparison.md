# Paired FM-v3 result comparison

Reference: `baseline_e16`. Candidate: `clip10_e6`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.439371 | 0.435424 | -0.00394759 | 15/11/22 |
| accuracy | max | 0.70222 | 0.693272 | -0.00894795 | 15/13/20 |
| macro_f1 | max | 0.41234 | 0.405308 | -0.00703273 | 16/7/25 |
| nll | min | 3.04972 | 3.05155 | +0.00183051 | 19/0/29 |
| multiclass_brier | min | 0.45981 | 0.471669 | +0.0118585 | 16/0/32 |
| ece_10 | min | 0.123013 | 0.133465 | +0.0104521 | 21/0/27 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00654208 | -0.00384615 | -0.00773116 |
| helpdesk | +0.00524786 | +0.00275862 | +0.00704188 |
| receipt | -0.0174769 | -0.0438525 | -0.034376 |
| roadtraffic100traces | +0.00323548 | +0.00423729 | +0.00431277 |
| sepsis | -0.00276575 | -0.0014 | -0.00214205 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1065.39 | 1076.19 | +10.7995 | 17/0/31 |
| rmse_hours | min | 1656.17 | 1660.96 | +4.79435 | 24/0/24 |
| median_absolute_error_hours | min | 663.315 | 678.463 | +15.148 | 15/0/33 |
| normalized_mae | min | 0.944673 | 0.9597 | +0.0150273 | 17/0/31 |
| mae_skill_vs_median | max | -0.257863 | -0.271785 | -0.0139219 | 17/0/31 |
| rmse_skill_vs_median | max | -0.0679785 | -0.0650527 | +0.00292582 | 24/0/24 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours |
|---|---:|---:|
| billing | -16.0759 | -40.2098 |
| helpdesk | -0.00496256 | -0.00645969 |
| receipt | +3.75745 | +1.89238 |
| roadtraffic100traces | +67.9216 | +78.4957 |
| sepsis | +9.82367 | -1.45983 |
