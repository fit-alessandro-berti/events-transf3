# Paired FM-v3 result comparison

Reference: `head_focused_e17`. Candidate: `accuracy`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.442306 | 0.437283 | -0.0050232 | 11/10/27 |
| accuracy | max | 0.706749 | 0.702591 | -0.00415721 | 12/14/22 |
| macro_f1 | max | 0.412097 | 0.408368 | -0.00372984 | 17/7/24 |
| nll | min | 3.03786 | 3.0863 | +0.0484417 | 11/0/37 |
| multiclass_brier | min | 0.456688 | 0.465863 | +0.00917499 | 20/0/28 |
| ece_10 | min | 0.123076 | 0.123571 | +0.000494727 | 20/0/28 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | +4.91704e-05 | -0.00673077 | +0.00211486 |
| helpdesk | -0.00328694 | -0.00275862 | -0.000947974 |
| receipt | -0.0182935 | -0.0114754 | -0.0157051 |
| roadtraffic100traces | -0.00347222 | -0.00423729 | -0.00303424 |
| sepsis | +0.000197634 | +0.0044 | -0.000937586 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1017.41 | 1058.18 | +40.7729 | 21/0/27 |
| rmse_hours | min | 1619.54 | 1652.66 | +33.116 | 29/0/19 |
| median_absolute_error_hours | min | 631.613 | 683.74 | +52.1275 | 15/0/33 |
| normalized_mae | min | 0.891932 | 0.930674 | +0.0387418 | 21/0/27 |
| mae_skill_vs_median | max | -0.196028 | -0.242731 | -0.0467035 | 21/0/27 |
| rmse_skill_vs_median | max | -0.0515293 | -0.0694081 | -0.0178787 | 29/0/19 |
| d2_absolute_error | max | -0.196028 | -0.242731 | -0.0467035 | 21/0/27 |
| r2 | max | -0.260866 | -0.337834 | -0.0769685 | 29/0/19 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +142.91 | +112.404 | -0.266121 |
| helpdesk | +0.00409998 | +0.00185197 | -0.0126322 |
| receipt | +2.56339 | +2.90797 | -0.073316 |
| roadtraffic100traces | +48.1798 | +62.2479 | -0.0319193 |
| sepsis | +11.6891 | -6.15524 | +0.0081558 |
