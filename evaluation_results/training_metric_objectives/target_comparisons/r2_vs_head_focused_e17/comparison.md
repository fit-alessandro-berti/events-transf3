# Paired FM-v3 result comparison

Reference: `head_focused_e17`. Candidate: `r2_e19`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.442306 | 0.433021 | -0.00928502 | 10/7/31 |
| accuracy | max | 0.706749 | 0.692464 | -0.0142842 | 6/12/30 |
| macro_f1 | max | 0.412097 | 0.403639 | -0.00845858 | 17/3/28 |
| nll | min | 3.03786 | 3.09734 | +0.0594761 | 7/0/41 |
| multiclass_brier | min | 0.456688 | 0.471884 | +0.0151967 | 8/0/40 |
| ece_10 | min | 0.123076 | 0.125852 | +0.00277566 | 20/0/28 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00353318 | -0.00913462 | -0.00270291 |
| helpdesk | -0.0043199 | -0.00758621 | -0.000526918 |
| receipt | -0.0174402 | -0.0258197 | -0.0127269 |
| roadtraffic100traces | -0.0243056 | -0.0317797 | -0.0267972 |
| sepsis | +0.000169636 | -0.0006 | -0.00320662 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1017.41 | 1058.39 | +40.9847 | 12/0/36 |
| rmse_hours | min | 1619.54 | 1634.43 | +14.8923 | 28/0/20 |
| median_absolute_error_hours | min | 631.613 | 675.666 | +44.0529 | 12/0/36 |
| normalized_mae | min | 0.891932 | 0.947744 | +0.055812 | 12/0/36 |
| mae_skill_vs_median | max | -0.196028 | -0.260192 | -0.064164 | 12/0/36 |
| rmse_skill_vs_median | max | -0.0515293 | -0.0553361 | -0.00380683 | 28/0/20 |
| d2_absolute_error | max | -0.196028 | -0.260192 | -0.064164 | 12/0/36 |
| r2 | max | -0.260866 | -0.287175 | -0.0263092 | 28/0/20 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +103.721 | +15.4234 | -0.0629593 |
| helpdesk | +0.00457576 | -0.000621082 | +0.00413634 |
| receipt | +6.73278 | +0.959938 | -0.0493376 |
| roadtraffic100traces | +63.9771 | +89.5707 | -0.0499518 |
| sepsis | +35.0864 | -16.5561 | +0.0218379 |
