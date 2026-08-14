# Paired FM-v3 result comparison

Reference: `selected_e44`. Candidate: `accuracy`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.443837 | 0.437283 | -0.00655463 | 15/7/26 |
| accuracy | max | 0.708084 | 0.702591 | -0.00549247 | 10/15/23 |
| macro_f1 | max | 0.413911 | 0.408368 | -0.00554315 | 15/1/32 |
| nll | min | 3.02119 | 3.0863 | +0.0651135 | 9/0/39 |
| multiclass_brier | min | 0.452898 | 0.465863 | +0.0129643 | 17/0/31 |
| ece_10 | min | 0.121629 | 0.123571 | +0.00194206 | 25/0/23 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00251855 | -0.00480769 | -0.00165463 |
| helpdesk | -0.00137729 | -0.00965517 | +0.00351037 |
| receipt | -0.0164067 | +0.00327869 | -0.0173064 |
| roadtraffic100traces | -0.00694444 | -0.00847458 | -0.00641049 |
| sepsis | -0.00560412 | -0.0084 | -0.00602809 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1010.55 | 1058.18 | +47.633 | 20/0/28 |
| rmse_hours | min | 1591.74 | 1652.66 | +60.9197 | 19/0/29 |
| median_absolute_error_hours | min | 630.614 | 683.74 | +53.1269 | 23/0/25 |
| normalized_mae | min | 0.872341 | 0.930674 | +0.0583325 | 20/0/28 |
| mae_skill_vs_median | max | -0.178455 | -0.242731 | -0.0642769 | 20/0/28 |
| rmse_skill_vs_median | max | -0.0356311 | -0.0694081 | -0.033777 | 19/0/29 |
| d2_absolute_error | max | -0.178455 | -0.242731 | -0.0642769 | 20/0/28 |
| r2 | max | -0.210744 | -0.337834 | -0.12709 | 19/0/29 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +246.809 | +304.149 | -0.521923 |
| helpdesk | +0.000108183 | -0.00337784 | +0.0193428 |
| receipt | +4.18808 | +5.3473 | -0.119669 |
| roadtraffic100traces | -37.79 | -17.5289 | +0.0101507 |
| sepsis | +7.87341 | -3.05469 | +0.00409532 |
