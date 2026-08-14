# Paired FM-v3 result comparison

Reference: `equilibrated_e16`. Candidate: `accuracy`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.43743 | 0.437283 | -0.000147201 | 15/15/18 |
| accuracy | max | 0.699246 | 0.702591 | +0.0033454 | 17/19/12 |
| macro_f1 | max | 0.40843 | 0.408368 | -6.24922e-05 | 20/9/19 |
| nll | min | 3.08854 | 3.0863 | -0.00224274 | 30/0/18 |
| multiclass_brier | min | 0.468698 | 0.465863 | -0.00283538 | 35/0/13 |
| ece_10 | min | 0.124678 | 0.123571 | -0.0011073 | 26/0/22 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | +0.00030181 | +0.000480769 | +0.00164782 |
| helpdesk | -0.000653236 | +0.00206897 | -0.00137783 |
| receipt | -0.00352313 | +0.0045082 | -0.00611826 |
| roadtraffic100traces | +0 | +0 | +1.88922e-05 |
| sepsis | +0.00316799 | +0.009 | +0.00553319 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1048.69 | 1058.18 | +9.49033 | 18/0/30 |
| rmse_hours | min | 1644.03 | 1652.66 | +8.62862 | 21/0/27 |
| median_absolute_error_hours | min | 674.461 | 683.74 | +9.27973 | 22/0/26 |
| normalized_mae | min | 0.922772 | 0.930674 | +0.00790147 | 18/0/30 |
| mae_skill_vs_median | max | -0.232636 | -0.242731 | -0.0100953 | 18/0/30 |
| rmse_skill_vs_median | max | -0.0634843 | -0.0694081 | -0.0059238 | 21/0/27 |
| d2_absolute_error | max | -0.232636 | -0.242731 | -0.0100953 | 18/0/30 |
| r2 | max | -0.312695 | -0.337834 | -0.0251391 | 21/0/27 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +38.3174 | +34.5614 | -0.0934439 |
| helpdesk | +0.00157281 | +0.00186013 | -0.0115917 |
| receipt | +0.275453 | +0.734809 | -0.0137933 |
| roadtraffic100traces | +8.7798 | +8.05678 | -0.00283915 |
| sepsis | -0.0646928 | -0.326147 | +0.000432669 |
