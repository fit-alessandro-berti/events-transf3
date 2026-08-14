# Paired FM-v3 result comparison

Reference: `mae_e14`. Candidate: `r2_e19`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.434161 | 0.433021 | -0.00114059 | 24/4/20 |
| accuracy | max | 0.693994 | 0.692464 | -0.00152993 | 23/7/18 |
| macro_f1 | max | 0.40507 | 0.403639 | -0.00143076 | 26/3/19 |
| nll | min | 3.09964 | 3.09734 | -0.00230127 | 27/0/21 |
| multiclass_brier | min | 0.474261 | 0.471884 | -0.00237653 | 30/0/18 |
| ece_10 | min | 0.133443 | 0.125852 | -0.00759103 | 33/0/15 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00203575 | -0.00336538 | -0.0016442 |
| helpdesk | +0.00368742 | -0.00206897 | +0.00240555 |
| receipt | +0.00150718 | +0.0135246 | +0.00294697 |
| roadtraffic100traces | -0.0197285 | -0.0275424 | -0.0221103 |
| sepsis | +0.00714916 | +0.0066 | +0.00711225 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1013.5 | 1058.39 | +44.8946 | 12/0/36 |
| rmse_hours | min | 1607.32 | 1634.43 | +27.116 | 27/0/21 |
| median_absolute_error_hours | min | 647.564 | 675.666 | +28.1017 | 13/0/35 |
| normalized_mae | min | 0.896862 | 0.947744 | +0.0508815 | 12/0/36 |
| mae_skill_vs_median | max | -0.199372 | -0.260192 | -0.0608204 | 12/0/36 |
| rmse_skill_vs_median | max | -0.0460659 | -0.0553361 | -0.00927024 | 27/0/21 |
| d2_absolute_error | max | -0.199372 | -0.260192 | -0.0608204 | 12/0/36 |
| r2 | max | -0.251366 | -0.287175 | -0.0358093 | 27/0/21 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | +112.911 | +52.2816 | -0.117419 |
| helpdesk | +0.00621597 | +0.00231599 | -0.0137956 |
| receipt | +4.65449 | -0.190316 | -0.00768949 |
| roadtraffic100traces | +73.3992 | +113.053 | -0.0615674 |
| sepsis | +39.2032 | -12.3788 | +0.0162739 |
