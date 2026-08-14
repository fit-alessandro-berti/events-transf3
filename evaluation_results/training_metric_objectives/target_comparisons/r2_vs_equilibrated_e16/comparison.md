# Paired FM-v3 result comparison

Reference: `equilibrated_e16`. Candidate: `r2_e19`.

## Classification (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| balanced_accuracy | max | 0.43743 | 0.433021 | -0.00440902 | 16/15/17 |
| accuracy | max | 0.699246 | 0.692464 | -0.0067816 | 14/18/16 |
| macro_f1 | max | 0.40843 | 0.403639 | -0.00479123 | 16/8/24 |
| nll | min | 3.08854 | 3.09734 | +0.00879164 | 22/0/26 |
| multiclass_brier | min | 0.468698 | 0.471884 | +0.00318629 | 28/0/20 |
| ece_10 | min | 0.124678 | 0.125852 | +0.00117363 | 24/0/24 |

### Per-log candidate-minus-reference deltas

| Log | balanced_accuracy | accuracy | macro_f1 |
|---|---:|---:|---:|
| billing | -0.00328054 | -0.00192308 | -0.00316996 |
| helpdesk | -0.0016862 | -0.00275862 | -0.000956772 |
| receipt | -0.00266986 | -0.00983607 | -0.00314008 |
| roadtraffic100traces | -0.0208333 | -0.0275424 | -0.0237441 |
| sepsis | +0.00313999 | +0.004 | +0.00326416 |

## Regression (48 paired rows)

| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| mae_hours | min | 1048.69 | 1058.39 | +9.70209 | 12/0/36 |
| rmse_hours | min | 1644.03 | 1634.43 | -9.59503 | 29/0/19 |
| median_absolute_error_hours | min | 674.461 | 675.666 | +1.20514 | 14/0/34 |
| normalized_mae | min | 0.922772 | 0.947744 | +0.0249717 | 12/0/36 |
| mae_skill_vs_median | max | -0.232636 | -0.260192 | -0.0275557 | 12/0/36 |
| rmse_skill_vs_median | max | -0.0634843 | -0.0553361 | +0.00814812 | 29/0/19 |
| d2_absolute_error | max | -0.232636 | -0.260192 | -0.0275557 | 12/0/36 |
| r2 | max | -0.312695 | -0.287175 | +0.0255202 | 29/0/19 |

### Per-log candidate-minus-reference deltas

| Log | mae_hours | rmse_hours | r2 |
|---|---:|---:|---:|
| billing | -0.871159 | -62.4189 | +0.109718 |
| helpdesk | +0.00204858 | -0.000612921 | +0.00517687 |
| receipt | +4.44484 | -1.21323 | +0.0101851 |
| roadtraffic100traces | +24.5771 | +35.3796 | -0.0208716 |
| sepsis | +23.3326 | -10.727 | +0.0141147 |
