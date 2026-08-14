# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 1.7831551378423518
- Fraction of best validation-loss improvement reached by epoch 3: 0.3666030841055635
- Fraction of best improvement reached by two-thirds: 0.9521550891770186
- Fraction of best invariant-metric improvement reached by epoch 3: 0.47690697730933934
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False
- Best `accuracy`: 0.3420437249270352 (epoch 19); overfitting: False
- Best `balanced_accuracy`: 0.33797370811755006 (epoch 19); overfitting: False
- Best `macro_f1`: 0.2949953250248324 (epoch 19); overfitting: False
- Best `nll`: 2.0361798258835875 (epoch 20); overfitting: False
- Best `brier`: 0.3845420926809311 (epoch 18); overfitting: False

### Regression

- Best validation epoch: 15
- Best validation loss: 1.6515414931557395
- Fraction of best validation-loss improvement reached by epoch 3: 0.09815024403843012
- Fraction of best improvement reached by two-thirds: 0.9823172421082182
- Fraction of best invariant-metric improvement reached by epoch 3: 0.16216008432055967
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: None
- Best `mae_hours`: 866.31227684021 (epoch 13); overfitting: False
- Best `rmse_hours`: 1169.4604244232178 (epoch 13); overfitting: False
- Best `r2`: 0.40539791909131134 (epoch 16); overfitting: False

## Detected bottlenecks

- `medium` `front_loaded_learning` (classification)
- `medium` `front_loaded_learning` (regression)
- `high` `frequent_gradient_clipping`
- `low` `transient_amp_overflow`
- `medium` `large_stagnant_auxiliary_loss` (classification)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `classification_overconfidence` (classification)
- `medium` `large_auxiliary_gradient` (classification)
- `high` `regression_pool_error_concentration` (regression)
