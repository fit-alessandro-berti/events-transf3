# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 1.848076506094499
- Fraction of best validation-loss improvement reached by epoch 3: 0.37111950939024935
- Fraction of best improvement reached by two-thirds: 0.9618369897351408
- Fraction of best invariant-metric improvement reached by epoch 3: 0.377158344612264
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False
- Best `accuracy`: 0.3461109074679288 (epoch 19); overfitting: False
- Best `balanced_accuracy`: 0.3400173888287761 (epoch 20); overfitting: False
- Best `macro_f1`: 0.29632288945669477 (epoch 20); overfitting: False
- Best `nll`: 2.0515077198988987 (epoch 17); overfitting: False
- Best `brier`: 0.3878869820724834 (epoch 17); overfitting: False

### Regression

- Best validation epoch: 15
- Best validation loss: 1.6504957242445513
- Fraction of best validation-loss improvement reached by epoch 3: 0.18496558744535316
- Fraction of best improvement reached by two-thirds: 0.9943550817269163
- Fraction of best invariant-metric improvement reached by epoch 3: -0.3949518594727336
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: None
- Best `mae_hours`: 864.5085204731334 (epoch 14); overfitting: False
- Best `rmse_hours`: 1168.5687137950551 (epoch 14); overfitting: False
- Best `r2`: 0.40484061566266144 (epoch 14); overfitting: False

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
