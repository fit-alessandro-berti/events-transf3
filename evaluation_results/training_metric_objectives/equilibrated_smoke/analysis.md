# Training debug analysis

Epochs analyzed: 1.

## Learning phases

### Classification

- Best validation epoch: 1
- Best validation loss: 2.405069416219538
- Fraction of best validation-loss improvement reached by epoch 3: None
- Fraction of best improvement reached by two-thirds: None
- Fraction of best invariant-metric improvement reached by epoch 3: None
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False

### Regression

- Best validation epoch: 1
- Best validation loss: 1.861283454028043
- Fraction of best validation-loss improvement reached by epoch 3: None
- Fraction of best improvement reached by two-thirds: None
- Fraction of best invariant-metric improvement reached by epoch 3: None
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: None

## Detected bottlenecks

- `high` `frequent_gradient_clipping`
- `medium` `large_auxiliary_gradient` (classification)
- `medium` `large_auxiliary_gradient` (classification)
- `high` `regression_pool_error_concentration` (regression)
