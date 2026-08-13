# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 3.3008090366016734
- Fraction of best validation-loss improvement reached by epoch 3: 0.4278520940657998
- Fraction of best improvement reached by two-thirds: 0.9580850991427755
- Fraction of best invariant-metric improvement reached by epoch 3: 0.43657424477679596
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False

### Regression

- Best validation epoch: 13
- Best validation loss: 1.868399739265442
- Fraction of best validation-loss improvement reached by epoch 3: 0.5920418384318036
- Fraction of best improvement reached by two-thirds: 0.9858126381177069
- Fraction of best invariant-metric improvement reached by epoch 3: 0.5844510757477488
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: None

## Detected bottlenecks

- `medium` `front_loaded_learning` (classification)
- `medium` `front_loaded_learning` (regression)
- `high` `frequent_gradient_clipping`
- `low` `transient_amp_overflow`
- `medium` `large_stagnant_auxiliary_loss` (classification)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `classification_overconfidence` (classification)
- `medium` `large_auxiliary_gradient` (regression)
- `high` `regression_pool_error_concentration` (regression)
