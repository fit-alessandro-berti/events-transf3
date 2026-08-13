# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 3.2659436139193447
- Fraction of best validation-loss improvement reached by epoch 3: 0.4731096434732579
- Fraction of best improvement reached by two-thirds: 0.9798331101467892
- Fraction of best invariant-metric improvement reached by epoch 3: 0.48515356040144586
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False

### Regression

- Best validation epoch: 20
- Best validation loss: 1.9218529571186413
- Fraction of best validation-loss improvement reached by epoch 3: 0.5431811616267842
- Fraction of best improvement reached by two-thirds: 0.9810140653864808
- Fraction of best invariant-metric improvement reached by epoch 3: 0.5425454732780441
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
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `near_uniform_example_selector` (classification)
- `medium` `near_uniform_regression_branch_mixture` (regression)
- `medium` `classification_overconfidence` (classification)
- `medium` `large_auxiliary_gradient` (regression)
- `high` `regression_pool_error_concentration` (regression)
