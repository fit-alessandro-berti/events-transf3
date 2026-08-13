# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 19
- Best validation loss: 3.479942126707597
- Fraction of best validation-loss improvement reached by epoch 3: 0.47078921899224635
- Fraction of best improvement reached by two-thirds: 0.9764899304333008
- Fraction of best invariant-metric improvement reached by epoch 3: 0.4796713455573381
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False

### Regression

- Best validation epoch: 15
- Best validation loss: 1.9368098432367498
- Fraction of best validation-loss improvement reached by epoch 3: 0.27513798288836605
- Fraction of best improvement reached by two-thirds: 0.9885460866218806
- Fraction of best invariant-metric improvement reached by epoch 3: 0.2695787702639317
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
