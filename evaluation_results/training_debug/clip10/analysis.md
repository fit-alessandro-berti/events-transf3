# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 3.243730241602117
- Fraction of best validation-loss improvement reached by epoch 3: 0.4729248930380975
- Fraction of best improvement reached by two-thirds: 0.9714124604954297
- Fraction of best invariant-metric improvement reached by epoch 3: 0.47497066851208913
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False

### Regression

- Best validation epoch: 19
- Best validation loss: 1.9323106353933162
- Fraction of best validation-loss improvement reached by epoch 3: 0.3296629170858139
- Fraction of best improvement reached by two-thirds: 0.9755736129929693
- Fraction of best invariant-metric improvement reached by epoch 3: 0.3778984549520119
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: None

## Detected bottlenecks

- `medium` `front_loaded_learning` (classification)
- `medium` `front_loaded_learning` (regression)
- `low` `transient_amp_overflow`
- `medium` `large_stagnant_auxiliary_loss` (classification)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `near_uniform_example_selector` (classification)
- `medium` `classification_overconfidence` (classification)
- `medium` `large_auxiliary_gradient` (regression)
- `high` `regression_pool_error_concentration` (regression)
