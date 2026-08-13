# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 3.3084068298339844
- Fraction of best improvement reached by two-thirds: 0.9830662810006799
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False

### Regression

- Best validation epoch: 16
- Best validation loss: 1.9258118651129983
- Fraction of best improvement reached by two-thirds: 0.966180301140999
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
