# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 1.771113233132796
- Fraction of best validation-loss improvement reached by epoch 3: 0.3463803191476251
- Fraction of best improvement reached by two-thirds: 0.9352961709840871
- Fraction of best invariant-metric improvement reached by epoch 3: 0.43525335202214277
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False
- Best `accuracy`: 0.3484368920326233 (epoch 19); overfitting: False
- Best `balanced_accuracy`: 0.34563519196076825 (epoch 19); overfitting: False
- Best `macro_f1`: 0.3003355452621525 (epoch 20); overfitting: False
- Best `nll`: 2.000054587046541 (epoch 20); overfitting: False
- Best `brier`: 0.38187013363296335 (epoch 20); overfitting: False

### Regression

- Best validation epoch: 11
- Best validation loss: 1.8121876824985852
- Fraction of best validation-loss improvement reached by epoch 3: 0.6618713663755139
- Fraction of best improvement reached by two-thirds: 0.9697298853568921
- Fraction of best invariant-metric improvement reached by epoch 3: 0.40697401246863985
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: True
- Invariant-metric generalization-gap signal: True
- Decision-metric overfitting signal: None
- Best `mae_hours`: 874.8233002749356 (epoch 11); overfitting: True
- Best `rmse_hours`: 1166.5505532351408 (epoch 11); overfitting: True
- Best `r2`: 0.3826504349708557 (epoch 11); overfitting: True

## Detected bottlenecks

- `medium` `front_loaded_learning` (classification)
- `high` `overfitting` (regression)
- `medium` `front_loaded_learning` (regression)
- `high` `frequent_gradient_clipping`
- `low` `transient_amp_overflow`
- `medium` `large_stagnant_auxiliary_loss` (classification)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `large_stagnant_auxiliary_loss` (regression)
- `medium` `classification_overconfidence` (classification)
- `medium` `large_auxiliary_gradient` (classification)
- `high` `regression_pool_error_concentration` (regression)
