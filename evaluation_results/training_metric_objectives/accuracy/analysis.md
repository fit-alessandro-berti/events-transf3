# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 1.8476837981830945
- Fraction of best validation-loss improvement reached by epoch 3: 0.3786865302818795
- Fraction of best improvement reached by two-thirds: 0.9557435753981098
- Fraction of best invariant-metric improvement reached by epoch 3: 0.4223050868032206
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False
- Best `accuracy`: 0.3356986262581565 (epoch 20); overfitting: False
- Best `balanced_accuracy`: 0.32536590878259053 (epoch 20); overfitting: False
- Best `macro_f1`: 0.2822770125825297 (epoch 20); overfitting: False
- Best `nll`: 2.1164309803612045 (epoch 18); overfitting: False
- Best `brier`: 0.3943152278661728 (epoch 18); overfitting: False

### Regression

- Best validation epoch: 20
- Best validation loss: 1.6489929827776821
- Fraction of best validation-loss improvement reached by epoch 3: 0.1166725712740296
- Fraction of best improvement reached by two-thirds: 0.9964127318952866
- Fraction of best invariant-metric improvement reached by epoch 3: 0.16978172493837693
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: None
- Best `mae_hours`: 864.7604720375755 (epoch 18); overfitting: False
- Best `rmse_hours`: 1175.4049797058105 (epoch 8); overfitting: False
- Best `r2`: 0.4035529819401828 (epoch 18); overfitting: False

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
