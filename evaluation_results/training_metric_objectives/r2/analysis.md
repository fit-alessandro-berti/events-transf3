# Training debug analysis

Epochs analyzed: 20.

## Learning phases

### Classification

- Best validation epoch: 20
- Best validation loss: 1.781649719585072
- Fraction of best validation-loss improvement reached by epoch 3: 0.3415767922350451
- Fraction of best improvement reached by two-thirds: 0.9444531436231783
- Fraction of best invariant-metric improvement reached by epoch 3: 0.4144504634704669
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: False
- Best `accuracy`: 0.35989217324690387 (epoch 18); overfitting: False
- Best `balanced_accuracy`: 0.35395433245734736 (epoch 19); overfitting: False
- Best `macro_f1`: 0.3109398998997428 (epoch 19); overfitting: False
- Best `nll`: 2.0276465895797133 (epoch 19); overfitting: False
- Best `brier`: 0.38203639401630923 (epoch 20); overfitting: False

### Regression

- Best validation epoch: 15
- Best validation loss: 1.3694312464107166
- Fraction of best validation-loss improvement reached by epoch 3: 0.609310290903193
- Fraction of best improvement reached by two-thirds: 0.9562028847308361
- Fraction of best invariant-metric improvement reached by epoch 3: -0.0687716864933403
- Automatic overfitting signal: False
- Invariant-metric overfitting signal: False
- Invariant-metric generalization-gap signal: False
- Decision-metric overfitting signal: None
- Best `mae_hours`: 861.5285396575928 (epoch 16); overfitting: False
- Best `rmse_hours`: 1120.9409708543258 (epoch 17); overfitting: False
- Best `r2`: 0.42707542939619586 (epoch 15); overfitting: False

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
