# FM-v3 training diagnostics and overfitting audit

## Purpose

Historical training printed one scalar: total loss averaged over the requested
number of episodes. That was not enough to diagnose a model that learns rapidly
for a few epochs and then stalls or overfits. In particular, the old number:

- mixed classification and regression losses with different meanings;
- divided by scheduled episodes even when an episode was skipped;
- hid every component of the multi-metric regression objective;
- did not expose classifier evidence, selector trust, transform branches, or
  expert-confidence/routing auxiliaries;
- did not measure gradient clipping, AMP overflow, per-component updates, or
  parameter drift; and
- had no case-disjoint source validation signal.

Structured telemetry is now opt-in through `training_diagnostics.enabled`.
When disabled, the historical data split and training step remain unchanged.

## Persisted artifacts

An instrumented checkpoint directory contains:

| File | Contents |
|---|---|
| `training_validation_split.json` | Per-source-log case counts, prefix counts, and SHA-256 of held-out case IDs; raw IDs are not persisted |
| `training_debug_steps.jsonl` | Sampled step records at `step_interval`, including gradients and AMP/clip state |
| `training_debug_epochs.jsonl` | Append-only aggregate record after every epoch |
| `training_debug_summary.json` | Readable consolidated history plus automatic overfitting signals |

All files use `schema_version: 1`. Metric leaves in epoch aggregates contain
`count`, `mean`, `std`, `min`, and `max`. Metrics appear globally and under
`task/...`, `expert/...`, `episode/...`, and `task/.../expert/...` prefixes.

## Collected signals

### Data and schedule

- requested batch size, effective queries, skipped queries, unique cases and
  labels;
- actual successful/skipped/non-finite step counts;
- task and expert step balance;
- base/head learning rates, retrieval k, negative-random fraction, and active
  contrastive/variance/covariance weights.

The console average is now divided by successful steps. The historical
scheduled-step denominator is still recorded explicitly for comparison.

### Classification head

- cross-entropy, confidence loss, routing loss, NCA, separation, contrastive,
  variance, and covariance terms, both raw and weighted;
- accuracy, NLL, true-class probability, maximum probability, normalized
  entropy, and top-two margin;
- local/pool class coverage, local/global evidence, gate values, and prototype
  variance;
- selector log weights, trust entropy, maximum trust, attention entropy, and
  effective support count.

### Regression head and loss

- total/primary/confidence/routing/gate-aux losses;
- raw and normalized contribution of MAE, RMSE, Huber, log-RMSE, relative MAE,
  bias, median AE, and quantile terms;
- the batch target normalizer in hours;
- raw-hour MAE, RMSE, median AE, bias, relative MAE, p90 error, prediction and
  target scales;
- neighbor similarities, uncertainty, selector behavior;
- each transform branch's raw-hour MAE, mean aggregation weight, and aggregate
  branch-weight entropy.

### Optimization and state

- sampled gradient L2/mean/max, finite fraction, and nonzero fraction for
  encoders, embedders, prefix/temporal/task adapters, both selectors, transform
  bank, confidence heads, router, projection, and remaining head parameters;
- total pre-clip gradient norm, clip incidence, AMP scale/overflow, and whether
  an optimizer step applied;
- per-epoch absolute and parameter-relative update norms for the same groups;
- parameter group norms and head-state scalars, including transform powers,
  scales, and prior weights.

## Case-disjoint validation

The diagnostic run reserves a deterministic fraction of whole cases from each
of the eleven source logs. The same case IDs are withheld from classification
and regression, preventing prefixes from one case appearing on both sides.
Validation uses the same task sampler and loss implementation as training but:

- disables gradients;
- fixes its random state and therefore repeats the same episodes each epoch;
- disables encoder/dropout randomness;
- disables regression scale augmentation; and
- retains raw training-time classification logits rather than applying
  inference temperature.

This is a source-log development signal, not the five target-log confirmation.
The target logs remain outside optimization and are evaluated separately after
candidate selection.

For each task, `training_debug_summary.json` reports the epoch with minimum
held-out loss and flags overfitting only when validation has degraded beyond a
configured relative tolerance for the configured patience while training loss
continues to improve. The flag is a screening diagnostic, not an automatic
stopping action.

## Full retraining audit

The initial full-run configuration is
`configs/fmv3/training_debug_full_retrain.yaml`. It initializes the complete
selected epoch-44 architecture from scratch, trains all parameters for the
historical 20-epoch/300-episode schedule, balances the two tasks 50/50, and
holds out 10% of source cases. Both selectors train at strength 1.0; the
classification strength 0.25 remains a later structured-inference calibration.

```bash
python main.py \
  --config configs/fmv3/training_debug_full_retrain.yaml \
  --checkpoint_dir checkpoints/fmv3/training_debug_full_retrain
```

The final training curves, bottleneck diagnosis, regularization choice,
matched rerun, and target-log confirmation will be added here after the full
audit completes.
