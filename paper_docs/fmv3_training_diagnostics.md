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
`task/...`, `expert/...`, `pool/...`, `episode/...`,
`task/.../expert/...`, and `task/.../pool/...` prefixes. The numeric pool index
maps to the source log in `training_validation_split.json`; this makes an
extreme process scale or error distribution visible instead of hiding it in an
eleven-log mean. Sampled step records also persist `pool` directly.

`analyze_training_debug.py` converts the verbose history into:

| File | Contents |
|---|---|
| `analysis.json` | Best epochs, generalization signals, bottlenecks, head/loss endpoints, gradient groups, and latest per-pool results |
| `analysis.md` | Short human-readable decision summary |
| `curves.csv` | Core task loss, accuracy/MAE, clipping, update success, and LR curves |
| `loss_curves.csv` | Long-form primary and weighted auxiliary/component losses |
| `head_curves.csv` | Long-form classifier, regressor, selector, branch, and calibration behavior |
| `pool_curves.csv` | Held-out source-log loss and task metrics when pool-level telemetry is available |
| `loss_gradient_curves.csv` | Sampled gradient L2 attributable to each differentiable loss component when enabled |

The analyzer automatically locates the validation manifest next to the summary:

```bash
python analyze_training_debug.py \
  --summary checkpoints/fmv3/training_debug_full_retrain/training_debug_summary.json \
  --output_dir evaluation_results/training_debug/full_retrain_baseline
```

## Collected signals

### Data and schedule

- requested batch size, effective queries, skipped queries, unique cases and
  labels;
- actual finite-loss, applied-optimizer, AMP-overflow, skipped, and non-finite
  step counts;
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
- optional sampled per-loss gradient L2 attribution for the primary objective,
  each weighted regression term, separation, confidence, gate, and routing
  auxiliaries (`loss_gradient_interval`; zero disables the extra autograd work);
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

The offline analyzer applies the same conservative rule to invariant output
metrics as well: raw classifier NLL and raw-hour regression MAE. This matters
when a reweighted auxiliary/composite objective improves while the deployed
error measure degrades. Both the objective and invariant flags remain in the
analysis output. Classification accuracy has its own maximize-oriented rule,
so falling validation accuracy can still flag memorization while NLL improves.
The analyzer also emits a lower-severity generalization-gap signal after the
same patience when the training invariant improves by at least 10% while the
held-out invariant fails to improve, even if degradation has not crossed the
strict 2% overfitting tolerance.

The analyzer additionally screens for large auxiliary losses that stay nearly
constant, selectors whose effective support remains indistinguishable from
uniform retrieval, classifier overconfidence, frequent gradient clipping, and
AMP overflow. An overflow confined to AMP scale warm-up is reported as a
low-severity transient event; only a mean overflow rate of at least 1% or an
overflow still present in the last epoch is high severity. These are
hypotheses for a matched intervention, not proof that a component should be
removed: scalar loss size and gradient influence are reported separately for
that reason. In particular, the analyzer flags an auxiliary component when its
mean isolated gradient norm reaches at least half the primary-loss gradient
norm, even if its scalar contribution looks small. It also flags a regression
transform mixture whose normalized weight entropy remains at least 0.99 while
the best and worst branch MAE differ by at least 5%; branch MAE and mean weight
are retained separately so a uniform mixture of genuinely equivalent branches
is not mislabeled. Finally, a source pool whose raw-hour MAE is at least 10×
the median pool is reported explicitly with its share of the summed per-pool
MAE. This prevents one long-duration process from silently defining the
aggregate regression conclusion.

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

The baseline completed all 20 epochs (6,000 sampled episodes) from a new random
initialization. Its case-held-out results separate rapid early learning from
strict overfitting:

| Invariant/decision metric | Best | Last (epoch 20) |
|---|---:|---:|
| Classification NLL | **1.908373 (e20)** | **1.908373** |
| Classification accuracy | **0.305672 (e16)** | 0.300665 |
| Regression MAE (hours) | **837.320 (e13)** | 846.715 |
| Regression RMSE (hours) | **1,129.434 (e13)** | 1,137.997 |

The composite validation losses are front-loaded: 98.3% of the eventual
classification improvement and 96.6% of the regression improvement are
already present by epoch 14, and their final three-epoch relative ranges are
0.16% and 0.24%. This confirms diminishing returns near the final third of the
schedule. It does **not** confirm sustained overfitting. Classification NLL
sets its best value at epoch 20; accuracy ends only 1.64% below its epoch-16
best. Regression MAE ends 1.12% above its epoch-13 best, while noisy sampled
training MAE does not continue to improve. All objective-, invariant-, and
decision-level overfitting rules remain false at the predeclared 2%/three-epoch
threshold.

The baseline joint invariant score is best at epoch 16. This audit therefore
retains per-task and joint best epochs instead of treating the last checkpoint
or lowest reweighted training objective as universally optimal. Compact final
curves and the machine-readable finding set are committed under
`evaluation_results/training_debug/full_retrain_baseline`.

The already-running baseline process predated the counter-label correction
that separates finite losses from applied AMP optimizer steps. Its
`successful_steps` field therefore means 6,000 finite-loss steps. The raw AMP
metric records five scale-warm-up overflows in epoch 1 and none afterwards, so
the applied-step total is 5,995. All matched ablations were launched after the
correction and persist finite, applied, skipped, overflow, and non-finite
counters separately.

### Matched diagnostic ablations

Four one-factor, full-schedule runs and one head-focused curriculum are
reserved for hypotheses that can be read directly from the telemetry. The
clip-10 dose-response control is conditional on clip 5 retaining a held-out
gain while still clipping a majority of steps:

| Configuration | Isolated change | Diagnostic hypothesis |
|---|---|---|
| `training_debug_clip5_retrain.yaml` | Global clip cap 1 → 5 | Near-100% clipping throttles otherwise finite gradients and front-loads effective learning |
| `training_debug_clip10_retrain.yaml` | Global clip cap 1 → 10 | Clip 5 improves invariant outputs but its residual majority clipping still throttles updates |
| `training_debug_smoothing010_retrain.yaml` | Classification label smoothing 0.05 → 0.10 | Held-out confidence grows faster than accuracy |
| `training_debug_regression_balanced_retrain.yaml` | Median/relative weights 0.40/0.05 → 0.20/0.025 | These two terms contribute disproportionate regression gradient energy |
| `training_debug_head_focused_retrain.yaml` | Smoothing 0.10 plus staged-module LR multipliers 5×/20× | Shared backbone overfits while selectors remain uniform under a joint LR 20× below their historical stage |

The seed, case split, architecture, task mixture, optimizer, and 20×300 schedule
remain matched. Reweighted total losses are not compared across the regression
ablation. Selection instead uses invariant outputs: held-out classification
accuracy/NLL and raw-hour regression MAE/RMSE, with pool-level results checked
for regressions hidden by the aggregate.

The regression-balanced control completed all 20 epochs. Halving the
median/relative coefficients reduced their mean isolated gradient norms to
1.782/0.407, versus approximately 3.24--3.29/0.74--0.77 in the other fully
instrumented controls. This did not yield a consistent invariant improvement:
best MAE worsened from 837.320 to 840.106 hours, while best RMSE improved
slightly from 1,129.434 to 1,127.514 hours; classification NLL was effectively
identical (1.90843 versus 1.90837). The joint score remains worse. The robust
terms are therefore retained; scalar loss share alone would have led to the
wrong removal decision.

The clip-5 and label-smoothing controls also completed all 20 epochs without a
strict overfitting signal. Clip 5 reduces mean clip incidence from 99.73% to
55.32% and improves the best joint invariant score from 0.806643 (baseline
e16) to 0.803059 (clip-5 e14). Its classification NLL continues to 1.87592 at
epoch 20, but best regression MAE is 841.872 versus baseline 837.320 hours and
its last confidence gap is larger (0.1731 versus 0.1660). The candidate is
therefore checkpointed at its source-selected joint epoch 14 and sent to the
fixed target screen rather than promoted from training loss alone.

Increasing label smoothing from 0.05 to 0.10 slightly narrows the final
confidence gap to 0.1604, but worsens best NLL (1.92285), accuracy (0.29949),
MAE (845.103 hours), and joint score (0.811981). This control is rejected:
post-training output-temperature calibration is better isolated from feature
learning than stronger smoothing for this model.

`compare_training_debug.py` creates a live or final matched comparison. Its
joint screening score gives equal weight to held-out classification NLL and
raw-hour regression MAE after normalizing both by baseline epoch 1. The two
constituent metrics, accuracy, RMSE, confidence gap, and clipping remain visible
and are the evidence used for a decision; the joint score is not a new training
objective.

The head-focused run uses `training_lr_multipliers`, keyed by the same stable
parameter-group names as the gradient/update diagnostics. An empty mapping is
the default and preserves the historical optimizer groups. Its selectors use
20× because their promoted task-isolated stages used LR 0.002 versus this
audit's base 0.0001; adapters, transform/confidence heads, and the task router
use the more conservative 5× ratio seen in earlier staged continuations. The
backbone LR and clip cap remain unchanged so this intervention moves capacity
toward small task heads rather than accelerating memorization in the encoder.

Candidates are selected exclusively from the source-case holdout, then run
once through `training_debug_target_screen_eval.yaml`. That overlay inherits
the established five-log, 96-row selector screen unchanged; target results do
not feed back into epoch selection. Dominated smoothing/loss-weight controls
are not screened merely because they completed training.

`compare_fmv3_results.py` then requires exact pairing on task, profile, log,
repetition, support scenario/budget, retrieval/prior mode and strength, and
retrieval k. It reports candidate-minus-reference means and row-level
wins/ties/losses separately for classification and regression; a missing or
extra target row is an error rather than an implicit unpaired comparison.

The source-selected baseline e16 and clip-5 e14 checkpoints completed this
screen. Clip 5 is not promoted. Relative to baseline, it improves ordinary
accuracy by 0.001895, NLL by 0.016678, Brier by 0.004843, and ECE by 0.000318,
but reduces balanced accuracy by 0.001082 and macro-F1 by 0.000455 while
worsening MAE/RMSE by 7.860/9.452 hours. Relative to the existing selected e44
endpoint, clip 5 is lower by 0.005548 balanced accuracy, 0.003969 accuracy, and
0.002026 macro-F1, and worse by 62.703 MAE hours and 73.880 RMSE hours. The
from-scratch baseline also remains below selected e44. The deployed checkpoint
therefore stays unchanged; relaxing clipping remains a useful optimization
finding, not a validated replacement model.

```bash
python compare_training_debug.py \
  --baseline baseline \
  --run baseline=checkpoints/fmv3/training_debug_full_retrain/training_debug_summary.json \
  --run clip5=checkpoints/fmv3/training_debug_clip5_retrain/training_debug_summary.json \
  --run clip10=checkpoints/fmv3/training_debug_clip10_retrain/training_debug_summary.json \
  --run smoothing010=checkpoints/fmv3/training_debug_smoothing010_retrain/training_debug_summary.json \
  --run regression_balanced=checkpoints/fmv3/training_debug_regression_balanced_retrain/training_debug_summary.json \
  --output_dir evaluation_results/training_debug/matched_comparison
```
