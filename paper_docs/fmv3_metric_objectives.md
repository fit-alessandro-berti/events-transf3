# Adaptable metric-target objectives

## Purpose

FM-v3 previously had one classification objective—label-smoothed cross
entropy—and a configurable but fixed regression mixture. That made evaluation
metrics visible after training without making the training intent explicit.
Metric-target profiles now provide one shared implementation for retrieval and
episodic training, preserve historical configurations, and expose every active
component and its gradient in the structured diagnostics.

The base/default FM-v3 configuration is `equilibrated`. Published experiment
roots are explicitly pinned to `legacy` classification; native-space sqrt-head
roots use `legacy` regression while learned-transform roots restore their
published `custom` mixture. Thus an old YAML file does not silently acquire a
new objective when reproduced.

## Classification profiles

`classification_objective.profile` accepts:

| Profile | Optimized differentiable surrogate |
|---|---|
| `equilibrated` | Initial-gradient-calibrated blend of all five metrics below |
| `accuracy` | Mean soft error, `1 - mean(p_true)` |
| `balanced_accuracy` | One minus mean per-class soft recall |
| `macro_f1` | One minus macro soft-F1 from episode-level soft TP/FP/FN |
| `nll` | Label-smoothed NLL divided by uniform-predictor entropy |
| `brier` | Half the multiclass Brier score, bounded to a comparable scale |
| `custom` | User-supplied non-negative `weights` mapping |
| `legacy` | Historical label-smoothed cross entropy at its original scale |

Balanced accuracy and macro-F1 are computed over the complete training episode.
The retrieval path first aligns query-specific prototype class columns into the
original label-ID universe, then builds differentiable episode-level counts.
Computing either metric independently for each query would collapse balanced
accuracy to accuracy and would not represent macro-F1.

The equilibrated classification profile uses weights `1.0/1.0/1.0/0.18/1.0`
for accuracy/balanced accuracy/macro-F1/NLL/Brier. Equal scalar weights looked
balanced but the smoke protocol showed NLL producing 5.9 times the smallest
metric gradient. The calibrated NLL weight balances initial optimization
pressure while leaving the single-NLL profile mathematically unchanged.

Example:

```yaml
classification_label_smoothing: 0.05
classification_objective:
  profile: custom
  weights:
    accuracy: 0.25
    balanced_accuracy: 1.0
    macro_f1: 1.0
    nll: 0.10
    brier: 0.25
```

Unknown metrics, negative/non-finite weights, and all-zero custom profiles are
configuration errors.

## Regression profiles

`fmv3_head.regression_objective_profile` accepts:

| Profile | Optimized differentiable surrogate |
|---|---|
| `equilibrated` | Initial-gradient-calibrated blend of all regression metrics |
| `mae` | Scale-normalized raw-hour MAE only |
| `rmse` | Scale-normalized raw-hour RMSE only |
| `r2` | Stabilized R2 surrogate only |
| `custom` | Existing flat weights plus optional `regression_metric_weights` overrides |
| `legacy` | Historical Huber loss in the head's native output space |

The R2 loss is `log1p(SSE/SST)`, equivalently a monotonic stabilization of
`1 - R2`. SSE and target variance use the same batch scale normalization; a
small normalized variance floor keeps constant-target batches finite. Scaling
or shifting both predictions and targets leaves the non-degenerate surrogate
unchanged. R2 is episode-relative, so the target screen still reports global
R2 alongside MAE/RMSE rather than treating training loss as evaluation R2.

Metric profiles also apply to the base `sqrt_knn` head. Its prediction and
label tensors are differentiably squared back to raw hours before MAE, RMSE,
R2, and the other metric terms are computed. Historical sqrt-space experiments
use the explicit `legacy` profile; learned-transform experiment roots switch
back to `custom`, preserving their published multi-metric objectives.

The equilibrated MAE/RMSE/Huber/log-RMSE/relative-MAE/bias/median-AE/quantile/R2
weights are `1.0/1.0/1.3/0.8/0.15/0.75/0.08/2.0/1.4`. These values compensate
for the initial gradient scale measured by the fixed smoke protocol. In
particular, equal scalar weights made median-AE pressure 23.6 times the
smallest regression-metric gradient. The full-MAE and full-R2 profiles each
remain exactly their named component because active weights are renormalized.

Examples:

```yaml
fmv3_head:
  regression_objective_profile: mae
```

```yaml
fmv3_head:
  regression_objective_profile: custom
  regression_metric_weights:
    mae: 2.0
    r2: 1.0
    median_ae: 0.1
```

## Diagnostics and selection

Every classification surrogate is persisted as raw and weighted loss, with
hard episode accuracy, balanced accuracy, and macro-F1. Regression now adds raw
and weighted R2 surrogate loss plus raw-hour R2. `loss_gradient_interval`
attributes gradients to each metric component. The analyzer reports a metric
gradient imbalance when the largest active component is at least five times
the smallest rather than mislabelling primary metric components as auxiliaries.

Metric profiles change training intent, not checkpoint-selection discipline.
Source-case-held-out accuracy, balanced accuracy, macro-F1, NLL, Brier, MAE,
RMSE, and R2 remain the selection evidence. Target logs are evaluated only
after a source-selected checkpoint has been fixed.

### End-to-end smoke validation

The equilibrated profile completed a real retrieval-training epoch over both
tasks with deterministic source-case validation and gradient attribution every
five steps. All metric losses and gradients were finite. Calibration reduced
the largest-to-smallest component-gradient ratio from 5.94 to 1.11 for
classification and from 23.60 to 1.17 for regression. The compact curves and
machine-readable gradient summary are committed under
`evaluation_results/training_metric_objectives/equilibrated_smoke`.

## Matched experiment matrix

All runs below share seed 42, the selected architecture, the deterministic 10%
source-case holdout, a 50/50 task mix, 20 epochs × 300 episodes, conservative
backbone LR/clip settings, and the previously validated module-specific head
LR multipliers:

| Configuration | Classification profile | Regression profile |
|---|---|---|
| `training_metric_equilibrated_retrain.yaml` | equilibrated | equilibrated |
| `training_metric_accuracy_retrain.yaml` | accuracy | equilibrated |
| `training_metric_balanced_accuracy_retrain.yaml` | balanced accuracy | equilibrated |
| `training_metric_mae_retrain.yaml` | equilibrated | MAE |
| `training_metric_r2_retrain.yaml` | equilibrated | R2 |

The experiment compares each targeted metric, all non-target metrics, gradient
balance, clipping, confidence, and per-source behavior. A profile is not
promoted merely because its own training surrogate falls fastest.

`compare_metric_objectives.py` reads each completed checkpoint directory,
reports best epochs for every metric, and selects a source-only profile-aligned
epoch. For an extreme task it uses the named metric while retaining the
equilibrated score for the other task; it also reports a fully equilibrated
epoch for comparison.

```bash
python compare_metric_objectives.py \
  --baseline equilibrated \
  --run equilibrated=checkpoints/fmv3/training_metric_equilibrated_retrain \
  --run accuracy=checkpoints/fmv3/training_metric_accuracy_retrain \
  --run balanced_accuracy=checkpoints/fmv3/training_metric_balanced_accuracy_retrain \
  --run mae=checkpoints/fmv3/training_metric_mae_retrain \
  --run r2=checkpoints/fmv3/training_metric_r2_retrain \
  --output_dir evaluation_results/training_metric_objectives/matched_comparison
```
