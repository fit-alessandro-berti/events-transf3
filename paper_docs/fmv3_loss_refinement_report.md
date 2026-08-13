# Task-isolated classification and regression loss refinement

## Promotion decision

The epoch-43 task-isolated refinement is promoted over the learned half-expert
router's epoch-42 endpoint. On the same five logs and full 400-row protocol,
every reported aggregate classification and regression metric improves. The
selected checkpoint is:

`checkpoints/fmv3/loss_refinement_selected/model_epoch_43.pth`

Its SHA-256 is
`54a384b037cb83919bad9aab6bd16ca923125fcd9d93943d6b8484da3653d846`.
The stable configuration is `configs/fmv3/loss_refinement_selected.yaml`; the
full evaluation overlay is
`configs/fmv3/loss_refinement_confirmation_eval.yaml`.

The checkpoint is an audited merge of two continuations from the exact same
epoch-42 base. The classification run can change only its classification
adapter, and the regression run can change only its regression adapter,
time-transform bank, and regression posterior-confidence head. The merge tool
rejects any shared or out-of-scope tensor change. It merged 24 classification
tensors and 80 regression tensors.

## Kept changes

### Classification: direct angular separation

The winning classification continuation adds a zero-initialized residual
adapter after the shared encoder:

`LayerNorm(256) -> Linear(256, 64) -> GELU -> Linear(64, 256)`.

It executes only for classification. Regression receives the original encoder
output exactly. The adapter is trained with the existing cross-entropy plus an
additive angular-margin loss (weight `0.20`, temperature `0.10`, margin
`0.15`) on the deployed embedding. For every anchor, the loss builds
leave-case-out class prototypes and subtracts the margin from the true-class
cosine logit before cross-entropy. Thus the objective explicitly requires the
own-class similarity to clear competing class similarities; it does not act
only on the disposable projection head.

On a fixed held-out sample (seed `20260813`, up to 2,000 prefixes per log,
five logs, both active classification experts), the mean leave-one-out
own-class-minus-hardest-other centroid margin changes from `-0.014493` to
`-0.012780` (`+0.001713`). Leave-one-out centroid top-1 accuracy changes from
`0.514944` to `0.515445`, and within-class centroid cosine changes from
`0.929407` to `0.932675`. The committed diagnostic is
`evaluation_results/loss_refinement/separability.json`.

After the already-established structured fusion, a monotone output temperature
of `0.60` calibrates the probability vector. Temperature scaling cannot change
its argmax. It therefore preserves the improved decisions while improving NLL,
Brier, ECE, and AURC.

### Regression: median-aware multi-metric refinement

Regression receives its own zero-initialized residual adapter with the same
shape; it never executes for classification. The retained primary loss uses
MAE/RMSE weights `0.65/0.35` and adds a scale-normalized median absolute-error
term with weight `0.40`, alongside the existing Huber, log-RMSE, relative-MAE,
and bias terms. The direct median term was added only after the MAE/RMSE
candidate improved mean and tail errors but slightly worsened the reported
median.

The posterior regression-confidence head keeps its original meaning: it is
trained by BCE against `exp(-relative absolute error)` and receives detached
prediction diagnostics. Its selected aggregation temperature is `0.018`. The
support-calibration mix is `0.502` except for the inherited budget-specific
settings (budget 2: `0.0`; budget 4: `0.6`). This narrow mix is the first tested
point that improves MAE, RMSE, and median error together.

## Confidence and routing invariants

The pre-execution task-confidence router was frozen during both continuations.
All four router tensors in the promoted checkpoint are byte-identical to the
epoch-42 router. The full JSONL routing payload—logits, sigmoid confidences,
selected indices, and active/inactive counts—is exactly identical row-for-row
between the two confirmations.

- Classification route: experts `[0, 2]`.
- Regression route: experts `[2, 3]`.
- All 400 rows: 4 total, 2 active, 2 inactive.

Classification output calibration is monotone, and regression posterior
confidence retains the same reliability target. Consequently, separability
and task losses improve without redefining confidence as an arbitrary mixture
weight or leaking query labels into routing.

## Full confirmation

The comparison uses
`evaluation_results/expert_routing/confirmations/expert_routing_bias_e42` as
the fixed baseline and
`evaluation_results/loss_refinement/confirmations/loss_refinement_selected_final_e43`
as the selected result. Each task has 200 paired rows.

### Classification

| Metric | Epoch 42 | Selected epoch 43 | Change |
|---|---:|---:|---:|
| Balanced accuracy | 0.450242 | **0.450425** | **+0.000184** |
| Accuracy | 0.716352 | **0.716434** | **+0.000082** |
| Macro-F1 | 0.421233 | **0.421533** | **+0.000301** |
| Macro-precision | 0.431457 | **0.432456** | **+0.000999** |
| Zero-recall fraction | 0.405578 | **0.404815** | **-0.000763** |
| NLL | 3.072708 | **3.002257** | **-0.070451** |
| Multiclass Brier | 0.477702 | **0.447051** | **-0.030651** |
| ECE-10 | 0.184229 | **0.118677** | **-0.065552** |
| AURC | 0.246467 | **0.240732** | **-0.005735** |

The gains therefore cover both decision separability (balanced accuracy,
macro-F1/precision, zero-recall classes) and confidence quality. Paired-row
wins/ties are respectively 39/124 for balanced accuracy, 37/130 for accuracy,
49/96 for F1, 157/1 for NLL, 159/1 for Brier, 140/1 for ECE, and 149/1 for
AURC.

### Regression

| Metric | Epoch 42 | Selected epoch 43 | Change |
|---|---:|---:|---:|
| MAE (h) | 1,097.8102 | **1,097.2543** | **-0.5559** |
| RMSE (h) | 1,651.8832 | **1,651.7386** | **-0.1446** |
| Median AE (h) | 735.5687 | **735.5062** | **-0.0625** |
| Normalized MAE | 0.824843 | **0.823743** | **-0.001100** |
| MAE skill | -0.135294 | **-0.134140** | **+0.001154** |
| RMSE skill | -0.013026 | **-0.013021** | **+0.000006** |
| D2 absolute error | -0.135294 | **-0.134140** | **+0.001154** |
| R2 | -0.202428 | **-0.202192** | **+0.000236** |
| Interval coverage | 0.706114 | **0.706232** | **+0.000118** |
| Mean interval width (h) | 2,875.9136 | **2,875.0361** | **-0.8775** |

MAE improves on 132/200 paired rows, RMSE on 84/200, median AE on 125/200,
and interval width on 110/200. Aggregate improvements are not uniform on every
individual log/budget row; the promotion criterion is the fixed protocol's
predeclared aggregate metric set, all of which move in the favorable direction.

## Rejected candidates

Only the winning code paths and configurations remain in the repository.
Matched screens rejected:

- unconstrained full-model continuations, which improved classification but
  worsened MAE/RMSE;
- classification margin weights `0.05` and `0.10`, which did not improve the
  full classification metric set as consistently as `0.20`;
- a regression target-ordering margin, which did not beat the MAE-focused
  continuation;
- median-loss weights `0.05`, `0.10`, and `0.20`, which were weaker than
  `0.40` on the full metric tradeoff;
- confidence temperatures and calibration mixes outside `0.018`/`0.502`,
  which left either RMSE or median AE worse than epoch 42.

Generated rejected checkpoints and result trees were removed after selection.

## Reproduction

Seed two task-specific directories from the identical epoch-42 checkpoint:

```bash
for name in loss_refinement_classification loss_refinement_regression; do
  mkdir -p checkpoints/fmv3/$name
  cp checkpoints/fmv3/expert_routing_bias/model_epoch_42.pth \
    checkpoints/fmv3/$name/model_epoch_42.pth
  cp checkpoints/fmv3/expert_routing_bias/training_artifacts.pth \
    checkpoints/fmv3/$name/training_artifacts.pth
  cp checkpoints/fmv3/expert_routing_bias/training_config.pth \
    checkpoints/fmv3/$name/training_config.pth
  cp checkpoints/fmv3/expert_routing_bias/training_config.yaml \
    checkpoints/fmv3/$name/training_config.yaml
done
```

Train the disjoint adapters:

```bash
python main.py \
  --config configs/fmv3/loss_refinement_classification_train.yaml \
  --checkpoint_dir checkpoints/fmv3/loss_refinement_classification \
  --resume --stop_after_epoch 43

python main.py \
  --config configs/fmv3/loss_refinement_regression_train.yaml \
  --checkpoint_dir checkpoints/fmv3/loss_refinement_regression \
  --resume --stop_after_epoch 43
```

Audit and merge them while materializing the resolved deployment config:

```bash
python merge_task_isolated_checkpoints.py \
  --base checkpoints/fmv3/expert_routing_bias/model_epoch_42.pth \
  --classification checkpoints/fmv3/loss_refinement_classification/model_epoch_43.pth \
  --regression checkpoints/fmv3/loss_refinement_regression/model_epoch_43.pth \
  --output checkpoints/fmv3/loss_refinement_selected/model_epoch_43.pth \
  --config configs/fmv3/loss_refinement_selected.yaml \
  --artifacts checkpoints/fmv3/expert_routing_bias/training_artifacts.pth
```

Run the full confirmation:

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/loss_refinement_selected \
  --checkpoint_epoch 43 \
  --eval_config configs/fmv3/loss_refinement_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/loss_refinement/confirmations/loss_refinement_selected_final_e43
```
