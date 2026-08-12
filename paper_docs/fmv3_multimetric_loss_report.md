# Promoted FM-v3 multi-metric remaining-time loss

## Decision

**Promotion decision: select the epoch-38 multi-metric checkpoint as the base
repository model, and use the epoch-40 regression-confidence endpoint for the
current best evaluation stack.** The base-checkpoint decision is based on its
paired improvement on the established five-log benchmark: balanced accuracy,
macro-F1, raw-hour MAE, RMSE, and median absolute error all improve over epoch
36. The later endpoint keeps that classifier path, adds the promoted
low-support structured-memory overlay for classification, and uses only the
regression expert-confidence head from epoch 40. The separately evaluated
10,000-case Road Traffic log remains an explicit transfer limitation: its mean
MAE and RMSE worsen even though classification is unchanged and median absolute
error improves. Promotion does not erase or override that evidence.

The promoted artifact is
`checkpoints/fmv3/loss_multimetric_gate_aux_005/model_epoch_38.pth`. New work
should start from this checkpoint and its resolved training artifacts unless
the experiment deliberately uses epoch 36 as an ablation control. The stable
base configuration entry point is `configs/fmv3/selected.yaml`.

The current best endpoint is
`checkpoints/fmv3/expert_confidence_heads/model_epoch_40.pth` evaluated with
`configs/fmv3/regression_confidence_low_support_confirmation_eval.yaml`. That
overlay deliberately disables the learned classification-confidence head and
keeps the learned regression-confidence head. The regression expert-confidence
softmax uses inference temperature `0.1`; the default temperature-`1.0`
variant is retained as a comparison artifact.

This conclusion separates two claims:

- **Verified repository-benchmark improvement:** yes, on the original five
  evaluation logs and paired rows; the endpoint improves the base selected
  stack on balanced accuracy, accuracy, macro-F1, MAE, RMSE, median absolute
  error, and R².
- **Verified cross-log robustness:** not yet; mean and tail regression errors
  worsen on the later, much larger Road Traffic log.

## Change tested

The historical learned-transform head optimizes the normalized average of
raw-hour MAE and RMSE. The promoted model keeps those two terms and adds four
complementary terms:

| Loss term | Weight | Purpose |
|---|---:|---|
| Scale-normalized MAE | 0.50 | Absolute-error pressure |
| Scale-normalized RMSE | 0.50 | Large-error pressure |
| Huber on normalized residuals | 0.15 | Smooth MAE/RMSE bridge |
| RMSE in `log1p(hours)` residual space | 0.15 | Multi-scale error pressure |
| Relative MAE | 0.05 | Target-relative absolute error |
| Absolute mean residual | 0.05 | Bias control |
| Quantile loss | 0.00 | Disabled in this experiment |

The weighted terms are divided by the sum of their weights. The existing
dynamic transform-gate auxiliary remains enabled with weight `0.05` and target
temperature `0.10`. Implementation and component diagnostics are in
`components/prototypical_head.py`; the exact experiment configuration is
`configs/fmv3/loss_multimetric_gate_aux_005.yaml`.

### How the learned rescaling branches are combined

The four learned time transforms are not averaged uniformly. Each branch owns
its power, time scale, and neighbor-attention scale; it transforms support
times, performs soft neighbor regression in that space, and inverts its result
back to raw hours. The promoted `dynamic` aggregation then computes
query-specific softmax weights and takes a convex weighted sum of the four
hour-valued branch predictions. The weights are nonnegative and sum to one.

The evaluator subsequently averages the four experts' head predictions. In
the full confirmation overlay, it also learns a support-only branch prior and
blends that calibrated prediction 50/50 with the trained query-dynamic result.
Thus uniform averaging is used across experts and for the final two-path blend,
but not across the four learned rescaling branches inside an expert.

The promoted complementary weights are now the defaults in `config.py`, the
head constructor, and `configs/fmv3/base.yaml`. Historical root configurations
pin all four weights to zero, and resolved checkpoint configurations remain
authoritative, so earlier two-term experiments stay reproducible.

### Raw-hour no-rescaling ablation

A simple inference-time ablation replaces the learned transform bank with
`regression_mode: raw_hours_knn` / `raw_prediction`. It converts stored
`sqrt(hours)` labels back to raw hours, then predicts a single
similarity-weighted mean of neighbor remaining-time hours. It uses no learned
target rescaling branches, no dynamic branch gate, and no support-only branch
calibration.

The ablation is not better on the established five-log confirmation. On the
current endpoint checkpoint, `raw_prediction` keeps the low-support classifier
overlay and learned regression expert-confidence aggregation, but bypasses the
remaining-time transform bank. Loading this ablation intentionally ignores the
unused transform-bank and disabled classification-confidence tensors. It leaves
classification exactly unchanged and worsens every primary regression metric:

| Head | MAE (h) | RMSE (h) | Median AE (h) | Normalized MAE | R² |
|---|---:|---:|---:|---:|---:|
| Current endpoint: learned transforms + regression confidence, T=0.1 | **1,107.614** | **1,658.117** | **742.639** | **0.841043** | **-0.211902** |
| Raw prediction + regression confidence, T=0.1 | 1,187.138 | 1,691.549 | 851.593 | 0.994111 | -0.279494 |
| Raw minus current endpoint | +79.524 | +33.431 | +108.954 | +0.153068 | -0.067592 |

Paired regression rows favor the current endpoint for MAE on 137/200 rows,
RMSE on 106/200 rows, median absolute error on 163/200 rows, and R² on
106/200 rows. The degradation is not just an ultra-low-data effect:
for budgets ≥4, raw prediction changes MAE/RMSE/median AE by
+101.260/+42.056/+139.283 h; for budgets ≥8, it changes them by
+94.212/+29.292/+128.406 h. Per-log MAE deltas are nonnegative on every
original log: billing +177.066 h, helpdesk +0.000 h, receipt +20.259 h,
roadtraffic100traces +25.534 h, and sepsis +158.180 h.

The same conclusion already held when isolating the selected epoch-38 base
checkpoint without the confidence continuation:

| Head | MAE (h) | RMSE (h) | Median AE (h) | Normalized MAE |
|---|---:|---:|---:|---:|
| Selected epoch-38 learned-transform head | **1,109.409** | **1,659.620** | **744.362** | **0.843444** |
| Raw-hour soft-kNN ablation | 1,188.114 | 1,692.908 | 852.999 | 0.994541 |
| Raw minus selected epoch-38 base | +78.705 | +33.288 | +108.636 | +0.151097 |

The separate Road Traffic 10,000-case check is mixed rather than dominant:
classification again stays unchanged, raw-hour soft-kNN changes MAE by
+14.259 h, RMSE by -52.099 h, and median absolute error by +142.428 h. This
suggests the raw average can soften some extreme errors on that one
distribution, but it is not a replacement for the learned rescaling head.

Reproduce the current-endpoint raw-prediction ablation with:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/expert_confidence_heads \
  --checkpoint_epoch 40 \
  --eval_config configs/fmv3/raw_prediction_regression_confidence_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/raw_hours_knn/confirmations/raw_prediction_temp010_regression_confidence_e40 \
  --device cuda:0
```

Reproduce the earlier epoch-38 raw-hour ablation with:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/loss_multimetric_gate_aux_005 \
  --checkpoint_epoch 38 \
  --eval_config configs/fmv3/raw_hours_knn_ablation.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/raw_hours_knn/confirmations/raw_hours_knn_ablation_e38 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/loss_multimetric_gate_aux_005 \
  --checkpoint_epoch 38 \
  --eval_config configs/fmv3/raw_hours_knn_ablation.yaml \
  --logs roadtraffic_10000 \
  --output_dir evaluation_results/raw_hours_knn/confirmations/raw_hours_knn_ablation_e38_roadtraffic_10000 \
  --device cuda:0
```

### Learned per-expert confidence ablation

A second post-promotion ablation adds two small confidence heads to each expert:
one predicts a general reliability logit for the expert's classification
posterior, and one predicts a reliability logit for the expert's remaining-time
prediction. At inference, the four expert outputs are aggregated with a
softmax over these learned logits instead of the uniform expert average. The
confidence heads are initialized to zero-logit output, so the checkpoint starts
from the selected model's uniform aggregation behavior.

The run trains only these new heads from the selected epoch-38 checkpoint:
`configs/fmv3/expert_confidence_heads.yaml` writes to
`checkpoints/fmv3/expert_confidence_heads/`. The final epoch-40 checkpoint has
1,032 trainable parameters in the confidence heads.

The symmetric two-head ablation is **not promoted**. It produces a very small
regression gain, but slightly lowers the primary classification accuracy-style
metrics on the full five-log confirmation:

| Model | Balanced accuracy | Accuracy | Macro-F1 | NLL | ECE-10 | MAE (h) | RMSE (h) | Median AE (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected epoch 38 | **0.447740** | **0.709221** | **0.419179** | 3.055258 | 0.170634 | 1,109.409 | 1,659.620 | 744.362 |
| Expert confidence epoch 39 | 0.447631 | 0.709208 | 0.419098 | 3.055216 | 0.170586 | 1,109.323 | 1,659.541 | 744.301 |
| Expert confidence epoch 40 | 0.447260 | 0.709200 | 0.418929 | **3.055162** | **0.170365** | **1,109.237** | **1,659.464** | **744.239** |
| Epoch 40 minus selected | -0.000480 | -0.000020 | -0.000250 | -0.000096 | -0.000268 | -0.172 | -0.156 | -0.124 |

On epoch 40, paired regression rows improve for MAE on 153/200 rows and RMSE
on 112/200 rows. Classification calibration also improves slightly, but the
classification decision metrics move in the wrong direction: balanced accuracy
improves on only 18/200 rows and ties on 167/200 rows, with a negative mean
delta. The learned expert weights are real but mild rather than decisive:
epoch-40 full-confirmation mean classification weights are
`[0.257895, 0.239091, 0.258187, 0.244826]`; mean regression weights are
`[0.241001, 0.249266, 0.249912, 0.259820]`.

The result should therefore be kept as an ablation rather than made the stable
`selected.yaml` target. It suggests that expert-level calibration may help
remaining-time aggregation marginally, but the classification-confidence half
of this formulation is too mixed to justify replacing the selected uniform
classification expert aggregation.

#### Regression-only confidence endpoint

A follow-up endpoint keeps the epoch-40 regression confidence head but disables
the classification confidence head at load time. It is combined with the
already promoted low-support structured-memory classifier. Loading prints
`Ignored 16 checkpoint parameters not used by current config`, corresponding
to the disabled classification-confidence tensors; the regression-confidence
tensors remain active. A final inference-only refinement sharpens the
regression expert-confidence softmax to temperature `0.1`.

This endpoint is a small Pareto improvement over the base selected stack on the
original five-log confirmation:

| Endpoint | Balanced accuracy | Accuracy | Macro-F1 | MAE (h) | RMSE (h) | Median AE (h) | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base selected epoch 38 | 0.447740 | 0.709221 | 0.419179 | 1,109.409 | 1,659.620 | 744.362 | -0.214691 |
| Low-support structured + regression confidence, T=1.0 | 0.451092 | 0.717033 | 0.422542 | 1,109.237 | 1,659.464 | 744.239 | -0.214383 |
| **Current endpoint, T=0.1** | **0.451092** | **0.717033** | **0.422542** | **1,107.614** | **1,658.117** | **742.639** | **-0.211902** |
| Current endpoint minus base selected | +0.003351 | +0.007812 | +0.003363 | -1.795 | -1.503 | -1.723 | +0.002789 |
| T=0.1 minus T=1.0 | 0.000000 | 0.000000 | 0.000000 | -1.623 | -1.347 | -1.600 | +0.002482 |

The classification metrics are exactly identical to the selected low-support
structured overlay because classification expert confidence is disabled. The
temperature-sharpened regression rows improve over the previous endpoint by
MAE **-1.623 h**, RMSE **-1.347 h**, median absolute error **-1.600 h**, and
R² **+0.002482**. Relative to the epoch-38 base, MAE improves on 156/200
paired rows, RMSE and R² improve on 105/200 rows, and median absolute error
improves on 141/200 rows. In the support range now used for continued work,
budget ≥4 rows change versus the base by MAE **-2.003 h**, RMSE **-1.688 h**,
and R² **+0.002737**; budget ≥8 rows change by MAE **-1.147 h**, RMSE
**-0.709 h**, and R² **+0.001293**.

The temperature screen is monotonic in this local range: flattening to `1.5`
or `2.0` worsens MAE/RMSE/R², while sharpening from `0.75` to `0.5`, `0.4`,
`0.333`, `0.25`, and `0.1` progressively improves them. The selected value is
the best screened value, not a newly trained checkpoint.

| Regression confidence temperature | MAE Δ vs T=1.0 (h) | RMSE Δ vs T=1.0 (h) | Median AE Δ (h) | R² Δ |
|---:|---:|---:|---:|---:|
| 0.10 | **-1.623** | **-1.347** | **-1.600** | **+0.002482** |
| 0.25 | -0.528 | -0.462 | -0.450 | +0.000888 |
| 0.333 | -0.351 | -0.309 | -0.358 | +0.000598 |
| 0.40 | -0.263 | -0.232 | -0.267 | +0.000451 |
| 0.50 | -0.175 | -0.155 | -0.128 | +0.000302 |
| 0.75 | -0.058 | -0.052 | -0.027 | +0.000101 |
| 1.50 | +0.058 | +0.052 | +0.034 | -0.000102 |
| 2.00 | +0.086 | +0.078 | +0.056 | -0.000153 |

Reproduce the endpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/expert_confidence_heads \
  --checkpoint_epoch 40 \
  --eval_config configs/fmv3/regression_confidence_low_support_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/expert_confidence/confirmations/regression_confidence_temp010_low_support_e40 \
  --device cuda:0
```

#### Structured threshold extension screen

The low-support structured classifier was also tested with stronger-rule
thresholds above the promoted eight-prefix cutoff on the current endpoint. The
screen used classification-only rows for budgets 1, 4, 8, 16, 32, 64, 128,
plus eligible full-support rows. Thresholds 16, 32, and 64 all reduce the
overall decision metrics versus the threshold-8 endpoint. The threshold-64
variant has a tiny balanced-accuracy gain on budgets ≥8 (`+0.000741`), but it
also loses accuracy (`-0.002168`), macro-F1 (`-0.002204`), and calibration.
The endpoint therefore keeps the threshold-8 overlay. Full screen details and
commands are in
[`structured_fmv3_report.md`](structured_fmv3_report.md#threshold-extension-screen-on-the-current-endpoint).

#### Support-calibration mix sweep

The evaluation-time branch prior was also swept by changing
`regression_calibration_mix`. Stronger support-only calibration improves
MAE/median error but trades off RMSE and R², so it is not promoted. On the full
confirmation, mix `0.75` changes MAE by **-6.969 h** but RMSE by **+2.948 h**
and R² by **-0.005796**. Mix `1.0` changes MAE by **-10.945 h** but RMSE by
**+8.294 h** and R² by **-0.014606**. Because the target includes R², the
selected endpoint keeps the established `regression_calibration_mix: 0.5`.

Reproduce with:

```bash
mkdir -p checkpoints/fmv3/expert_confidence_heads
cp checkpoints/fmv3/loss_multimetric_gate_aux_005/model_epoch_38.pth \
   checkpoints/fmv3/loss_multimetric_gate_aux_005/training_artifacts.pth \
   checkpoints/fmv3/loss_multimetric_gate_aux_005/training_config.pth \
   checkpoints/fmv3/loss_multimetric_gate_aux_005/training_config.yaml \
   checkpoints/fmv3/expert_confidence_heads/

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/fmv3/expert_confidence_heads.yaml \
  --checkpoint_dir checkpoints/fmv3/expert_confidence_heads \
  --resume \
  --stop_after_epoch 40

CUDA_VISIBLE_DEVICES=0 python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/expert_confidence_heads \
  --checkpoint_epoch 40 \
  --eval_config configs/fmv3/expert_confidence_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/expert_confidence/confirmations/expert_confidence_e40 \
  --device cuda:0
```

## Controlled retraining protocol

The promoted model and matched control both start from the byte-identical selected
epoch-36 state-aware checkpoint:

```text
476506d7678c3446be076cf50aec73013218bff7636cac2fdb89fb18a15de876
```

Both runs continue for epochs 37 and 38 with seed 42, 200 retrieval episodes
per epoch, 70% classification episodes, learning rate `0.001`, weight decay
`0.001`, and `trainable_scope: temporal_prefix_joint`. The matched control is
`loss_gate_aux_005`; the only intended objective difference is the four
non-zero complementary primary-loss weights above.

Across four experts, 991,900 parameters in 116 tensors are trainable. A direct
epoch-36/epoch-38 tensor comparison confirms that exactly 116 tensors changed:

| Scope | Changed tensors |
|---|---:|
| State-aware prefix pool | 36 |
| Independent observable-clock encoders | 40 |
| Remaining-time target-transform banks | 40 |
| Outside the declared scope | **0** |

The promoted model was trained with:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/fmv3/loss_multimetric_gate_aux_005.yaml \
  --checkpoint_dir checkpoints/fmv3/loss_multimetric_gate_aux_005 \
  --resume \
  --stop_after_epoch 38
```

Epoch 37 was screened post hoc after the epoch-38 full and robustness results,
to check whether the external regression was merely terminal-epoch
degradation. Epoch 38 remained stronger on balanced accuracy, macro-F1, MAE,
RMSE, and median absolute error, while epoch 37 retained slightly higher
ordinary accuracy. Because this was not a predeclared stopping comparison, it
is supporting evidence rather than an independent checkpoint-selection claim.

## Evaluation protocol

The full original confirmation contains 400 rows: 200 classification and 200
regression rows over five logs, five repetitions, nested natural-support case
budgets, case-disjoint query sets, classification retrieval `k=20`, and
regression retrieval `k=50`. All comparisons are one-to-one paired on task,
log, repetition, support scenario, case budget, and retrieval `k`.

Because repeated support draws within a log are not independent, uncertainty
is reported with a deterministic 10,000-resample paired cluster bootstrap over
the five log-level mean deltas (seed 42). Five clusters give coarse intervals;
they should not be read as high-powered significance tests.

The robustness check uses the same full protocol on `roadtraffic_10000`, kept
separate from the original means. It contributes 40 classification and 40
regression rows. It was not folded into the original benchmark after the fact.

## Full result: original five logs

Higher is better for classification; lower is better for the three error
metrics.

| Model | Balanced accuracy | Accuracy | Macro-F1 | MAE (h) | RMSE (h) | Median AE (h) |
|---|---:|---:|---:|---:|---:|---:|
| Previous selected, epoch 36 | 0.447473 | **0.709352** | 0.418885 | 1,112.291 | 1,660.682 | 746.830 |
| Matched historical-loss control, epoch 38 | 0.447239 | 0.709350 | 0.418675 | 1,111.283 | 1,660.211 | 746.468 |
| **Promoted multi-metric model, epoch 38** | **0.447740** | 0.709221 | **0.419179** | **1,109.409** | **1,659.620** | **744.362** |

### Promoted model minus previous selected model

Negative error deltas are improvements.

| Metric | Mean delta | 95% log-cluster bootstrap interval | Paired rows better / tied / worse |
|---|---:|---:|---:|
| Balanced accuracy | +0.000268 | [-0.000081, +0.000585] | 28 / 130 / 42 |
| Accuracy | -0.000131 | [-0.000530, +0.000195] | 27 / 138 / 35 |
| Macro-F1 | +0.000294 | [-0.000087, +0.000710] | 41 / 107 / 52 |
| MAE | **-2.882 h** | **[-5.137, -0.477] h** | 144 / 0 / 56 |
| RMSE | **-1.062 h** | **[-2.793, -0.020] h** | 79 / 0 / 121 |
| Median absolute error | -2.468 h | [-8.166, +4.881] h | 144 / 0 / 56 |

For errors, the row count is shown as lower / tied / higher. The aggregate
MAE and RMSE improvements survive the five-log cluster bootstrap, while the
classification deltas and median-error delta do not exclude zero. Ordinary
accuracy has a very small negative point estimate.

### Promoted model minus matched epoch-38 control

This comparison isolates the effect of the primary-loss mixture from the
benefit of simply continuing training for two epochs.

| Metric | Mean delta | 95% log-cluster bootstrap interval |
|---|---:|---:|
| Balanced accuracy | +0.000502 | [+0.000179, +0.000783] |
| Accuracy | -0.000129 | [-0.000347, +0.000129] |
| Macro-F1 | +0.000504 | [+0.000186, +0.000772] |
| MAE | **-1.874 h** | **[-3.255, -0.345] h** |
| RMSE | -0.591 h | [-1.965, +0.126] h |
| Median absolute error | -2.105 h | [-5.782, +2.352] h |

The richer objective therefore adds a reproducible five-log MAE benefit over
continued training alone. Its incremental RMSE benefit has a negative point
estimate but a cluster interval that crosses zero.

## Separate 10,000-case Road Traffic robustness check

Classification predictions are exactly unchanged on all 40 paired rows. The
regression result is mixed and is an accepted limitation of the promotion:

| Metric | Previous epoch 36 | Multi-metric epoch 38 | Change |
|---|---:|---:|---:|
| Balanced accuracy | 0.538713 | 0.538713 | 0.000000 |
| Accuracy | 0.860833 | 0.860833 | 0.000000 |
| Macro-F1 | 0.513469 | 0.513469 | 0.000000 |
| MAE | **5,751.430 h** | 5,755.408 h | **+3.979 h (worse)** |
| RMSE | **7,927.730 h** | 7,935.488 h | **+7.758 h (worse)** |
| Median absolute error | 4,186.404 h | **4,178.586 h** | **-7.818 h (better)** |

The simultaneous median improvement and mean/RMSE regression suggests that
the objective helped typical errors but worsened some large-horizon errors on
this distribution. The checkpoint is therefore selected for the established
repository benchmark, not claimed to dominate epoch 36 on every target log.

## Verification and artifacts

The complete unit suite passes: 48 tests, including finite component values,
sqrt-label/raw-hour equivalence, historical two-term compatibility, gradient
flow into the transform bank, and rejection of an all-zero primary objective.
Python compilation and `git diff --check` also pass.

| Artifact | SHA-256 |
|---|---|
| Previous epoch-36 checkpoint | `476506d7678c3446be076cf50aec73013218bff7636cac2fdb89fb18a15de876` |
| Matched control epoch-38 checkpoint | `7ce06c25cdbeaca5cc523378c27fa4078f5de2fb3975a565a1c6576dcad2e4c9` |
| Promoted epoch-38 checkpoint | `7893ac0b68f6fb66d16aa1d47e779bd40b248e232df90ce33ed7471c38708aa3` |
| Promoted loader artifacts | `490f0e3e89dca3c2d890474bb6181611ef91ad7278fc4daad3ba7230d108b455` |
| Promoted source YAML | `3038cd6b56d4fcf47aff7e55af616e19f0f0bafb407d781efaa0d5d3494a0b75` |

Primary result files:

- Previous five-log result:
  `evaluation_results/prefix_attention/confirmations/prefix_state_attention_joint_e36/results.csv`
- Matched control:
  `evaluation_results/loss_audit/confirmations/loss_gate_aux_005_e38/results.csv`
- Promoted five-log result:
  `evaluation_results/loss_multimetric/confirmations/loss_multimetric_gate_aux_005_e38/results.csv`
- Promoted Road Traffic robustness result:
  `evaluation_results/loss_multimetric/confirmations/loss_multimetric_gate_aux_005_e38_roadtraffic_10000/results.csv`
- Previous Road Traffic classification and regression results:
  `evaluation_results/prefix_attention/confirmations/prefix_state_attention_joint_e36_roadtraffic_10000/`
  and
  `evaluation_results/prefix_attention/confirmations/prefix_state_attention_joint_e36_roadtraffic_10000_reg/`

Reproduce the two confirmations with:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/loss_multimetric_gate_aux_005 \
  --checkpoint_epoch 38 \
  --eval_config configs/fmv3/prefix_attention_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/loss_multimetric/confirmations/loss_multimetric_gate_aux_005_e38 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/loss_multimetric_gate_aux_005 \
  --checkpoint_epoch 38 \
  --eval_config configs/fmv3/prefix_attention_confirmation_eval.yaml \
  --logs roadtraffic_10000 \
  --output_dir evaluation_results/loss_multimetric/confirmations/loss_multimetric_gate_aux_005_e38_roadtraffic_10000 \
  --device cuda:0
```

## Starting point for subsequent work

Use the promoted epoch-38 checkpoint as the base for the next architecture
change. Target the observed tail-transfer failure directly—for example with
log-stratified validation or a loss whose large-error weight is normalized
across logs—and continue reporting `roadtraffic_10000` separately so a later
aggregate gain cannot hide the same transfer regression.
