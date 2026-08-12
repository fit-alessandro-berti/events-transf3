# Promoted FM-v3 multi-metric remaining-time loss

## Decision

**Promotion decision: select the epoch-38 multi-metric checkpoint as the
current repository model.** The decision is based on its paired improvement on
the established five-log benchmark: balanced accuracy, macro-F1, raw-hour MAE,
RMSE, and median absolute error all improve over epoch 36. The separately
evaluated 10,000-case Road Traffic log remains an explicit transfer limitation:
its mean MAE and RMSE worsen even though classification is unchanged and median
absolute error improves. Promotion does not erase or override that evidence.

The promoted artifact is
`checkpoints/fmv3/loss_multimetric_gate_aux_005/model_epoch_38.pth`. New work
should start from this checkpoint and its resolved training artifacts unless
the experiment deliberately uses epoch 36 as an ablation control. The stable
configuration entry point is `configs/fmv3/selected.yaml`.

This conclusion separates two claims:

- **Verified repository-benchmark improvement:** yes, on the original five
  evaluation logs and paired rows.
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
`regression_mode: raw_hours_knn`. It keeps the same selected epoch-38 embeddings
and prefix projection, converts the stored `sqrt(hours)` labels back to raw
hours, then predicts a single softmax-weighted mean of neighbor remaining-time
hours. It uses no learned target rescaling branches, no dynamic branch gate, and
no support-only branch calibration. Loading this ablation from the selected
checkpoint intentionally ignores the 40 superseded transform-bank tensors.

The ablation is not better on the established five-log confirmation. It leaves
classification exactly unchanged, but worsens all primary regression errors:

| Head | MAE (h) | RMSE (h) | Median AE (h) | Normalized MAE |
|---|---:|---:|---:|---:|
| Selected learned-transform head | **1,109.409** | **1,659.620** | **744.362** | **0.843444** |
| Raw-hour soft-kNN ablation | 1,188.114 | 1,692.908 | 852.999 | 0.994541 |
| Raw minus selected | +78.705 | +33.288 | +108.636 | +0.151097 |

Paired regression rows favor the selected head for MAE on 137/200 rows, RMSE
on 106/200 rows, and median absolute error on 158/200 rows. The per-log MAE
deltas are also nonnegative on every original log: billing +175.281 h,
helpdesk +0.000 h, receipt +19.959 h, roadtraffic100traces +25.594 h, and
sepsis +156.347 h.

The separate Road Traffic 10,000-case check is mixed rather than dominant:
classification again stays unchanged, raw-hour soft-kNN changes MAE by
+14.259 h, RMSE by -52.099 h, and median absolute error by +142.428 h. This
suggests the raw average can soften some extreme errors on that one
distribution, but it is not a replacement for the learned rescaling head.

Reproduce with:

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
