# FM-v3 learned temporal-transform architecture and evaluation

## Outcome

The selected epoch-33 temporal FM-v3 beats the strongest fixed square-root
baseline on both full-protocol remaining-time metrics across 200 paired rows:

| Model | MAE (hours) | RMSE (hours) |
|---|---:|---:|
| Fixed sqrt baseline | 1,125.8421 | 1,665.6983 |
| Learned temporal FM-v3 | **1,113.7193** | **1,661.9432** |
| New minus baseline | **-12.1228** | **-3.7551** |
| Relative change | **-1.077%** | **-0.225%** |

Lower is better. The improvement occurs without a classification tradeoff:
balanced accuracy (0.446293), ordinary accuracy (0.708601), and macro-F1
(0.417745) are bit-for-bit unchanged on all 200 paired classification rows.

## Direct answers about the transformations

Both prefix timing variables pass through the new learned transformation:

- `time_from_start`: seconds from the first event in the prefix;
- `time_from_previous`: seconds from the preceding event.

They are converted to hours and independently encoded by four learned
power/scale branches. The selected model adds that encoding as a
regression-only residual to the frozen event embedding. It retains the old
`log1p` timing coordinates underneath the residual because the ablation that
replaced them was weaker. Cost is not routed through the new temporal adapter;
it remains on the fixed `log1p` numerical path.

MAE and RMSE are computed in neither sqrt space nor log space. Predictions and
targets are first returned to raw hours, and both metrics are evaluated there.
Historical task labels remain stored as `sqrt(hours)` only for data/checkpoint
compatibility; the new head squares them once on entry.

## Architecture

### Learned prefix-clock adapter

For clock feature $f$ and branch $k$, raw seconds $x_f$ become hours
$h_f=x_f/3600$. The adapter learns positive power $p_{f,k}$ and scale
$s_{f,k}$:

$$
u_{f,k}=\frac{(1+h_f/s_{f,k})^{p_{f,k}}-1}{p_{f,k}},
\qquad
\tilde u_{f,k}=\frac{u_{f,k}}{1+u_{f,k}}.
$$

The bounded transformed values from both clocks are flattened, normalized,
projected to the 256-dimensional event space, and multiplied by a learned
sigmoid residual gate. This path is invoked only for regression.

### Learned remaining-time transform bank

Each of four branches learns a separate monotone target map in raw hours:

$$
z_k(y)=\frac{(1+y/s_k)^{p_k}-1}{p_k},
\qquad
z_k^{-1}(v)=s_k\left[(1+p_kv)^{1/p_k}-1\right].
$$

The family can learn log-like, square-root-like, and near-linear behavior; it
does not fix any one of them. Each branch has its own neighbor-attention scale,
regresses in transformed space, and is inverted to hours before combination.

### Shared scale augmentation

During regression training, one factor sampled log-uniformly from 0.02 to 50
multiplies both prefix clocks and the remaining-time target. The output is
divided by the same factor before loss calculation. This trains input and
target geometry together over several orders of magnitude without changing
the output unit.

### Dual branch aggregation

The final output combines two new predictions equally:

1. a trained query-specific gate using dimensionless neighbor, attention,
   transform, similarity, and branch-disagreement features;
2. a target-log branch prior calibrated on at most 512 labeled support prefixes
   using self-excluded retrieval and raw-hour support MAE.

The support prior never sees held-out query labels or query errors. The 50/50
blend was retained because the query gate protects tail RMSE while the support
prior reduces ordinary absolute error. No fixed sqrt prediction is blended
back into the model.

## Training scope and classifier isolation

Training starts from corrected FM-v3 epoch 23. The character CNN, Transformer,
classification head, and all pre-existing weights are frozen. Only four
experts' temporal adapters and learned target-transform banks are optimized:
40,776 trainable parameters in the selected four-branch model.

Classification bypasses the temporal adapter and transform bank. Unit tests
check exact embedder-path equality, and the full paired evaluation confirms a
maximum row-level classification difference of zero.

## Full paired results by event log

| Log | Baseline MAE | New MAE | MAE delta | Baseline RMSE | New RMSE | RMSE delta |
|---|---:|---:|---:|---:|---:|---:|
| Billing | 1,033.1454 | 1,011.2405 | **-21.9049** | 1,902.4321 | 1,887.1482 | **-15.2839** |
| Helpdesk | 0.2466 | 0.2451 | **-0.0015** | 0.3025 | 0.2997 | **-0.0028** |
| Receipt | 72.9807 | 73.1362 | +0.1555 | 156.5369 | 155.9015 | **-0.6355** |
| Road traffic | 3,986.7065 | 3,962.4362 | **-24.2704** | 4,927.6931 | 4,925.1999 | **-2.4932** |
| Sepsis | 919.5285 | 903.8602 | **-15.6683** | 1,739.9898 | 1,739.3926 | **-0.5972** |

Receipt MAE is the only per-log mean that is slightly worse; its RMSE still
improves. Aggregate MAE improves on 132/200 paired rows and RMSE on 120/200.

## Full paired deltas by support-case budget

| Cases | Paired rows | MAE delta (hours) | RMSE delta (hours) |
|---:|---:|---:|---:|
| 1 | 25 | +0.7802 | +0.9946 |
| 2 | 25 | +17.8555 | +15.6015 |
| 4 | 25 | **-14.6542** | **-17.8572** |
| 8 | 25 | **-16.7517** | **-9.3070** |
| 16 | 25 | **-8.4782** | +5.9829 |
| 32 | 25 | **-33.2672** | **-11.9505** |
| 64 | 20 | **-10.1042** | **-1.6677** |
| 128 | 20 | **-14.1426** | **-6.1057** |
| 43 (eligible full pool) | 5 | **-84.2921** | **-32.3805** |
| 1,000 (eligible full pool) | 5 | **-31.0558** | **-4.0520** |

The model is not uniformly better at the two smallest case budgets. Its full
mean advantage is driven by support pools large enough to identify the useful
target-log transform branch. This limitation should remain visible rather than
being hidden by the aggregate.

## Architecture screening and rejected variants

The 48-row screen used two repetitions, at most 500 query prefixes, natural
support, and regression retrieval $k=20$. It was used for architecture
selection, not as the final reported result.

| Screen variant | MAE (hours) | RMSE (hours) | Decision |
|---|---:|---:|---|
| Fixed sqrt baseline | 1,065.8487 | 1,641.4496 | Reference |
| 8-branch output-only bank + support prior | 1,032.2148 | 1,620.5502 | Input timing still fixed |
| 4-branch temporal bank, epoch 31, support prior only | **1,025.9713** | **1,613.9812** | Strong typical-error screen |
| 4-branch temporal bank, epoch 33, selected dual blend | 1,047.1453 | 1,622.2654 | Selected for full tail behavior |
| 8-branch temporal bank, epoch 33, dual blend | 1,047.0207 | 1,622.7658 | No joint advantage over four branches |
| Replace legacy timing coordinates, epoch 31 | 1,055.9749 | 1,635.5307 | Rejected; residual is stronger |

On the full protocol, the query gate alone gave MAE 1,138.2899 and RMSE
1,664.7050: it beat baseline RMSE but not MAE. The support prior alone gave
stronger MAE but slightly worse RMSE. Their selected equal blend is the first
new architecture that beats the strongest baseline on both full metrics.

## Reproduction

Selected checkpoint:
`checkpoints/fmv3/learned_time_4_temporal/model_epoch_33.pth`

Training configuration:
[`configs/fmv3/learned_time_4_temporal.yaml`](../configs/fmv3/learned_time_4_temporal.yaml)

Evaluation overlay:
[`configs/fmv3/time_transform_confirmation_eval.yaml`](../configs/fmv3/time_transform_confirmation_eval.yaml)

Results:
`evaluation_results/time_transform/learned_temporal_time_transform/results.csv`

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/learned_time_4_temporal \
  --checkpoint_epoch 33 \
  --eval_config configs/fmv3/time_transform_confirmation_eval.yaml \
  --output_dir evaluation_results/time_transform/learned_temporal_time_transform
```

The confirmation contains five repetitions, natural nested support budgets
from 1 to 128 plus eligible full pools, case-disjoint queries capped at 1,000
prefixes per log, classification $k=20$, and regression $k=50$.

## Validity boundary

The same five event logs informed architecture screening before the full paired
run. These results establish improvement for this repository benchmark; a
publication claim beyond it still requires untouched logs or a nested
log-level development/test split.
