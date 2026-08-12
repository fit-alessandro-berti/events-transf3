# Events Transformer

Events Transformer is a meta-learning Transformer for process mining event logs (XES). It learns from multiple event logs and adapts to new logs with few examples, targeting two tasks:

- **Next-activity prediction** (classification).
- **Remaining-time prediction** (regression).

It supports two embedding strategies for event attributes:

- **learned**: character-level CNN embeddings.
- **pretrained**: Sentence-Transformers embeddings for activity/resource names.

## Project layout

- `main.py`: training entry point.
- `testing.py`: evaluation entry point.
- `config.py`: default configuration and log paths.
- `data_generator.py`: XES loader and feature/embedding preparation.
- `components/`: model components (Transformer, MoE, meta-learner heads).
- `evaluation/`: evaluation routines (meta-learning and retrieval-augmented).
- `logs/`: sample XES logs and a simulation script.

## Setup

1) Create and activate a Python environment.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

## Data expectations

The loader expects XES logs with these event attributes:

- `concept:name` (activity label)
- `time:timestamp` (event timestamp)
- `org:resource` (resource name; missing values default to `Unknown`)
- `amount` (cost; missing values default to 0.0)

Default training/testing logs are configured in `config.py` under `CONFIG['log_paths']`. Sample logs are already in `logs/`.

## Training

Run training with the defaults:

```bash
python main.py --checkpoint_dir ./checkpoints
```

Common options:

- `--embedding_strategy learned|pretrained`
- `--training_strategy episodic|retrieval|mixed`
- `--resume` (resume from latest checkpoint)
- `--stop_after_epoch N`

The script saves checkpoints and artifacts in `--checkpoint_dir`.

## Evaluation

Evaluate a trained model against a test log key from `config.py`:

```bash
python testing.py --checkpoint_dir ./checkpoints --test_log_name D_unseen
```

You can also pass a direct path to a `.xes` or `.xes.gz` file:

```bash
python testing.py --checkpoint_dir ./checkpoints --test_log_name ./logs/00013_clos2rep.xes.gz
```

To run retrieval-augmented evaluation:

```bash
python testing.py \
  --checkpoint_dir ./checkpoints \
  --test_log_name D_unseen \
  --test_mode retrieval_augmented \
  --test_retrieval_k 1 5 10 20 \
  --test_retrieval_prediction_mode proto_head \
  --test_retrieval_report_confidence_buckets
```

Use `--test_retrieval_prediction_mode foundation_knn` to bypass prototypical heads and predict directly with kNN over foundation-model feature embeddings.
Confidence-bucket reporting uses 5 dynamic buckets (equal-sized by confidence rank) and is applied only when `--test_retrieval_prediction_mode proto_head`.

## FM-v3 experiments

FM-v3 is configured through composable YAML files under `configs/fmv3/`. Any scalar or list can also be overridden without editing code:

For a complete explanation of the selected architecture—including what changed
at training time, what changed only at inference, the equations for coverage
fallback and structured-memory fusion, and the rejected alternatives—start
with [`paper_docs/fmv3_architecture_changes.md`](paper_docs/fmv3_architecture_changes.md).

```bash
python main.py \
  --config configs/fmv3/06_full_fmv3.yaml \
  --checkpoint_dir checkpoints/fmv3/06_full_fmv3 \
  --set fmv3_head.prior_mode=natural
```

Train the complete ablation manifest on four GPUs:

```bash
python run_fmv3_training.py
```

Run the repeated, case-level low-data evaluation on all logs in `logs_eval/`:

```bash
python run_fmv3_evaluation.py --resume
python run_fmv3_baselines.py --resume
python generate_fmv3_report.py
```

The post-audit corrected checkpoint and its focused comparison are available at:

- `checkpoints/fmv3/corrected_fmv3/model_epoch_23.pth`
- `configs/fmv3/corrected_fmv3.yaml`
- `paper_docs/fmv3_improvement_report.md`

That checkpoint is the neural base of the final structured FM-v3. The
improvement report documents the intermediate neural correction; it is not the
final end-to-end result by itself.

Regenerate the paired improvement report with:

```bash
python generate_fmv3_improvement_report.py
```

The stronger structured-memory inference branch combines the corrected FM-v3
posterior with a log-local, class-balanced activity-transition memory. Run its
frozen full protocol and regenerate the paired report with:

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/corrected_fmv3 \
  --logs_dir logs_eval \
  --eval_config configs/fmv3/structured_memory_eval.yaml \
  --output_dir evaluation_results/fmv3_improved/structured_fmv3
python generate_structured_fmv3_report.py
```

The transition branch backs off from the last three activities to shorter
suffixes. Its contribution is shrunk by observed context support and becomes
zero for an unseen context, leaving the foundation-model prediction unchanged.
The primary full-protocol comparison is documented in
`paper_docs/structured_fmv3_report.md`.

### State-aware prefix attention and independent clocks

**Outcome:** yes, this change improves the repository benchmark. Compared with
the immediate independent-temporal predecessor, the selected epoch-36 model is
better on every aggregate primary metric. Higher is better for the first three
columns; lower is better for MAE and RMSE.

| Model | Balanced accuracy | Accuracy | Macro-F1 | MAE (h) | RMSE (h) |
|---|---:|---:|---:|---:|---:|
| Independent temporal, epoch 34 | 0.445968 | 0.708158 | 0.417730 | 1,113.1992 | 1,661.3121 |
| **State-aware joint, epoch 36** | **0.447473** | **0.709352** | **0.418885** | **1,112.2914** | **1,660.6820** |
| Change | **+0.001504** | **+0.001194** | **+0.001155** | **-0.9078** | **-0.6301** |

The current architecture keeps parameter-disjoint learned components for the
two observable prefix clocks: elapsed time from case start and time since the
previous event. Their residuals augment the legacy `log1p` coordinates before
the Transformer and are consumed by both tasks. Cost keeps its existing
numerical path.

After the Transformer, a state-aware prefix adapter now supplements the
historical static pooling query. It builds a dynamic query from the CLS state
and last valid event, applies a learned task-specific ordinal-recency bias, and
adds a small gated residual to the historical prefix vector. Classification
and regression have separate query offsets, gates, and recency strengths. The
old projection remains the frozen anchor, and the feature is disabled by
default for historical configurations.

The regression head is a third independent learned component. It learns four
monotone target transforms and returns every branch to raw hours before
aggregation. Consequently, MAE and RMSE are computed in raw hours—not in
square-root or log space.

“Time from the end” is not an input feature: the true time until case end is
the remaining-time label, so passing it to either task would leak the answer.
The second observable prefix clock is `time_from_previous`.

Start with
[`paper_docs/fmv3_prefix_attention_report.md`](paper_docs/fmv3_prefix_attention_report.md)
for the current design, bottleneck audit, equations, ablations, diagnostics,
evaluation, and reproduction commands. The full component history is in
[`paper_docs/fmv3_architecture_changes.md`](paper_docs/fmv3_architecture_changes.md),
while the immediate predecessor is recorded in
[`paper_docs/fmv3_independent_temporal_report.md`](paper_docs/fmv3_independent_temporal_report.md)
and
[`paper_docs/fmv3_time_transform_report.md`](paper_docs/fmv3_time_transform_report.md)
records the superseded shared, regression-only temporal adapter.

Reproduce the selected confirmation with:

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/prefix_state_attention_joint \
  --checkpoint_epoch 36 \
  --eval_config configs/fmv3/prefix_attention_confirmation_eval.yaml \
  --output_dir evaluation_results/prefix_attention/confirmations/prefix_state_attention_joint_e36
```

On the full paired protocol, the selected checkpoint reaches balanced accuracy
`0.447473`, ordinary accuracy `0.709352`, macro-F1 `0.418885`, MAE
`1,112.2914` hours, and RMSE `1,660.6820` hours. Relative to the immediate
independent-temporal predecessor, all five primary means improve: `+0.001504`,
`+0.001194`, `+0.001155`, `-0.9078` hours, and `-0.6301` hours, respectively.
MAE and RMSE are `13.5507` and `5.0163` hours below the fixed-sqrt structured
baseline.

The three-way corrected baseline/control/FM-v3 evaluation manifest is
`configs/fmv3/improved_evaluation_manifest.yaml`.

The corrected `coverage_fallback` head keeps FM-v2's centered local decision
rule and admits a globally available but locally missing label only when its
prototype clears a configurable margin. Evaluation retrieves neighbours in
each expert's own embedding space.

The selected run is a continuation of `00_fmv2` epoch 20. To reproduce the
training schedule in a new directory, seed that directory with the epoch-20
checkpoint and training artifacts, then resume with the corrected config and
retain epoch 23:

```bash
cp -a checkpoints/fmv3/00_fmv2 checkpoints/fmv3/corrected_fmv3_reproduction
python main.py \
  --config configs/fmv3/corrected_fmv3.yaml \
  --checkpoint_dir checkpoints/fmv3/corrected_fmv3_reproduction \
  --resume \
  --stop_after_epoch 23
```

The primary classification endpoint is balanced accuracy. The evaluator also records ordinary accuracy, macro-F1/precision, per-class recall, zero-recall classes, pool and retrieval label coverage, frequency-stratified recall, NLL, multiclass Brier score, reliability bins, risk–coverage curves, and case-bootstrap intervals. Remaining-time outputs include raw-hour MAE and RMSE, median absolute error, normalized MAE, MAE/RMSE skill, D² absolute-error score, R², and interval coverage/width.

## Simulating new logs (optional)

Generate synthetic XES logs using pm4py:

```bash
python logs/perform_simulation.py --output logs/simulated_log.xes.gz --num-logs 3
```

Then point `config.py` to the new files or pass them directly to `testing.py`.

## Notes

- GPU is optional; the code uses CUDA if available.
- The pretrained embedding strategy downloads the Sentence-Transformers model specified in `config.py`.

## Feature Dependency Analysis

You can analyze how strongly activity/path indicators depend on time features in prefix vectors:

```bash
python files/prefix_activity_time_dependency.py ./logs/00013_clos2rep.xes.gz
```

The script reports distance correlation, MI summaries, and predictive dependence with permutation gaps.
