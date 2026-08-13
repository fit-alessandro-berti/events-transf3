# Learned task confidence and half-expert routing

## Outcome

FM-v3 now learns a pre-execution confidence for every expert and task. At test
time, classification and regression are routed independently to exactly two of
the four experts; the other two experts are not encoded or evaluated for that
task. The promoted endpoint is:

- configuration: `configs/fmv3/expert_routing_selected.yaml`;
- checkpoint: `checkpoints/fmv3/expert_routing_bias/model_epoch_42.pth`;
- full evaluation overlay: `configs/fmv3/expert_routing_confirmation_eval.yaml`;
- full result: `evaluation_results/expert_routing/confirmations/expert_routing_bias_e42/results.csv`.

The checkpoint SHA-256 is
`334e5543c39d604a9e591515e873f71c3d075f83129242a1dfcd04a8b2825683`.

## Why a second confidence mechanism is necessary

The older confidence heads consume an expert's classification posterior or
remaining-time prediction. They can weight an ensemble, but all experts have
already run by the time those confidences exist. The new task router consumes a
16-value, expert-independent descriptor made from information available before
execution: task type, support/query sizes, prefix-length statistics, observable
elapsed/inter-event/cost statistics, and support-label shape. Query labels are
never read.

Each expert owns its own `TaskConfidenceHead`. The head predicts a reliability
logit and is trained with binary cross-entropy against:

- mean correctness for a classification episode; or
- mean `exp(-absolute_error / max(|target|, 1))` for a regression episode.

At inference, the router ranks all four cheap logits and activates the top
`ceil(num_experts * expert_active_fraction)`. The selected configuration uses
`expert_active_fraction: 0.5`, hence exactly 2/4 experts. Stable index-based
tie-breaking makes neutral initialization deterministic. Output-dependent
confidence remains available as second-stage weighting among only the selected
experts.

In the FM-v3 evaluator, routing occurs before `encode_tasks`, so rejected
experts do no encoder, retrieval-head, or aggregation work. Every JSONL row
records the learned logits/confidences, selected indices, and active/inactive
counts.

## Architecture comparison

All three variants started from the same expert-confidence epoch-40 checkpoint.
Only router parameters were trainable for epochs 41--42 (300 episodes per
epoch); the backbone and prediction heads remained frozen.

| Router | Total router parameters (4 experts) | Selected class experts | Selected regression experts | Balanced accuracy | Macro-F1 | MAE (h) | RMSE (h) |
|---|---:|---|---|---:|---:|---:|---:|
| Task-type bias | 8 | 0, 2 | 2, 3 | **0.442989** | **0.412667** | **1,027.509** | **1,601.543** |
| Linear descriptor | 68 | 1, 2 | 1, 2 | 0.441487 | 0.411859 | 1,040.758 | 1,621.305 |
| Two-layer MLP descriptor | 6,660 | 1, 2 | task-dependent | 0.441487 | 0.411859 | 1,028.703 | 1,601.686 |
| All four experts (screen reference) | n/a | 0, 1, 2, 3 | 0, 1, 2, 3 | 0.444757 | 0.415529 | 1,039.719 | 1,616.092 |

These are matched five-log development-screen means (48 classification and 48
regression rows per endpoint). The task-bias router wins among the half-expert
architectures on all four selection metrics and is promoted. The MLP did learn
log-specific regression routing (Sepsis selected experts 1,2 while the other
logs selected 2,3), but that extra capacity did not improve aggregate metrics.

## Full confirmation

The promoted bias router was then evaluated with the established 400-row
five-log protocol and the current regression-confidence/low-support overlay.

| Endpoint | Active experts | Balanced accuracy | Accuracy | Macro-F1 | MAE (h) | RMSE (h) | Normalized MAE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous current endpoint | 4/4 | 0.451092 | 0.717033 | 0.422542 | 1,099.517 | 1,652.928 | 0.827663 | -0.202875 |
| **Selected learned router** | **2/4** | 0.450242 | 0.716352 | 0.421233 | **1,097.810** | **1,651.883** | **0.824843** | **-0.202428** |
| Router minus previous | -2 | -0.000850 | -0.000680 | -0.001310 | **-1.706** | **-1.045** | **-0.002820** | **+0.000447** |

All 400 selected-endpoint JSONL rows contain `active_expert_count == 2`,
`inactive_expert_count == 2`, and `total_expert_count == 4`. Classification
selects experts 0 and 2; regression selects experts 2 and 3. Thus the requested
50% inference activation is directly verified rather than inferred from an
aggregation weight near zero.

## Reproduction

Train all three isolated router architectures:

```bash
for variant in expert_routing_bias expert_routing_linear expert_routing_mlp; do
  mkdir -p checkpoints/fmv3/$variant
  cp checkpoints/fmv3/expert_confidence_heads/model_epoch_40.pth \
    checkpoints/fmv3/$variant/model_epoch_40.pth
  cp checkpoints/fmv3/expert_confidence_heads/training_artifacts.pth \
    checkpoints/fmv3/expert_confidence_heads/training_config.pth \
    checkpoints/fmv3/expert_confidence_heads/training_config.yaml \
    checkpoints/fmv3/$variant/
done

python run_fmv3_training.py \
  --manifest configs/fmv3/expert_routing_manifest.yaml \
  --checkpoint_root checkpoints/fmv3 \
  --gpus 0 1 2 \
  --resume \
  --stop_after_epoch 42
```

Run the selected full confirmation:

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/expert_routing_bias \
  --checkpoint_epoch 42 \
  --eval_config configs/fmv3/expert_routing_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/expert_routing/confirmations/expert_routing_bias_e42
```

The core implementation is in `components/task_confidence.py`; routing and
top-half execution are in `components/moe_model.py`; evaluator-level
pre-encoding routing is in `evaluation/fmv3_protocol.py`; and training targets
are wired in both files under `training_strategies/`. The legacy meta-learning
and retrieval-augmented test utilities in `evaluation/eval_meta.py` and
`evaluation/eval_retrieval.py` also route before embedding. Both were smoke
tested on the unseen `D_unseen` log and reported the same learned split (0,2
for classification and 2,3 for regression) with two inactive experts.
