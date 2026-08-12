# FM-v3 experiment manifest

For the architectural relationship among these runs, the selected final
pipeline, and the distinction between training and inference changes, see
[`fmv3_architecture_changes.md`](fmv3_architecture_changes.md).

| Checkpoint | Change relative to preceding conceptual baseline |
|---|---|
| `minus1_fmv1_retrained` | FM-v1-style balanced episodic pretraining under the current data/code environment |
| `00_fmv2` | Historical summed local support mass and guaranteed-positive balanced retrieval training |
| `01_realistic_episodes` | Natural, long-tail, and random-shot retrieval episodes |
| `02_count_neutral` | Count-normalized local evidence (γ=1) and balanced prior |
| `03_global_prototypes` | Full support pool defines candidates via global class prototypes |
| `04_global_shrinkage` | Learned count-dependent prototype shrinkage |
| `05_global_local` | Fixed-gate fusion of local instances and global prototypes |
| `06_full_fmv3` | Learned γ, dynamic gate, missing-label training, and abstention |
| `07_full_no_pretraining` | Randomly initialized full architecture, saved at epoch 0 |
| `08_gamma0` | Frequency-sensitive local aggregation control |
| `09_gamma_learned` | Learned count normalization control |

## Post-audit improvement experiments

| Checkpoint | Purpose |
|---|---|
| `10_fmv2_cls80` | FM-v2 continuation with 80% classification steps; stronger training control |
| `11_corrected_fallback_m05` | Conservative coverage fallback with margin 0.5 |
| `12_corrected_fallback_m10` | Conservative coverage fallback with margin 1.0 |
| `13_corrected_fallback_centered` | Margin-1 fallback with centered global prototypes |
| `corrected_fmv3` | Selected margin-1 epoch-23 checkpoint with adaptive fallback calibration |

## Final inference-only augmentation

| Variant | Purpose |
|---|---|
| `structured_fmv3` | Frozen `corrected_fmv3` checkpoint plus class-balanced order-1--3 transition memory and support-count-gated probability fusion |

`structured_fmv3` is not a separately trained checkpoint. It uses
`configs/fmv3/structured_memory_eval.yaml` as an evaluation-only overlay and
stores results under `evaluation_results/fmv3_improved/structured_fmv3/`.

The post-audit runs continue the common `00_fmv2` epoch-20 checkpoint. The
screening manifest is `configs/fmv3/improvement_manifest.yaml`; the selected
configuration is `configs/fmv3/corrected_fmv3.yaml`, and the confirmation
manifest is `configs/fmv3/improved_evaluation_manifest.yaml`. Paired results
and the implementation audit are in `paper_docs/fmv3_improvement_report.md`.
The final end-to-end comparison is in
`paper_docs/structured_fmv3_report.md`.

## Temporal architecture sequence

| Checkpoint or variant | Purpose | Status |
|---|---|---|
| `learned_time_4_temporal`, epoch 33 | Shared two-clock adapter used only by regression plus four learned target transforms | Superseded input architecture; historical full result retained |
| `learned_time_independent_4`, epoch 34 | Separate four-branch start/previous encoders used by both tasks; legacy logged clocks retained | Screen ablation |
| `learned_time_independent_8`, epoch 34 | Eight-branch capacity ablation for both independent input clocks | Screen ablation |
| `learned_time_independent_4_cls70`, epoch 34 | Four-branch architecture with 70% classification training episodes | Selected immediate predecessor |
| `learned_time_independent_4_replace`, epoch 34 | Separate four-branch input encoders for both tasks; old logged clock coordinates removed | Classification-favoring full-confirmation ablation |
| `prefix_state_attention_70`, epoch 35 | Dynamic CLS/last-state query and learned recency; only the prefix adapter trained | Full-confirmation ablation |
| `prefix_state_attention_no_recency`, epoch 35 | Same state path with near-zero initial recency | Full-confirmation ablation |
| `prefix_state_attention_80`, epochs 35--38 | Prefix-only path, 80% classification sampling, stronger initial recency | Screen ablation |
| `prefix_state_attention_joint`, epoch 35 | Prefix adapter plus both clock encoders and regression bank | Full-confirmation checkpoint candidate |
| `prefix_state_attention_joint`, epoch 36 | Same joint scope after a second continuation epoch | Selected Stage-6 architecture; predecessor to promoted model |
| Original state-aware epoch-38 checkpoints | End-of-schedule candidates without the promoted loss | Rejected by development screen |
| `loss_multimetric_gate_aux_005`, epoch 38 | Epoch-36 continuation with the multi-metric primary loss and 0.05 gate auxiliary | **Selected current checkpoint** |
| `expert_confidence_heads`, epochs 39--40 | Confidence-only continuation from the selected checkpoint; learns per-expert aggregation logits for classification and regression | Full-confirmation ablation; not promoted |

The selected independent-temporal predecessor completed epoch 34 and was
intentionally stopped during epoch 35 when that experiment was closed. Its
architecture, migration rules,
screen comparison, raw-hour metric definition, and reproduction commands are
documented in
[`fmv3_independent_temporal_report.md`](fmv3_independent_temporal_report.md).
Screen results live under `evaluation_results/independent_temporal/screens/`;
the selected full confirmation lives under
`evaluation_results/independent_temporal/confirmation_learned_time_independent_4_cls70_e34/`.

The state-aware runs were seeded from that epoch-34 checkpoint and continued
through epoch 38 without modifying the frozen Transformer or historical prefix
path. Four variants were screened at epochs 35, 36, and 38; four informative
checkpoints received the full paired confirmation. Joint epoch 36 became the
Stage-6 architecture base. The promoted loss experiment then continued that
exact checkpoint through epoch 38 and is now the selected repository model.
Its promotion evidence, Road Traffic limitation, raw-hour no-rescaling
ablation, and learned per-expert confidence ablation are in
[`fmv3_multimetric_loss_report.md`](fmv3_multimetric_loss_report.md); the
architecture audit remains in
[`fmv3_prefix_attention_report.md`](fmv3_prefix_attention_report.md). Screen
results live under `evaluation_results/prefix_attention/screens/` and
`evaluation_results/loss_multimetric/screens/`; full results live under their
corresponding `confirmations/` directories. The confidence-head ablation
results live under `evaluation_results/expert_confidence/`.

`configs/fmv3/selected.yaml` is the stable configuration alias for the
promoted checkpoint; it resolves to `loss_multimetric_gate_aux_005` with
`selected_checkpoint_epoch: 38`.

Historical manifest folders contain their resolved `training_config.yaml`,
serialized loader artifacts, and final `model_epoch_*.pth`. Independent
temporal continuation folders retain both the seeded epoch-33 checkpoint and
the completed epoch-34 checkpoint so the migration remains reproducible. The
state-aware folders retain the epoch-34 seed and saved continuation checkpoints.
The original manifest is machine-readable at `configs/fmv3/manifest.yaml`;
the newer temporal and prefix variants are the explicit YAML files listed
above.
