# FM-v3 experiment manifest

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

The post-audit runs continue the common `00_fmv2` epoch-20 checkpoint. The
screening manifest is `configs/fmv3/improvement_manifest.yaml`; the selected
configuration is `configs/fmv3/corrected_fmv3.yaml`, and the confirmation
manifest is `configs/fmv3/improved_evaluation_manifest.yaml`. Paired results
and the implementation audit are in `paper_docs/fmv3_improvement_report.md`.

Every folder contains its resolved `training_config.yaml`, serialized loader artifacts, and one final `model_epoch_*.pth`. The manifest is machine-readable at `configs/fmv3/manifest.yaml`.
