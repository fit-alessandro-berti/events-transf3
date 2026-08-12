# Structured FM-v3 architecture and paired evaluation

> **Selected final classifier.** For a component-by-component explanation of the
> FM-v2 baseline, the corrected neural head, the structured-memory equations,
> training versus inference changes, and rejected alternatives, see
> [`fmv3_architecture_changes.md`](fmv3_architecture_changes.md).
> Remaining-time inference was redesigned subsequently; see
> [`fmv3_time_transform_report.md`](fmv3_time_transform_report.md).

## Outcome

The frozen structured-memory FM-v3 improves the corrected FM-v2 evaluator by
**+0.0327 balanced accuracy**, **+0.0324 accuracy**,
and **+0.0242 macro-F1** across 200 identical
support/query rows. The isolated increment over the intermediate corrected
FM-v3 is **+0.0269**, **+0.0249**, and **+0.0191**,
respectively.
After averaging the nested curves within each log, the end-to-end
FM-v2-to-structured-FM-v3 balanced-accuracy gain is **+0.0329**,
with a 95% cluster-bootstrap interval of **[+0.0229, +0.0418]**.
The isolated structured-memory gain is **+0.0268** with interval
**[+0.0186, +0.0350]** (5 log-level clusters;
10,000 resamples each).

## Architectural bottleneck

FM-v3 compresses a prefix into a generic embedding and retrieves globally by
cosine similarity. That representation does not enforce the discrete process
state already observed in the prefix: its last activity and recent activity
suffix. Consequently, semantically similar prefixes can retrieve examples from
different outgoing transitions, especially as the support pool grows.

The new branch stores log-local suffix-to-next-activity counts for orders 1--3.
It uses a uniform next-class prior and a smoothed class-conditional suffix
likelihood, preferring the longest observed suffix and backing off when it is
unseen. This explicitly optimizes rare-class evidence rather than replaying the
natural class frequency. Its posterior is mixed with corrected FM-v3 using
`lambda(s) = 0.75 * n(s) / (n(s) + 0.5)`, where `n(s)` is support count for the
selected suffix. An unseen suffix has zero weight, so the branch collapses to
the FM prediction rather than guessing.

The order, smoothing, mixture weight, and shrinkage constant were selected on a
smaller two-repetition diagnostic protocol. They were then frozen before this
five-repetition full evaluation. Across the confirmation rows, the structured
memory covers **93.3%** of queries, chooses suffix order
**2.55** on average, and receives mean effective mixture
weight **0.606** after support shrinkage.

### Low-support refinement on the current selected checkpoint

A later small inference-only refinement targets the lowest-data regime without
changing trained weights. The standard fusion remains
`0.75 * n(s) / (n(s) + 0.5)`, but when the labeled support pool has at most
eight prefixes it temporarily uses a stronger low-support setting:
`1.0 * n(s) / (n(s) + 0.25)`. This threshold was chosen because, in the
five-log screen, case-budget 1 contained only 2--7 support prefixes while
case-budget 4 started at 9 support prefixes. The refinement is implemented by
`structured_low_support_threshold`, `structured_low_support_weight`, and
`structured_low_support_tau`; the default threshold is zero, so historical
evaluations are unchanged unless the overlay enables it.

On the current selected epoch-38 checkpoint, the full classification-only
confirmation improves the primary classification metrics while leaving all
medium/high-support rows identical:

| Evaluation overlay | Balanced accuracy | Accuracy | Macro-F1 | NLL | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Selected structured fusion | 0.447740 | 0.709221 | 0.419179 | **3.055258** | **0.170634** |
| Low-support adaptive fusion | **0.451092** | **0.717033** | **0.422542** | 3.063743 | 0.184301 |
| Adaptive minus selected | +0.003351 | +0.007812 | +0.003363 | +0.008485 | +0.013667 |

The improvement is concentrated where intended. At case budget 1, balanced
accuracy changes by **+0.020244**, ordinary accuracy by **+0.046329**, and
macro-F1 by **+0.021149**. At case budget 2, the changes are **+0.006568**,
**+0.016167**, and **+0.005754**. Rows at budgets 8 and above are exactly
unchanged; one budget-4 support pool crossed the prefix threshold but did not
change predictions. Calibration metrics worsen slightly, so this refinement is
a decision-quality improvement for low support, not a calibration improvement.

### Threshold extension screen on the current endpoint

Because subsequent work no longer optimizes only the ultra-small support
regime, the stronger low-support rule was also screened on the current
regression-confidence endpoint with thresholds 16, 32, and 64. The test used
classification-only rows on the same five logs, support scenarios, and
repetitions, with budgets 1, 4, 8, 16, 32, 64, 128, plus eligible full-support
rows. It is not promoted.

| Strong-rule threshold | Rows | Balanced accuracy Δ | Accuracy Δ | Macro-F1 Δ | NLL Δ | ECE-10 Δ | Brier Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 prefixes | 175 | -0.001026 | -0.002051 | -0.000824 | +0.007436 | +0.004425 | +0.003207 |
| 32 prefixes | 175 | -0.002121 | -0.003546 | -0.002560 | +0.013937 | +0.007777 | +0.006329 |
| 64 prefixes | 175 | -0.000714 | -0.004476 | -0.002714 | +0.025455 | +0.016212 | +0.012488 |

Threshold 64 gives a very small balanced-accuracy lift when restricted to
budgets ≥8 (`+0.000741`), but that comes with lower accuracy (`-0.002168`),
lower macro-F1 (`-0.002204`), and worse NLL/ECE/Brier. The endpoint therefore
keeps the threshold-8 rule: it captures the low-support decision gain without
allowing structured suffix counts to perturb medium-support rows.

Reproduce with:

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/loss_multimetric_gate_aux_005 \
  --checkpoint_epoch 38 \
  --eval_config configs/fmv3/structured_low_support_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/structured_tuning/confirmations/adaptive_low_support_w100_tau025_thr8 \
  --set 'fmv3_evaluation.tasks=[classification]'
```

Reproduce the threshold-extension screens by changing the threshold override
and output directory:

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/expert_confidence_heads \
  --checkpoint_epoch 40 \
  --eval_config configs/fmv3/regression_confidence_low_support_confirmation_eval.yaml \
  --set 'fmv3_evaluation.tasks=[classification]' \
  --set 'fmv3_evaluation.case_budgets=[1,4,8,16,32,64,128]' \
  --set fmv3_evaluation.structured_low_support_threshold=64 \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/structured_tuning/screens/endpoint_thr64_w100_tau025
```

The committed screen artifacts also include the threshold-16 and threshold-32
directories under `evaluation_results/structured_tuning/screens/`.

## Classification means

| variant                   |   balanced_accuracy |   accuracy |   macro_f1 |   zero_recall_fraction |    nll |   multiclass_brier |   ece_10 |   aurc |
|:--------------------------|--------------------:|-----------:|-----------:|-----------------------:|-------:|-------------------:|---------:|-------:|
| Corrected FM-v2 evaluator |              0.4136 |     0.6762 |     0.3936 |                 0.4219 | 3.3469 |             0.4901 |   0.1388 | 0.2072 |
| Corrected FM-v3           |              0.4194 |     0.6837 |     0.3986 |                 0.4209 | 3.2898 |             0.4945 |   0.1586 | 0.2165 |
| Structured FM-v3          |              0.4463 |     0.7086 |     0.4177 |                 0.4075 | 3.0583 |             0.4746 |   0.1724 | 0.2364 |

## Paired structured-minus-corrected deltas

For error metrics (zero recall, NLL, Brier, ECE, and AURC), negative values are
improvements. Sign counts use the corresponding favorable direction.

| metric               |   mean_delta |   improved_rows |   tied_rows |   regressed_rows |
|:---------------------|-------------:|----------------:|------------:|-----------------:|
| balanced_accuracy    |       0.0269 |             176 |           6 |               18 |
| accuracy             |       0.0249 |             158 |          19 |               23 |
| macro_f1             |       0.0191 |             157 |           6 |               37 |
| zero_recall_fraction |      -0.0134 |              29 |         165 |                6 |
| nll                  |      -0.2315 |             158 |           1 |               41 |
| multiclass_brier     |      -0.0199 |             134 |           1 |               65 |
| ece_10               |       0.0138 |              92 |           1 |              107 |
| aurc                 |       0.0199 |             120 |           2 |               78 |

## Per-log deltas

| log                  |   balanced_accuracy_delta |   accuracy_delta |   macro_f1_delta |
|:---------------------|--------------------------:|-----------------:|-----------------:|
| billing              |                    0.0368 |           0.0275 |           0.0279 |
| helpdesk             |                    0.0147 |           0.0124 |           0.0129 |
| receipt              |                    0.0390 |           0.0445 |           0.0291 |
| roadtraffic100traces |                    0.0200 |           0.0242 |           0.0104 |
| sepsis               |                    0.0235 |           0.0169 |           0.0150 |

## Learning-curve deltas

|   case_budget |   paired_rows |   context_coverage |   mean_effective_weight |   balanced_accuracy_delta |   accuracy_delta |   macro_f1_delta |
|--------------:|--------------:|-------------------:|------------------------:|--------------------------:|-----------------:|-----------------:|
|        1.0000 |       25.0000 |             0.8098 |                  0.4097 |                    0.0085 |           0.0213 |           0.0067 |
|        2.0000 |       25.0000 |             0.8859 |                  0.5026 |                    0.0133 |           0.0357 |           0.0130 |
|        4.0000 |       25.0000 |             0.9123 |                  0.5648 |                    0.0220 |           0.0396 |           0.0181 |
|        8.0000 |       25.0000 |             0.9432 |                  0.6196 |                    0.0244 |           0.0307 |           0.0212 |
|       16.0000 |       25.0000 |             0.9634 |                  0.6539 |                    0.0247 |           0.0123 |           0.0162 |
|       32.0000 |       25.0000 |             0.9714 |                  0.6791 |                    0.0340 |           0.0178 |           0.0234 |
|       43.0000 |        5.0000 |             1.0000 |                  0.7331 |                    0.0347 |           0.0508 |           0.0237 |
|       64.0000 |       20.0000 |             0.9833 |                  0.6913 |                    0.0382 |           0.0132 |           0.0263 |
|      128.0000 |       20.0000 |             0.9921 |                  0.7107 |                    0.0569 |           0.0172 |           0.0421 |
|     1000.0000 |        5.0000 |             1.0000 |                  0.7431 |                    0.0252 |           0.0370 |          -0.0240 |

## Interpretation and limitations

The gain is not a calibration-only effect: ordinary accuracy and macro-F1 rise
alongside balanced accuracy, and all five logs improve. NLL and Brier score also
improve, although ECE and AURC worsen; calibration and selective prediction
should therefore remain separate validation-only steps. The largest gains occur at medium/high support,
consistent with the structured memory becoming reliable as transition counts
accumulate.

This classification experiment is a no-gradient target-memory augmentation of
the already trained corrected checkpoint; remaining-time predictions were
deliberately unchanged in this historical comparison. The later temporal
reports evaluate that task separately. The current independent temporal input
architecture also feeds classification; see
[`fmv3_independent_temporal_report.md`](fmv3_independent_temporal_report.md).
The same five logs were used for architectural screening, so a publication
claim still requires confirmation on additional untouched logs or a nested
development/test split.

## Reproduction

Checkpoint: `checkpoints/fmv3/corrected_fmv3/model_epoch_23.pth`

Evaluation overlay: `configs/fmv3/structured_memory_eval.yaml`

Results: `evaluation_results/fmv3_improved/structured_fmv3`

Protocol: five unseen event logs, five repeated nested natural-support samples,
absolute case budgets 1--128 plus eligible full-support rows, a case-disjoint
query set capped at 1,000 prefixes per log, balanced prior, and retrieval k=20.
