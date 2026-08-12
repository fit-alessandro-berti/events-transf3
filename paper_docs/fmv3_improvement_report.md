# FM-v3 architecture audit and improvement report

> **Architecture role:** this report covers the intermediate corrected neural
> checkpoint. The selected final system adds structured transition memory on
> top. See [`fmv3_architecture_changes.md`](fmv3_architecture_changes.md) for the
> complete before/after architecture and
> [`structured_fmv3_report.md`](structured_fmv3_report.md) for the final result.
> The later learned remaining-time architecture and raw-hour MAE/RMSE
> confirmation are documented in
> [`fmv3_time_transform_report.md`](fmv3_time_transform_report.md).

## Outcome

The corrected FM-v3 checkpoint improves the primary metric over FM-v2 by **+0.0059 balanced accuracy** across 200 paired natural-support rows. Ordinary accuracy changes by **+0.0075** and macro-F1 by **+0.0050**. It also improves balanced accuracy by **+0.0016** over the stronger classification-focused continuation control.

After averaging the nested learning curves within each log, the paired balanced-accuracy gain is **+0.0061** with a 95% cluster-bootstrap interval of **[+0.0022, +0.0104]** (5 log-level clusters; 10,000 resamples).

The selected artifact is `checkpoints/fmv3/corrected_fmv3/model_epoch_23.pth`; its resolved configuration is `checkpoints/fmv3/corrected_fmv3/training_config.yaml`, with the reusable source configuration at `configs/fmv3/corrected_fmv3.yaml`.

## Defects found in the original FM-v3 implementation

1. `learn_temperature: true` enabled gradients on `logit_scale`, but the FM-v3 evidence paths divided by fixed Python temperatures and never used that parameter. Temperature learning was therefore a no-op.
2. The local FM-v3 paths dropped FM-v2's neighbourhood mean-centering while supposedly changing only aggregation. The γ=0 control consequently changed both evidence aggregation and similarity geometry.
3. Evaluation retrieved neighbours in expert 0's embedding space and reused those indices for all four experts, despite each expert being trained and encoded independently.
4. Global–local `logaddexp` fusion allowed noisy global prototypes to perturb every locally supported decision. Learned shrinkage, gate, γ, and abstention parameters barely moved, while missing-pool losses were disproportionately large.
5. Classification and remaining-time tasks were sampled 50/50 even though low-data next-activity balanced accuracy is the stated FM-v3 objective.

## Corrective design

The new `coverage_fallback` head retains the centered legacy local ordering. A class absent from top-k can enter only if its full-pool prototype exceeds the best locally represented prototype by a fixed margin. Thus global memory addresses candidate coverage without rewriting strong local decisions. Training uses 80% classification steps, removes the unstable missing-pool abstention objective, and assigns 25% of classification episodes to deliberate missing-local-label failures. Retrieval is expert-specific at evaluation. An inference-only fallback temperature calibrates rows where global fallback candidates are present.

The selected checkpoint continues the common FM-v2 epoch-20 checkpoint through epoch 23. Epoch 25 was evaluated and rejected because balanced accuracy had begun to regress.

## Primary classification means

| variant                             |   balanced_accuracy |   accuracy |   macro_f1 |   zero_recall_fraction |    nll |   brier |    ece |   aurc |
|:------------------------------------|--------------------:|-----------:|-----------:|-----------------------:|-------:|--------:|-------:|-------:|
| Corrected FM-v2 evaluator           |              0.4136 |     0.6762 |     0.3936 |                 0.4219 | 3.3469 |  0.4901 | 0.1388 | 0.2072 |
| Classification-focused continuation |              0.4178 |     0.6828 |     0.3971 |                 0.4208 | 3.3170 |  0.4812 | 0.1371 | 0.2010 |
| Corrected FM-v3                     |              0.4194 |     0.6837 |     0.3986 |                 0.4209 | 3.2898 |  0.4945 | 0.1586 | 0.2165 |

## Paired corrected FM-v3 minus corrected FM-v2

Positive deltas are improvements for accuracy metrics and regressions for zero-recall/calibration-error metrics. Sign counts are over identical log, repetition, and case-budget rows.

| metric                                     |   mean_delta |   positive_rows |   zero_rows |   negative_rows |
|:-------------------------------------------|-------------:|----------------:|------------:|----------------:|
| balanced_accuracy                          |       0.0059 |             111 |          41 |              48 |
| accuracy                                   |       0.0075 |             104 |          53 |              43 |
| macro_f1                                   |       0.0050 |             114 |          25 |              61 |
| zero_recall_fraction                       |      -0.0010 |               5 |         187 |               8 |
| macro_decision_given_retrieval             |       0.0098 |             110 |          35 |              55 |
| conditional_balanced_accuracy_pool_covered |       0.0093 |             111 |          41 |              48 |
| nll                                        |      -0.0570 |              81 |           1 |             118 |
| multiclass_brier                           |       0.0045 |              99 |           1 |             100 |
| ece_10                                     |       0.0198 |             118 |           1 |              81 |
| aurc                                       |       0.0093 |             119 |           1 |              80 |

## Per-log predictive deltas

| log                  |   balanced_accuracy_delta |   accuracy_delta |   macro_f1_delta |   zero_recall_delta |
|:---------------------|--------------------------:|-----------------:|-----------------:|--------------------:|
| billing              |                    0.0081 |           0.0030 |           0.0076 |             -0.0075 |
| helpdesk             |                   -0.0002 |           0.0021 |          -0.0003 |              0.0036 |
| receipt              |                    0.0037 |           0.0169 |           0.0035 |             -0.0013 |
| roadtraffic100traces |                    0.0140 |           0.0140 |           0.0110 |              0.0000 |
| sepsis               |                    0.0049 |           0.0029 |           0.0041 |             -0.0000 |

## Learning-curve deltas

|   case_budget |   n_logs |   balanced_accuracy_delta |   accuracy_delta |   macro_f1_delta |
|--------------:|---------:|--------------------------:|-----------------:|-----------------:|
|        1.0000 |   5.0000 |                    0.0139 |           0.0213 |           0.0148 |
|        2.0000 |   5.0000 |                    0.0082 |           0.0149 |           0.0079 |
|        4.0000 |   5.0000 |                    0.0073 |           0.0127 |           0.0065 |
|        8.0000 |   5.0000 |                    0.0043 |           0.0082 |           0.0041 |
|       16.0000 |   5.0000 |                    0.0008 |           0.0040 |           0.0012 |
|       32.0000 |   5.0000 |                    0.0012 |           0.0005 |           0.0005 |
|       43.0000 |   1.0000 |                   -0.0259 |          -0.0339 |          -0.0376 |
|       64.0000 |   4.0000 |                    0.0122 |           0.0053 |           0.0103 |
|      128.0000 |   4.0000 |                    0.0101 |           0.0034 |           0.0082 |
|     1000.0000 |   1.0000 |                   -0.0080 |          -0.0088 |          -0.0099 |

## Remaining-time deltas

Negative MAE deltas and positive skill/D²/R² deltas are improvements.

| metric                      |   mean_delta |
|:----------------------------|-------------:|
| mae_hours                   |      -1.6767 |
| median_absolute_error_hours |      -1.7381 |
| mae_skill_vs_median         |       0.0040 |
| d2_absolute_error           |       0.0040 |
| r2                          |       0.0039 |

## Protocol and scope

The confirmation uses five event logs, five repeated nested natural-support samples, absolute case budgets 1–128 plus eligible full-support rows, a fixed case-disjoint query set capped at 1,000 prefixes, balanced prior, k=20, and 200 case-bootstrap repetitions per row. All comparisons are paired on identical support/query rows. The experiment improves the repository benchmark; the same five logs were used during architecture screening, so claims beyond this benchmark require additional external logs.
