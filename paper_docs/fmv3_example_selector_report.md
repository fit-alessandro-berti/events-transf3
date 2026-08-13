# Explainable task-specific example selection

## Promotion decision

The epoch-44 task-specific example selectors are promoted over the epoch-43
loss-refinement endpoint. The selected checkpoint is:

`checkpoints/fmv3/example_selector_selected/model_epoch_44.pth`

Its SHA-256 is
`231dc6b00d5840498c8469fd470b736a35ebe46294714d5c04c540c0284b9314`.
The stable configuration is `configs/fmv3/example_selector_selected.yaml`; the
full evaluation overlay is
`configs/fmv3/example_selector_confirmation_eval.yaml`.

This change addresses a specific weakness exposed by the earlier kNN
comparison: a retrieved prefix should not be trusted merely because it is in
the top-k set. The new components learn a bounded, task-specific log weight
for every retrieved support example. They retain the transparency and
case-disjoint adaptation behavior of kNN while allowing the head to reject a
geometrically or target-inconsistent neighbor.

## Method

For support example `i`, a small MLP maps scalar diagnostics `f_i` to a
bounded residual log weight:

```text
delta_i = a * tanh(Linear(GELU(Linear(f_i))))
```

The final layer is initialized to zero, so enabling an untrained selector is
an exact identity. The maximum absolute learned log weight is 2.0. A separate
non-negative deployment strength multiplies `delta_i`; this permits
calibration after the selector is trained without changing its explanation.

The classification head adds `delta_i` to its local class-evidence logits
before the per-class log-sum-exp. Its six features are:

1. raw query/support cosine;
2. centered cosine used by the deployed head;
3. within-neighborhood similarity z-score;
4. support-example centrality;
5. leave-one-out same-class coherence; and
6. normalized support count for the example's class.

The regression head adds its own `delta_i` before neighbor attention in every
learned target-transform branch. Its seven features are the first four
geometric features above plus:

1. robust absolute log-target deviation from the neighborhood median;
2. disagreement with the geometrically nearest other support example; and
3. signed robust position relative to the support-target median.

No query activity label, query remaining time, future event, or time from the
end is available to either selector. Classification uses only retrieved
support labels; regression uses only retrieved support targets. These are the
same labeled target-log examples already authorized for few-shot adaptation.

Each classifier selector has 129 parameters per expert (516 stored across
four experts); each regressor selector has 145 parameters per expert (580
stored). Routing materializes two experts per task, so only two selectors of
the appropriate task execute for a prediction.

## Task-isolated training and merge audit

Both runs start from the exact epoch-43 checkpoint with SHA-256
`54a384b037cb83919bad9aab6bd16ca923125fcd9d93943d6b8484da3653d846`.
Each runs one 300-episode continuation epoch with learning rate `0.002` and
zero weight decay:

- classification trains only `classification_example_selector` tensors;
- regression trains only `regression_example_selector` tensors.

The merge audit accepted 16 new tensors from each continuation and zero
changes to existing tensors. Encoders, temporal components, prefix adapters,
classification/regression heads, transform bank, confidence heads, and router
are therefore byte-identical to epoch 43. The selected strengths are 0.25 for
classification and 1.0 for regression. The smaller classification value is
needed because the downstream structured-memory mixture already supplies 75%
of the final posterior.

The post-selector classification temperature is 0.65. It is a monotone map and
cannot change a selected class. The regression interval multiplier is 1.73;
it changes neither the point prediction nor the learned regression confidence.

## What the selectors learned

The committed diagnostic
`evaluation_results/example_selector/selector_diagnostics.json` scores real
top-20 neighborhoods at the 128-case natural-support budget (43 cases for the
smaller Road Traffic log), using up to 256 fixed queries per log. It covers
1,824 expert/query pairs and 36,480 support-example scores for each task.
Permutation importance means the absolute change in deployed log weight after
one observed feature is shuffled; correlated-feature effects are descriptive,
not causal.

### Classification selector

| Feature | Permutation importance | Correlation with log weight |
|---|---:|---:|
| Centered cosine | 0.405 | +0.825 |
| Neighborhood z-score | 0.373 | +0.827 |
| Normalized class support | 0.161 | -0.275 |
| Same-class coherence | 0.043 | -0.269 |
| Raw cosine | 0.011 | +0.087 |
| Support centrality | 0.008 | -0.118 |

The selector primarily rewards relative, centered relevance rather than the
nearly saturated raw cosine. Its negative association with class support
counteracts locally dominant classes, an explainable response to the
classification imbalance that made plain kNN brittle. At deployed strength,
the p90/p10 example-weight ratio is 1.164 and the selector-only effective
support count is 19.95 of 20: this is deliberately a gentle correction, not
hard pruning.

### Regression selector

| Feature | Permutation importance | Correlation with log weight |
|---|---:|---:|
| Signed target position | 0.497 | -0.823 |
| Robust target deviation | 0.171 | -0.416 |
| Neighborhood z-score | 0.136 | +0.322 |
| Centered cosine | 0.108 | +0.325 |
| Nearest-target disagreement | 0.084 | -0.319 |
| Raw cosine | 0.003 | -0.014 |
| Support centrality | 0.002 | -0.067 |

Regression retains geometrically relevant neighbors while downweighting
robust target outliers and examples that disagree with their nearest geometric
neighbor. It also learned a lower-target correction, consistent with reducing
the predecessor's overprediction on the fixed benchmark. Its p90/p10 weight
ratio is 1.825 and its selector-only effective support count is 19.23 of 20.

## Screening evidence

On the matched 48-row classification screen, the selected strength improves
balanced accuracy by `+0.000828`, accuracy by `+0.000525`, and macro-F1 by
`+0.000454`, with zero-recall unchanged. Temperature 0.65 then improves NLL by
`-0.005277`, Brier by `-0.001460`, and ECE by `-0.003226`. Macro-precision
changes by `-0.000056` and AURC by `+0.001867`.

Before structured-memory fusion, the full-strength learned classifier selector
improves the neural head's balanced accuracy by `+0.004510`, macro-F1 by
`+0.005069`, and macro-precision by `+0.005315`. This is the direct head-level
test: the selector improves the learned head itself, while the reduced
deployment strength prevents double-counting after structured fusion.

The regression screen improves MAE by `14.371` hours, RMSE by `9.131` hours,
median absolute error by `10.679` hours, normalized MAE by `0.02417`, D2 by
`0.02682`, and R2 by `0.01637`. With the preselected interval multiplier,
coverage rises by `0.00492` while mean width falls by `16.187` hours.

## Full fixed-query confirmation

The comparison uses 200 paired rows per task. The fixed baseline is
`evaluation_results/loss_refinement/confirmations/loss_refinement_selected_final_e43`;
the promoted result is
`evaluation_results/example_selector/confirmations/example_selector_selected_e44`.

### Classification

| Metric | Epoch 43 | Selected epoch 44 | Change |
|---|---:|---:|---:|
| Balanced accuracy | 0.450425 | **0.450954** | **+0.000528** |
| Accuracy | 0.716434 | **0.716476** | **+0.000042** |
| Macro-F1 | 0.421533 | **0.421786** | **+0.000252** |
| Macro-precision | **0.432456** | 0.432446 | -0.000010 |
| Zero-recall fraction | 0.404815 | **0.404552** | **-0.000263** |
| NLL | 3.002257 | **2.999596** | **-0.002662** |
| Multiclass Brier | 0.447051 | **0.446726** | **-0.000325** |
| ECE-10 | 0.118677 | **0.114190** | **-0.004487** |
| AURC | **0.240732** | 0.241919 | +0.001187 |

The main classification decision metrics improve, and the zero-recall gain is
one improvement with 199 ties. Macro-precision is effectively flat. AURC is a
real small tradeoff: the selector improves calibrated posterior quality but
does not improve the ordering of all errors by maximum probability.

### Regression

| Metric | Epoch 43 | Selected epoch 44 | Change |
|---|---:|---:|---:|
| MAE (h) | 1,097.254 | **1,088.594** | **-8.660** |
| RMSE (h) | 1,651.739 | **1,647.497** | **-4.242** |
| Median AE (h) | 735.506 | **729.576** | **-5.931** |
| Normalized MAE | 0.823743 | **0.808938** | **-0.014805** |
| MAE skill | -0.134140 | **-0.118670** | **+0.015470** |
| RMSE skill | **-0.013021** | -0.015077 | -0.002057 |
| D2 absolute error | -0.134140 | **-0.118670** | **+0.015470** |
| R2 | -0.202192 | **-0.200364** | **+0.001828** |
| Interval coverage | 0.706232 | **0.707504** | **+0.001273** |
| Mean interval width (h) | 2,875.036 | **2,836.335** | **-38.701** |

All primary raw-hour errors improve. RMSE skill is the one regression
tradeoff: small relative-scale losses on some rows outweigh its absolute RMSE
gain in the row-averaged normalized score. Lower selector strengths were
tested on the full protocol. Strength 0.25 reduces the RMSE-skill change to
`-0.000082`, but gives up more than half of the MAE gain and widens intervals
by `86.63` hours; strengths 0.50 and 0.75 are also dominated on the selected
primary errors/width tradeoff.

The selected model is not uniformly better on every log. Road Traffic MAE
worsens by `1.763` hours while its RMSE improves by `35.059` hours; Billing,
Receipt, and Helpdesk have mixed small RMSE changes. These limitations are
retained rather than averaged away.

## Confidence and routing invariants

The complete routing payload is exactly identical row-for-row across all 400
baseline/candidate rows. Classification still activates experts `[0, 2]` and
regression `[2, 3]`; every row has two active and two inactive experts.

Selector log weights are internal support-trust adjustments, not a redefined
posterior confidence. Classification confidence remains the maximum final
calibrated class probability. Regression expert confidence retains its
original supervised meaning and its `0.018` aggregation temperature. The
interval multiplier only maps the unchanged predictive standard deviation to
reported bounds.

## Reproduction

Seed both task-specific directories from epoch 43, copy its training artifacts
and configuration, then run:

```bash
python main.py \
  --config configs/fmv3/example_selector_classification_train.yaml \
  --checkpoint_dir checkpoints/fmv3/example_selector_classification \
  --resume --stop_after_epoch 44

python main.py \
  --config configs/fmv3/example_selector_regression_train.yaml \
  --checkpoint_dir checkpoints/fmv3/example_selector_regression \
  --resume --stop_after_epoch 44
```

Merge with the existing scope-auditing tool:

```bash
python merge_task_isolated_checkpoints.py \
  --base checkpoints/fmv3/loss_refinement_selected/model_epoch_43.pth \
  --classification checkpoints/fmv3/example_selector_classification/model_epoch_44.pth \
  --regression checkpoints/fmv3/example_selector_regression/model_epoch_44.pth \
  --output checkpoints/fmv3/example_selector_selected/model_epoch_44.pth \
  --config configs/fmv3/example_selector_selected.yaml \
  --artifacts checkpoints/fmv3/loss_refinement_selected/training_artifacts.pth
```

Run the confirmation and explanation diagnostic:

```bash
python evaluate_fmv3.py \
  --checkpoint_dir checkpoints/fmv3/example_selector_selected \
  --checkpoint_epoch 44 \
  --eval_config configs/fmv3/example_selector_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output_dir evaluation_results/example_selector/confirmations/example_selector_selected_e44

python analyze_example_selectors.py \
  --checkpoint_dir checkpoints/fmv3/example_selector_selected \
  --checkpoint_epoch 44 \
  --eval_config configs/fmv3/example_selector_confirmation_eval.yaml \
  --logs billing helpdesk receipt roadtraffic100traces sepsis \
  --output evaluation_results/example_selector/selector_diagnostics.json
```

Development screening and the confirmation use the repository's established
five target logs. The result supports promotion within that benchmark; a
claim of broad external transfer still requires untouched logs or a nested
log-level development/test split.
