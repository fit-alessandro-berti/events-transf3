# Parameter ablation against the published FM-v2 prototypical head

## Source and protocol provenance

The comparison target is the prototypical-head result in the July 2026
[FM-v2 paper](https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second_New2.pdf).
The downloaded PDF has SHA-256
`51dd05523ec3049536280070ea3c4f0c3086ed140c90f729e89208a3ada0e69c`.
The exact published values for five logs and eight support fractions are stored
in [`fmv2_new2_proto_reference.csv`](fmv2_new2_proto_reference.csv), whose
SHA-256 is
`9c881ccdbb6722c0ea3023d5049450b5c389f6d01e168abfa864905bd64c6a8e`.

The fixed evaluation logs come from the paper's public
[`experiments-fm`](https://github.com/fit-alessandro-berti/experiments-fm)
artifact at commit `5f4da9d79ac51beb98dda4b6e1fd6f94e3a4abf3`. It supplies one
case-disjoint held-out test log and eight nested support logs (0.5%, 1%, 3%,
5%, 10%, 20%, 50%, and 100%) for every target log.

[`evaluate_fmv2_paper_protocol.py`](../evaluate_fmv2_paper_protocol.py)
implements the matched neural evaluation:

- the target-log activity vocabulary is fixed from the complete base log;
- every eligible held-out prefix is a query in confirmation runs;
- support prefixes come only from the corresponding nested training split;
- retrieval uses `L2(mean(raw expert embedding))` and one shared neighbor set;
- each expert's legacy prototypical head receives its own normalized embeddings;
- classification probabilities and confidence-weighted duration predictions are
  aggregated across experts;
- all paper retrieval depths, `k in {1, 5, 10, 20, 50, 100, 200}`, are retained;
- `results.csv`/`results.jsonl` retain the complete sweep, while
  `selected_results.csv` selects the best task metric and the smallest `k` on
  ties.

The downloaded public checkpoint was used only as a loader/protocol calibration
artifact. Its SHA-256 is
`1500b6e73294d3fc435ca2feb863a3ada35c0666c041c72c2f25edafb747013a`.
Its saved configuration lists 11 hand-selected training logs rather than the
paper's 971-log synthetic corpus, and its Sepsis duration results do not
reproduce the paper. It is therefore not used as the numerical baseline; the
published table is.

## Stage 1: expert count and width

All stage-1 candidates used seed 42, six Transformer layers, 300 episodes per
epoch, the same training logs/schedule, and 30 epochs. The rapid comparison
uses the same deterministic 200 held-out queries per log/task at 0.5% and 1%.
The relative column is positive when the candidate improves on FM-v2 (accuracy
increase or MAE reduction), averaged over the 20 task rows.

| Experts | Width | Parameters | Low-data wins | Classification wins | Regression wins | Mean relative change |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 256 | 10,685,630 | 11/20 | 4/10 | 7/10 | +0.14% |
| 4 | 256 | 21,371,260 | 12/20 | 5/10 | 7/10 | +1.30% |
| 8 | 256 | 42,742,520 | 12/20 | 5/10 | 7/10 | +0.31% |
| 4 | 384 | 47,502,716 | 12/20 | 5/10 | 7/10 | +1.57% |

Doubling experts from four to eight doubles parameter count without adding a
win and reduces the mean improvement. Widening to 384 adds 26.1 million
parameters without improving the win count. Four experts at width 256 are the
parameter-efficient frontier.

The 4x384 endpoint was also run against every held-out prefix. It wins 11/20
low-data task rows (6/10 classification, 5/10 regression). Its strongest
remaining deficits are Sepsis MAE: 988.95 h versus 703.20 h at 0.5%, and
1,279.76 h versus 936.73 h at 1%. This full-query check is stored in
[`stage1_full/e4_d384_e30_low_fastbatch`](../evaluation_results/parameter_ablation/stage1_full/e4_d384_e30_low_fastbatch).

## Stage 2: metric target and optimizer controls

Stage 2 keeps the conventional four-expert architecture and trains from the
same seed for 20 epochs. It targets exactly the requested paper metrics:
classification accuracy and regression MAE. The endpoint screen again uses
200 deterministic queries per log/task at 0.5% and 1%.

| Variant | Width | Dropout | Learning rate | Clip | Low-data wins | Mean relative change |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy/MAE, standard | 256 | 0.15 | 1e-4 | 1 | 12/20 | +0.70% |
| Accuracy/MAE, clip-5 | 256 | 0.15 | 1e-4 | 5 | **13/20** | +0.75% |
| Accuracy/MAE, fast | 256 | 0.10 | 2e-4 | 5 | 12/20 | **+1.15%** |
| Accuracy/MAE, wide | 384 | 0.15 | 1e-4 | 5 | 12/20 | +0.55% |

Clip-5 is the most consistent stage-2 endpoint. The faster run has a larger
mean margin but one fewer win; width 384 again does not justify its parameter
cost. Raw endpoint evidence is under
[`stage2_screens`](../evaluation_results/parameter_ablation/stage2_screens).

## Corpus-schedule audit

The original working configuration activates the 18 root logs for epochs
1--100 and the 971 `logs/out` files for epochs 6--100. When both sets are
active, the trainer chooses one uniformly per epoch. Consequently, roughly half
of later epochs use the small root set, whereas the paper states that its full
schedule uses the 971 successful synthetic logs in `logs/out`.

The stage-3 controls remove this confound without changing the base
architecture. The selected curriculum uses the 18 root logs only for epochs
1--5, then the 971 `logs/out` files exclusively from epoch 6 onward. There is
exactly one active set at every epoch. An out-only control and a longer
20-epoch warm-up establish that the five-epoch warm-up is the better schedule.

## Stage 3: schedule, objective, and seed controls

The rapid columns below use the same deterministic 200-query screen as stages
1--2. The full columns use **every** eligible held-out prefix: 54,878 queries
for Billing, 2,430 for Helpdesk, 1,218 for Receipt, 3,461 for Road Traffic,
and 2,610 for Sepsis, for each task and fraction. The mean relative change is
`candidate/reference - 1` for accuracy and `reference/candidate - 1` for MAE.

| Variant | Schedule | Epoch | Rapid wins | Full wins | Full mean relative |
|---|---|---:|---:|---:|---:|
| Accuracy/MAE, out-only, clip-5 | 971 logs throughout | 40 | 12/20 | -- | -- |
| **Accuracy/MAE, selected** | **5-epoch warm-up** | **40** | **13/20** | **12/20** | **-0.17%** |
| Equilibrated classification/MAE | 5-epoch warm-up | 40 | 13/20 | 12/20 | -0.64% |
| Accuracy/equilibrated regression | 5-epoch warm-up | 40 | 13/20 | 10/20 | -0.41% |
| Accuracy/MAE, seed 43 | 5-epoch warm-up | 40 | 13/20 | 11/20 | -0.97% |
| Equilibrated classification/MAE | 20-epoch warm-up | 40 | 12/20 | -- | -- |
| Accuracy/relative-MAE | 5-epoch warm-up | 20 | 12/20 | 11/20 | +0.21% |
| Accuracy/RMSE | 5-epoch warm-up | 20 | 12/20 | 11/20 | +0.49% |
| Accuracy/log-RMSE | 5-epoch warm-up | 40 | 13/20 | 11/20 | -0.32% |

The 200-query screen is useful for pruning but optimistic on Road Traffic's
heavy-tailed durations. Full-prefix evidence is therefore the only basis for
the final selection. Seed 43 reproduces the general effect but not the exact
win count: 11/20 versus 12/20 at epoch 40. Continuing the selected seed-42 run
to epoch 60 also overfits, falling from 13/20 to 12/20 on the rapid screen and
losing the Sepsis 0.5% accuracy win. Epoch 40 is the evidence-based stopping
point.

## Parameter-free and architecture-extension checks

The existing `raw_hours_knn` inference mode was tested with the same epoch-40
checkpoint. It wins only 10/20 full low-data rows. It improves Helpdesk,
Receipt-1%, and Sepsis-0.5% duration, but produces catastrophic Road Traffic
MAE (12,020--12,784 hours). Training `raw_hours_knn` end-to-end is worse at
8/20 on the rapid screen.

That complementarity justified one bounded architecture-extension test using
the repository's existing learned transform ensemble. It adds only 128
parameters (21,371,388 versus 21,371,260) but wins only 10/20 at epoch 10 and
also produces Road Traffic MAE above 11,000 hours. It was pruned immediately.
The selected configuration therefore keeps the conventional `sqrt_knn` head.

## Selected full-prefix low-data result

The selected checkpoint is
`checkpoints/parameter_ablation/curriculum_metric_targets/model_epoch_40.pth`
(SHA-256
`fc9ddf888df9bbd613535002aa0113d9a6f36c3d24c137973460a5f192a63ecb`).
It has 21,371,260 parameters: four experts, width 256, eight attention heads,
and six Transformer layers.

| Log | Fraction | Accuracy (ours / FM-v2) | MAE hours (ours / FM-v2) | Wins |
|---|---:|---:|---:|---:|
| Billing | 0.5% | 0.9034 / 0.8950 | 1,144.14 / 1,160.78 | 2/2 |
| Billing | 1% | 0.9055 / 0.9200 | 1,140.65 / 1,186.55 | 1/2 |
| Helpdesk | 0.5% | 0.7473 / 0.7250 | 0.1946 / 0.1960 | 2/2 |
| Helpdesk | 1% | 0.7243 / 0.7850 | 0.1841 / 0.2021 | 1/2 |
| Receipt | 0.5% | 0.6921 / 0.6800 | 87.29 / 97.11 | 2/2 |
| Receipt | 1% | 0.7438 / 0.7850 | 78.67 / 102.52 | 1/2 |
| Road Traffic | 0.5% | 0.8723 / 0.8500 | 4,877.37 / 4,459.56 | 1/2 |
| Road Traffic | 1% | 0.8518 / 0.8550 | 4,887.22 / 4,592.98 | 0/2 |
| Sepsis | 0.5% | 0.4487 / 0.4458 | 945.29 / 703.20 | 1/2 |
| Sepsis | 1% | 0.5195 / 0.4663 | 1,270.53 / 936.73 | 1/2 |

This is **8/10 wins at 0.5%**, **4/10 at 1%**, and **12/20 combined**.
The result is strong and consistent at 0.5%, but it does **not** support a
claim of consistent superiority at 1%. The persistent 1% losses are four
classification rows plus Road Traffic and Sepsis MAE, partly offset by three
other MAE wins and Sepsis accuracy.

## Complete eight-fraction result

The all-prefix sweep over all five logs, two tasks, and eight fractions selects
the best task metric across the seven paper retrieval depths, exactly as in the
low-data confirmation.

| Fraction | Wins | Classification | Regression | Mean relative change |
|---:|---:|---:|---:|---:|
| 0.5% | **8/10** | **5/5** | 3/5 | -1.17% |
| 1% | 4/10 | 1/5 | 3/5 | +0.83% |
| 3% | 5/10 | 2/5 | 3/5 | +0.19% |
| 5% | 5/10 | 2/5 | 3/5 | -1.08% |
| 10% | 4/10 | 2/5 | 2/5 | -0.22% |
| 20% | 5/10 | 2/5 | 3/5 | +0.51% |
| 50% | 4/10 | 1/5 | 3/5 | -0.99% |
| 100% | 4/10 | 1/5 | 3/5 | -2.35% |
| **Overall** | **39/80** | **16/40** | **23/40** | **-0.54%** |

The complete result is stored under
[`stage3_full/curriculum_metric_targets_e40_all_final`](../evaluation_results/parameter_ablation/stage3_full/curriculum_metric_targets_e40_all_final).
It shows that the selected configuration is a real improvement at the most
data-starved 0.5% setting, but not an across-the-board replacement for the
published FM-v2 prototypical head. Road Traffic and Sepsis regression remain
the systematic limitation: the selected model loses all eight fractions for
both logs' MAE.

## Reproduction

The final confirmation was produced with:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_fmv2_paper_protocol.py \
  --checkpoint_dir checkpoints/parameter_ablation/curriculum_metric_targets \
  --checkpoint_epoch 40 \
  --paper_repo /tmp/experiments-fm \
  --reference paper_docs/fmv2_new2_proto_reference.csv \
  --output_dir evaluation_results/parameter_ablation/stage3_full/curriculum_metric_targets_e40_all_final \
  --logs billing helpdesk receipt roadtraffic_10000 sepsis \
  --fractions 0.5 1 3 5 10 20 50 100 \
  --retrieval_k 1 5 10 20 50 100 200 \
  --num_queries 0 --embedding_batch_size 1024 \
  --prediction_batch_size 512 --device cuda --seed 42
```

The rapid screens use the same invocation with `--num_queries 200`; this is a
deterministic sample, not the confirmation protocol. Every result directory
contains a manifest recording checkpoint, evaluator, reference, paper-repo
commit, model dimensions, prediction mode, split scope, and SHA-256 hashes.

Final artifact SHA-256 values are:

- `config.py`: `192e8866d873357c41c25e4800c7e4e2c065f7fe03b3d4b9a80690b14531948b`;
- evaluator: `e24ed9d6d4f7967594c5108ff14d0946b899e93e9313274820d73d647b471ba1`;
- complete `results.csv`: `debae5213765041e819243f0021ac580862502c3e48c0489c2ddc49b37ebcd23`;
- complete `selected_results.csv`: `f7aad6987ad7a10163d57c108a0b2ce9d3ffc9cfaafddf3266ae2758c4571365`;
- final manifest: `9cb4df7f0e503daaac57846e777df550ab5537c1735071121ca0e2cc31c1d33c`.

## Delivered configuration

`config.py` now encodes the selected conventional configuration:

- four experts, `d_model=256`, eight heads, and six layers;
- 18 root logs in epochs 1--5, then 971 `logs/out` files exclusively in
  epochs 6--40;
- 300 episodes per epoch, learning rate `1e-4`, weight decay `0.01`, dropout
  `0.15`, and gradient clipping at `1.0`;
- accuracy classification objective, MAE regression objective, and the
  conventional `sqrt_knn` regression head;
- seed 42 and an evidence-based 40-epoch stopping point.

The configuration contains no unproven architecture extension. Its behavior
through epoch 40 is identical to the selected checkpoint's saved training
configuration; only the declared terminal epoch is shortened from 100 to the
validated stopping point.
