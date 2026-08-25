# Current foundation-model retrain versus published FM-v2

## Outcome

The current repository model was retrained once from scratch with the fixed
seed-42, 40-epoch configuration and evaluated at the terminal checkpoint. It
beats the published FM-v2 prototypical-head value in **20 of 80** matched
log/fraction/task cells: **5/40 classification-accuracy cells** and **15/40
regression-MAE cells**. It therefore does not beat FM-v2 overall.

The lowest-percentage result is mixed:

- **0.5%: 5/10 wins** (3/5 classification, 2/5 regression);
- **1%: 2/10 wins** (0/5 classification, 2/5 regression);
- **0.5% and 1% combined: 7/20 wins**.

At 0.5%, the wins are Billing MAE, Helpdesk accuracy, both Receipt tasks, and
Road Traffic accuracy. At 1%, only Billing and Receipt MAE win. Sepsis loses
both low-percentage tasks. These results do not support a claim that the new
model is consistently better in the low-data regime.

## Win summary

The mean-relative column is positive for improvement. It uses
`candidate/reference - 1` for accuracy and `reference/candidate - 1` for MAE,
then averages the task cells without weighting by log size.

| Support fraction | Wins | Classification | Regression | Mean relative change |
|---:|---:|---:|---:|---:|
| 0.5% | **5/10** | **3/5** | 2/5 | -3.36% |
| 1% | **2/10** | 0/5 | **2/5** | -1.82% |
| 3% | 3/10 | 1/5 | 2/5 | -1.48% |
| 5% | 2/10 | 0/5 | 2/5 | -2.18% |
| 10% | 2/10 | 1/5 | 1/5 | -1.86% |
| 20% | 2/10 | 0/5 | 2/5 | -1.73% |
| 50% | 2/10 | 0/5 | 2/5 | -2.03% |
| 100% | 2/10 | 0/5 | 2/5 | -2.93% |
| **Overall** | **20/80** | **5/40** | **15/40** | **-2.17%** |

| Log | Wins | Classification | Regression | Mean relative change |
|---|---:|---:|---:|---:|
| Billing | 7/16 | 0/8 | 7/8 | +0.68% |
| Helpdesk | 1/16 | 1/8 | 0/8 | -3.85% |
| Receipt | **9/16** | 1/8 | **8/8** | **+8.52%** |
| Road Traffic 10k | 2/16 | 2/8 | 0/8 | -4.15% |
| Sepsis | 1/16 | 1/8 | 0/8 | -12.06% |

The gain is concentrated in duration prediction: Receipt MAE wins at all
eight fractions and Billing MAE wins at every fraction except 10%. The main
systematic failures are Billing accuracy (0/8), Road Traffic MAE (0/8), and
Sepsis MAE (0/8). Accuracy wins occur only for Helpdesk 0.5%, Receipt 0.5%,
Road Traffic 0.5% and 3%, and Sepsis 10%.

## Low-percentage detail

Values are `retrain / published FM-v2`. A win requires strictly higher
accuracy or strictly lower MAE. `k` is the selected retrieval depth.

| Log | Fraction | Accuracy (k) | MAE hours (k) | Wins |
|---|---:|---:|---:|---:|
| Billing | 0.5% | 0.8771 / 0.8950 (50) | 1,148.48 / 1,160.78 (100) | 1/2 |
| Billing | 1% | 0.8859 / 0.9200 (50) | 1,150.51 / 1,186.55 (200) | 1/2 |
| Helpdesk | 0.5% | 0.7444 / 0.7250 (20) | 0.2105 / 0.1960 (20) | 1/2 |
| Helpdesk | 1% | 0.7284 / 0.7850 (100) | 0.2031 / 0.2021 (10) | 0/2 |
| Receipt | 0.5% | 0.7061 / 0.6800 (20) | 93.89 / 97.11 (50) | **2/2** |
| Receipt | 1% | 0.7250 / 0.7850 (10) | 78.20 / 102.52 (50) | 1/2 |
| Road Traffic 10k | 0.5% | 0.8720 / 0.8500 (20) | 4,775.20 / 4,459.56 (50) | 1/2 |
| Road Traffic 10k | 1% | 0.8391 / 0.8550 (50) | 4,833.57 / 4,592.98 (200) | 0/2 |
| Sepsis | 0.5% | 0.4203 / 0.4458 (20) | 949.68 / 703.20 (100) | 0/2 |
| Sepsis | 1% | 0.4609 / 0.4663 (20) | 1,255.85 / 936.73 (200) | 0/2 |

## Complete matched result

| Log | Fraction | Accuracy: retrain / FM-v2 (k) | MAE hours: retrain / FM-v2 (k) | Wins |
|---|---:|---:|---:|---:|
| Billing | 0.5% | 0.8771 / 0.8950 (50) | 1,148.4790 / 1,160.7777 (100) | 1/2 |
| Billing | 1% | 0.8859 / 0.9200 (50) | 1,150.5056 / 1,186.5530 (200) | 1/2 |
| Billing | 3% | 0.8978 / 0.9200 (100) | 1,126.7179 / 1,194.6041 (200) | 1/2 |
| Billing | 5% | 0.9092 / 0.9150 (50) | 1,110.5702 / 1,163.6401 (200) | 1/2 |
| Billing | 10% | 0.9140 / 0.9250 (100) | 1,121.7443 / 1,120.5362 (200) | 0/2 |
| Billing | 20% | 0.9184 / 0.9400 (100) | 1,102.1484 / 1,157.2576 (200) | 1/2 |
| Billing | 50% | 0.9242 / 0.9400 (100) | 1,077.8357 / 1,117.6198 (200) | 1/2 |
| Billing | 100% | 0.9280 / 0.9450 (100) | 1,068.0098 / 1,099.8401 (200) | 1/2 |
| Helpdesk | 0.5% | 0.7444 / 0.7250 (20) | 0.2105 / 0.1960 (20) | 1/2 |
| Helpdesk | 1% | 0.7284 / 0.7850 (100) | 0.2031 / 0.2021 (10) | 0/2 |
| Helpdesk | 3% | 0.7638 / 0.8000 (50) | 0.1864 / 0.1772 (20) | 0/2 |
| Helpdesk | 5% | 0.7580 / 0.8100 (200) | 0.1873 / 0.1848 (100) | 0/2 |
| Helpdesk | 10% | 0.7531 / 0.8100 (200) | 0.1785 / 0.1741 (50) | 0/2 |
| Helpdesk | 20% | 0.7613 / 0.8050 (100) | 0.1723 / 0.1686 (50) | 0/2 |
| Helpdesk | 50% | 0.7794 / 0.8300 (50) | 0.1702 / 0.1670 (100) | 0/2 |
| Helpdesk | 100% | 0.7872 / 0.8400 (100) | 0.1681 / 0.1661 (200) | 0/2 |
| Receipt | 0.5% | 0.7061 / 0.6800 (20) | 93.8945 / 97.1067 (50) | 2/2 |
| Receipt | 1% | 0.7250 / 0.7850 (10) | 78.2044 / 102.5240 (50) | 1/2 |
| Receipt | 3% | 0.7348 / 0.8000 (20) | 83.5396 / 101.7245 (100) | 1/2 |
| Receipt | 5% | 0.7479 / 0.7850 (50) | 85.2479 / 99.9050 (200) | 1/2 |
| Receipt | 10% | 0.7521 / 0.8050 (50) | 80.2279 / 100.8993 (100) | 1/2 |
| Receipt | 20% | 0.7775 / 0.8050 (50) | 81.4784 / 99.0821 (200) | 1/2 |
| Receipt | 50% | 0.8202 / 0.8500 (50) | 77.9323 / 98.0772 (100) | 1/2 |
| Receipt | 100% | 0.8333 / 0.8500 (50) | 77.5444 / 94.4677 (100) | 1/2 |
| Road Traffic 10k | 0.5% | 0.8720 / 0.8500 (20) | 4,775.1963 / 4,459.5550 (50) | 1/2 |
| Road Traffic 10k | 1% | 0.8391 / 0.8550 (50) | 4,833.5723 / 4,592.9824 (200) | 0/2 |
| Road Traffic 10k | 3% | 0.8636 / 0.8550 (100) | 5,039.2188 / 4,477.5697 (200) | 1/2 |
| Road Traffic 10k | 5% | 0.8792 / 0.8800 (50) | 4,647.2178 / 4,440.8163 (200) | 0/2 |
| Road Traffic 10k | 10% | 0.8792 / 0.8850 (100) | 4,891.4170 / 4,456.0001 (200) | 0/2 |
| Road Traffic 10k | 20% | 0.8818 / 0.8950 (100) | 4,752.9438 / 4,402.3713 (200) | 0/2 |
| Road Traffic 10k | 50% | 0.8830 / 0.8900 (200) | 4,793.1445 / 4,301.9524 (200) | 0/2 |
| Road Traffic 10k | 100% | 0.8853 / 0.9000 (200) | 4,756.7227 / 4,287.3581 (200) | 0/2 |
| Sepsis | 0.5% | 0.4203 / 0.4458 (20) | 949.6776 / 703.1954 (100) | 0/2 |
| Sepsis | 1% | 0.4609 / 0.4663 (20) | 1,255.8475 / 936.7253 (200) | 0/2 |
| Sepsis | 3% | 0.5375 / 0.5422 (50) | 891.3367 / 788.3009 (200) | 0/2 |
| Sepsis | 5% | 0.5180 / 0.5496 (50) | 902.5202 / 718.7791 (200) | 0/2 |
| Sepsis | 10% | 0.5609 / 0.5581 (100) | 887.1484 / 727.6845 (200) | 1/2 |
| Sepsis | 20% | 0.5632 / 0.5715 (200) | 895.8151 / 713.3536 (200) | 0/2 |
| Sepsis | 50% | 0.5839 / 0.5927 (200) | 878.5525 / 666.9328 (200) | 0/2 |
| Sepsis | 100% | 0.5962 / 0.5980 (200) | 891.4970 / 614.9354 (200) | 0/2 |

## Reference and matched protocol

The numerical baseline is the prototypical-head result in the July 2026
revised [FM-v2 paper](https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second_New2.pdf).
The downloaded PDF has SHA-256
`51dd05523ec3049536280070ea3c4f0c3086ed140c90f729e89208a3ada0e69c`.
The frozen 80-cell transcription is
[`fmv2_new2_proto_reference.csv`](fmv2_new2_proto_reference.csv), SHA-256
`9c881ccdbb6722c0ea3023d5049450b5c389f6d01e168abfa864905bd64c6a8e`.
Classification uses ordinary accuracy; regression uses MAE in hours.

The paper declares eight support fractions: 0.5%, 1%, 3%, 5%, 10%, 20%, 50%,
and 100%. Its compact printed result rows expose six fractions directly; the
frozen transcription supplies the exact 1% and 50% plotted/companion values.
This limitation should be kept in mind when auditing those two columns.

No GitHub page, source tree, checkpoint, or reported score was consulted for
this retrain. The paper values above are the numerical reference. Evaluation
uses the pre-existing local copy of the paper's fixed split logs only as input
data. The comparison uses:

- the same five target logs at every percentage: Billing, Helpdesk, Receipt,
  Road Traffic 10k, and Sepsis;
- the same case-disjoint held-out test log for all eight nested support logs;
- a fixed minimum of two observed events per prefix, independent of the
  retrained model's training-time prefix setting;
- every eligible held-out prefix: Billing 54,878, Helpdesk 2,430, Receipt
  1,218, Road Traffic 3,461, and Sepsis 2,610 queries per task/fraction;
- all paper retrieval depths, `k in {1, 5, 10, 20, 50, 100, 200}`;
- best accuracy or lowest MAE across `k`, choosing the smallest `k` on ties.

The fixed support-prefix counts are recorded per fraction in
[`selected_results.csv`](../evaluation_results/foundation_retrained_current/fmv2_paper_all/selected_results.csv).
They exactly match the established paper-protocol cohort.

## Retraining record

This was a fixed from-scratch run, not a hyperparameter screen. No intermediate
checkpoint was evaluated to choose an endpoint. The saved configuration uses:

- four experts with a shared six-layer, width-256 Transformer backbone and
  lightweight expert adapters (6,365,946 trainable parameters);
- seed 42, 40 epochs, and 300 successful optimizer updates per epoch;
- 12 content-addressed source logs and 971 content-addressed synthetic logs,
  mixed with the repository's declared 70/30, 40/60, and 25/75 schedules;
- an accuracy classification objective, an MAE regression objective, learning
  rate `1e-4`, weight decay `0.01`, and gradient clipping at 5;
- retrieval training with balanced task/expert quotas and AMP.

The terminal run applied 12,000/12,000 finite optimizer updates (6,000 per
task), processed 1,504,661 examples, and recorded zero non-finite loss steps.
There were 28 ordinary rejected/retried episodes and seven AMP-overflow retries;
all epochs still reached exactly 300 successful updates. Training metrics are
in [`training_metrics.jsonl`](../checkpoints/foundation_retrained_current/training_metrics.jsonl).

Before the successful run, the accuracy-only objective produced no optimizer
updates because an unused NLL surrogate could be infinite under AMP and
`0 * inf` contaminated the active loss with NaN. The implementation now omits
zero-weight components from loss arithmetic and reports their weighted
diagnostic as zero. This is an objective-correctness fix, not a changed
hyperparameter. Evaluation also needed two protocol fixes: preserving sparse
historical-memory records as dictionaries, and pinning the paper's two-event
prefix floor independently of the saved training config. Regression tests
cover all three conditions.

## Reproduction and artifacts

The final evaluation command was:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_fmv2_paper_protocol.py \
  --checkpoint_dir checkpoints/foundation_retrained_current \
  --checkpoint_epoch 40 \
  --paper_repo /tmp/experiments-fm \
  --reference paper_docs/fmv2_new2_proto_reference.csv \
  --output_dir evaluation_results/foundation_retrained_current/fmv2_paper_all \
  --logs billing helpdesk receipt roadtraffic_10000 sepsis \
  --fractions 0.5 1 3 5 10 20 50 100 \
  --retrieval_k 1 5 10 20 50 100 200 \
  --num_queries 0 --embedding_batch_size 1024 \
  --prediction_batch_size 512 --device cuda --seed 42
```

Primary artifacts:

- terminal checkpoint: [`model_epoch_40.pth`](../checkpoints/foundation_retrained_current/model_epoch_40.pth),
  SHA-256 `4193cdb2c46a48ba529138974c5097ee7e59f15c8178e5550b748752c0a21183`;
- saved training configuration:
  [`training_config.yaml`](../checkpoints/foundation_retrained_current/training_config.yaml),
  SHA-256 `c8bcfdbcdb853c47ab9a0f9c208e38a2a43e72d2814fcb5a50ef0c5378277ce0`;
- all 560 k-level rows:
  [`results.csv`](../evaluation_results/foundation_retrained_current/fmv2_paper_all/results.csv),
  SHA-256 `40e3bd0735cb088672d2f4cf6d58ec526901f5ac66c5f3e9a025a1a2e0fb372a`;
- selected 80-cell comparison:
  [`selected_results.csv`](../evaluation_results/foundation_retrained_current/fmv2_paper_all/selected_results.csv),
  SHA-256 `a76579fbb88c2038b853625a31ae7093979b51737e2f58c6bb92aeb30a2fd34e`;
- evaluation manifest:
  [`manifest.json`](../evaluation_results/foundation_retrained_current/fmv2_paper_all/manifest.json),
  SHA-256 `7de9790f7b38444ee3ac8bfa44d2b8a8b7788e008026472c065d3a9f14d819be`;
- evaluator SHA-256:
  `bf895da1c8b83b42c95473cb1efdc3ed29d1a30e9f0f5a3883161e157fd65c31`.

Verification completed with `PYTHONPATH=. pytest -q`: **142 passed**, with
seven existing Transformer nested-tensor warnings. `git diff --check` also
passes.

## Conclusion

The retrained current model improves the published FM-v2 result mainly on
Receipt and Billing duration prediction, but its accuracy and its Road Traffic
and Sepsis duration errors prevent broad superiority. The fairest summary is
**20/80 overall, 5/10 at 0.5%, 2/10 at 1%, and 7/20 across the two lowest
fractions**. It is a useful specialized improvement, not a general replacement
for the published FM-v2 prototypical head.
