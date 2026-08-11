# FM-v3 design recommendations

> **Document status:** this is the pre-implementation design rationale, not a
> description of the selected final system. See
> [`fmv3_architecture_changes.md`](fmv3_architecture_changes.md) for what was
> actually implemented, retained, corrected, or rejected, and
> [`structured_fmv3_report.md`](structured_fmv3_report.md) for the final paired
> result.

## Balanced accuracy should become the primary classification metric

For multiclass next-activity prediction,

[
\operatorname{BAcc}
===================

\frac{1}{|\mathcal C|}
\sum_{c\in\mathcal C}
\frac{TP_c}{TP_c+FN_c}.
]

Thus, multiclass balanced accuracy is exactly **macro-averaged recall**: every activity contributes equally, regardless of how frequently it occurs. Ordinary accuracy instead weights activities according to their prevalence in the test set. ([Scikit-learn][1])

That distinction is important in PPM. The SPICE study reports, for example:

| Event log and model              | Accuracy | Balanced accuracy |
| -------------------------------- | -------: | ----------------: |
| Helpdesk, Camargo et al.         |    0.865 |             0.398 |
| BPI 2020d, Camargo et al.        |    0.912 |             0.496 |
| Hospital Billing, Camargo et al. |    0.942 |             0.584 |

So a model can look excellent according to accuracy while predicting uncommon deviations very poorly. 

FM-v2 currently reports accuracy for next-activity prediction, while using MAE and (R^2) for remaining time. Re-evaluating FM-v1 and FM-v2 with balanced accuracy could therefore change some of the conclusions and possibly some method rankings. 

However, **only replacing accuracy with balanced accuracy would probably not justify FM-v3**. It would be a valuable evaluation correction or benchmark contribution. The stronger paper is one in which the model is explicitly redesigned for the balanced, low-data objective.

I would therefore report:

* **Balanced accuracy:** primary next-activity metric.
* **Ordinary accuracy:** secondary operational metric.
* **Macro-F1:** secondary imbalance-aware metric, because it additionally exposes precision problems.
* **Per-class recall:** diagnostic, especially for zero-, one-, and few-support activities.

There is no need to report “macro recall” separately because it is identical to balanced accuracy under the standard multiclass definition.

---

# Recommended coherent FM-v3 idea

## Decouple label coverage, class evidence, and class prevalence

At present, FM-v2 couples three different functions:

1. Retrieval determines which examples are locally relevant.
2. The labels of those retrieved examples determine which activities can be predicted.
3. The number of retrieved examples per activity influences the class score.

The FM-v2 paper explicitly notes that the correct activity is unreachable when it is absent from the retrieved support set, making support coverage a central bottleneck. It also finds that kNN over FM representations often matches or exceeds the support-conditioned head, suggesting that head design remains improvable. 

A coherent FM-v3 could be a:

> **Global–local, prior-controllable foundation model for low-data PPM**

The central architectural change would be to let the **whole target support pool define the candidate activity space**, while using retrieval only to provide additional local evidence.

## 1. Global–local support memory

For each target event log, construct two memories without target-log gradient updates:

### Global class memory

For every activity (c) occurring as a next-activity label anywhere in the available target support pool, construct a prototype:

[
\mu_c =
\frac{1}{n_c}
\sum_{i:y_i=c}z_i.
]

Therefore, every activity observed anywhere in the support pool remains predictable, even when it is absent from the local top-(k) retrieval.

### Local instance memory

Retain the FM-v2 nearest-neighbour retrieval over individual prefixes. This captures local variants and multimodal behaviour that a single global prototype may miss.

For a query (q), one possible combined score is:

[
e_c(q)
======

\lambda_c(q),e_c^{\mathrm{local}}(q)
+
\bigl(1-\lambda_c(q)\bigr)e_c^{\mathrm{global}}(q),
]

where:

[
e_c^{\mathrm{global}}(q)=\operatorname{sim}(z_q,\tilde{\mu}_c),
]

and

[
e_c^{\mathrm{local}}(q)
=======================

\log
\left[
\frac{1}{n_{c,k}}
\sum_{\substack{i\in R_k(q)\y_i=c}}
\exp\left(\frac{\operatorname{sim}(z_q,z_i)}{T}\right)
\right].
]

When activity (c) is absent from the retrieved neighbourhood, the model falls back to the global prototype. The gate (\lambda_c(q)) could depend on:

* the closest local similarity for class (c);
* the number of local examples of (c);
* global support count (n_c);
* retrieval entropy;
* agreement between FM experts.

This would change retrieval from a **hard candidate generator** into a **local evidence provider**. That is a conceptually clean extension of FM-v2.

---

## 2. Explicitly separate evidence from the class prior

In the uploaded implementation, `PrototypicalHead.forward_classification` first sums attention mass for all retrieved examples belonging to each class:

```python
class_mass.scatter_add_(...)
logits = torch.log(class_mass)
logits = logits + count_prior * torch.log(counts)
```

Even when `count_prior == 0`, summing support mass means that a class represented by more retrieved examples can receive more evidence merely because it has more examples.

For FM-v3, separate:

* **class-conditional evidence**, estimated by count-normalized local/global similarity;
* **the desired class prior**, supplied explicitly.

For example:

[
s_c(q)=e_c(q)+\beta\log \pi_c.
]

Then the same pretrained model can support two operating modes:

### Balanced operating mode

[
\pi_c=\frac{1}{|\mathcal C|}.
]

This is aligned with balanced accuracy, because balanced accuracy corresponds to evaluating classification under an equal class prior.

### Natural-frequency operating mode

Use a smoothed target-log prior:

[
\hat{\pi}_c=
\frac{n_c+\alpha}{N+\alpha|\mathcal C|}.
]

This is aligned with ordinary accuracy under the naturally observed target-log distribution.

This gives FM-v3 a particularly attractive property: it does not need to sacrifice natural-distribution accuracy permanently to improve balanced accuracy. Instead, it estimates class evidence once and applies an explicit prior appropriate to the application.

A simpler first implementation would replace the present summed class mass with:

[
e_c =
\operatorname{logsumexp}_{i:y_i=c}(s_i)
---------------------------------------

\gamma\log n_c.
]

Here:

* (\gamma=0) approximates the current frequency-sensitive aggregation;
* (\gamma=1) produces approximately count-neutral mean evidence;
* (\gamma) can be learned or conditioned on the support size.

Balanced Softmax and logit adjustment would be natural external baselines for this component, although the FM-v3 formulation would differ because the class counts arise inside an in-context support set rather than a conventional fixed training set. ([NeurIPS Proceedings][2])

---

## 3. Uncertainty-aware prototype estimation

In low-data regimes, prototypes based on one or two examples have high variance.

The current prototype implementation performs some centroid shrinkage, but it disables shrinkage for classes with fewer than three examples. FM-v3 could instead learn a count-dependent prototype denoiser specifically for the low-data regime:

[
\tilde{\mu}_c
=============

w_c\mu_c+(1-w_c)\mu_c^{\mathrm{prior}},
\qquad
w_c=\frac{n_c}{n_c+\kappa}.
]

The prior could be:

* a learned task-level prototype prior;
* a meta-network conditioned on the activity prototype, support count, support variance, and log-level statistics;
* optionally, an embedding of the activity label when activity names are semantically meaningful.

The important property is that shrinkage is strongest at one-shot and gradually disappears as (n_c) increases.

A learned prototype denoiser is preferable to adding several hand-designed corrections because it gives the paper a clear low-data mechanism: **partial pooling across support examples and pretrained knowledge**.

---

## 4. Train under the failures that occur at deployment

This may be the most important change suggested by the uploaded code.

In `training_strategies/retrieval_strategy.py`:

* classification batches are constructed to contain multiple examples per selected class;
* every processed classification query is explicitly supplied with at least one same-label positive support;
* queries with no same-label positive are skipped.

That is useful for representation learning, but it creates a train–test mismatch. During FM-v2 evaluation, ordinary top-(k) retrieval can fail to include the correct activity, while the training classification head is not exposed to that situation.

FM-v3 training episodes should mix:

1. **Balanced episodes**, preserving effective rare-class representation learning.
2. **Natural long-tail episodes**, matching real target logs.
3. **Random-shot episodes**, with different support counts per activity.
4. **Missing-local-label episodes**, where the correct class exists globally but is deliberately absent from the local retrieved context.
5. **Missing-pool-label episodes**, where the query class is absent altogether and the model must abstain or return an “uncovered” prediction.

Few-shot methods do not automatically become imbalance-robust simply because they use episodic or meta-learning; prior work found substantial degradation under task-level imbalance and showed that explicit rebalancing remains useful. 

For the synthetic pretraining generator, I would also vary:

* branch probabilities and long-tail intensity;
* number of cases per synthetic log;
* proportion of one-shot and zero-shot activities;
* rare loops and exceptional paths;
* support/query distribution mismatch.

That makes the pretraining distribution genuinely low-data-oriented rather than merely evaluating the same model on a smaller fraction afterward.

---

# A useful error decomposition for the paper

For FM-v2, define for a query whose true class is (c):

* (A_c): class (c) occurs somewhere in the available support pool;
* (R_c): class (c) occurs in the retrieved top-(k);
* (D_c): the prediction head chooses (c), conditional on (c) being retrieved.

Because FM-v2 cannot correctly predict (c) when it is absent from the retrieved support:

[
\operatorname{Recall}_c
=======================

P(A_c\mid y=c)
,
P(R_c\mid A_c,y=c)
,
P(D_c\mid R_c,y=c).
]

Therefore:

[
\operatorname{BAcc}
===================

\frac{1}{|\mathcal C|}
\sum_c
A_cR_cD_c.
]

This decomposition could become one of the paper’s main methodological contributions. It separates:

* **data-availability failure**;
* **retrieval failure**;
* **classification-head failure**.

With the proposed global–local head, top-(k) retrieval would no longer impose a hard candidate ceiling. The corresponding recall becomes approximately:

[
\operatorname{Recall}^{\mathrm{FM-v3}}_c
========================================

P(A_c\mid y=c)
,
P(D'_c\mid A_c,y=c).
]

That gives a very precise hypothesis:

> FM-v3 improves low-data balanced accuracy primarily by eliminating retrieval-induced zero recall for activities that exist in the support pool but do not occur in the nearest neighbourhood.

---

# Recommended metric suite

| Research question                                                     | Measurement                                                                 |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Does the model treat activities equally?                              | **Balanced accuracy**, primary                                              |
| Does it remain effective under natural workload frequencies?          | Ordinary accuracy                                                           |
| Does rare-class recall come at the cost of excessive false positives? | Macro-F1 and macro precision                                                |
| Are some activities completely ignored?                               | Number/proportion of classes with zero recall; 10th-percentile class recall |
| Is the correct class available at all?                                | Support-pool label availability                                             |
| Does top-(k) retrieve the correct candidate?                          | Macro label-recall@(k)                                                      |
| Is the head effective when the label is available?                    | Conditional balanced accuracy given pool/retrieval coverage                 |
| How does support frequency affect results?                            | Recall for (n_c=0), (1), (2!-!5), and (>5) support examples                 |
| How quickly does the model improve with data?                         | Area under the learning curve over (\log_2) case budget                     |
| How many cases are needed for a target performance?                   | Cases-to-threshold                                                          |
| Are probabilities reliable?                                           | NLL, multiclass Brier score, reliability curves                             |
| Can unreliable predictions be rejected?                               | Risk–coverage curve and area under selective-risk curve                     |

The Brier score and log loss are preferable to reporting only average confidence because they assess probability predictions rather than simply ranking predictions into confidence buckets. FM-v2 itself notes that its confidence is useful for stratification but is not calibrated. ([Scikit-learn][1])

I would also report an **accuracy–balanced-accuracy Pareto plot** while changing the prior-strength parameter (\beta) or count-normalization parameter (\gamma). This could show that FM-v3 offers a controllable operating point rather than simply shifting errors from common activities to rare ones.

---

# Low-data evaluation protocol

## Use absolute case budgets

FM-v2’s percentage-based fractions are useful, but (0.5%) can represent very different absolute amounts of evidence across logs. Add:

[
1,2,4,8,16,32,64,128
]

complete target cases, whenever the log size permits.

Plot balanced accuracy against both:

* number of cases;
* number of resulting support prefixes.

## Evaluate two support-sampling scenarios

### Natural support sampling

Randomly sample complete cases. This is the main realistic experiment and preserves natural activity imbalance.

### Class-aware support sampling

Construct approximately balanced or coverage-maximizing support sets. Treat this as a diagnostic or acquisition upper bound, not the main result.

The difference between the two reveals how much of the low-data limitation is caused by model quality versus simple activity non-coverage.

## Keep the query test set fixed

Across all support budgets and repetitions:

* use the same held-out query cases;
* use the same test activity universe;
* vary only the support subset;
* retain classes absent from the support in the balanced-accuracy denominator, giving them zero recall rather than silently excluding them.

Otherwise, balanced accuracy at very small budgets can appear artificially good because difficult missing classes disappear from evaluation.

## Repeat support sampling

Use nested support subsets and repeated random seeds. All methods should receive exactly the same support cases in each repetition. Confidence intervals should resample **cases**, not individual prefixes, because multiple prefixes generated from the same case are dependent.

Report:

* mean per event log;
* macro-average across event logs;
* lower-quartile or worst-log performance;
* case-level bootstrap intervals.

Do not pool all prefixes from all logs into one global balanced-accuracy value, because large logs would again dominate the evaluation.

Standard balanced accuracy should remain the interpretable primary result. Chance-adjusted balanced accuracy, where random performance is mapped to zero, can be a supplementary cross-log summary when logs have very different numbers of activities. ([Scikit-learn][1])

---

# Essential ablation study

A clean ablation sequence would be:

| Variant                                    | Purpose                                                   |
| ------------------------------------------ | --------------------------------------------------------- |
| FM-v2, newly evaluated with all metrics    | Establish whether metric choice alone changes conclusions |
| FM-v2 + realistic low-data episodes        | Is the train–test support mismatch important?             |
| FM-v2 + count-neutral/prior-separated head | Is support-frequency bias important?                      |
| FM-v2 + global class prototypes            | Is top-(k) label omission the principal bottleneck?       |
| Global prototypes + learned shrinkage      | Does uncertainty-aware estimation help one-shot classes?  |
| Global–local head                          | Are global coverage and local specificity complementary?  |
| Full FM-v3                                 | Combined result                                           |
| Full FM-v3 without pretraining             | How much improvement comes from foundation pretraining?   |

Within the head ablation, compare:

[
\gamma=0,\qquad
\gamma=1,\qquad
\gamma\text{ learned}.
]

Within retrieval, compare:

* current local top-(k);
* global prototypes only;
* global plus local;
* global plus dynamically expanded local retrieval.

Relevant conventional baselines should include class-weighted cross-entropy, logit adjustment, and Balanced Softmax in addition to the existing FM-v1, FM-v2, foundation-kNN, TabPFN, and classical baselines. At least one log-specific LSTM or transformer should also be included under identical low-data case budgets; FM-v2 already identifies the absence of dedicated per-log sequence baselines and the five-log benchmark as limitations. 

---

# Suggested research questions

**RQ1.** Does FM-v3 improve balanced accuracy and macro-F1 in low-data target logs while retaining competitive ordinary accuracy?

**RQ2.** How much error is attributable to support-pool availability, retrieval coverage, and conditional classification?

**RQ3.** Does separating class evidence from the target class prior improve robustness to support imbalance?

**RQ4.** Does global–local support memory reduce the number of activities with zero recall?

**RQ5.** How many target cases are required to reach a fixed proportion of full-data balanced accuracy?

**RQ6.** Are FM-v3 confidence estimates reliable enough for dynamic retrieval or selective prediction?

---

# Remaining-time prediction

Balanced accuracy applies only to next-activity classification. For remaining time, I would keep MAE as the main interpretable metric but add:

* median absolute error for robustness to extreme durations;
* normalized MAE or an MAE skill score against a median or prefix-length-conditioned baseline;
* (D^2) absolute-error score;
* interval coverage and interval width if FM-v3 produces predictive uncertainty.

Raw (R^2) should remain secondary because its variance is dataset-dependent and it is not necessarily meaningful to average across heterogeneous event logs. A (D^2) absolute-error score provides a clearer zero point corresponding to a constant median predictor. ([Scikit-learn][1])

The strongest and most coherent FM-v3 paper would nevertheless make **low-data next-activity prediction** the central contribution and use remaining-time results mainly to verify that the new representation does not regress substantially.

## Recommended paper framing

A suitable title would be:

> **Beyond Accuracy: A Coverage- and Prior-Aware Foundation Model for Low-Data Predictive Process Monitoring**

The central claim would be:

> FM-v3 improves performance per target case by decoupling the target label space from local retrieval, separating similarity evidence from noisy support-class priors, and training under realistic long-tailed and missing-label support conditions.

That is considerably stronger than “we now report balanced accuracy”: the new metric directly motivates the new architecture, training distribution, diagnostic decomposition, and evaluation protocol.

[1]: https://scikit-learn.org/stable/modules/model_evaluation.html "https://scikit-learn.org/stable/modules/model_evaluation.html"
[2]: https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html?utm_source=chatgpt.com "Balanced Meta-Softmax for Long-Tailed Visual Recognition"
