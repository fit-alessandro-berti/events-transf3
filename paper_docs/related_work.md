# Related work for a low-data FM-v3 in predictive process monitoring

This is a targeted literature review of work available through **11 August 2026**, rather than a formal systematic review. It covers the literature most directly relevant to an FM-v3 that improves next-activity prediction when only a small number of target-log cases are available. The review focuses on six intersecting areas:

1. predictive process monitoring on small event logs;
2. data augmentation and self-supervised learning;
3. transfer learning and process foundation models;
4. few-shot metric and prototype learning;
5. retrieval-augmented in-context prediction;
6. long-tail classification, balanced evaluation, and uncertainty.

The central conclusion is that the strongest contribution is not merely a new low-data benchmark or a change from accuracy to balanced accuracy. A defensible FM-v3 would combine **coverage-aware retrieval, prior-neutral class evidence, uncertainty-aware prototype estimation, and deployment-realistic episodic training**.

---

## 1. Executive synthesis

Research on low-data predictive process monitoring, or PPM, has so far followed several mostly separate paths.

Small-log studies train conventional predictive models directly on reduced target logs and investigate when their performance deteriorates. Data-augmentation approaches attempt to increase the effective target sample size by generating traces or transforming existing ones. Transfer-learning approaches reuse representations learned from another process but generally depend on source–target compatibility and often require target-specific optimization. LLM-based methods exploit semantic priors through prompting, feature construction, augmentation, or retrieval-augmented generation. More recent process foundation models pretrain reusable representations across process data, but they differ substantially in their adaptation mechanisms and assumptions about the downstream label space. ([Springer][1])

FM-v1 introduced an in-context foundation model for event-log PPM. FM-v2 subsequently added retrieval-based support construction and a support-conditioned prototypical prediction mechanism. The distinctive aspect of this line is that a previously unseen target log can be handled through labeled support examples without fitting a completely separate predictive model for that log. FM-v2 also permits a target-local activity vocabulary rather than requiring every process to share a single global set of activity identifiers. FM-v2 is currently a July 2026 preprint and should therefore be identified as non-peer-reviewed when discussing it. ([IEEE Xplore][2])

The literature does not appear to contain a PPM method that simultaneously:

* pretrains an event-native representation across heterogeneous event logs;
* adapts to a new event log without target-specific gradient updates;
* instantiates a new activity vocabulary from target support examples;
* keeps every globally observed support label reachable even when it is absent from the local nearest-neighbour set;
* separates class-conditional evidence from the desired deployment class prior;
* estimates the uncertainty of one- and few-example class representations;
* and evaluates these mechanisms using explicit support-coverage and balanced-recall measurements.

This combination is the strongest novelty opportunity for FM-v3. The individual ingredients have precedents in neighbouring literatures, but their integration is not a straightforward application of an existing method.

A suitable positioning would therefore be:

> **FM-v3 is a coverage- and prior-aware in-context foundation model for low-data PPM. It separates global label availability from local retrieval evidence, explicitly controls the class prior, and models uncertainty in few-example class representations.**

---

# 2. What exactly constitutes a low-data problem?

The term *low data* is used too broadly in the existing literature. At least five different failure mechanisms should be separated.

| Failure mechanism             | Definition                                                                                    | Can architecture alone solve it?                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Target-case scarcity**      | Only a few complete cases from the target process are available                               | Partly, through cross-process pretraining                            |
| **Support-pool non-coverage** | The true next activity never appears as a label anywhere in the available target support pool | Generally no, unless external semantics or constraints are available |
| **Support imbalance**         | The class is present, but has far fewer examples than common activities                       | Yes, through evidence normalization, prior correction, and shrinkage |
| **Retrieval omission**        | The class occurs in the global support pool but not in the locally retrieved top-(k) examples | Yes, through global–local memory or coverage-aware retrieval         |
| **Few-example uncertainty**   | One or two examples give an unreliable estimate of the class distribution                     | Partly, through Bayesian or shrinkage-based prototype estimation     |

These mechanisms should not be combined into one generic “few-shot performance” number.

For example, suppose an activity occurs in the target process but is absent from all support cases. When activity identifiers are opaque and no process model, textual semantics, or external constraints are provided, the model cannot infer the existence of that activity from the support set. This is an information-availability problem rather than a classification failure. Work on predicting unseen process behaviour using compliance constraints illustrates how external process knowledge can make some genuinely unseen behaviour predictable, but this is a different setting from support-only in-context prediction. ([Springer][3])

FM-v3 should therefore distinguish:

* **uncovered classes**, absent from the full support pool;
* **covered but unretrieved classes**, present globally but omitted locally;
* **retrieved but misclassified classes**, where the representation or head fails.

This distinction leads to a useful error decomposition. Let:

* (A_c) denote that class (c) is available somewhere in the global target support pool;
* (R_c) denote that (c) occurs in the locally retrieved context;
* (D_c) denote successful discrimination once (c) is present in that context.

For a candidate-restricted retrieval head,

[
\operatorname{Recall}_c =
P(A_c\mid y=c)
P(R_c\mid A_c,y=c)
P(D_c\mid R_c,y=c).
]

Because balanced accuracy is the mean of the class recalls,

[
\operatorname{BAcc}
===================

\frac{1}{|\mathcal C|}
\sum_{c\in\mathcal C}
\operatorname{Recall}_c,
]

retrieval omission can create entire classes with zero recall even when those classes are represented elsewhere in the target support pool.

A global–local FM-v3 would remove local retrieval as a hard candidate-availability condition:

[
\operatorname{Recall}^{\mathrm{FM-v3}}_c
========================================

P(A_c\mid y=c)
P(D'_c\mid A_c,y=c).
]

Local retrieval would still affect the quality of the evidence, but it would no longer determine whether the class is reachable at all.

---

# 3. Predictive process monitoring and the generalization problem

## 3.1 Conventional predictive process monitoring

PPM predicts future properties of running cases from event-log prefixes. Common tasks include next-activity prediction, suffix prediction, remaining-time estimation, outcome prediction, and risk prediction. Traditional solutions train a separate model for each event log, commonly using recurrent networks, transformers, temporal convolutional networks, tree ensembles, or tabular encodings of case prefixes. ([Springer][4])

This per-log training paradigm has several consequences:

* it assumes enough historical target cases are available;
* architecture and hyperparameter selection may have to be repeated for each process;
* activity identifiers and process structures are usually log-specific;
* the model may learn process-specific correlations that do not transfer;
* and performance can be highly sensitive to the prefix encoding and validation design.

Recent work on generalization in next-activity prediction distinguishes performance on prefixes similar to the training distribution from generalization to less frequent or structurally novel behaviour. Limited behavioural coverage, process drift, prefix-distribution changes, and overfitting all contribute to the problem. ([Springer][5])

A recent comparative preprint covering sequence models, tabular foundation models, and LLM approaches reports that conventional sequence models remain particularly strong for next-activity prediction, while tabular foundation models can be competitive for some temporal targets. General-purpose LLM approaches often incur substantially greater computational cost without consistently outperforming specialized sequence models. This means FM-v3 should not rely solely on “foundation model” status as evidence of superiority; it must demonstrate a measurable sample-efficiency or adaptation advantage. ([arXiv][6])

## 3.2 Generalization is not equivalent to low-data adaptation

A model can generalize poorly despite abundant data, and it can also perform well with little data when the small support set adequately represents the test behaviour. The FM-v3 study should therefore separate:

* **sample efficiency:** how rapidly performance improves as target cases are added;
* **behavioural coverage:** how much of the test activity and transition space appears in the support set;
* **cross-log transfer:** how much is contributed by pretraining on other processes;
* **robustness to support composition:** how performance varies across different support samples of the same size.

This also implies that the effective sample unit should be a **complete case**, not an individual prefix. A log with ten cases may yield hundreds of correlated prefixes, but those prefixes do not provide hundreds of independent observations.

## 3.3 Evaluation leakage and ground-truth ambiguity

PPM evaluations are particularly vulnerable to leakage because multiple prefixes from the same case are strongly related. Randomly distributing such prefixes across training and test sets can allow nearly identical information to occur on both sides of the split. Temporal leakage and feature leakage can similarly produce optimistic results. Leakage-safe case-level splitting and careful validation sampling are therefore essential for a credible low-data study. ([Springer][7])

Next-activity labels may also be intrinsically ambiguous. A given prefix can have several valid continuations, even when the benchmark records only one realized continuation as the “correct” answer. This label-ambiguity problem means that poor top-1 accuracy does not always imply an unreasonable prediction. It motivates top-(m) coverage, probability calibration, prediction sets, and process-conformance-aware evaluation as supplementary analyses. ([Springer][8])

The SPICE benchmark and library further emphasize reproducibility, consistent data preprocessing, and the use of metrics that do not allow frequent classes to dominate the reported result. These considerations are especially important in the low-data regime, where rare-class inclusion can vary substantially between support samples. ([arXiv][9])

---

# 4. Direct work on small event logs

## 4.1 Training conventional models on small logs

Käppel, Jablonski, and Schönig explicitly evaluated predictive monitoring approaches on small event logs. Their work shows that model rankings and reliability can change as the available training log becomes small, motivating methods designed specifically for small-sample process learning rather than merely applying architectures optimized for large logs. Their related small-sample-learning framework treats the scarcity problem as a first-class BPM concern. ([Springer][1])

This literature is directly relevant to FM-v3’s empirical motivation, but it differs in adaptation mechanism. Conventional small-log approaches still train a target-specific model. They therefore answer:

> How effectively can a model be fitted from a small target log?

FM-v3 instead asks:

> How effectively can a pretrained model infer from a small target support set without fitting a separate target model?

These are not equivalent questions. A method may perform well after extensive optimization on ten traces but still be unsuitable for instantaneous in-context adaptation.

## 4.2 Event-log sampling is not the same as genuine scarcity

Performance-preserving event-log sampling reduces a large event log while attempting to retain the predictive information needed by a downstream model. This can lower computation and storage costs and may identify redundant cases. However, the method begins from access to a larger log and can deliberately preserve important behaviour. It does not reproduce the information constraints of a genuinely new process for which only a few cases have ever been observed. ([Springer][10])

FM-v3 should therefore distinguish two experimental settings:

1. **cold-start acquisition:** sample the first or a random small set of complete cases without looking at the rest of the log;
2. **subset condensation:** select a small but optimized subset from a larger available log.

The first is the primary low-data setting. The second can be reported as an acquisition or support-selection upper bound.

## 4.3 Data augmentation

Model-agnostic event-log augmentation attempts to generate additional training information without changing the downstream predictor. Process-aware augmentation is particularly important because arbitrary perturbations may create impossible control-flow sequences, inconsistent timestamps, or unrealistic resource relationships. ([Springer][11])

Generative adversarial approaches have also been proposed for next-event prediction. Such methods can increase the apparent sample size by generating synthetic process behaviour, but they require sufficient original data to learn a realistic generator and may reproduce the imbalance or omissions already present in the original log. ([Springer][11])

SiamSA combines statistically grounded event-log transformations with Siamese or self-supervised representation learning. This is closely related to FM-v3’s representation-learning objective because it attempts to make prefix representations invariant to suitable process-preserving transformations. Its principal difference is that the augmentation and representation learning remain oriented toward target-log predictive training rather than heterogeneous-log in-context adaptation. ([Springer][12])

A recent experimental comparison of event-log augmentation techniques found that process-aware methods and transformation choices matter substantially. Generic oversampling or interpolation methods can be harmful when they create representations that do not correspond to feasible process traces. The study also reinforces that balancing the training set is not sufficient by itself: synthetic examples must preserve control-flow and temporal plausibility. This comparison is currently available as a preprint. ([arXiv][13])

### Implication for FM-v3

Augmentation could be used in two different places:

* **during heterogeneous pretraining**, to create long-tailed, low-shot, and missing-label episodes;
* **during target inference**, to augment a one-example class with process-preserving variants.

The first is safer and more coherent. Target-time synthetic augmentation risks amplifying an unrepresentative one-shot observation. A learned prototype prior or shrinkage estimator may offer a cleaner solution than generating several nearly duplicate traces.

## 4.4 LLM-based augmentation and small-log prediction

Recent work uses LLMs to generate or augment scarce event-log data for next-activity prediction. The premise is that a general-purpose language model may encode semantic and procedural knowledge that is absent from a small target log. Reported benefits suggest that LLM-generated traces can improve a conventional predictor in some scarce-data settings. However, validity checks are required because syntactically plausible activity sequences may violate the actual process. ([ScienceDirect][14])

Other work directly prompts LLMs for predictive monitoring on small event logs, reporting meaningful performance with very few training traces in selected settings. A related preprint studies LLM-generated features for temporal and activity-based prediction from approximately one hundred traces. ([Springer][15])

These approaches offer three capabilities that an event-native FM may lack:

* semantic understanding of activity names;
* world knowledge about likely procedural sequences;
* zero- or low-shot reasoning over natural-language descriptions.

They also introduce three disadvantages:

* activity labels and attributes must be textualized;
* performance can be sensitive to prompt format and model version;
* inference can be much more expensive than event-native embedding and retrieval.

The strongest FM-v3 comparison would therefore include an LLM baseline when activity labels have meaningful text, while also evaluating an anonymized-label condition. The anonymized condition tests whether the method truly learns event-log structure rather than exploiting activity-name semantics.

---

# 5. Retrieval-augmented generation in process prediction

Casciani and colleagues proposed retrieval-augmented generation for next-activity prediction. Their method retrieves process-relevant context and supplies it to a generative model, bringing the general RAG paradigm into process mining. This is one of the most directly adjacent works to FM-v2, but the representation and prediction mechanisms differ: language-model RAG constructs textual context for a generative model, whereas FM-v2 retrieves event-native support examples for a support-conditioned predictive head. ([ScienceDirect][16])

This distinction matters because retrieval can play at least three roles:

1. **information retrieval:** provide examples or knowledge that improve the query representation;
2. **distribution estimation:** approximate the local conditional label distribution;
3. **candidate construction:** determine which labels the model is permitted to output.

Most RAG and nearest-neighbour methods use retrieval primarily for the first two roles. In a support-defined activity vocabulary, retrieval can additionally perform the third role. This creates a harsher failure mode: an otherwise known target activity may become impossible to predict because it was not selected into the local context.

That candidate-coverage effect is a core opportunity for FM-v3 and should be explicitly measured rather than treated as ordinary classification error.

---

# 6. Transfer learning across event logs

## 6.1 Pairwise source-to-target transfer

Transfer-learning research in PPM investigates whether a predictor or representation learned on one event log can improve prediction on another. Experiments on suffix prediction and broader source-to-target transfer show that transfer can help, but gains depend on the compatibility of the source and target processes, the degree of shared behaviour, and the ability to align activity or feature semantics. ([Springer][17])

Pairwise transfer creates several methodological questions:

* How is a suitable source process selected?
* Must source and target activity names overlap?
* Are only encoder parameters transferred, or also prediction layers?
* How much target fine-tuning is required?
* Can negative transfer be detected before deployment?

A heterogeneous foundation model is intended to reduce the need to answer these questions separately for every source–target pair. It can learn from a distribution of synthetic or real processes rather than transferring from one chosen source.

## 6.2 Fine-tuning versus in-context adaptation

Transfer-learning methods generally adapt through gradient updates, parameter freezing, or head replacement. In-context adaptation instead retains fixed parameters and conditions predictions on a support set.

This difference has practical and scientific significance:

| Property                         | Fine-tuning                        | In-context adaptation                      |
| -------------------------------- | ---------------------------------- | ------------------------------------------ |
| Target optimization              | Required                           | Not required                               |
| Deployment latency               | Includes training                  | Primarily indexing and inference           |
| Risk of overfitting a tiny log   | High                               | Shifted to support selection and retrieval |
| New target label vocabulary      | Usually requires head modification | Can be instantiated from support           |
| Repeated adaptation to many logs | Potentially costly                 | Naturally amortized                        |
| Support-example sensitivity      | Usually indirect                   | Direct and potentially severe              |

FM-v3 should therefore compare not only final accuracy but also:

* target adaptation time;
* number of target-specific trainable parameters;
* number of target optimization steps;
* memory required for the support index;
* and performance variance across support samples.

---

# 7. Process-specific foundation models

## 7.1 The case for process foundation models

The proposal for business-process-specific foundation models argues that process data have structures not fully represented in ordinary text, vision, or tabular foundation models. Event order, case structure, resource interactions, timestamps, and process variants motivate process-native pretraining objectives and architectures. ([Springer][18])

This perspective supports FM-v3’s event-native design. It also suggests that the model should demonstrate benefits beyond applying a generic foundation model to a flattened prefix table.

## 7.2 ProcessGFM

ProcessGFM uses graph-oriented pretraining for predictive process monitoring, representing relationships among events, cases, resources, or other process entities. Its hierarchical graph design and self-supervised pretraining demonstrate another route to reusable process representations. Downstream use nevertheless follows a representation-pretraining and adaptation pattern rather than the same support-conditioned local-vocabulary mechanism as FM-v1 and FM-v2. ([MDPI][19])

ProcessGFM is consequently an important conceptual baseline, but a direct experimental comparison may require substantial engineering because the data representation, pretraining corpus, and downstream adaptation protocol differ.

## 7.3 Tabular and time-series foundation models

TabPFN demonstrates that pretraining on a distribution of synthetic tabular tasks can yield strong predictions on small datasets through in-context inference. This is highly relevant conceptually: synthetic task distributions can amortize learning across many downstream datasets, and inference can be conditioned on a small labeled table. However, event logs contain structured sequences, repeated case-level observations, timestamps, and log-specific activity vocabularies that are not naturally preserved by generic tabularization. ([Nature][20])

Time-series foundation models have also been investigated for process-model forecasting. This concerns forecasting aggregate or evolving process behaviour rather than predicting the continuation of an individual running case, but it further demonstrates growing interest in adapting pretrained sequence models to process-science tasks. ([Springer][21])

## 7.4 FM-v1 and FM-v2

FM-v1 is the most direct starting point for the proposed work because it frames PPM as an in-context task over event logs. FM-v2 adds retrieval to improve the relevance of support examples and introduces a more explicitly retrieval-conditioned prediction mechanism. ([IEEE Xplore][2])

The natural progression is:

* **FM-v1:** establish cross-log in-context prediction;
* **FM-v2:** improve target conditioning through retrieval;
* **FM-v3:** make retrieval robust when target support is small, imbalanced, incomplete, or uncertain.

The third contribution must therefore address a failure mode introduced or exposed by FM-v2, not merely add another generic encoder component.

The strongest such failure mode is:

> Local retrieval improves relevance, but it can reduce label coverage. This is particularly damaging in low-data and long-tailed logs because one omitted rare-class example can eliminate the only locally available representation of that class.

---

# 8. Few-shot and metric learning

## 8.1 Matching and prototypical networks

Matching Networks predict a query label using attention over support examples. Prototypical Networks represent each class through the mean embedding of its support examples and classify queries by distance to these class prototypes. These methods established support-conditioned classification as a standard few-shot-learning paradigm. They are therefore foundational to the interpretation of FM-v2’s support-conditioned head.

A standard prototype is

[
\mu_c=\frac{1}{n_c}\sum_{i:y_i=c}z_i,
]

with predictions based on a distance or similarity such as

[
p(y=c\mid q)
\propto
\exp\left[-d(z_q,\mu_c)/T\right].
]

Prototypes are attractive for process-local activity vocabularies because the classifier does not require a permanent global output neuron for every possible activity.

However, standard few-shot benchmarks often construct balanced (N)-way (K)-shot episodes in which every candidate class is guaranteed to occur in the support set. Real event logs violate both assumptions:

* activity frequencies are strongly imbalanced;
* the target activity may be absent from the available cases;
* retrieval can omit a class even when the full support pool contains it;
* and different activity classes may require different numbers of prototypes.

## 8.2 Multimodal classes

A single activity can occur after several structurally different prefix types. For example, the same administrative activity may follow a normal path, a rework loop, or an exceptional escalation. Averaging all corresponding support prefixes into one centroid may place the prototype in a representation region that does not correspond to any actual variant.

Infinite Mixture Prototypes address a similar issue by allowing a class to be represented by multiple prototypes and interpolating between prototype-style and nearest-neighbour prediction. ([Proceedings of Machine Learning Research][22])

For FM-v3, this motivates:

* one global prototype per activity for guaranteed coverage;
* zero or more local sub-prototypes derived from retrieved examples;
* a query-dependent gate between global and local evidence;
* prototype splitting only when support size and dispersion justify it.

This is more coherent than choosing between global prototypes and instance-level retrieval as mutually exclusive alternatives.

## 8.3 Bayesian and uncertainty-aware prototypes

Amortized Bayesian prototype methods model class prototypes as latent distributions rather than deterministic means. Their predictive uncertainty can become greater when a class has little or inconsistent support. ([Proceedings of Machine Learning Research][23])

A simpler FM-v3 alternative would be empirical-Bayes shrinkage:

[
\widetilde{\mu}_c
=================

w_c\mu_c+(1-w_c)\mu_c^{\text{prior}},
\qquad
w_c=\frac{n_c}{n_c+\kappa}.
]

Here:

* (n_c) is the number of target support examples for activity (c);
* (\mu_c) is the target activity prototype;
* (\mu_c^{\text{prior}}) is a prototype prior predicted from pretraining;
* (\kappa) controls how rapidly target evidence overrides the prior.

For a one-shot activity, the model strongly regularizes the prototype. For a well-represented activity, the target prototype dominates.

The prior could be conditioned on:

* the one-shot embedding;
* within-class support variance;
* prefix length;
* activity-name embedding, when available;
* log-level statistics;
* agreement among FM experts.

## 8.4 Episodic training does not automatically solve imbalance

Research on episodic few-shot learning shows that episode construction materially affects learned representations and conclusions. Results obtained with fixed balanced episodes do not necessarily transfer to natural long-tailed tasks. ([IEEE Xplore][24])

Work specifically addressing few-shot learning with class imbalance further demonstrates that meta-learning methods are not automatically robust to imbalanced support or query sets. Explicit rebalancing or imbalance-aware objectives can still be necessary. ([IEEE Xplore][24])

FM-v3’s pretraining episodes should therefore mix:

* balanced (N)-way (K)-shot episodes;
* natural long-tailed episodes;
* random-shot episodes with different (n_c);
* episodes where the correct class exists globally but is removed from local retrieval;
* episodes where the true class is absent from the support pool;
* support–query prior-shift episodes;
* and episodes with multimodal within-class prefixes.

The missing-local-label episodes are especially important. A model trained only with guaranteed same-class positives may never learn how to combine a weak global prototype with locally retrieved evidence.

---

# 9. Retrieval-based learning beyond process mining

## 9.1 Example retrieval for in-context learning

Prompt-retrieval research shows that the choice of in-context examples can have a large effect on language-model performance. Learned retrievers and similarity-based example selection generally outperform arbitrary prompt examples when the retrieved items are relevant to the query. ([ACL Anthology][25])

The lesson for FM-v3 is that support-set size alone is insufficient. Two support sets with the same number of cases can provide very different:

* activity coverage;
* transition coverage;
* prefix-length coverage;
* temporal-condition coverage;
* and local similarity to the query.

## 9.2 Retrieval in tabular prediction

TabR combines learned tabular representations with nearest-neighbour retrieval, showing that retrieving examples from the training set can complement parametric prediction. Retrieval and local fine-tuning have also been used to scale in-context tabular models beyond their original context-size limitations. ([OpenReview][26])

These methods support the general claim that a foundation model need not compress all target data into its parameters. A non-parametric support memory can preserve target-specific information while the pretrained encoder supplies transferable structure.

However, tabular retrieval generally assumes a fixed target column and does not face a new activity vocabulary for every dataset. FM-v3’s local class space makes candidate management a more central architectural issue.

## 9.3 Retrieval relevance versus label coverage

Pure top-(k) nearest-neighbour retrieval optimizes similarity, not class coverage. In an imbalanced support pool, frequent classes occupy denser representation regions and may dominate the retrieved neighbourhood. This can occur even when the query belongs to a rare class with a few moderately similar examples elsewhere.

Several alternatives should be compared:

### Ordinary top-(k)

Retrieve the globally most similar (k) examples.

### Class-balanced retrieval

Retrieve up to (k_c) examples per candidate activity. This improves coverage but may include many irrelevant classes and becomes expensive when the target vocabulary is large.

### Two-stage retrieval

First retrieve a larger approximate neighbourhood; then aggregate or re-rank at class level.

### Global–local retrieval

Maintain one global prototype for every class and retrieve only additional local examples or sub-prototypes.

### Adaptive retrieval

Start with a small neighbourhood and expand it when:

* predictive entropy is high;
* global and local predictions disagree;
* the nearest support similarities are weak;
* or the predicted class has very low support.

The global–local design is the strongest fit for FM-v3 because it separates two objectives:

* **global memory guarantees candidate coverage;**
* **local retrieval supplies query-specific evidence.**

---

# 10. Long-tailed activity distributions

## 10.1 Class imbalance in PPM

Activity distributions in event logs are often highly skewed. Common routine activities may constitute most next-event labels, while exceptional transitions, rework actions, escalations, cancellations, and rare outcomes occur infrequently.

PPM-specific diagnostic work shows that high aggregate performance can conceal weak predictions for minority activities. Cost-sensitive PPM similarly attempts to prevent frequent activities from dominating the training objective and the resulting decision rule. ([ScienceDirect][27])

This problem becomes more severe in a low-data regime. If an activity has a natural probability of 1%, a random support sample of fifty cases may contain no examples of it. Even when it appears once, its prototype has much greater variance than that of an activity represented hundreds of times.

## 10.2 Balanced Softmax and logit adjustment

Long-tail classification research distinguishes the evidence learned for a class from the class prior encoded in the training distribution.

Balanced Meta-Softmax modifies the softmax formulation to account for class frequencies in long-tailed training. Logit adjustment adds or subtracts class-prior terms from logits and provides theoretical connections to minimizing balanced classification error. ([NeurIPS Proceedings][28])

These methods provide important inspiration, but direct application to FM-v3 is not trivial. Conventional long-tail methods usually assume:

* a fixed global class vocabulary;
* stable global training class counts;
* and a parametric classifier with one output per class.

In FM-v3:

* the target vocabulary is log-specific;
* global support counts may be very small;
* local retrieval counts are query-dependent;
* and the number of retrieved examples of a class is partly an artefact of the retriever.

Consequently, a class should not necessarily receive more evidence simply because it appears more times in the retrieved set.

## 10.3 Separating evidence and prior

Let (R_k(q)) be the retrieved examples for query (q). A count-neutral local evidence score can be defined as

[
e_c^{\text{local}}(q)
=====================

\log
\left[
\frac{1}{n_{c,k}}
\sum_{\substack{i\in R_k(q)\y_i=c}}
\exp\left(\frac{s(z_q,z_i)}{T}\right)
\right],
]

where (n_{c,k}) is the number of retrieved examples belonging to class (c).

The corresponding global score could be

[
e_c^{\text{global}}(q)
======================

s(z_q,\widetilde{\mu}_c).
]

The two are combined through a query- and class-dependent gate:

[
e_c(q)
======

\lambda_c(q)e_c^{\text{local}}(q)
+
\left[1-\lambda_c(q)\right]e_c^{\text{global}}(q).
]

Only after estimating class-conditional evidence is the desired prior introduced:

[
\ell_c(q)=e_c(q)+\beta\log \pi_c.
]

This provides two deployment modes.

### Balanced mode

[
\pi_c=\frac{1}{|\mathcal C|}.
]

This treats activities equally and is naturally aligned with balanced accuracy.

### Natural-frequency mode

[
\pi_c=
\frac{n_c+\alpha}{N+\alpha|\mathcal C|}.
]

This uses a smoothed target-log prior and is more closely aligned with ordinary accuracy under the observed operating distribution.

Instead of claiming that one mode is universally correct, FM-v3 can report an **accuracy–balanced-accuracy Pareto curve** while varying (\beta). This shows whether the model supports different operational objectives without retraining.

---

# 11. Balanced accuracy and complementary measurements

Balanced accuracy is defined as the mean class recall:

[
\operatorname{BAcc}
===================

\frac{1}{|\mathcal C|}
\sum_{c=1}^{|\mathcal C|}
\frac{TP_c}{TP_c+FN_c}.
]

For ordinary multiclass classification, this is equivalent to macro-averaged recall. Each activity contributes equally regardless of frequency. ([IEEE Xplore][29])

This makes balanced accuracy an appropriate **primary metric** for FM-v3, because the main hypothesis concerns the recovery of poorly represented activity classes.

However, balanced accuracy should not replace every other metric.

## 11.1 Why accuracy should remain

Ordinary accuracy measures the probability that an arbitrary observed test prefix is classified correctly under the natural test distribution. A model that improves rare-class recall by producing many false alarms may increase balanced accuracy while reducing operational usefulness.

Accuracy should therefore remain a secondary metric, not disappear.

## 11.2 Why macro-F1 is also necessary

Balanced accuracy evaluates recall but does not directly penalize poor precision for rare classes. A model could assign a rare label to many common-class cases and still improve that rare class’s recall.

Macro-F1 adds class-level precision sensitivity. It should be reported alongside:

* macro precision;
* macro recall or balanced accuracy;
* per-class precision and recall;
* confusion matrices for representative logs.

There is no need to report both macro recall and balanced accuracy as separate headline metrics because, under the standard multiclass definition, they contain the same information.

## 11.3 Coverage-aware metrics

The following measurements would be particularly novel and informative:

| Measurement                                   | Question answered                                                            |
| --------------------------------------------- | ---------------------------------------------------------------------------- |
| **Support-pool label coverage**               | Is the correct class present anywhere in available support?                  |
| **Macro retrieval label-recall@(k)**          | Does local retrieval contain the correct class equally across classes?       |
| **Conditional BAcc given global coverage**    | How well does the model perform when prediction is informationally possible? |
| **Conditional BAcc given retrieved coverage** | How well does the head discriminate once retrieval succeeds?                 |
| **Zero-recall activity rate**                 | How many test activities are never predicted correctly?                      |
| **10th-percentile class recall**              | How poor is performance for the weakest classes?                             |
| **Recall by support count**                   | What happens for 0-, 1-, 2–5-, and (>5)-example activities?                  |
| **Learning-curve area**                       | How efficiently does performance improve as target cases are added?          |
| **Cases-to-threshold**                        | How many cases are required to reach a defined performance level?            |

Support-pool coverage should be treated as a property of the available data. Retrieval coverage should be treated as a property of the method. Combining them hides where improvement originates.

---

# 12. Uncertainty, calibration, and abstention

## 12.1 Sources of uncertainty in PPM

PPM contains both epistemic and aleatoric uncertainty.

**Epistemic uncertainty** arises because the model has insufficient evidence. In FM-v3 this includes:

* few target cases;
* low support count for a class;
* weak similarity to available examples;
* disagreement between global and local representations;
* and out-of-distribution prefix structures.

**Aleatoric uncertainty** arises because the process itself permits several plausible next activities. Even a perfect representation cannot know which valid branch will be realized in a particular case.

Work on uncertainty-aware neural PPM explicitly distinguishes uncertainty sources and investigates predictive intervals or uncertainty estimates. ([ScienceDirect][30])

## 12.2 Probability calibration

Modern neural networks can be highly accurate while producing overconfident probabilities. Temperature scaling and other calibration methods can improve the correspondence between reported confidence and empirical correctness. ([Proceedings of Machine Learning Research][31])

For FM-v3, probability calibration should be evaluated using:

* negative log-likelihood;
* multiclass Brier score;
* expected calibration error;
* classwise calibration error;
* reliability diagrams;
* and calibration stratified by support count.

The last measurement is particularly important. A model may be well calibrated for common activities but severely overconfident for one-shot classes.

## 12.3 Conformal prediction

Conformal prediction has been introduced into PPM to produce prediction sets with statistical coverage guarantees under suitable assumptions. Rather than returning one next activity, the model can return a set expected to contain the realized activity at a specified coverage level. ([Springer][32])

This is useful when several process continuations are plausible. Nevertheless, conformal prediction requires calibration observations, and reliable class-conditional calibration can be difficult when the target log is extremely small. FM-v3 should therefore distinguish:

* **global conformal calibration**, potentially learned across logs;
* **target-log calibration**, using held-out target cases;
* **rolling calibration**, updated as new cases complete;
* **unregularized prediction-set size**, which may become large under severe uncertainty.

Conformal evaluation can be supplementary rather than part of the first FM-v3 contribution.

## 12.4 Uncertainty-guided retrieval

Uncertainty can also control computation. For example:

1. perform prediction using global prototypes and a small local neighbourhood;
2. estimate predictive and prototype uncertainty;
3. expand retrieval only when uncertainty exceeds a threshold;
4. abstain when support coverage or similarity remains inadequate.

This yields a risk–cost trade-off:

* confident cases use little retrieval;
* ambiguous cases receive more context;
* unsupported cases are identified rather than assigned an unjustified high-confidence label.

The evaluation should include a **risk–coverage curve**, where coverage is the fraction of cases for which the model predicts and risk is the error rate among accepted predictions.

---

# 13. Evaluation protocol for a credible FM-v3 study

## 13.1 Use absolute target-case budgets

FM-v2’s percentage-based data fractions provide continuity with earlier results, but percentages can obscure large differences among event logs. A 1% sample may mean a handful of cases in one log and thousands in another.

FM-v3 should include absolute budgets such as:

[
1,\ 2,\ 4,\ 8,\ 16,\ 32,\ 64,\ 128
]

complete support cases, whenever the log permits.

Percentage budgets can additionally be reported for comparability:

[
0.5%,\ 1%,\ 2%,\ 5%,\ 10%,\ 25%,\ 50%,\ 100%.
]

The horizontal axis of the primary learning curve should be (\log_2) of the number of support cases.

## 13.2 Keep the test set fixed

For a given event log:

* define one leakage-safe test partition;
* keep those test cases unchanged across all support budgets;
* vary only the support subset;
* and ensure that all compared models receive exactly the same support cases.

Otherwise, apparent improvement may be caused by an easier test sample rather than additional support.

## 13.3 Use nested support samples

For each seed, construct nested supports:

[
S_1\subset S_2\subset S_4\subset S_8\subset\cdots.
]

This produces interpretable learning curves because increasing the budget only adds information. Independent samples at each budget can produce non-monotonic results simply because support composition changes.

Multiple nested sequences should still be sampled to quantify support-set variance.

## 13.4 Retain uncovered test classes

When a test class is absent from a small support pool, do not remove that class from the balanced-accuracy calculation. Its recall should be zero in the unconditional result.

Then separately report:

[
\operatorname{BAcc}_{\text{covered}}
]

over classes or examples for which the true activity is present in the support pool.

This yields two transparent conclusions:

* how well the complete low-data system performs;
* how well the model performs conditional on sufficient label availability.

## 13.5 Natural and controlled support sampling

Two support scenarios are useful.

### Natural support sampling

Sample complete cases randomly or chronologically. This preserves realistic class imbalance and is the primary deployment scenario.

### Coverage-maximizing sampling

Select cases to maximize activity or directly-follows-relation coverage under the same budget. This is an acquisition upper bound and can motivate active data collection.

The gap between these scenarios quantifies how much could be gained through better support-case selection without changing the model.

## 13.6 Statistical analysis

Because many prefixes come from the same case, confidence intervals should resample at the **case level**, not the prefix level.

Recommended reporting:

* mean and standard deviation across support seeds;
* case-level bootstrap confidence intervals;
* paired comparisons using identical support samples;
* per-log results;
* macro-average across logs;
* median and lower-quartile performance across logs.

Pooling all prefixes from all logs into one metric should be avoided because large logs would dominate the conclusion.

---

# 14. Appropriate baselines

A convincing paper needs baselines from several families, but not every published model has to be reproduced.

## 14.1 Internal lineage baselines

These are indispensable:

* FM-v1;
* FM-v2;
* (k)-nearest neighbours over FM embeddings;
* FM-v2 with balanced accuracy added but no architectural change;
* global-prototype-only FM;
* local-retrieval-only FM;
* full global–local FM-v3.

This separates improvement due to metric choice from improvement due to the new method.

## 14.2 Target-specific sequence models

At least one strong per-log model should be trained at every support budget:

* LSTM or GRU;
* transformer encoder;
* optionally a strong published PPM architecture.

This tests whether the foundation model genuinely improves sample efficiency relative to direct target fitting.

## 14.3 Small-log and augmentation baselines

A representative event-log augmentation method and the Siamese/self-supervised approach would show whether FM-v3’s pretrained prior is more useful than increasing the apparent target sample size. ([Springer][11])

## 14.4 Transfer-learning baseline

A source-to-target transfer model should be included where activity alignment is feasible. This establishes whether universal heterogeneous pretraining improves over selecting and adapting a particular source process. ([Springer][33])

## 14.5 Imbalance-aware baselines

Recommended variants include:

* class-weighted cross-entropy;
* cost-sensitive PPM;
* prior or logit adjustment;
* count-normalized prototype evidence;
* natural-prior and uniform-prior operating modes.

## 14.6 LLM or process-RAG baseline

An LLM/RAG comparison is valuable for logs with meaningful activity names. It should ideally be evaluated in both original-label and anonymized-label conditions. ([ScienceDirect][16])

---

# 15. Essential FM-v3 ablations

| Ablation                           | Scientific question                                                         |
| ---------------------------------- | --------------------------------------------------------------------------- |
| FM-v2 with new metrics only        | Do earlier conclusions change simply because of balanced evaluation?        |
| Global candidate prototypes        | Is local retrieval omission a principal bottleneck?                         |
| Count-normalized local aggregation | Does repeated retrieval of frequent classes improperly inflate their score? |
| Explicit prior term                | Can balanced and natural-frequency operation be controlled separately?      |
| Count-aware prototype shrinkage    | Does regularization improve one- and few-example classes?                   |
| Multiple local prototypes          | Are some activities structurally multimodal?                                |
| Natural long-tail episodes         | Does deployment-realistic episode construction matter?                      |
| Missing-local-label episodes       | Does training for retrieval failure improve fallback behaviour?             |
| Adaptive (k)                       | Can uncertainty-guided retrieval improve the accuracy–cost trade-off?       |
| Removal of cross-log pretraining   | How much improvement is truly contributed by foundation pretraining?        |

The most important interaction ablation is:

[
\text{global memory}
\times
\text{prior separation}
\times
\text{low-data episode training}.
]

This determines whether the components solve complementary problems or merely provide redundant regularization.

---

# 16. Relationship among the literature families

| Literature family             | Main low-data mechanism                                                   |        Target-specific training? | Principal limitation relative to FM-v3                                          |
| ----------------------------- | ------------------------------------------------------------------------- | -------------------------------: | ------------------------------------------------------------------------------- |
| Small-log PPM                 | Fit simpler or regularized model on few cases                             |                              Yes | Repeats optimization for every process                                          |
| Event-log augmentation        | Increase effective target sample size                                     |                              Yes | Synthetic traces may be invalid or amplify bias                                 |
| LLM augmentation              | Use language-model semantic prior                                         | Usually yes for downstream model | Cost and process-validity concerns                                              |
| LLM few-shot prompting        | Predict directly from textual examples                                    |                    No or limited | Not event-native; prompt and label semantics matter                             |
| Process RAG                   | Retrieve process examples for an LLM                                      |        Usually no target fitting | Retrieval supplies text but does not explicitly solve local-vocabulary coverage |
| Pairwise transfer learning    | Reuse a source-process representation                                     |                      Usually yes | Source selection, alignment, and negative transfer                              |
| ProcessGFM                    | Graph-based process pretraining                                           |    Usually downstream adaptation | Different adaptation and output-space assumptions                               |
| Generic prototypical learning | Infer novel classes from support prototypes                               |                No target fitting | Balanced episodes commonly guarantee class support                              |
| Long-tail classification      | Correct fixed-class priors or losses                                      |                              Yes | Assumes a stable global label vocabulary                                        |
| FM-v1                         | Cross-log event-native in-context prediction                              |         No separate target model | Limited target-context specialization                                           |
| FM-v2                         | Retrieval-augmented support conditioning                                  |         No separate target model | Local retrieval may reduce rare-label reachability                              |
| **Proposed FM-v3**            | Global coverage, local evidence, prior control, and prototype uncertainty |    No target-gradient adaptation | Must demonstrate robust gains across genuinely small support pools              |

---

# 17. Defensible novelty statement

A careful novelty claim would be:

> Existing research has separately studied predictive monitoring on small event logs, event-log augmentation, source-to-target transfer, process-specific pretraining, retrieval-augmented process prediction, metric-based few-shot classification, and long-tail correction. In the literature reviewed through August 2026, we did not identify a PPM method that jointly performs target-gradient-free adaptation to a previously unseen event log, instantiates a target-local activity vocabulary from support examples, preserves global support-label reachability independently of local retrieval, explicitly separates class evidence from the target prior, and models the uncertainty of few-example activity representations. FM-v3 addresses this intersection.

This is stronger and more defensible than claiming that no previous work has studied low-data PPM.

---

# 18. Recommended research questions

### RQ1 — Sample efficiency

> Does FM-v3 improve next-activity balanced accuracy under small absolute target-case budgets compared with FM-v1, FM-v2, target-trained models, and transfer-learning baselines?

### RQ2 — Error decomposition

> How much low-data error is attributable to support-pool non-coverage, retrieval omission, and discrimination failure?

### RQ3 — Prior control

> Does separating class-conditional evidence from the class prior improve minority-activity recall while retaining competitive natural-distribution accuracy?

### RQ4 — Prototype uncertainty

> Do count-dependent prototype shrinkage and multiple local prototypes improve prediction for activities represented by one to five support examples?

### RQ5 — Training–deployment alignment

> Does pretraining with natural long-tailed and missing-local-label episodes improve robustness compared with conventional balanced episodic training?

### RQ6 — Selective prediction

> Can prototype uncertainty and global–local disagreement support effective adaptive retrieval and abstention?

---

# 19. Testable hypotheses

**H1.** Global candidate memory will reduce the proportion of activities with zero recall, particularly at budgets below 32 target cases.

**H2.** Count-neutral evidence with a uniform deployment prior will improve balanced accuracy and macro-F1 compared with summed support mass.

**H3.** A smoothed natural-prior mode will recover most of the ordinary-accuracy difference between the balanced and frequency-sensitive variants.

**H4.** Prototype shrinkage will yield its largest gains for (n_c=1) and (n_c=2!-!5), with diminishing effects for well-supported activities.

**H5.** Missing-local-label pretraining episodes will improve conditional accuracy when the true class is absent from the initial top-(k) retrieval.

**H6.** Uncertainty-guided retrieval will obtain a superior error–retrieval-cost curve compared with a fixed large (k).

---

# 20. Recommended paper framing

A coherent title would be:

> **Coverage- and Prior-Aware In-Context Foundation Models for Low-Data Predictive Process Monitoring**

An alternative emphasizing the FM-v2 limitation is:

> **Beyond Local Retrieval: A Low-Data Foundation Model for Predictive Process Monitoring**

The central method should contain four tightly connected components:

1. **Global class memory:** every activity occurring in the target support pool receives a reachable class representation.
2. **Local support retrieval:** nearest prefixes or sub-prototypes supply query-specific evidence.
3. **Evidence–prior separation:** retrieved multiplicity does not implicitly define the deployment prior.
4. **Uncertainty-aware prototype estimation:** one- and few-example classes are shrunk toward learned priors and can trigger adaptive retrieval or abstention.

The headline empirical result should not merely be “FM-v3 has higher balanced accuracy.” It should establish:

> FM-v3 achieves more balanced performance **per available target case**, primarily by converting covered-but-unretrieved activities from impossible predictions into uncertain but reachable candidates.

---

# 21. Paper-ready related-work synthesis

Predictive process monitoring has traditionally relied on event-log-specific models trained from historical prefixes. Research on generalization has shown that predictive performance depends not only on model architecture but also on behavioural coverage, data distribution, validation design, and the relationship between observed and future process variants. Small-log studies further demonstrate that models developed for large event logs may not preserve their rankings or reliability when only a few complete cases are available. ([Springer][34])

Existing data-efficient PPM approaches can be divided into augmentation, transfer, prompting, and pretraining methods. Event-log augmentation generates or transforms traces before training a target-specific predictor, while Siamese and self-supervised methods improve representations through process-preserving transformations. LLM-based approaches use semantic priors either to augment scarce logs, construct predictive features, or perform direct few-shot prediction. Retrieval-augmented generation additionally supplies query-relevant process examples to a language model. These approaches can perform well on small logs, but they either retain target-specific training, rely on textual activity semantics, or do not explicitly distinguish global support coverage from local retrieval. ([Springer][11])

Transfer-learning and process-foundation-model research instead attempt to reuse representations across event logs. Pairwise transfer can improve target predictions when source and target processes are sufficiently compatible, but requires source selection, activity alignment, and often target fine-tuning. Process-specific foundation-model proposals and ProcessGFM demonstrate the value of pretraining over structured process data. FM-v1 and FM-v2 differ by supporting event-native in-context adaptation to a target log without fitting a separate target model; FM-v2 further uses retrieval to construct query-specific support. ([Springer][33])

Metric-based few-shot learning provides useful mechanisms for this setting. Prototypical networks instantiate classifiers from support examples, mixture-prototype methods represent multimodal classes, and Bayesian prototype methods model uncertainty under scarce support. Nevertheless, standard episodic evaluations often guarantee that every candidate class occurs in a relatively balanced support set. Real target event logs instead contain long-tailed activity distributions, and local retrieval can omit the true class even when it occurs elsewhere in the support pool. Research on imbalanced few-shot learning confirms that episodic or metric learning is not automatically robust to such imbalance. ([Proceedings of Machine Learning Research][22])

Long-tail classification offers complementary mechanisms such as cost-sensitive learning, balanced softmax formulations, and logit adjustment. In a target-local in-context classifier, however, support multiplicity and class prior must be treated carefully because retrieved counts are query-dependent and may reflect neighbourhood density rather than the deployment distribution. This motivates FM-v3’s separation of count-normalized class evidence from an explicitly selected target prior. Balanced accuracy is then the natural primary measure because it averages recall across activities, while ordinary accuracy and macro-F1 remain necessary to quantify operational frequency weighting and false-positive trade-offs. ([ScienceDirect][27])

```bibtex
@article{Berti2026InContextFM,
  author  = {Berti, Alessandro and van der Aalst, Wil M. P.},
  title   = {An In-Context Foundation Model for Predictive Process Monitoring on Event Logs},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {16959--16983},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3658877}
}

@article{Berti2026RetrievalFM,
  author  = {Berti, Alessandro and van der Aalst, Wil M. P.},
  title   = {Retrieval-Augmented In-Context Foundation Model for Predictive Process Monitoring},
  journal = {Preprints},
  year    = {2026},
  doi     = {10.20944/preprints202607.0705.v1},
  note    = {Preprint; not peer reviewed}
}

@article{Ceravolo2024PPM,
  author  = {Ceravolo, Paolo and Comuzzi, Marco and De Weerdt, Jochen and Di Francescomarino, Chiara and Maggi, Fabrizio Maria},
  title   = {Predictive Process Monitoring: Concepts, Challenges, and Future Research Directions},
  journal = {Process Science},
  volume  = {1},
  number  = {1},
  pages   = {2},
  year    = {2024},
  doi     = {10.1007/s44311-024-00002-4}
}

@incollection{DiFrancescomarino2022PPM,
  author    = {Di Francescomarino, Chiara and Ghidini, Chiara},
  title     = {Predictive Process Monitoring},
  booktitle = {Process Mining Handbook},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {448},
  pages     = {320--346},
  year      = {2022},
  doi       = {10.1007/978-3-031-08848-3_10}
}

@inproceedings{Abb2024Generalization,
  author    = {Abb, Luka and Pfeiffer, Peter and Fettke, Peter and Rehse, Jana-Rebecca},
  title     = {A Discussion on Generalization in Next-Activity Prediction},
  booktitle = {Business Process Management Workshops},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {492},
  pages     = {18--30},
  year      = {2024},
  doi       = {10.1007/978-3-031-50974-2_2}
}

@article{Pfeiffer2025Learning,
  author  = {Pfeiffer, Peter and Abb, Luka and Fettke, Peter and Rehse, Jana-Rebecca},
  title   = {Learning from the Data to Predict the Process},
  journal = {Business \& Information Systems Engineering},
  year    = {2025},
  doi     = {10.1007/s12599-025-00936-4}
}

@inproceedings{Weytjens2022Leakage,
  author    = {Weytjens, Hans and De Weerdt, Jochen},
  title     = {Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring},
  booktitle = {Business Process Management Workshops},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {436},
  pages     = {18--29},
  year      = {2022},
  doi       = {10.1007/978-3-030-94343-1_2}
}

@article{Peeperkorn2024Validation,
  author  = {Peeperkorn, Jari and vanden Broucke, Seppe and De Weerdt, Jochen},
  title   = {Validation Set Sampling Strategies for Predictive Process Monitoring},
  journal = {Information Systems},
  volume  = {121},
  pages   = {102330},
  year    = {2024},
  doi     = {10.1016/j.is.2023.102330}
}

@misc{Fertig2026FoundationBenchmark,
  author        = {Fertig, Lennart and Kirchdorfer, Lukas and Sesterhenn, Tobias},
  title         = {Revisiting Predictive Process Monitoring in the Age of Foundation Models: A Comparative Study of Sequence, Tabular, and {LLM} Approaches},
  year          = {2026},
  eprint        = {2607.27797},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Preprint}
}

@misc{Stritzel2025SPICE,
  author        = {Stritzel, Oliver and Hühnerbein, Nick and Rauch, Simon and Zarate, Itzel and Fleischmann, Lukas and Buck, Moike and Lischka, Attila and Frey, Christian},
  title         = {Towards Reproducibility in Predictive Process Mining: {SPICE}---A Deep Learning Library},
  year          = {2025},
  eprint        = {2512.16715},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Preprint}
}

@inproceedings{Pfeiffer2023LabelAmbiguity,
  author    = {Pfeiffer, Peter and Lahann, Johannes and Fettke, Peter},
  title     = {The Label Ambiguity Problem in Process Prediction},
  booktitle = {Business Process Management Workshops},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {460},
  pages     = {37--44},
  year      = {2023},
  doi       = {10.1007/978-3-031-25383-6_4}
}

@inproceedings{Kaeppel2021SmallLogs,
  author    = {Käppel, Martin and Jablonski, Stefan and Schönig, Stefan},
  title     = {Evaluating Predictive Business Process Monitoring Approaches on Small Event Logs},
  booktitle = {Quality of Information and Communications Technology},
  series    = {Communications in Computer and Information Science},
  volume    = {1439},
  pages     = {167--182},
  year      = {2021},
  doi       = {10.1007/978-3-030-85347-1_13}
}

@article{Kaeppel2021SmallSample,
  author  = {Käppel, Martin and Schönig, Stefan and Jablonski, Stefan},
  title   = {Leveraging Small Sample Learning for Business Process Management},
  journal = {Information and Software Technology},
  volume  = {132},
  pages   = {106472},
  year    = {2021},
  doi     = {10.1016/j.infsof.2020.106472}
}

@article{FaniSani2023Sampling,
  author  = {Fani Sani, Mohammadreza and Vazifehdoostirani, Mozhgan and Park, Gyunam and Pegoraro, Marco and van Zelst, Sebastiaan J. and van der Aalst, Wil M. P.},
  title   = {Performance-Preserving Event Log Sampling for Predictive Monitoring},
  journal = {Journal of Intelligent Information Systems},
  volume  = {61},
  pages   = {53--82},
  year    = {2023},
  doi     = {10.1007/s10844-022-00775-9}
}

@inproceedings{Kaeppel2023Augmentation,
  author    = {Käppel, Martin and Jablonski, Stefan},
  title     = {Model-Agnostic Event Log Augmentation for Predictive Process Monitoring},
  booktitle = {Advanced Information Systems Engineering},
  series    = {Lecture Notes in Computer Science},
  volume    = {13901},
  pages     = {381--397},
  year      = {2023},
  doi       = {10.1007/978-3-031-34560-9_23}
}

@inproceedings{Taymouri2020GAN,
  author    = {Taymouri, Farbod and La Rosa, Marcello and Erfani, Sarah and Dasht Bozorgi, Zahra and Verenich, Ilya},
  title     = {Predictive Business Process Monitoring via Generative Adversarial Nets: The Case of Next Event Prediction},
  booktitle = {Business Process Management},
  pages     = {237--256},
  year      = {2020},
  doi       = {10.1007/978-3-030-58666-9_14}
}

@inproceedings{vanStraten2026SiamSA,
  author    = {van Straten, Sjoerd and Padella, Alessandro and Hassani, Marwan},
  title     = {Leveraging Data Augmentation and Siamese Learning for Predictive Process Monitoring},
  booktitle = {Cooperative Information Systems},
  series    = {Lecture Notes in Computer Science},
  volume    = {15535},
  pages     = {70--87},
  year      = {2026},
  doi       = {10.1007/978-3-032-15538-2_5}
}

@misc{Padella2025AugmentationComparison,
  author        = {Padella, Alessandro and Vinci, Francesco and de Leoni, Massimiliano},
  title         = {An Experimental Comparison of Alternative Techniques for Event-Log Augmentation},
  year          = {2025},
  eprint        = {2511.01896},
  archivePrefix = {arXiv},
  note          = {Preprint}
}

@article{Kaeppel2026LLMAugmentation,
  author  = {Käppel, Martin and Weinzierl, Sven and Ackermann, Lars and Matzner, Martin and Jablonski, Stefan},
  title   = {Improving Next Process Activity Prediction with Scarce Event Log Data Using Data Augmentation with Large Language Models},
  journal = {Information Systems},
  volume  = {140},
  pages   = {102717},
  year    = {2026},
  doi     = {10.1016/j.is.2026.102717}
}

@inproceedings{Padella2026SmallScaleLLM,
  author    = {Padella, Alessandro and Frazzetto, Paolo and Navarin, Nicolò and de Leoni, Massimiliano},
  title     = {Enhancing Predictive Process Monitoring on Small-Scale Event Logs Using {LLM}s},
  booktitle = {Business Process Management Forum},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {564},
  pages     = {274--290},
  year      = {2026},
  doi       = {10.1007/978-3-032-02929-4_16}
}

@misc{Padella2026LLMFeatures,
  author        = {Padella, Alessandro and de Leoni, Massimiliano and Dumas, Marlon},
  title         = {Exploring {LLM} Features in Predictive Process Monitoring for Small-Scale Event-Logs},
  year          = {2026},
  eprint        = {2601.11468},
  archivePrefix = {arXiv},
  note          = {Preprint}
}

@article{Casciani2026RAG,
  author  = {Casciani, Angelo and Bernardi, Mario Luca and Cimitile, Marta and Marrella, Andrea},
  title   = {Enhancing Next Activity Prediction in Process Mining with Retrieval-Augmented Generation},
  journal = {Information Systems},
  volume  = {137},
  pages   = {102642},
  year    = {2026},
  doi     = {10.1016/j.is.2025.102642}
}

@inproceedings{Chen2023UnseenBehavior,
  author    = {Chen, Qian and Winter, Karolin and Rinderle-Ma, Stefanie},
  title     = {Predicting Unseen Process Behavior Based on Context Information from Compliance Constraints},
  booktitle = {Business Process Management Forum},
  pages     = {127--144},
  year      = {2023},
  doi       = {10.1007/978-3-031-41623-1_8}
}

@inproceedings{vanLuijken2024Transfer,
  author    = {van Luijken, Mathieu and Ketykó, István and Mannhardt, Felix},
  title     = {An Experiment on Transfer Learning for Suffix Prediction on Event Logs},
  booktitle = {Business Process Management Workshops},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {492},
  pages     = {31--43},
  year      = {2024},
  doi       = {10.1007/978-3-031-50974-2_3}
}

@article{Weinzierl2025Transfer,
  author  = {Weinzierl, Sven and Zilker, Sandra and Liessmann, Annina and Käppel, Martin and Wang, Weixin and Matzner, Martin},
  title   = {From Source to Target: Leveraging Transfer Learning for Predictive Process Monitoring in Organizations},
  journal = {Business \& Information Systems Engineering},
  year    = {2025},
  doi     = {10.1007/s12599-025-00969-9}
}

@inproceedings{Rizk2024ProcessFoundationModels,
  author    = {Rizk, Yara and Venkateswaran, Praveen and Isahagian, Vatche and Narcomey, Austin and Muthusamy, Vinod},
  title     = {A Case for Business Process-Specific Foundation Models},
  booktitle = {Business Process Management Workshops},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {492},
  pages     = {44--56},
  year      = {2024},
  doi       = {10.1007/978-3-031-50974-2_4}
}

@article{Hu2025ProcessGFM,
  author  = {Hu, Yikai and Lu, Jian and Zhao, Xuhai and Li, Yimeng and Tian, Zhen and Li, Zhiping},
  title   = {{ProcessGFM}: A Domain-Specific Graph Pretraining Prototype for Predictive Process Monitoring},
  journal = {Mathematics},
  volume  = {13},
  number  = {24},
  pages   = {3991},
  year    = {2025},
  doi     = {10.3390/math13243991}
}

@article{Hollmann2025TabPFN,
  author  = {Hollmann, Noah and Müller, Samuel and Purucker, Lennart and Krishnakumar, Arjun and Körfer, Max and Hoo, Shi Bin and Schirrmeister, Robin Tibor and Hutter, Frank},
  title   = {Accurate Predictions on Small Data with a Tabular Foundation Model},
  journal = {Nature},
  volume  = {637},
  pages   = {319--326},
  year    = {2025},
  doi     = {10.1038/s41586-024-08328-6}
}

@inproceedings{Yu2026TSFM,
  author    = {Yu, Yongbo and Peeperkorn, Jari and De Smedt, Johannes and De Weerdt, Jochen},
  title     = {Time Series Foundation Models for Process Model Forecasting},
  booktitle = {Advanced Information Systems Engineering},
  series    = {Lecture Notes in Computer Science},
  volume    = {16558},
  pages     = {78--97},
  year      = {2026},
  doi       = {10.1007/978-3-032-28110-4_5}
}

@inproceedings{Vinyals2016MatchingNetworks,
  author    = {Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and Kavukcuoglu, Koray and Wierstra, Daan},
  title     = {Matching Networks for One Shot Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {29},
  pages     = {3630--3638},
  year      = {2016}
}

@inproceedings{Snell2017Prototypical,
  author    = {Snell, Jake and Swersky, Kevin and Zemel, Richard S.},
  title     = {Prototypical Networks for Few-Shot Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {4077--4087},
  year      = {2017}
}

@inproceedings{Allen2019InfiniteMixture,
  author    = {Allen, Kelsey and Shelhamer, Evan and Shin, Hanul and Tenenbaum, Joshua},
  title     = {Infinite Mixture Prototypes for Few-Shot Learning},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {232--241},
  year      = {2019}
}

@inproceedings{Sun2021BayesianPrototype,
  author    = {Sun, Zhuo and Wu, Jijie and Li, Xiaoxu and Yang, Wenming and Xue, Jing-Hao},
  title     = {Amortized Bayesian Prototype Meta-Learning: A New Probabilistic Meta-Learning Approach to Few-Shot Image Classification},
  booktitle = {Proceedings of the 24th International Conference on Artificial Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  volume    = {130},
  pages     = {1414--1422},
  year      = {2021}
}

@inproceedings{Laenen2021Episodes,
  author    = {Laenen, Steinar and Bertinetto, Luca},
  title     = {On Episodes, Prototypical Networks, and Few-Shot Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  pages     = {24581--24592},
  year      = {2021}
}

@article{Ochal2023ImbalancedFewShot,
  author  = {Ochal, Mateusz and Patacchiola, Massimiliano and Vazquez, Jose and Storkey, Amos and Wang, Sen},
  title   = {Few-Shot Learning with Class Imbalance},
  journal = {IEEE Transactions on Artificial Intelligence},
  volume  = {4},
  number  = {5},
  pages   = {1348--1358},
  year    = {2023},
  doi     = {10.1109/TAI.2023.3298303}
}

@inproceedings{Rubin2022PromptRetrieval,
  author    = {Rubin, Ohad and Herzig, Jonathan and Berant, Jonathan},
  title     = {Learning to Retrieve Prompts for In-Context Learning},
  booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages     = {2655--2671},
  year      = {2022},
  doi       = {10.18653/v1/2022.naacl-main.191}
}

@inproceedings{Liu2022GoodExamples,
  author    = {Liu, Jiachang and Shen, Dinghan and Zhang, Yizhe and Dolan, Bill and Carin, Lawrence and Chen, Weizhu},
  title     = {What Makes Good In-Context Examples for {GPT}-3?},
  booktitle = {Proceedings of Deep Learning Inside Out},
  pages     = {100--114},
  year      = {2022},
  doi       = {10.18653/v1/2022.deelio-1.10}
}

@inproceedings{Gorishniy2024TabR,
  author    = {Gorishniy, Yury and Rubachev, Ivan and Kartashev, Nikolay and Shlenskii, Daniil and Kotelnikov, Akim and Babenko, Artem},
  title     = {{TabR}: Tabular Deep Learning Meets Nearest Neighbors},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=rhgIgTSSxW}
}

@misc{Thomas2024RetrievalTabular,
  author        = {Thomas, Valentin and Ma, Junwei and Hosseinzadeh, Rasa and Golestan, Keyvan and Yu, Guangwei and Volkovs, Maksims and Caterini, Anthony L.},
  title         = {Retrieval \& Fine-Tuning for In-Context Tabular Models},
  year          = {2024},
  eprint        = {2406.05207},
  archivePrefix = {arXiv}
}

@article{Kim2021ImbalanceDiagnostic,
  author  = {Kim, Jongchan and Comuzzi, Marco},
  title   = {A Diagnostic Framework for Imbalanced Classification in Business Process Predictive Monitoring},
  journal = {Expert Systems with Applications},
  volume  = {184},
  pages   = {115536},
  year    = {2021},
  doi     = {10.1016/j.eswa.2021.115536}
}

@inproceedings{Kaeppel2021CostSensitive,
  author    = {Käppel, Martin and Jablonski, Stefan and Schönig, Stefan},
  title     = {Cost-Sensitive Predictive Business Process Monitoring},
  booktitle = {New Trends in Database and Information Systems},
  series    = {Communications in Computer and Information Science},
  volume    = {1450},
  pages     = {14--26},
  year      = {2021},
  doi       = {10.1007/978-3-030-85082-1_2}
}

@inproceedings{Ren2020BalancedMetaSoftmax,
  author    = {Ren, Jiawei and Yu, Cunjun and Sheng, Shunan and Ma, Xiao and Zhao, Haiyu and Yi, Shuai and Li, Hongsheng},
  title     = {Balanced Meta-Softmax for Long-Tailed Visual Recognition},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}

@inproceedings{Menon2021LogitAdjustment,
  author    = {Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv},
  title     = {Long-Tail Learning via Logit Adjustment},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://openreview.net/forum?id=37nvvqkCo5}
}

@inproceedings{Brodersen2010BalancedAccuracy,
  author    = {Brodersen, Kay H. and Ong, Cheng Soon and Stephan, Klaas E. and Buhmann, Joachim M.},
  title     = {The Balanced Accuracy and Its Posterior Distribution},
  booktitle = {2010 20th International Conference on Pattern Recognition},
  pages     = {3121--3124},
  year      = {2010},
  doi       = {10.1109/ICPR.2010.764}
}

@article{Weytjens2022Uncertainty,
  author  = {Weytjens, Hans and De Weerdt, Jochen},
  title   = {Learning Uncertainty with Artificial Neural Networks for Predictive Process Monitoring},
  journal = {Applied Soft Computing},
  volume  = {125},
  pages   = {109134},
  year    = {2022},
  doi     = {10.1016/j.asoc.2022.109134}
}

@inproceedings{Skouvas2024Conformal,
  author    = {Skouvas, Fotios and Papadopoulos, Harris and Andreou, Andreas S.},
  title     = {Enhancing Predictive Process Monitoring with Conformal Prediction},
  booktitle = {Artificial Intelligence Applications and Innovations},
  series    = {IFIP Advances in Information and Communication Technology},
  volume    = {712},
  pages     = {201--214},
  year      = {2024},
  doi       = {10.1007/978-3-031-63215-0_15}
}

@inproceedings{Guo2017Calibration,
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  title     = {On Calibration of Modern Neural Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {70},
  pages     = {1321--1330},
  year      = {2017}
}
```

[1]: https://link.springer.com/chapter/10.1007/978-3-030-85347-1_13 "https://link.springer.com/chapter/10.1007/978-3-030-85347-1_13"
[2]: https://ieeexplore.ieee.org/document/11366646/ "https://ieeexplore.ieee.org/document/11366646/"
[3]: https://link.springer.com/article/10.1007/s44311-026-00048-6 "https://link.springer.com/article/10.1007/s44311-026-00048-6"
[4]: https://link.springer.com/chapter/10.1007/978-3-031-08848-3_10 "https://link.springer.com/chapter/10.1007/978-3-031-08848-3_10"
[5]: https://link.springer.com/chapter/10.1007/978-3-031-50974-2_2 "https://link.springer.com/chapter/10.1007/978-3-031-50974-2_2"
[6]: https://arxiv.org/abs/2607.27797 "https://arxiv.org/abs/2607.27797"
[7]: https://link.springer.com/chapter/10.1007/978-3-030-94343-1_2 "https://link.springer.com/chapter/10.1007/978-3-030-94343-1_2"
[8]: https://link.springer.com/chapter/10.1007/978-3-031-25383-6_4 "https://link.springer.com/chapter/10.1007/978-3-031-25383-6_4"
[9]: https://arxiv.org/abs/2512.16715 "https://arxiv.org/abs/2512.16715"
[10]: https://link.springer.com/article/10.1007/s10844-022-00775-9 "https://link.springer.com/article/10.1007/s10844-022-00775-9"
[11]: https://link.springer.com/chapter/10.1007/978-3-031-34560-9_23 "https://link.springer.com/chapter/10.1007/978-3-031-34560-9_23"
[12]: https://link.springer.com/chapter/10.1007/978-3-032-15538-2_5 "https://link.springer.com/chapter/10.1007/978-3-032-15538-2_5"
[13]: https://arxiv.org/html/2511.01896v1 "https://arxiv.org/html/2511.01896v1"
[14]: https://www.sciencedirect.com/journal/information-systems/vol/140/suppl/C "https://www.sciencedirect.com/journal/information-systems/vol/140/suppl/C"
[15]: https://link.springer.com/chapter/10.1007/978-3-032-02929-4_16 "https://link.springer.com/chapter/10.1007/978-3-032-02929-4_16"
[16]: https://www.sciencedirect.com/science/article/pii/S0306437925001280 "https://www.sciencedirect.com/science/article/pii/S0306437925001280"
[17]: https://link.springer.com/chapter/10.1007/978-3-031-50974-2_3 "https://link.springer.com/chapter/10.1007/978-3-031-50974-2_3"
[18]: https://link.springer.com/chapter/10.1007/978-3-031-50974-2_4 "https://link.springer.com/chapter/10.1007/978-3-031-50974-2_4"
[19]: https://www.mdpi.com/2227-7390/13/24/3991 "https://www.mdpi.com/2227-7390/13/24/3991"
[20]: https://www.nature.com/articles/s41586-024-08328-6 "https://www.nature.com/articles/s41586-024-08328-6"
[21]: https://link.springer.com/chapter/10.1007/978-3-032-28110-4_5 "https://link.springer.com/chapter/10.1007/978-3-032-28110-4_5"
[22]: https://proceedings.mlr.press/v97/allen19b.html "https://proceedings.mlr.press/v97/allen19b.html"
[23]: https://proceedings.mlr.press/v130/sun21a.html "https://proceedings.mlr.press/v130/sun21a.html"
[24]: https://ieeexplore.ieee.org/document/10192558/ "https://ieeexplore.ieee.org/document/10192558/"
[25]: https://aclanthology.org/2022.naacl-main.191/ "https://aclanthology.org/2022.naacl-main.191/"
[26]: https://openreview.net/forum?id=rhgIgTSSxW "https://openreview.net/forum?id=rhgIgTSSxW"
[27]: https://www.sciencedirect.com/science/article/pii/S095741742100943X "https://www.sciencedirect.com/science/article/pii/S095741742100943X"
[28]: https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html "https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html"
[29]: https://ieeexplore.ieee.org/document/5597285 "https://ieeexplore.ieee.org/document/5597285"
[30]: https://www.sciencedirect.com/science/article/pii/S1568494622004008 "https://www.sciencedirect.com/science/article/pii/S1568494622004008"
[31]: https://proceedings.mlr.press/v70/guo17a.html "https://proceedings.mlr.press/v70/guo17a.html"
[32]: https://link.springer.com/chapter/10.1007/978-3-031-63215-0_15 "https://link.springer.com/chapter/10.1007/978-3-031-63215-0_15"
[33]: https://link.springer.com/article/10.1007/s12599-025-00969-9 "https://link.springer.com/article/10.1007/s12599-025-00969-9"
[34]: https://link.springer.com/article/10.1007/s44311-024-00002-4 "https://link.springer.com/article/10.1007/s44311-024-00002-4"
