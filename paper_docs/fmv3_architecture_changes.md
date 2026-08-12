# FM-v3 architecture changes: from FM-v2 to structured and temporal FM-v3

This document is the source of truth for the architecture currently selected
for the paper. It explains what changed, why each change was made, which parts
were trained, which parts are inference-only, and which experiments were
rejected. Numerical results live in
[`fmv3_improvement_report.md`](fmv3_improvement_report.md) and
[`structured_fmv3_report.md`](structured_fmv3_report.md). The subsequent
remaining-time redesign and its paired results are in
[`fmv3_time_transform_report.md`](fmv3_time_transform_report.md).

## Short version

The final system is not the original `06_full_fmv3` model. It has two selected
layers:

1. **Corrected FM-v3 checkpoint:** retain FM-v2's reliable centered local
   decision, use full-pool prototypes only to recover labels absent from the
   local top-k neighborhood, fix temperature and multi-expert retrieval bugs,
   and train primarily on classification with explicit missing-local-label
   episodes.
2. **Structured FM-v3 inference:** combine that neural posterior with a
   class-balanced, target-log transition memory keyed by the last one to three
   activities. The transition branch backs off to shorter contexts and is
   automatically suppressed when its context has little or no support.
3. **Learned temporal remaining-time head:** pass both elapsed time from case
   start and time since the previous event through a regression-only learned
   multi-scale adapter, predict raw hours through four learned monotone target
   transforms, and combine a query-specific gate with a support-only branch
   prior. Training uses shared scale augmentation for the two input clocks and
   the remaining-time target.

The Transformer encoder and its four-expert mixture remain the learned
representation backbone. The structured transition memory remains
classification-only. The temporal adapter and target-transform bank are
regression-only, so their parameters and forward path cannot change
classification output.

## Five-stage comparison

| Dimension | FM-v2 baseline | Original full FM-v3 | Corrected FM-v3 | Structured FM-v3 | Temporal-transform FM-v3 |
|---|---|---|---|---|---|
| Encoder | Four Transformer experts | Same backbone, continued training | Same backbone, continued from FM-v2 epoch 20 | Frozen corrected epoch-23 checkpoint | Frozen corrected encoder plus regression-only temporal residual |
| Local geometry | Neighborhood-centered cosine | Uncentered cosine | Neighborhood-centered cosine restored | Same as corrected FM-v3 | Same classifier; regression retrieval uses temporal-adapted embeddings |
| Local aggregation | Summed soft-kNN mass | Log-sum-exp with learned count normalization | Log-sum-exp with fixed $\gamma=0$ | Same as corrected FM-v3 | Same classifier; four regression transform branches |
| Candidate labels | Local top-k only | Full support pool | Full support pool, but global evidence used only for locally missing labels | Neural candidates plus structured transition evidence | Same classification rule |
| Global combination | None | Learned/fixed global-local fusion for all classes | Margin-gated coverage fallback | Same fallback inside $p_{FM}$ | Same classification rule; dual-gated regression combination |
| Class prior | Implicit local frequency | Explicit balanced/natural mode | Balanced | Balanced neural head plus uniform structured class prior | Same classification prior |
| Abstention | None | Learned missing-pool abstention | Removed | Removed | Removed |
| Target-log adaptation | Embedding retrieval | Embedding retrieval and prototypes | Corrected retrieval and prototypes | Corrected neural memory plus order-1--3 transition memory | Labeled-support branch calibration for regression |
| Target gradients | None | None | None | None | None at adaptation time |
| Remaining-time target rule | Fixed square-root neighbor regression | Same | Same | Same | Four learned monotone branches, inverted to raw hours |
| Prefix timing inputs | Fixed `log1p` coordinates | Same | Same | Same | Fixed coordinates plus learned multi-scale elapsed/inter-event adapter |
| Status | Authoritative baseline | Rejected combined design | Selected neural base | **Selected classifier** | **Selected remaining-time head** |

## Architecture at a glance

```mermaid
flowchart LR
    Q[Query prefix] --> E[Four pretrained Transformer experts]
    S[Labeled target support cases] --> SE[Support embeddings per expert]
    E --> R[Expert-specific cosine top-k retrieval]
    SE --> R
    R --> L[Centered local class evidence]
    SE --> G[Full-pool class prototypes]
    L --> C[Conservative coverage fallback]
    G --> C
    C --> A[Mean posterior across experts: p_FM]

    S --> T[Order 1-3 activity-transition counts]
    Q --> X[Recent activity suffix]
    T --> B[Longest-match suffix backoff]
    X --> B
    B --> P[Balanced structured posterior: p_str]
    B --> W[Support-count reliability]

    A --> F[Reliability-gated probability mixture]
    P --> F
    W --> F
    F --> Y[Next-activity prediction]
```

The regression path is separate:

```mermaid
flowchart LR
    Q[Query prefix clocks] --> I[Four learned power/scale maps per clock]
    I --> E[Regression-only residual into frozen expert encoder]
    S[Labeled support prefixes] --> SE[Temporal-adapted support embeddings]
    E --> R[Expert-specific top-50 retrieval]
    SE --> R
    R --> B[Four learned monotone target branches]
    B --> D[Trained query-specific gate]
    S --> C[Self-excluded support calibration]
    C --> P[Log-level branch prior]
    D --> M[50/50 convex prediction blend]
    P --> M
    M --> H[Remaining time in raw hours]
```

The two memory systems are complementary:

- The **embedding memory** transfers similarity learned across source logs and
  can generalize between prefixes that are not exact symbolic matches.
- The **transition memory** enforces the explicit target-process state and
  prevents cosine retrieval from overlooking a well-observed outgoing
  transition.

## What stayed unchanged

The following are not new architectural claims:

- Event prefixes are still encoded by the existing event-native Transformer.
- The learned configuration still uses four independently trained experts.
- Activity identifiers remain **log-local**. An identifier has no shared
  semantic meaning across logs.
- Support and query cases remain disjoint, and target-log adaptation remains
  gradient-free for both tasks.
- The selected structured branch operates only on next-activity
  classification.
- The selected temporal-transform branch operates only on remaining-time
  regression. It does not alter activity/resource encoding or classifier
  evidence.

This distinction matters: neither improvement increases encoder depth or
width. Classification gains come from how target support memory is queried;
remaining-time gains come from learned input/target geometry and support-only
branch adaptation around the frozen encoder.

## Stage 0: FM-v2 baseline

FM-v2 retrieves a local top-k neighborhood for a query embedding. It centers
the query and retrieved embeddings by the neighborhood mean, computes cosine
similarities, and sums softmax attention mass for examples with the same
label. Only labels present in top-k are candidates.

This design is effective when retrieval contains the correct activity, but it
couples three separate concerns:

1. semantic relevance of examples;
2. reachability of an activity label;
3. frequency of that label among retrieved examples.

If a label exists elsewhere in the target support pool but is absent from the
query's top-k neighborhood, its prediction probability is exactly zero.

Reference configuration:
[`configs/fmv3/00_fmv2.yaml`](../configs/fmv3/00_fmv2.yaml).

## Stage 1: the original full FM-v3 and why it was not selected

The initial `06_full_fmv3` design attempted to decouple local evidence, global
coverage, and class prevalence. It introduced:

- realistic balanced, natural, long-tail, random-shot, rare-path, and
  missing-label episodes;
- count-normalized local evidence;
- full-support class prototypes;
- count-dependent prototype shrinkage;
- fixed and learned global-local gates;
- explicit balanced or natural class priors;
- dynamic retrieval expansion;
- an abstention output for support-pool-unseen labels.

Those are useful ablation concepts, but the combined checkpoint was not a
successful final architecture. Its main problems were both conceptual and
implementation-related:

| Problem | Consequence |
|---|---|
| Learned temperature was disconnected from the FM-v3 evidence paths | `learn_temperature: true` did not change those logits |
| FM-v2 neighborhood centering was silently removed | The comparison changed similarity geometry as well as aggregation |
| Expert 0's neighbor indices were reused for every expert | Experts 1-3 were evaluated with neighborhoods from the wrong embedding space |
| Global-local `logaddexp` fusion modified every class score | Noisy global prototypes could overturn strong local decisions even when no coverage repair was needed |
| Shrinkage, gate, count-normalization, and abstention parameters moved little | Added capacity did not become a reliable adaptive mechanism |
| Missing-pool-label training asked an opaque-label model to predict an unavailable class | The task is information-theoretically impossible without external semantics or constraints |
| Training sampled classification and regression equally | Only half the updates optimized the paper's primary next-activity endpoint |

The historical pre-audit results are retained in
[`fmv3_evaluation_report.md`](fmv3_evaluation_report.md), but that report does
not describe the selected final architecture.

## Stage 2: corrected FM-v3

### 2.1 Reconnected learned temperature

The local and global evidence scales now depend on the learned `logit_scale`:

$$
\operatorname{scale}_{local}
= \frac{\operatorname{clip}(s,1,20)/s_0}{T_{local}},
\qquad
\operatorname{scale}_{global}
= \frac{\operatorname{clip}(s,1,20)/s_0}{T_{global}}.
$$

Here $s$ is the learned scale and $s_0$ is its initialization reference. This
makes the configuration flag and its gradient path agree. The implementation
is in [`components/prototypical_head.py`](../components/prototypical_head.py).

### 2.2 Restored centered local geometry

The selected head uses `local_centering: true`. For retrieved support vectors
$z_i$ and query $q$, it subtracts the local support mean and renormalizes before
computing similarities. Count normalization is fixed at $\gamma=0$, so local
class evidence is

$$
L_c(q)=\log\sum_{i\in\mathcal N_k(q):y_i=c}
\exp\bigl(\operatorname{scale}_{local}\,
\operatorname{cos}(q,z_i)\bigr).
$$

At a matched similarity scale, omitting the common softmax normalizer makes this
the same class score and ordering as FM-v2's summed attention mass. Continued
training may adjust the learned scale, but the local geometry and aggregation
form are restored. The corrected experiment therefore changes candidate
coverage without needlessly replacing a strong local rule.

### 2.3 Made retrieval expert-specific

Each expert now retrieves top-k neighbors using its own query and support
embeddings. The four posterior vectors are aligned to the same target-log class
space and averaged only after each expert has evaluated its own neighborhood.

This is an evaluator correction, but it changes effective model behavior: an
ensemble of different embedding spaces is no longer forced to share expert
0's neighborhood. The batched implementation is in
[`evaluation/fmv3_protocol.py`](../evaluation/fmv3_protocol.py).

### 2.4 Replaced global-local fusion with conservative coverage fallback

Global prototypes are still formed from every labeled support prefix:

$$
\mu_c = \operatorname{normalize}\left(
\frac{1}{N_c}\sum_{i:y_i=c} z_i
\right),
\qquad
G_c(q)=\operatorname{scale}_{global}\,q^\top\mu_c.
$$

However, global evidence no longer perturbs classes already represented in the
local neighborhood. For a locally present class, the final evidence is simply
$L_c$. For a support-covered but locally missing class,

$$
E_c = \max_{j\in C_{local}}L_j
      +G_c
      -\max_{j\in C_{local}}G_j
      -m,
$$

where the selected margin is $m=1.0$. Thus a missing class can win only when
its global prototype exceeds the best locally present prototype by more than
the margin. This makes global memory a **candidate-recovery mechanism**, not a
second classifier that rewrites all decisions.

The selected head disables prototype shrinkage and abstention because neither
was reliably learned in the original full model. A class absent from the
entire target support pool remains unavailable; the model does not pretend to
infer an opaque unseen label.

### 2.5 Changed the training distribution

The corrected continuation uses:

- 80% classification and 20% remaining-time steps, rather than 50/50;
- 25% deliberately constructed `missing_local_label` classification episodes;
- no `missing_pool_label` episodes;
- balanced, natural, long-tail, random-shot, and rare-path episodes for the
  remainder;
- a 5x prototypical-head learning-rate multiplier after warmup.

The classification episode mixture is exactly:

| Episode type | Probability within classification steps |
|---|---:|
| Balanced | 0.25 |
| Natural | 0.15 |
| Long-tail | 0.10 |
| Random-shot | 0.15 |
| Rare-path | 0.10 |
| Missing local label | 0.25 |
| Missing pool label | 0.00 |

In a missing-local-label episode, the query's correct label is removed from its
retrieved local neighbors but remains in the global support pool. This directly
trains the failure mode that coverage fallback can actually solve.

The run continues the common FM-v2 epoch-20 checkpoint. Epoch 23 was selected;
epoch 25 was evaluated and rejected after balanced accuracy began to regress.
The configuration chain is:

1. [`11_corrected_fallback_m05.yaml`](../configs/fmv3/11_corrected_fallback_m05.yaml)
2. [`12_corrected_fallback_m10.yaml`](../configs/fmv3/12_corrected_fallback_m10.yaml)
3. [`corrected_fmv3.yaml`](../configs/fmv3/corrected_fmv3.yaml)

### 2.6 Separated decision temperature from fallback calibration

The selected checkpoint uses ordinary inference temperature 1.0. Queries for
which a globally covered label is missing locally use fallback temperature
0.4. This changes probability sharpness only in the regime where the fallback
candidate set is active; it does not change which evidence sources are
available.

## Stage 3: structured transition memory

### 3.1 Bottleneck after correcting the neural head

Corrected FM-v3 still maps the complete prefix to one generic embedding and
retrieves by cosine similarity. The embedding may consider activities,
resources, costs, and time, but retrieval does not guarantee that neighbors
share the query's explicit process state. In particular, it can mix prefixes
with different recent activity suffixes and therefore different outgoing
transitions.

The final improvement adds an exact, log-local transition memory alongside the
embedding memory.

### 3.2 Stored state

For every labeled support prefix ending at activity $a_t$, the memory stores
the next activity $c$ under suffixes of order one through three:

$$
s_n=(a_{t-n+1},\ldots,a_t), \qquad n\in\{1,2,3\}.
$$

Only activity identifiers are used. Resource-conditioned keys did not improve
the diagnostic result, and elapsed-time buckets were harmful, so both were
discarded.

### 3.3 Balanced class-conditional score

Let $N_n(s,c)$ be the number of support prefixes with suffix $s$ followed by
class $c$, $N(c)$ the total support count of class $c$, and $V_n$ the number of
distinct order-$n$ suffixes in the memory. With smoothing $\alpha=0.5$,

$$
\ell_c(s)=
\log\bigl(N_n(s,c)+\alpha\bigr)
-\log\bigl(N(c)+\alpha V_n\bigr),
$$

and

$$
p_{str}(c\mid s)=\operatorname{softmax}_c\ell_c(s).
$$

This estimates suffix likelihood given the next class, $P(s\mid c)$, and then
uses a uniform class prior. It is deliberately different from the natural
transition frequency $P(c\mid s)$: a suffix observed once for a rare class is
stronger evidence than the same observation for a globally frequent class.
That choice aligns the memory with balanced accuracy.

Classes absent from the support pool receive no probability mass. The
evaluation class universe is used only to align probability columns and score
zero recall correctly; it does not make an unseen target label predictable.

### 3.4 Longest-suffix backoff

At prediction time the memory checks the order-3 suffix first, then order 2,
then order 1. It uses the longest suffix observed at least once in support. If
no suffix is observed, its own posterior is uniform over support-covered
classes, but its fusion weight is forced to zero, so the final output is exactly
the neural FM posterior.

### 3.5 Reliability-gated fusion

If the selected suffix occurs $n_s$ times, its effective mixture weight is

$$
\lambda(s)=w\frac{n_s}{n_s+\tau},
\qquad w=0.75,\quad\tau=0.5.
$$

The final posterior is

$$
p(c\mid q,s)=
\bigl(1-\lambda(s)\bigr)p_{FM}(c\mid q)
+\lambda(s)p_{str}(c\mid s).
$$

The gate has an interpretable behavior:

| Context support $n_s$ | Effective structured weight $\lambda(s)$ |
|---:|---:|
| 0 | 0.000 |
| 1 | 0.500 |
| 2 | 0.600 |
| 5 | 0.682 |
| many | approaches 0.750 |

This is a probability mixture, not a logit sum. The selected mode is
`fm_structured_mix`; a product/log-linear alternative was screened but was
weaker.

### 3.6 No additional training

Structured memory is constructed from the same target support cases already
available to the FM head. It introduces:

- no new learned parameters;
- no target-log gradient steps;
- no use of held-out query labels when fitting transition counts; the observed
  query prefix is used only to look up its context at inference;
- no changes to the frozen epoch-23 checkpoint;
- no changes to remaining-time inference.

Its configuration is an evaluation-only overlay with no `extends` clause:
[`structured_memory_eval.yaml`](../configs/fmv3/structured_memory_eval.yaml).
This is intentional: loading it must not overwrite the checkpoint's resolved
encoder or corrected-head configuration.

The protocol does supply a fixed target-log activity universe for probability
alignment and for scoring zero recall consistently across support budgets.
Support-absent labels still receive zero evidence. For deployment, this fixed
universe should come from the declared process schema or observed target-log
vocabulary; it is not learned by the transition counts.

## Stage 4: learned temporal input and target geometry

### 4.1 Why transform the two prefix clocks

The original embedder passed cost, elapsed seconds from the first case event,
and seconds from the previous event through a fixed `log1p` map. Replacing only
the remaining-time target would leave the two most relevant timing covariates
in a different geometry. The selected regression path therefore adds a learned
adapter for both `time_from_start` and `time_from_previous`; cost deliberately
stays on the stable fixed path.

For raw seconds $x_f$, feature $f$ is first converted to hours
$h_f=x_f/3600$. Each of four branches has independently learned positive power
$p_{f,k}$ and characteristic scale $s_{f,k}$:

$$
u_{f,k}=\frac{(1+h_f/s_{f,k})^{p_{f,k}}-1}{p_{f,k}},
\qquad
\tilde u_{f,k}=\frac{u_{f,k}}{1+u_{f,k}}.
$$

The rational bound keeps extreme durations finite without breaking
monotonicity. The eight resulting values (two clocks times four branches) pass
through a small normalized MLP and enter the event embedding as a learned
residual. The residual gate starts near 0.12, so the new path begins as a
conservative correction around the frozen representation.

The selected model retains the original logged clock coordinates underneath
this residual. A stricter ablation zeroed those two old coordinates for
regression, but it was materially weaker. Thus both timing variables do pass
through the new transform; retaining the fixed path is an empirical residual
design choice, not an omission.

### 4.2 Learned target-transform bank

Remaining-time labels are stored as $\sqrt{\text{hours}}$ in the historical
task files for checkpoint compatibility. The new head immediately squares
them back to raw nonnegative hours. For branch $k$ it then learns

$$
z_k(y)=\frac{(1+y/s_k)^{p_k}-1}{p_k},
\qquad
z_k^{-1}(v)=s_k\left[(1+p_kv)^{1/p_k}-1\right].
$$

This is not a fixed log or square-root target. As $p_k$ approaches zero it
approaches a log-like map; other learned powers cover square-root-like through
near-linear regimes. Each branch performs its own soft neighbor regression in
its transformed space and is inverted before aggregation, so every branch
prediction is in raw hours.

During training, one factor $a\sim\operatorname{LogUniform}(0.02,50)$ is shared
by the two prefix clocks and the remaining-time target. Predictions are divided
by $a$ before loss calculation. This exposes the model to several orders of
magnitude while preserving the reported unit.

### 4.3 Query gate plus support-only branch prior

A trained dynamic gate scores the four branch predictions separately for each
query using dimensionless neighborhood statistics, attention concentration,
branch disagreement, transform power/scale, and similarity summaries. A
second prior adapts to the target log using labeled support prefixes only:

1. predict up to 512 support prefixes with self-excluded nearest neighbors;
2. measure each branch's support MAE in raw hours;
3. turn relative support errors into a branch prior with temperature 100;
4. make no use of held-out query labels or query errors.

The selected prediction is the equal convex blend of the trained
query-specific prediction and the support-calibrated branch prediction. The
former protects tail RMSE; the latter reduces typical absolute error. Both
components use only learned transform branches—there is no fixed sqrt anchor
in the new output.

### 4.4 Constrained training and classifier isolation

Training starts from corrected FM-v3 epoch 23 and freezes the character CNN,
Transformer, classification head, and all existing parameters. Only the four
experts' temporal input adapters and target-transform banks are trainable. The
selected four-branch model has 40,776 trainable parameters.

The temporal adapter is called only when `task_type == "regression"`.
Classification calls the unchanged embedder path, and the unit suite verifies
bit-exact equality with and without the adapter present. The full confirmation
also re-evaluates classification to verify that balanced accuracy, ordinary
accuracy, and macro-F1 are unchanged.

### 4.5 MAE and RMSE unit

Both training diagnostics and final `mae_hours`/`rmse_hours` compare
predictions and targets after inversion in raw hours. Neither metric is
computed in sqrt space or log space. The only square operation is the one-time
conversion of legacy stored labels back to hours.

## Final classification inference algorithm

For each target log and support budget:

1. Split support and query by case; a case cannot occur in both sets.
2. Encode every materialized support/query prefix independently with each of
   the four experts.
3. For each expert, retrieve its own $k=20$ cosine-nearest support prefixes.
4. Compute centered local evidence and full-pool global prototypes.
5. Apply coverage fallback only to labels missing from that expert's local
   neighborhood.
6. Average the aligned four-expert probabilities to obtain $p_{FM}$.
7. Build order-1--3 transition counts from all labeled support prefixes.
8. Select the longest observed suffix for the query and compute the balanced
   structured posterior $p_{str}$.
9. Compute the support-count gate $\lambda(s)$ and mix the two posterior
   vectors.
10. Predict the activity with maximum final probability and retain the final
    maximum as confidence.

Across the full confirmation protocol, structured contexts covered 93.3% of
queries, the mean selected suffix order was 2.55, and the mean effective
structured weight was 0.606.

## Final remaining-time inference algorithm

For the same case-disjoint target support/query split:

1. Encode support and query prefixes with each frozen expert plus its
   regression-only temporal residual.
2. Retrieve the 50 nearest support prefixes independently in each expert's
   embedding space.
3. Run all four learned target transforms, their neighbor-attention scales,
   and inverse maps to obtain branch predictions in hours.
4. Produce the trained query-specific convex prediction in each expert and
   average across experts.
5. On labeled support prefixes only, repeat retrieval with the predicted
   prefix itself excluded and estimate the target-log branch prior.
6. Average each branch across experts, apply the support prior, and form the
   calibrated prediction.
7. Average the query-gated and support-calibrated predictions 50/50 and report
   raw hours.

## Training-time versus inference-time changes

| Change | Training | Inference | Selected final system? |
|---|:---:|:---:|:---:|
| Learned temperature connected to evidence scales | Yes | Yes | Yes |
| Restored local neighborhood centering | Yes | Yes | Yes |
| Expert-specific neighbor retrieval | No | Yes | Yes |
| Conservative coverage fallback | Yes | Yes | Yes |
| 80% classification task sampling | Yes | No | Yes |
| Missing-local-label episodes | Yes | No | Yes |
| Missing-pool abstention objective | Yes | Yes | **Removed** |
| Learned shrinkage and dynamic global-local gate | Yes | Yes | **Removed** |
| Order-1--3 structured transition memory | No | Yes | Yes |
| Reliability-gated posterior mixture | No | Yes | Yes |
| Four learned target transforms | Yes | Yes | Yes |
| Learned elapsed/inter-event residual | Yes | Yes | Yes |
| Shared temporal scale augmentation | Yes | No | Yes |
| Query-specific regression gate | Yes | Yes | Yes |
| Support-only regression branch prior | No | Yes | Yes |
| 50/50 regression prediction blend | No | Yes | Yes |

## What each result measures

| Variant | Balanced accuracy | Accuracy | Macro-F1 | Interpretation |
|---|---:|---:|---:|---|
| Corrected FM-v2 evaluator | 0.4136 | 0.6762 | 0.3936 | Re-evaluated neural baseline |
| Corrected FM-v3 | 0.4194 | 0.6837 | 0.3986 | Training, head, and evaluator corrections |
| Structured FM-v3 | 0.4463 | 0.7086 | 0.4177 | Corrected checkpoint plus structured inference memory |

The corrected checkpoint contributes +0.0059 balanced accuracy over FM-v2.
Structured memory then contributes +0.0269 over corrected FM-v3, for a total
+0.0327 row-averaged improvement. After first averaging within each independent
log, the end-to-end gain is +0.0329 with a 95% log-level bootstrap interval of
[+0.0229, +0.0418]. Every evaluated log improves.

NLL and multiclass Brier score improve, but ECE and AURC worsen. The method
therefore supports a predictive-performance claim, not a stronger calibration
or selective-risk claim. Calibration should be fitted separately using only a
dedicated validation set.

The selected regression redesign is evaluated on the same 200 paired
support/query rows used by the strongest fixed-sqrt baseline:

| Remaining-time model | MAE (hours) | RMSE (hours) |
|---|---:|---:|
| Fixed sqrt baseline | 1,125.8421 | 1,665.6983 |
| Learned temporal-transform FM-v3 | **1,113.7193** | **1,661.9432** |
| Delta | **-12.1228** | **-3.7551** |

Per-log and per-budget results, including the smallest-budget regressions, are
reported in
[`fmv3_time_transform_report.md`](fmv3_time_transform_report.md).

## Rejected or non-final alternatives

The following remain useful ablations but are not part of the selected method:

- the original full dynamic global-local gate;
- learned global-prototype shrinkage;
- missing-pool abstention training;
- global centering for the fallback prototypes;
- epoch 25 of the corrected continuation;
- structured-only prediction without the FM reliability fallback;
- product/log-linear FM-structured fusion;
- resource-conditioned structured keys;
- elapsed-time-bucket structured keys;
- a structured remaining-time median branch, which gave only a small aggregate
  screen improvement and regressed Billing;
- eight target/input branches, which did not improve the joint MAE/RMSE screen;
- replacing rather than supplementing the legacy logged timing coordinates;
- the query gate alone, which protected RMSE but regressed MAE;
- the support branch prior alone, which improved MAE but slightly regressed
  full-protocol RMSE.

Documenting these exclusions prevents an ablation or discarded screen from
being mistaken for the final architecture.

## Code and configuration map

| Concern | Source |
|---|---|
| Neural local/global/fallback head | [`components/prototypical_head.py`](../components/prototypical_head.py) |
| Classification/regression task sampling | [`training.py`](../training.py) |
| Retrieval episode construction and missing-label logic | [`training_strategies/retrieval_strategy.py`](../training_strategies/retrieval_strategy.py) |
| Case-disjoint protocol, per-expert retrieval, structured memory, and fusion | [`evaluation/fmv3_protocol.py`](../evaluation/fmv3_protocol.py) |
| Corrected checkpoint configuration | [`configs/fmv3/corrected_fmv3.yaml`](../configs/fmv3/corrected_fmv3.yaml) |
| Structured inference overlay | [`configs/fmv3/structured_memory_eval.yaml`](../configs/fmv3/structured_memory_eval.yaml) |
| Regression temporal input adapter | [`components/temporal_adapter.py`](../components/temporal_adapter.py) |
| Learned target-transform bank and query gate | [`components/prototypical_head.py`](../components/prototypical_head.py) |
| Four-branch temporal training configuration | [`configs/fmv3/learned_time_4_temporal.yaml`](../configs/fmv3/learned_time_4_temporal.yaml) |
| Temporal confirmation overlay | [`configs/fmv3/time_transform_confirmation_eval.yaml`](../configs/fmv3/time_transform_confirmation_eval.yaml) |
| Unit tests for fallback and structured memory | [`tests/test_fmv3_protocol.py`](../tests/test_fmv3_protocol.py) |
| Unit tests for transform inversion, gradients, scope, and classifier isolation | [`tests/test_fmv3_head.py`](../tests/test_fmv3_head.py) |
| Full structured result generator | [`generate_structured_fmv3_report.py`](../generate_structured_fmv3_report.py) |

## Reproduction and validity boundary

The exact commands are in the repository
[`README.md`](../README.md#fm-v3-experiments). The primary protocol uses five
unseen event logs, five nested natural-support repetitions, absolute case
budgets 1--128 plus eligible full-support rows and fixed case-disjoint queries.
Classification uses balanced-prior structured inference at $k=20$; the
selected remaining-time head uses $k=50$.

Architecture screening used the same five logs before the final full run.
Although the mixture parameters were frozen before confirmation, a publication
claim beyond this repository benchmark still requires additional untouched
logs or a nested log-level development/test split.
