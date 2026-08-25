# FM-v2 paper comparison data

## Current matched reference

`fmv2_new2_proto_reference_annotated.csv` is the auditable reference for the
epoch-7 retrain comparison. It contains the prototypical-head accuracy and MAE
for all five target logs and all eight support fractions. The `reference_basis`
column distinguishes the six fractions numerically printed in the paper's
compact tables (0.5%, 3%, 5%, 10%, 20%, and 100%) from 1% and 50%, which are
declared/plotted in the paper but use the frozen local paper-curve companion
transcription because the PDF does not print their per-log numbers.

`fmv2_new2_proto_reference.csv` preserves the same eight-fraction values in the
older unannotated format used by the parameter ablation.

Source: <https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second_New2.pdf>

The matched evaluator uses the paper artifact's fixed 80/20 case-disjoint logs,
nested support subsets, every held-out prefix, and the complete retrieval-depth
sweep. See `parameter_ablation_report.md` and
`../evaluate_fmv2_paper_protocol.py`.

## Historical six-fraction transcription

`fmv2_paper_reference.csv` transcribes both FM-v2 arms from the six numeric
rows in classification Figure 5 and regression Figure 7 of the July 2026
revision of Berti and van der Aalst, *Retrieval-Augmented In-Context Foundation
Model for Predictive Process Monitoring*:

- `proto_head`: FM-v2's support-conditioned prediction head.
- `foundation_knn`: direct kNN on FM-v2 embeddings.
- Classification values are ordinary accuracy; regression values are MAE in hours.

Source: <https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second_New2.pdf>

These 60 cells are the strict PDF-numeric comparison. They also let the
epoch-7 report compare against the prototypical head, foundation kNN, and the
better arm per cell without relying on values inferred from plot geometry.

The repository's `logs_eval/roadtraffic100traces.xes` contains 100 cases, while the paper reports the 10,000-case Road Traffic subset. Those rows are not directly comparable even for ordinary accuracy/MAE and must be labeled as contextual only.
