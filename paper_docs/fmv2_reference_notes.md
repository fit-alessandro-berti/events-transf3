# FM-v2 paper comparison data

## Current matched reference

`fmv2_new2_proto_reference.csv` is the authoritative reference for the
parameter ablation in this repository. It contains the prototypical-head
accuracy and MAE for all five target logs and all eight support fractions,
including 1% and 50%, from the July 2026 revised paper and its public result
artifact.

Source: <https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second_New2.pdf>

The matched evaluator uses the paper artifact's fixed 80/20 case-disjoint logs,
nested support subsets, every held-out prefix, and the complete retrieval-depth
sweep. See `parameter_ablation_report.md` and
`../evaluate_fmv2_paper_protocol.py`.

## Historical six-fraction transcription

`fmv2_paper_reference.csv` transcribes the two FM columns from Figures 5 and 6 of Berti and van der Aalst, *Retrieval-Augmented In-Context Foundation Model for Predictive Process Monitoring* (2026 preprint):

- `proto_head`: FM-v2's support-conditioned prediction head.
- `foundation_knn`: direct kNN on FM-v2 embeddings.
- Classification values are ordinary accuracy; regression values are MAE in hours.

Source: <https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second.pdf>

These older values remain contextual for the historical FM-v3 absolute-budget
protocol. They are not the source used by the new matched percentage-based
ablation.

The repository's `logs_eval/roadtraffic100traces.xes` contains 100 cases, while the paper reports the 10,000-case Road Traffic subset. Those rows are not directly comparable even for ordinary accuracy/MAE and must be labeled as contextual only.
