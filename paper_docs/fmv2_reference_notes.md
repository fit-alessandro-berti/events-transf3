# FM-v2 paper comparison data

`fmv2_paper_reference.csv` transcribes the two FM columns from Figures 5 and 6 of Berti and van der Aalst, *Retrieval-Augmented In-Context Foundation Model for Predictive Process Monitoring* (2026 preprint):

- `proto_head`: FM-v2's support-conditioned prediction head.
- `foundation_knn`: direct kNN on FM-v2 embeddings.
- Classification values are ordinary accuracy; regression values are MAE in hours.

Source: <https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second.pdf>

These values are contextual, not paired estimates for the new protocol. The paper uses an 80/20 case split and percentage-based support fractions (0.5%, 3%, 5%, 10%, 20%, 100%), whereas FM-v3's primary protocol uses repeated nested absolute case budgets and balanced accuracy. Direct claims must therefore use FM-v2 re-evaluation from checkpoint `00_fmv2`; the published figures are reported as an external sanity check for ordinary accuracy and MAE only.

The repository's `logs_eval/roadtraffic100traces.xes` contains 100 cases, while the paper reports the 10,000-case Road Traffic subset. Those rows are not directly comparable even for ordinary accuracy/MAE and must be labeled as contextual only.
