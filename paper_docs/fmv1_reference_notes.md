# FM-v1 comparison data

`fmv1_paper_reference.csv` transcribes Table 2 of the 2026 FM-v2 preprint for the earlier FM-v1 model at 1, 5, 10, and 20 shots. The source explicitly states that these are provided FM-v1 result files, not a matched rerun.

Source: <https://www.preprints.org/manuscript/202607.0705>

The new experiment therefore reports two different controls:

1. `minus1_fmv1_retrained`, an FM-v1-style balanced episodic model trained in the current repository and evaluated under the matched case-budget protocol.
2. The published FM-v1 values, used only for historical context because their support construction, queries, and metric suite differ.

The historical table has ordinary accuracy, MAE, and R² only; it cannot establish FM-v1 balanced accuracy.
