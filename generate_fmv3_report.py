#!/usr/bin/env python3
"""Generate the paper-facing FM-v3 report from machine-readable results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py


VARIANT_LABELS = {
    "minus1_fmv1_retrained": "FM-v1-style episodic retraining",
    "00_fmv2": "FM-v2 (re-evaluated)",
    "01_realistic_episodes": "FM-v2 + realistic episodes",
    "02_count_neutral": "Count-neutral local head (γ=1)",
    "03_global_prototypes": "Global prototypes",
    "04_global_shrinkage": "Global + learned shrinkage",
    "05_global_local": "Global–local head",
    "06_full_fmv3": "Full FM-v3",
    "07_full_no_pretraining": "Full FM-v3, no pretraining",
    "08_gamma0": "Local head (γ=0)",
    "09_gamma_learned": "Local head (learned γ)",
}


def _markdown_table(frame, floatfmt=".4f"):
    if frame.empty:
        return "_No completed rows._"
    rendered = frame.to_markdown(index=False, floatfmt=floatfmt)
    return re.sub(r"(?<!\w)(?:nan|inf|-inf)(?!\w)", "—", rendered)


def _read_results(root):
    rows = []
    for path in Path(root).glob("**/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _dataset_table(logs_dir="logs_eval"):
    rows = []
    paths = sorted(list(Path(logs_dir).glob("*.xes")) + list(Path(logs_dir).glob("*.xes.gz")))
    for path in paths:
        frame = pm4py.read_xes(str(path))
        case_sizes = frame.groupby("case:concept:name").size()
        rows.append({
            "log": path.name.replace(".xes.gz", "").replace(".xes", ""),
            "cases": int(case_sizes.size), "events": int(len(frame)),
            "activities": int(frame["concept:name"].nunique()),
            "median_events_per_case": float(case_sizes.median()),
        })
    return pd.DataFrame(rows)


def _aggregate_classification(frame):
    classif = frame[frame.task == "classification"].copy()
    if classif.empty:
        return classif
    for column in [
        "accuracy", "balanced_accuracy", "adjusted_balanced_accuracy", "macro_precision",
        "macro_f1", "recall_p10", "zero_recall_fraction", "support_pool_availability",
        "macro_label_recall_at_k", "macro_retrieval_given_pool", "macro_decision_given_retrieval",
        "conditional_balanced_accuracy_pool_covered",
        "conditional_balanced_accuracy_retrieval_covered", "nll", "multiclass_brier",
        "ece_10", "aurc",
    ]:
        classif[column] = pd.to_numeric(classif.get(column), errors="coerce")
    return classif


def _learning_efficiency(main):
    rows = []
    keys = ["experiment", "variant", "log", "repetition"]
    for key, group in main.groupby(keys):
        curve = group.groupby("case_budget", as_index=False).balanced_accuracy.mean().sort_values("case_budget")
        if curve.empty:
            continue
        budgets = curve.case_budget.to_numpy(dtype=float)
        values = curve.balanced_accuracy.to_numpy(dtype=float)
        x = np.log2(budgets)
        aulc = float(np.trapezoid(values, x) / max(float(x[-1] - x[0]), 1e-12)) if len(x) > 1 else float(values[0])
        target = 0.9 * float(values[-1])
        reached = budgets[values >= target]
        rows.append({
            "experiment": key[0], "variant": key[1], "log": key[2], "repetition": key[3],
            "log2_case_budget_aulc": aulc,
            "cases_to_90pct_own_max": float(reached.min()) if len(reached) else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("variant", as_index=False).agg(
        log2_case_budget_aulc=("log2_case_budget_aulc", "mean"),
        cases_to_90pct_own_max=("cases_to_90pct_own_max", "mean"),
    )


def _frequency_table(full):
    rows = []
    for _, row in full.iterrows():
        values = row.get("frequency_bin_recall")
        if not isinstance(values, dict):
            continue
        rows.append({"case_budget": row.case_budget, **values})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in ["n=0", "n=1", "n=2-5", "n>5"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame.groupby("case_budget", as_index=False)[["n=0", "n=1", "n=2-5", "n>5"]].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="evaluation_results/fmv3")
    parser.add_argument("--output", default="paper_docs/fmv3_evaluation_report.md")
    args = parser.parse_args()
    results = _read_results(args.results_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if results.empty:
        output.write_text("# FM-v3 evaluation report\n\n_Evaluation is not complete._\n", encoding="utf-8")
        return

    classif = _aggregate_classification(results)
    for column, default in {
        "evaluation_profile": "main", "retrieval_mode": "configured",
        "prior_mode": "balanced", "prior_strength": 1.0, "retrieval_k": 20,
    }.items():
        if column not in classif:
            classif[column] = default
    datasets = _dataset_table()
    regression = results[results.task == "regression"].copy()
    model_rows = classif[classif.experiment.isin(VARIANT_LABELS)]
    baseline_rows = classif[~classif.experiment.isin(VARIANT_LABELS)]
    main = model_rows[
        (model_rows.evaluation_profile == "main")
        & (model_rows.retrieval_mode == "configured")
        & (model_rows.prior_mode == "balanced")
        & (pd.to_numeric(model_rows.prior_strength, errors="coerce") == 1.0)
        & (pd.to_numeric(model_rows.retrieval_k, errors="coerce") == 20)
        & (model_rows.support_scenario == "natural")
    ].copy()
    main["variant"] = main.experiment.map(VARIANT_LABELS)

    low_budget_rows = []
    for log_name, group in main.groupby("log"):
        available = sorted(group.case_budget.unique())
        selected_budgets = sorted(set([available[0], *[value for value in [8, 32, 128] if value in available], available[-1]]))
        subset = group[group.case_budget.isin(selected_budgets)]
        agg = subset.groupby(["variant", "case_budget"], as_index=False)[["balanced_accuracy", "accuracy", "macro_f1"]].mean()
        agg.insert(0, "log", log_name)
        low_budget_rows.append(agg)
    per_log = pd.concat(low_budget_rows, ignore_index=True) if low_budget_rows else pd.DataFrame()

    macro = main.groupby(["variant", "case_budget"], as_index=False).agg(
        n_logs=("log", "nunique"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
    )
    per_log_balanced = main.groupby(
        ["variant", "case_budget", "log"], as_index=False
    ).balanced_accuracy.mean()
    robust = per_log_balanced.groupby(["variant", "case_budget"], as_index=False).agg(
        n_logs=("log", "nunique"),
        mean_log=("balanced_accuracy", "mean"),
        lower_quartile_log=("balanced_accuracy", lambda values: values.quantile(0.25)),
        worst_log=("balanced_accuracy", "min"),
    )
    coverage = main.groupby(["variant", "case_budget"], as_index=False)[[
        "support_pool_availability", "macro_label_recall_at_k",
        "macro_retrieval_given_pool", "macro_decision_given_retrieval",
        "conditional_balanced_accuracy_pool_covered",
        "conditional_balanced_accuracy_retrieval_covered", "recall_p10",
        "zero_recall_fraction",
    ]].mean()
    calibration = main.groupby("variant", as_index=False)[["nll", "multiclass_brier", "ece_10", "aurc"]].mean()
    learning_efficiency = _learning_efficiency(main)

    retrieval_ablation = model_rows[
        (model_rows.experiment == "06_full_fmv3")
        & (model_rows.evaluation_profile == "retrieval_ablation")
    ].groupby("retrieval_mode", as_index=False)[["balanced_accuracy", "accuracy", "macro_f1"]].mean()
    prior_pareto = model_rows[
        (model_rows.experiment == "06_full_fmv3")
        & (model_rows.evaluation_profile == "prior_pareto")
    ].groupby("prior_strength", as_index=False)[["balanced_accuracy", "accuracy", "macro_f1"]].mean()
    sampling_comparison = model_rows[
        (model_rows.experiment == "06_full_fmv3")
        & (model_rows.evaluation_profile == "main")
        & (model_rows.prior_mode == "balanced")
        & (pd.to_numeric(model_rows.prior_strength, errors="coerce") == 1.0)
        & (pd.to_numeric(model_rows.retrieval_k, errors="coerce") == 20)
    ].groupby("support_scenario", as_index=False)[["balanced_accuracy", "accuracy", "support_pool_availability"]].mean()

    regression_table = pd.DataFrame()
    if not regression.empty:
        regression["variant"] = regression.experiment.map(VARIANT_LABELS).fillna(regression.experiment)
        regression_table = regression[
            (regression.support_scenario == "natural")
        ].groupby(["variant", "case_budget"], as_index=False)[
            [
                "mae_hours", "median_absolute_error_hours", "normalized_mae",
                "mae_skill_vs_median", "d2_absolute_error", "r2",
                "interval_coverage", "mean_interval_width_hours",
            ]
        ].mean()

    baseline_table = pd.DataFrame()
    if not baseline_rows.empty:
        baseline_table = baseline_rows[
            baseline_rows.support_scenario == "natural"
        ].groupby(["experiment", "case_budget"], as_index=False).agg(
            n_logs=("log", "nunique"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
        )

    paper_reference = pd.read_csv("paper_docs/fmv2_paper_reference.csv")
    fmv1_reference = pd.read_csv("paper_docs/fmv1_paper_reference.csv")
    paper_class = paper_reference[paper_reference.task == "classification"]
    paper_reg = paper_reference[paper_reference.task == "regression"]
    paper_summary = paper_class.groupby("log", as_index=False)[["proto_head", "foundation_knn"]].agg(["min", "max"])
    paper_summary.columns = ["log", "proto_min", "proto_max", "knn_min", "knn_max"]
    paper_regression_summary = paper_reg.groupby("log", as_index=False)[["proto_head", "foundation_knn"]].agg(["min", "max"])
    paper_regression_summary.columns = ["log", "proto_mae_min", "proto_mae_max", "knn_mae_min", "knn_mae_max"]
    fmv1_summary = fmv1_reference.groupby("log", as_index=False).agg(
        accuracy_min=("accuracy", "min"), accuracy_max=("accuracy", "max"),
        mae_min=("mae_hours", "min"), mae_max=("mae_hours", "max"),
    )

    full = main[main.experiment == "06_full_fmv3"]
    fmv2 = main[main.experiment == "00_fmv2"]
    current_paper_comparison = []
    for log_name, log_rows in pd.concat([fmv2, full], ignore_index=True).groupby("log"):
        record = {"log": "roadtraffic" if str(log_name).startswith("roadtraffic") else log_name}
        for experiment, prefix in [("00_fmv2", "current_fmv2"), ("06_full_fmv3", "current_fmv3")]:
            subset = log_rows[log_rows.experiment == experiment]
            if subset.empty:
                continue
            subset = subset[subset.case_budget == subset.case_budget.max()]
            record[f"{prefix}_case_budget"] = int(subset.case_budget.max())
            record[f"{prefix}_accuracy"] = float(subset.accuracy.mean())
            record[f"{prefix}_balanced_accuracy"] = float(subset.balanced_accuracy.mean())
        current_paper_comparison.append(record)
    current_paper_comparison = pd.DataFrame(current_paper_comparison)
    if not current_paper_comparison.empty:
        current_paper_comparison = current_paper_comparison.merge(paper_summary, on="log", how="left")
    frequency = _frequency_table(full)
    ci_table = pd.DataFrame()
    ci_rows = pd.concat([full, fmv2], ignore_index=True)
    if not ci_rows.empty and "balanced_accuracy_ci" in ci_rows:
        ci_rows = ci_rows.copy()
        ci_rows["ci_lower"] = ci_rows.balanced_accuracy_ci.map(
            lambda value: value.get("lower") if isinstance(value, dict) else np.nan
        )
        ci_rows["ci_upper"] = ci_rows.balanced_accuracy_ci.map(
            lambda value: value.get("upper") if isinstance(value, dict) else np.nan
        )
        ci_rows["variant"] = ci_rows.experiment.map(VARIANT_LABELS)
        ci_table = ci_rows.groupby(["variant", "case_budget"], as_index=False)[
            ["balanced_accuracy", "ci_lower", "ci_upper"]
        ].mean()
    paired = full.merge(
        fmv2,
        on=["log", "repetition", "support_scenario", "case_budget", "retrieval_k"],
        suffixes=("_v3", "_v2"),
    ) if not full.empty and not fmv2.empty else pd.DataFrame()
    delta_text = "Paired FM-v2/FM-v3 rows were unavailable."
    rq_lines = {}
    if not paired.empty:
        bacc_delta = float((paired.balanced_accuracy_v3 - paired.balanced_accuracy_v2).mean())
        acc_delta = float((paired.accuracy_v3 - paired.accuracy_v2).mean())
        macro_f1_delta = float((paired.macro_f1_v3 - paired.macro_f1_v2).mean())
        variant_means = main.groupby("variant", as_index=False).balanced_accuracy.mean()
        best_variant = variant_means.loc[variant_means.balanced_accuracy.idxmax()]
        delta_text = (
            f"Across paired natural-support rows, full FM-v3 changed balanced accuracy by "
            f"**{bacc_delta:+.4f}**, macro-F1 by **{macro_f1_delta:+.4f}**, and ordinary accuracy "
            f"by **{acc_delta:+.4f}** on average relative to the re-evaluated FM-v2 checkpoint. "
            f"The highest mean balanced accuracy across all evaluated rows was achieved by "
            f"**{best_variant.variant}** ({best_variant.balanced_accuracy:.4f})."
        )
        if bacc_delta < 0:
            delta_text += (
                " The evaluated evidence therefore does not support the full FM-v3 configuration "
                "as a replacement for FM-v2 under this protocol."
            )
        zero_delta = float((paired.zero_recall_fraction_v3 - paired.zero_recall_fraction_v2).mean())
        rq_lines[1] = (
            f"- **RQ1:** paired balanced accuracy changed by {bacc_delta:+.4f}, macro-F1 by "
            f"{macro_f1_delta:+.4f}, and ordinary accuracy by {acc_delta:+.4f}; full FM-v3 did not "
            "retain competitive aggregate predictive performance."
        )
        rq_lines[4] = (
            f"- **RQ4:** the zero-recall class fraction changed by {zero_delta:+.4f}. The global–local "
            "mechanism reduced completely ignored classes slightly, but that gain did not translate "
            "into higher balanced accuracy."
        )
    if not full.empty:
        max_mask = full.case_budget == full.groupby("log").case_budget.transform("max")
        full_max = full[max_mask]
        rq_lines[2] = (
            "- **RQ2:** at each log's largest evaluated budget, mean pool availability was "
            f"{full_max.support_pool_availability.mean():.4f}, conditional retrieval P(R|A) was "
            f"{full_max.macro_retrieval_given_pool.mean():.4f}, and conditional decision P(D|R) was "
            f"{full_max.macro_decision_given_retrieval.mean():.4f}."
        )
    if not prior_pareto.empty:
        best_bacc = prior_pareto.loc[prior_pareto.balanced_accuracy.idxmax()]
        best_acc = prior_pareto.loc[prior_pareto.accuracy.idxmax()]
        rq_lines[3] = (
            f"- **RQ3:** the best swept β for balanced accuracy was {best_bacc.prior_strength:g}; "
            f"the best β for ordinary accuracy was {best_acc.prior_strength:g}."
        )
    if not learning_efficiency.empty and "Full FM-v3" in set(learning_efficiency.variant):
        row = learning_efficiency[learning_efficiency.variant == "Full FM-v3"].iloc[0]
        rq_lines[5] = (
            f"- **RQ5:** full FM-v3 reached 90% of its own largest-budget balanced accuracy at "
            f"{row.cases_to_90pct_own_max:.2f} cases on average (nested-budget interpolation was not used)."
        )
    if not calibration.empty and "Full FM-v3" in set(calibration.variant):
        row = calibration[calibration.variant == "Full FM-v3"].iloc[0]
        rq_lines[6] = (
            f"- **RQ6:** full FM-v3 mean NLL={row.nll:.4f}, Brier={row.multiclass_brier:.4f}, "
            f"ECE={row.ece_10:.4f}, and AURC={row.aurc:.4f}. In particular, ECE={row.ece_10:.4f} "
            "does not support a strong calibration claim; risk–coverage coordinates are in JSONL."
        )
    rq_text = "\n".join(rq_lines[key] for key in sorted(rq_lines)) if rq_lines else "_The full paired result matrix is not yet available._"

    text = f"""# FM-v3 evaluation report

> **Historical pre-audit ablation report.** This report evaluates the original
> `06_full_fmv3` design, not the selected final system. Several implementation
> defects were corrected afterward, and structured transition memory was then
> added. See [`fmv3_architecture_changes.md`](fmv3_architecture_changes.md) for
> the full evolution and [`structured_fmv3_report.md`](structured_fmv3_report.md)
> for the final paired result.

## Executive summary

{delta_text}

Balanced accuracy is the primary classification endpoint. Ordinary accuracy, macro-F1, coverage decomposition, calibration, and selective risk are secondary endpoints. Results are macro-averaged by event log; prefixes are never pooled across logs for the headline result.

## Implemented method

FM-v3 decouples the full support-pool label space from top-k local evidence. Its configurable head provides count-normalized local log-mean-exp evidence, global class prototypes, count-dependent prototype shrinkage, a fixed or learned global–local gate, explicit balanced/natural priors, and an uncovered-label abstention output. Pretraining mixes balanced, natural, long-tail, random-shot, missing-local-label, and missing-pool-label episodes. All target-log adaptation remains gradient-free.

The checkpoint sequence is an additive ablation: FM-v2; realistic episodes; count-neutral local evidence; global prototypes; learned shrinkage; global–local fusion; full FM-v3; no-pretraining control; and γ=0/1/learned controls.

## Evaluation protocol

- Event logs: Hospital Billing, Helpdesk, Receipt, Sepsis, and the repository's 100-case Road Traffic subset.
- Fixed case-wise held-out query set per log; support and query cases are disjoint.
- Nested absolute support budgets: 1, 2, 4, 8, 16, 32, 64, and 128 cases where available.
- Natural sampling is primary; class-aware sampling is a coverage/acquisition diagnostic.
- The query activity universe remains fixed as support shrinks; an absent support class receives zero recall.
- Repeated support sampling uses identical seeds/subsets for all methods.
- Confidence intervals resample cases, not dependent prefixes.
- The largest support pool is bounded at 128 cases for very large logs; true full-support rows are additionally included only when the support split contains at most 1,000 cases.

{_markdown_table(datasets)}

## Primary classification results

The following are repetition means for the configured head, balanced prior, k=20, and natural support.

{_markdown_table(per_log)}

### Macro-average learning curves

{_markdown_table(macro)}

### Lower-quartile and worst-log performance

Each log contributes equally. `lower_quartile_log` and `worst_log` are computed after averaging repetitions within each event log.

{_markdown_table(robust)}

### Case-level uncertainty

Intervals are percentile intervals formed by resampling complete query cases, preserving dependence among prefixes from the same case. The table averages row-level interval endpoints across logs and repetitions.

{_markdown_table(ci_table)}

### Learning efficiency

AULC integrates balanced accuracy against log2 case budget. Cases-to-threshold is the first nested budget reaching 90% of that run's largest-budget performance.

{_markdown_table(learning_efficiency)}

### Coverage decomposition

`support_pool_availability` estimates P(A), `macro_retrieval_given_pool` estimates P(R|A), and `macro_decision_given_retrieval` estimates P(D|R). `macro_label_recall_at_k` is the unconditional top-k candidate recall, so it also includes support-pool absence.

{_markdown_table(coverage)}

Full FM-v3 recall stratified by the number of support prefixes for the true activity:

{_markdown_table(frequency)}

### Calibration and selective prediction

NLL, multiclass Brier score, and ECE assess probability quality. AURC is area under the selective-risk curve; lower is better. Full reliability-bin and risk–coverage coordinates remain in the JSONL artifacts.

{_markdown_table(calibration)}

### Retrieval and head ablation

{_markdown_table(retrieval_ablation)}

### Accuracy–balanced-accuracy prior trade-off

The natural-prior strength β is swept without retraining; β=0 is prior-free and larger β increasingly follows observed support prevalence.

{_markdown_table(prior_pareto)}

### Natural versus class-aware support acquisition

{_markdown_table(sampling_comparison)}

## Conventional low-data baselines

Per-log LSTM variants use natural CE, class-weighted CE, logit adjustment, or Balanced Softmax. Weighted logistic regression, balanced random forest, Gaussian Naive Bayes, and TabPFN-v2 use the same fixed handcrafted prefix representation (activity counts, last activity, cost, and time features). For support pools exceeding TabPFN-v2's native ten-class limit, the official many-class extension applies error-correcting output codes. All methods receive the same nested support cases and fixed queries.

{_markdown_table(baseline_table)}

## Remaining-time results

MAE remains primary. Median absolute error, MAE skill versus the query-set median, D² absolute-error score, R², and empirical interval coverage are supplementary.

{_markdown_table(regression_table)}

## Relationship to published FM-v2 results

The published FM-v2 study reported ordinary accuracy and MAE under percentage-based support fractions. Its reported classification ranges are included only as context:

{_markdown_table(paper_summary)}

The closest descriptive comparison uses each current run's largest available absolute case budget. Published values remain ranges over percentage-based support fractions, so the table is contextual rather than a matched effect estimate:

{_markdown_table(current_paper_comparison)}

Published FM-v2 remaining-time MAE ranges:

{_markdown_table(paper_regression_summary)}

They are not paired comparisons: the new experiment uses absolute repeated case budgets and balanced accuracy, and `roadtraffic100traces.xes` has 100 cases rather than the paper's 10,000-case subset. The re-trained `00_fmv2` checkpoint under the new protocol is therefore the authoritative baseline.

Published FM-v1 ranges (historical context only):

{_markdown_table(fmv1_summary)}

## Answers to the research questions

{rq_text}

## Interpretation and validity

The architecture hypothesis is supported only if improvements in balanced accuracy coincide with reduced zero-recall classes and a smaller gap between pool availability and retrieval coverage. An accuracy gain without those changes is not evidence for the proposed coverage mechanism. Natural-prior rows quantify the ordinary-accuracy operating point; balanced-prior rows quantify the equal-prior operating point. Class-aware support is an acquisition upper bound, not the primary deployment estimate.

The no-pretraining checkpoint separates architectural effects from foundation pretraining. Results on five logs should not be generalized to all enterprise event logs, and the capped largest-case regime on Billing/Helpdesk should be described as a low-data study rather than a full-data benchmark.

## Reproducibility

Raw per-run results are in `evaluation_results/fmv3/<variant>/*.jsonl`; flattened tables and learning-curve summaries are stored alongside them. The external FM-v2 transcription is in `paper_docs/fmv2_paper_reference.csv`. Experiment YAML files are under `configs/fmv3/`.

Sources: [FM-v2 preprint](https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second.pdf), [balanced accuracy definition](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [Balanced Meta-Softmax](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html), [official TabPFN repository](https://github.com/PriorLabs/TabPFN).
"""
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
