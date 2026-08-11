#!/usr/bin/env python3
"""Generate the paired report for reliability-gated structured FM-v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_KEYS = ["log", "repetition", "support_scenario", "case_budget"]
METRICS = [
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "zero_recall_fraction",
    "nll",
    "multiclass_brier",
    "ece_10",
    "aurc",
]


def _classification(directory: Path) -> pd.DataFrame:
    path = directory / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return frame[frame.task == "classification"].copy()


def _fmt(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evaluation_results/fmv3_improved")
    parser.add_argument("--fmv2", default="00_fmv2_corrected_eval")
    parser.add_argument("--baseline", default="corrected_fmv3")
    parser.add_argument("--candidate", default="structured_fmv3")
    parser.add_argument("--output", default="paper_docs/structured_fmv3_report.md")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    root = Path(args.root)
    fmv2 = _classification(root / args.fmv2)
    baseline = _classification(root / args.baseline)
    candidate = _classification(root / args.candidate)
    pair = candidate.merge(baseline, on=PAIR_KEYS, suffixes=("_structured", "_base"))
    if len(pair) != len(candidate) or len(pair) != len(baseline):
        raise ValueError(
            f"Expected one-to-one paired rows, got candidate={len(candidate)}, "
            f"baseline={len(baseline)}, paired={len(pair)}"
        )
    fmv2_pair = candidate.merge(fmv2, on=PAIR_KEYS, suffixes=("_structured", "_fmv2"))
    if len(fmv2_pair) != len(candidate) or len(fmv2_pair) != len(fmv2):
        raise ValueError(
            f"Expected one-to-one FM-v2 pairs, got candidate={len(candidate)}, "
            f"fmv2={len(fmv2)}, paired={len(fmv2_pair)}"
        )

    means = []
    for name, frame in [
        ("Corrected FM-v2 evaluator", fmv2),
        ("Corrected FM-v3", baseline),
        ("Structured FM-v3", candidate),
    ]:
        means.append({"variant": name, **{metric: frame[metric].mean() for metric in METRICS}})
    means = pd.DataFrame(means)

    deltas = []
    for metric in METRICS:
        values = pair[f"{metric}_structured"] - pair[f"{metric}_base"]
        deltas.append({
            "metric": metric,
            "mean_delta": values.mean(),
            "improved_rows": int((values > 1e-12).sum()) if metric not in {
                "zero_recall_fraction", "nll", "multiclass_brier", "ece_10", "aurc"
            } else int((values < -1e-12).sum()),
            "tied_rows": int((values.abs() <= 1e-12).sum()),
            "regressed_rows": int((values < -1e-12).sum()) if metric not in {
                "zero_recall_fraction", "nll", "multiclass_brier", "ece_10", "aurc"
            } else int((values > 1e-12).sum()),
        })
    deltas = pd.DataFrame(deltas)

    per_log = []
    for log_name, group in pair.groupby("log"):
        per_log.append({
            "log": log_name,
            **{
                f"{metric}_delta": (
                    group[f"{metric}_structured"] - group[f"{metric}_base"]
                ).mean()
                for metric in ["balanced_accuracy", "accuracy", "macro_f1"]
            },
        })
    per_log = pd.DataFrame(per_log)

    per_budget = []
    for budget, group in pair.groupby("case_budget"):
        candidate_rows = candidate[candidate.case_budget == budget]
        per_budget.append({
            "case_budget": int(budget),
            "paired_rows": len(group),
            "context_coverage": candidate_rows.structured_context_coverage.mean(),
            "mean_effective_weight": candidate_rows.structured_mean_effective_weight.mean(),
            **{
                f"{metric}_delta": (
                    group[f"{metric}_structured"] - group[f"{metric}_base"]
                ).mean()
                for metric in ["balanced_accuracy", "accuracy", "macro_f1"]
            },
        })
    per_budget = pd.DataFrame(per_budget).sort_values("case_budget")

    # Repetitions reuse a log's fixed test cases/query prefixes, so the log—not
    # the repeated support draw—is the independent resampling unit.
    cluster = pair[["log"]].copy()
    cluster["delta"] = pair.balanced_accuracy_structured - pair.balanced_accuracy_base
    cluster_values = cluster.groupby("log").delta.mean().to_numpy()
    rng = np.random.default_rng(42)
    bootstrap = np.asarray([
        rng.choice(cluster_values, len(cluster_values), replace=True).mean()
        for _ in range(args.bootstrap_samples)
    ])
    cluster_mean = float(cluster_values.mean())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    fmv2_cluster = fmv2_pair[["log"]].copy()
    fmv2_cluster["delta"] = (
        fmv2_pair.balanced_accuracy_structured - fmv2_pair.balanced_accuracy_fmv2
    )
    fmv2_cluster_values = (
        fmv2_cluster.groupby("log").delta.mean().to_numpy()
    )
    rng = np.random.default_rng(42)
    fmv2_bootstrap = np.asarray([
        rng.choice(fmv2_cluster_values, len(fmv2_cluster_values), replace=True).mean()
        for _ in range(args.bootstrap_samples)
    ])
    fmv2_cluster_mean = float(fmv2_cluster_values.mean())
    fmv2_lower, fmv2_upper = np.quantile(fmv2_bootstrap, [0.025, 0.975])

    bacc_delta = float(
        (pair.balanced_accuracy_structured - pair.balanced_accuracy_base).mean()
    )
    accuracy_delta = float((pair.accuracy_structured - pair.accuracy_base).mean())
    f1_delta = float((pair.macro_f1_structured - pair.macro_f1_base).mean())
    fmv2_bacc_delta = float(
        (fmv2_pair.balanced_accuracy_structured - fmv2_pair.balanced_accuracy_fmv2).mean()
    )
    fmv2_accuracy_delta = float(
        (fmv2_pair.accuracy_structured - fmv2_pair.accuracy_fmv2).mean()
    )
    fmv2_f1_delta = float(
        (fmv2_pair.macro_f1_structured - fmv2_pair.macro_f1_fmv2).mean()
    )
    context_coverage = float(candidate.structured_context_coverage.mean())
    mean_selected_order = float(candidate.structured_mean_selected_order.mean())
    mean_effective_weight = float(candidate.structured_mean_effective_weight.mean())
    report = f"""# Structured FM-v3 architecture and paired evaluation

## Outcome

The frozen structured-memory FM-v3 improves the corrected FM-v2 evaluator by
**{fmv2_bacc_delta:+.4f} balanced accuracy**, **{fmv2_accuracy_delta:+.4f} accuracy**,
and **{fmv2_f1_delta:+.4f} macro-F1** across {len(pair)} identical
support/query rows. The isolated increment over the intermediate corrected
FM-v3 is **{bacc_delta:+.4f}**, **{accuracy_delta:+.4f}**, and **{f1_delta:+.4f}**,
respectively.
After averaging the nested curves within each log, the end-to-end
FM-v2-to-structured-FM-v3 balanced-accuracy gain is **{fmv2_cluster_mean:+.4f}**,
with a 95% cluster-bootstrap interval of **[{fmv2_lower:+.4f}, {fmv2_upper:+.4f}]**.
The isolated structured-memory gain is **{cluster_mean:+.4f}** with interval
**[{lower:+.4f}, {upper:+.4f}]** ({len(cluster_values)} log-level clusters;
{args.bootstrap_samples:,} resamples each).

## Architectural bottleneck

FM-v3 compresses a prefix into a generic embedding and retrieves globally by
cosine similarity. That representation does not enforce the discrete process
state already observed in the prefix: its last activity and recent activity
suffix. Consequently, semantically similar prefixes can retrieve examples from
different outgoing transitions, especially as the support pool grows.

The new branch stores log-local suffix-to-next-activity counts for orders 1--3.
It uses a uniform next-class prior and a smoothed class-conditional suffix
likelihood, preferring the longest observed suffix and backing off when it is
unseen. This explicitly optimizes rare-class evidence rather than replaying the
natural class frequency. Its posterior is mixed with corrected FM-v3 using
`lambda(s) = 0.75 * n(s) / (n(s) + 0.5)`, where `n(s)` is support count for the
selected suffix. An unseen suffix has zero weight, so the branch collapses to
the FM prediction rather than guessing.

The order, smoothing, mixture weight, and shrinkage constant were selected on a
smaller two-repetition diagnostic protocol. They were then frozen before this
five-repetition full evaluation. Across the confirmation rows, the structured
memory covers **{context_coverage:.1%}** of queries, chooses suffix order
**{mean_selected_order:.2f}** on average, and receives mean effective mixture
weight **{mean_effective_weight:.3f}** after support shrinkage.

## Classification means

{_fmt(means)}

## Paired structured-minus-corrected deltas

For error metrics (zero recall, NLL, Brier, ECE, and AURC), negative values are
improvements. Sign counts use the corresponding favorable direction.

{_fmt(deltas)}

## Per-log deltas

{_fmt(per_log)}

## Learning-curve deltas

{_fmt(per_budget)}

## Interpretation and limitations

The gain is not a calibration-only effect: ordinary accuracy and macro-F1 rise
alongside balanced accuracy, and all five logs improve. NLL and Brier score also
improve, although ECE and AURC worsen; calibration and selective prediction
should therefore remain separate validation-only steps. The largest gains occur at medium/high support,
consistent with the structured memory becoming reliable as transition counts
accumulate.

This is a no-gradient target-memory augmentation of the already trained
corrected checkpoint; remaining-time predictions are deliberately unchanged in
the primary comparison. The same five logs were used for architectural
screening, so a publication claim still requires confirmation on additional
untouched logs or a nested development/test split.

## Reproduction

Checkpoint: `checkpoints/fmv3/corrected_fmv3/model_epoch_23.pth`

Evaluation overlay: `configs/fmv3/structured_memory_eval.yaml`

Results: `evaluation_results/fmv3_improved/structured_fmv3`

Protocol: five unseen event logs, five repeated nested natural-support samples,
absolute case budgets 1--128 plus eligible full-support rows, a case-disjoint
query set capped at 1,000 prefixes per log, balanced prior, and retrieval k=20.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
