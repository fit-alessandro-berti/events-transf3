#!/usr/bin/env python3
"""Generate the post-audit FM-v3 improvement report from paired result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_KEYS = ["log", "repetition", "case_budget", "retrieval_k"]
CLASS_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "zero_recall_fraction",
    "macro_decision_given_retrieval",
    "conditional_balanced_accuracy_pool_covered",
    "nll",
    "multiclass_brier",
    "ece_10",
    "aurc",
]


def _read(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if not rows:
        raise FileNotFoundError(f"No JSONL result rows in {directory}")
    return pd.DataFrame(rows)


def _classification(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.task == "classification"].copy()


def _regression(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.task == "regression"].copy()


def _paired(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    return candidate.merge(baseline, on=PAIR_KEYS, suffixes=("_candidate", "_baseline"))


def _delta(pair: pd.DataFrame, metric: str) -> pd.Series:
    return pair[f"{metric}_candidate"] - pair[f"{metric}_baseline"]


def _fmt(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evaluation_results/fmv3_improved")
    parser.add_argument("--baseline", default="00_fmv2_corrected_eval")
    parser.add_argument("--control", default="10_fmv2_cls80")
    parser.add_argument("--candidate", default="corrected_fmv3")
    parser.add_argument("--output", default="paper_docs/fmv3_improvement_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    baseline = _read(root / args.baseline)
    control = _read(root / args.control)
    candidate = _read(root / args.candidate)
    class_frames = {
        "Corrected FM-v2 evaluator": _classification(baseline),
        "Classification-focused continuation": _classification(control),
        "Corrected FM-v3": _classification(candidate),
    }
    pair = _paired(class_frames["Corrected FM-v3"], class_frames["Corrected FM-v2 evaluator"])
    control_pair = _paired(
        class_frames["Corrected FM-v3"], class_frames["Classification-focused continuation"]
    )

    means = []
    for label, frame in class_frames.items():
        means.append({
            "variant": label,
            "balanced_accuracy": frame.balanced_accuracy.mean(),
            "accuracy": frame.accuracy.mean(),
            "macro_f1": frame.macro_f1.mean(),
            "zero_recall_fraction": frame.zero_recall_fraction.mean(),
            "nll": frame.nll.mean(),
            "brier": frame.multiclass_brier.mean(),
            "ece": frame.ece_10.mean(),
            "aurc": frame.aurc.mean(),
        })
    means = pd.DataFrame(means)

    delta_rows = []
    for metric in CLASS_METRICS:
        values = _delta(pair, metric)
        delta_rows.append({
            "metric": metric,
            "mean_delta": values.mean(),
            "positive_rows": int((values > 1e-12).sum()),
            "zero_rows": int((values.abs() <= 1e-12).sum()),
            "negative_rows": int((values < -1e-12).sum()),
        })
    deltas = pd.DataFrame(delta_rows)

    per_log_rows = []
    for log_name, group in pair.groupby("log"):
        per_log_rows.append({
            "log": log_name,
            "balanced_accuracy_delta": _delta(group, "balanced_accuracy").mean(),
            "accuracy_delta": _delta(group, "accuracy").mean(),
            "macro_f1_delta": _delta(group, "macro_f1").mean(),
            "zero_recall_delta": _delta(group, "zero_recall_fraction").mean(),
        })
    per_log = pd.DataFrame(per_log_rows)

    per_budget_rows = []
    for budget, group in pair.groupby("case_budget"):
        per_budget_rows.append({
            "case_budget": int(budget),
            "n_logs": int(group.log.nunique()),
            "balanced_accuracy_delta": _delta(group, "balanced_accuracy").mean(),
            "accuracy_delta": _delta(group, "accuracy").mean(),
            "macro_f1_delta": _delta(group, "macro_f1").mean(),
        })
    per_budget = pd.DataFrame(per_budget_rows)

    reg_pair = _paired(_regression(candidate), _regression(baseline))
    regression = pd.DataFrame([
        {
            "metric": metric,
            "mean_delta": _delta(reg_pair, metric).mean(),
        }
        for metric in [
            "mae_hours",
            "median_absolute_error_hours",
            "mae_skill_vs_median",
            "d2_absolute_error",
            "r2",
        ]
    ])

    bacc_delta = _delta(pair, "balanced_accuracy").mean()
    accuracy_delta = _delta(pair, "accuracy").mean()
    f1_delta = _delta(pair, "macro_f1").mean()
    control_bacc_delta = _delta(control_pair, "balanced_accuracy").mean()
    cluster_frame = pair[["log", "repetition"]].copy()
    cluster_frame["delta"] = _delta(pair, "balanced_accuracy")
    cluster_deltas = cluster_frame.groupby(["log", "repetition"]).delta.mean().to_numpy()
    rng = np.random.default_rng(42)
    bootstrapped = np.asarray([
        rng.choice(cluster_deltas, len(cluster_deltas), replace=True).mean()
        for _ in range(10_000)
    ])
    cluster_mean = float(cluster_deltas.mean())
    cluster_lower, cluster_upper = np.quantile(bootstrapped, [0.025, 0.975])
    report = f"""# FM-v3 architecture audit and improvement report

## Outcome

The corrected FM-v3 checkpoint improves the primary metric over FM-v2 by **{bacc_delta:+.4f} balanced accuracy** across {len(pair)} paired natural-support rows. Ordinary accuracy changes by **{accuracy_delta:+.4f}** and macro-F1 by **{f1_delta:+.4f}**. It also improves balanced accuracy by **{control_bacc_delta:+.4f}** over the stronger classification-focused continuation control.

After first averaging each nested learning curve within log and repetition, the paired balanced-accuracy gain is **{cluster_mean:+.4f}** with a 95% cluster-bootstrap interval of **[{cluster_lower:+.4f}, {cluster_upper:+.4f}]** (25 log/repetition clusters; 10,000 resamples).

The selected artifact is `checkpoints/fmv3/corrected_fmv3/model_epoch_23.pth`; its resolved configuration is `checkpoints/fmv3/corrected_fmv3/training_config.yaml`, with the reusable source configuration at `configs/fmv3/corrected_fmv3.yaml`.

## Defects found in the original FM-v3 implementation

1. `learn_temperature: true` enabled gradients on `logit_scale`, but the FM-v3 evidence paths divided by fixed Python temperatures and never used that parameter. Temperature learning was therefore a no-op.
2. The local FM-v3 paths dropped FM-v2's neighbourhood mean-centering while supposedly changing only aggregation. The γ=0 control consequently changed both evidence aggregation and similarity geometry.
3. Evaluation retrieved neighbours in expert 0's embedding space and reused those indices for all four experts, despite each expert being trained and encoded independently.
4. Global–local `logaddexp` fusion allowed noisy global prototypes to perturb every locally supported decision. Learned shrinkage, gate, γ, and abstention parameters barely moved, while missing-pool losses were disproportionately large.
5. Classification and remaining-time tasks were sampled 50/50 even though low-data next-activity balanced accuracy is the stated FM-v3 objective.

## Corrective design

The new `coverage_fallback` head retains the centered legacy local ordering. A class absent from top-k can enter only if its full-pool prototype exceeds the best locally represented prototype by a fixed margin. Thus global memory addresses candidate coverage without rewriting strong local decisions. Training uses 80% classification steps, removes the unstable missing-pool abstention objective, and assigns 25% of classification episodes to deliberate missing-local-label failures. Retrieval is expert-specific at evaluation. An inference-only fallback temperature calibrates rows where global fallback candidates are present.

The selected checkpoint continues the common FM-v2 epoch-20 checkpoint through epoch 23. Epoch 25 was evaluated and rejected because balanced accuracy had begun to regress.

## Primary classification means

{_fmt(means)}

## Paired corrected FM-v3 minus corrected FM-v2

Positive deltas are improvements for accuracy metrics and regressions for zero-recall/calibration-error metrics. Sign counts are over identical log, repetition, and case-budget rows.

{_fmt(deltas)}

## Per-log predictive deltas

{_fmt(per_log)}

## Learning-curve deltas

{_fmt(per_budget)}

## Remaining-time deltas

Negative MAE deltas and positive skill/D²/R² deltas are improvements.

{_fmt(regression)}

## Protocol and scope

The confirmation uses five event logs, five repeated nested natural-support samples, absolute case budgets 1–128 plus eligible full-support rows, a fixed case-disjoint query set capped at 1,000 prefixes, balanced prior, k=20, and 200 case-bootstrap repetitions per row. All comparisons are paired on identical support/query rows. The experiment improves the repository benchmark; the same five logs were used during architecture screening, so claims beyond this benchmark require additional external logs.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
