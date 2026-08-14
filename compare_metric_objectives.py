#!/usr/bin/env python3
"""Compare metric-target training runs and select source-held-out epochs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


METRICS = {
    "classification_accuracy": (
        "task/classification/head/classification/episode_accuracy",
        "max",
    ),
    "classification_balanced_accuracy": (
        "task/classification/head/classification/episode_balanced_accuracy",
        "max",
    ),
    "classification_macro_f1": (
        "task/classification/head/classification/episode_macro_f1",
        "max",
    ),
    "classification_nll": (
        "task/classification/head/classification/nll",
        "min",
    ),
    "classification_brier": (
        "task/classification/loss/classification/brier_surrogate_raw",
        "min",
    ),
    "regression_mae_hours": (
        "task/regression/head/regression/mae_hours",
        "min",
    ),
    "regression_rmse_hours": (
        "task/regression/head/regression/rmse_hours",
        "min",
    ),
    "regression_r2": (
        "task/regression/head/regression/r2",
        "max",
    ),
}
CLASSIFICATION_METRICS = tuple(
    name for name in METRICS if name.startswith("classification_")
)
REGRESSION_METRICS = tuple(
    name for name in METRICS if name.startswith("regression_")
)
CLASSIFICATION_PROFILE_METRIC = {
    "accuracy": "classification_accuracy",
    "balanced_accuracy": "classification_balanced_accuracy",
    "macro_f1": "classification_macro_f1",
    "nll": "classification_nll",
    "brier": "classification_brier",
}
REGRESSION_PROFILE_METRIC = {
    "mae": "regression_mae_hours",
    "rmse": "regression_rmse_hours",
    "r2": "regression_r2",
}


def _mean(record, key):
    value = record.get("validation", {}).get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _profile(config, task):
    if task == "classification":
        return str(
            (config.get("classification_objective", {}) or {}).get(
                "profile", "equilibrated"
            )
        ).lower()
    return str(
        (config.get("fmv3_head", {}) or {}).get(
            "regression_objective_profile", "equilibrated"
        )
    ).lower()


def _metric_error(metric, value):
    if value is None:
        return None
    return 1.0 - value if METRICS[metric][1] == "max" else value


def _normalized_error(metric, value, reference):
    error = _metric_error(metric, value)
    denominator = _metric_error(metric, reference.get(metric))
    if error is None or denominator is None:
        return None
    return error / max(abs(denominator), 1e-8)


def _average(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def compare(runs, baseline_label):
    """Compare ``{label: (summary, config)}`` on source validation metrics."""

    if baseline_label not in runs:
        raise ValueError(f"Unknown baseline run {baseline_label!r}")
    rows = []
    profiles = {}
    for label, (summary, config) in runs.items():
        profiles[label] = {
            "classification": _profile(config, "classification"),
            "regression": _profile(config, "regression"),
        }
        for record in summary.get("epochs", []):
            row = {"run": label, "epoch": int(record["epoch"])}
            for metric, (key, _) in METRICS.items():
                value = _mean(record, key)
                # Training uses half Brier solely to keep the surrogate bounded;
                # report the conventional multiclass Brier scale here.
                row[metric] = 2.0 * value if metric == "classification_brier" and value is not None else value
            rows.append(row)

    baseline_rows = [row for row in rows if row["run"] == baseline_label]
    if not baseline_rows:
        raise ValueError("Baseline has no epoch records")
    reference = {metric: baseline_rows[0].get(metric) for metric in METRICS}

    for row in rows:
        class_errors = [
            _normalized_error(metric, row.get(metric), reference)
            for metric in CLASSIFICATION_METRICS
        ]
        regression_errors = [
            _normalized_error(metric, row.get(metric), reference)
            for metric in REGRESSION_METRICS
        ]
        row["classification_equilibrated_score"] = _average(class_errors)
        row["regression_equilibrated_score"] = _average(regression_errors)
        row["multitask_equilibrated_score"] = _average(
            [
                row["classification_equilibrated_score"],
                row["regression_equilibrated_score"],
            ]
        )
        run_profiles = profiles[row["run"]]
        class_metric = CLASSIFICATION_PROFILE_METRIC.get(
            run_profiles["classification"]
        )
        regression_metric = REGRESSION_PROFILE_METRIC.get(
            run_profiles["regression"]
        )
        class_score = (
            _normalized_error(class_metric, row.get(class_metric), reference)
            if class_metric
            else row["classification_equilibrated_score"]
        )
        regression_score = (
            _normalized_error(
                regression_metric, row.get(regression_metric), reference
            )
            if regression_metric
            else row["regression_equilibrated_score"]
        )
        row["profile_aligned_score"] = _average(
            [class_score, regression_score]
        )

    result_runs = {}
    for label in runs:
        run_rows = [row for row in rows if row["run"] == label]
        metric_bests = {}
        for metric, (_, direction) in METRICS.items():
            valid = [row for row in run_rows if row.get(metric) is not None]
            if valid:
                best = (max if direction == "max" else min)(
                    valid, key=lambda item: item[metric]
                )
                metric_bests[metric] = {
                    "epoch": best["epoch"], "value": best[metric]
                }
        selectable = [
            row for row in run_rows if row.get("profile_aligned_score") is not None
        ]
        selected = min(selectable, key=lambda row: row["profile_aligned_score"])
        balanced = min(
            (
                row for row in run_rows
                if row.get("multitask_equilibrated_score") is not None
            ),
            key=lambda row: row["multitask_equilibrated_score"],
        )
        result_runs[label] = {
            "epochs": len(run_rows),
            "profiles": profiles[label],
            "metric_bests": metric_bests,
            "profile_selected_epoch": selected["epoch"],
            "profile_selected_score": selected["profile_aligned_score"],
            "profile_selected_metrics": {
                metric: selected.get(metric) for metric in METRICS
            },
            "equilibrated_selected_epoch": balanced["epoch"],
            "equilibrated_selected_score": balanced[
                "multitask_equilibrated_score"
            ],
        }
    return {
        "schema_version": 1,
        "baseline": baseline_label,
        "reference_epoch": 1,
        "reference": reference,
        "runs": result_runs,
        "rows": rows,
    }


def _markdown(result):
    lines = [
        "# Matched metric-objective comparison",
        "",
        f"Normalization baseline: `{result['baseline']}` epoch 1. Lower profile/equilibrated scores are better.",
        "",
        "| Run | Classification | Regression | Profile epoch | Equilibrated epoch | Accuracy | Balanced accuracy | Macro-F1 | NLL | Brier | MAE h | RMSE h | R2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, run in result["runs"].items():
        values = run["profile_selected_metrics"]
        cell = lambda name: "" if values.get(name) is None else f"{values[name]:.6g}"
        lines.append(
            f"| {label} | {run['profiles']['classification']} | "
            f"{run['profiles']['regression']} | {run['profile_selected_epoch']} | "
            f"{run['equilibrated_selected_epoch']} | "
            f"{cell('classification_accuracy')} | "
            f"{cell('classification_balanced_accuracy')} | "
            f"{cell('classification_macro_f1')} | "
            f"{cell('classification_nll')} | "
            f"{cell('classification_brier')} | "
            f"{cell('regression_mae_hours')} | "
            f"{cell('regression_rmse_hours')} | {cell('regression_r2')} |"
        )
    lines.append("")
    lines.append(
        "Profile epoch selection uses the named metric for an extreme task and "
        "the equilibrated score for the other task; target data is not used."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=CHECKPOINT_DIR"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    runs = {}
    for specification in args.run:
        if "=" not in specification:
            parser.error(f"Invalid --run {specification!r}; expected LABEL=DIR")
        label, raw_directory = specification.split("=", 1)
        directory = Path(raw_directory)
        summary = json.loads(
            (directory / "training_debug_summary.json").read_text(encoding="utf-8")
        )
        config = yaml.safe_load(
            (directory / "training_config.yaml").read_text(encoding="utf-8")
        )
        runs[label] = (summary, config)
    result = compare(runs, args.baseline)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "comparison.md").write_text(_markdown(result), encoding="utf-8")
    if result["rows"]:
        with (output / "comparison_curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(result["rows"][0])
            )
            writer.writeheader()
            writer.writerows(result["rows"])
    print(f"Wrote metric-objective comparison to {output}")


if __name__ == "__main__":
    main()

