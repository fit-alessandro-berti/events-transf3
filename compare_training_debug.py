#!/usr/bin/env python3
"""Compare matched structured-training runs on invariant held-out metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = {
    "classification_train_nll": (
        "train",
        "task/classification/head/classification/nll",
    ),
    "classification_validation_nll": (
        "validation",
        "task/classification/head/classification/nll",
    ),
    "classification_train_accuracy": (
        "train",
        "task/classification/head/classification/accuracy",
    ),
    "classification_validation_accuracy": (
        "validation",
        "task/classification/head/classification/accuracy",
    ),
    "classification_validation_max_probability": (
        "validation",
        "task/classification/head/classification/max_probability_mean",
    ),
    "regression_train_mae_hours": (
        "train",
        "task/regression/head/regression/mae_hours",
    ),
    "regression_validation_mae_hours": (
        "validation",
        "task/regression/head/regression/mae_hours",
    ),
    "regression_validation_rmse_hours": (
        "validation",
        "task/regression/head/regression/rmse_hours",
    ),
    "gradient_clip_fraction": (
        "train",
        "optimization/gradient_clip_fraction",
    ),
    "gradient_total_preclip": (
        "train",
        "optimization/gradient_total_preclip",
    ),
    "amp_overflow_fraction": ("train", "optimization/amp_overflow"),
}


def _mean(record, section, key):
    value = record.get(section, {}).get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rows(label, summary):
    rows = []
    for record in summary.get("epochs", []):
        row = {"run": label, "epoch": int(record["epoch"])}
        for metric, (section, key) in METRICS.items():
            row[metric] = _mean(record, section, key)
        accuracy = row.get("classification_validation_accuracy")
        confidence = row.get("classification_validation_max_probability")
        row["classification_validation_confidence_gap"] = (
            confidence - accuracy
            if confidence is not None and accuracy is not None
            else None
        )
        rows.append(row)
    return rows


def _best(rows, metric, maximize=False):
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        return None
    row = (max if maximize else min)(valid, key=lambda item: item[metric])
    return {"epoch": row["epoch"], "value": row[metric]}


def compare(summaries, baseline_label):
    rows = []
    for label, summary in summaries.items():
        rows.extend(_rows(label, summary))
    baseline_rows = [row for row in rows if row["run"] == baseline_label]
    if not baseline_rows:
        raise ValueError(f"Baseline run {baseline_label!r} has no epoch records")
    reference = {
        "classification_validation_nll": baseline_rows[0][
            "classification_validation_nll"
        ],
        "regression_validation_mae_hours": baseline_rows[0][
            "regression_validation_mae_hours"
        ],
    }
    for row in rows:
        class_nll = row.get("classification_validation_nll")
        regression_mae = row.get("regression_validation_mae_hours")
        if class_nll is not None and regression_mae is not None:
            row["joint_invariant_score"] = 0.5 * (
                class_nll / reference["classification_validation_nll"]
                + regression_mae / reference["regression_validation_mae_hours"]
            )
        else:
            row["joint_invariant_score"] = None

    run_results = {}
    for label in summaries:
        run_rows = [row for row in rows if row["run"] == label]
        run_results[label] = {
            "epochs": len(run_rows),
            "best_classification_nll": _best(
                run_rows, "classification_validation_nll"
            ),
            "best_classification_accuracy": _best(
                run_rows, "classification_validation_accuracy", maximize=True
            ),
            "best_regression_mae_hours": _best(
                run_rows, "regression_validation_mae_hours"
            ),
            "best_regression_rmse_hours": _best(
                run_rows, "regression_validation_rmse_hours"
            ),
            "best_joint_invariant_score": _best(
                run_rows, "joint_invariant_score"
            ),
            "last_confidence_gap": (
                run_rows[-1].get("classification_validation_confidence_gap")
                if run_rows
                else None
            ),
            "mean_clip_fraction": (
                sum(
                    row["gradient_clip_fraction"]
                    for row in run_rows
                    if row.get("gradient_clip_fraction") is not None
                )
                / max(
                    1,
                    sum(
                        row.get("gradient_clip_fraction") is not None
                        for row in run_rows
                    ),
                )
            ),
        }
    return {
        "schema_version": 1,
        "baseline": baseline_label,
        "reference": reference,
        "runs": run_results,
        "rows": rows,
    }


def _markdown(result):
    lines = [
        "# Matched training-debug comparison",
        "",
        f"Baseline: `{result['baseline']}`.",
        "",
        "The joint score equally weights held-out classification NLL and raw-hour "
        "regression MAE after normalization by baseline epoch 1. Lower is better.",
        "",
        "| Run | Epochs | Best class NLL | Best class accuracy | Best regression MAE | Best joint score | Last confidence gap | Mean clip fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, run in result["runs"].items():
        def cell(name):
            item = run.get(name)
            return (
                f"{item['value']:.6g} (e{item['epoch']})" if item else ""
            )

        lines.append(
            f"| {label} | {run['epochs']} | {cell('best_classification_nll')} | "
            f"{cell('best_classification_accuracy')} | "
            f"{cell('best_regression_mae_hours')} | "
            f"{cell('best_joint_invariant_score')} | "
            f"{run['last_confidence_gap']:.6g} | {run['mean_clip_fraction']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=SUMMARY_JSON",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summaries = {}
    for specification in args.run:
        if "=" not in specification:
            parser.error(f"Invalid --run {specification!r}; expected LABEL=PATH")
        label, path = specification.split("=", 1)
        summaries[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    if args.baseline not in summaries:
        parser.error("--baseline must match one of the --run labels")
    result = compare(summaries, args.baseline)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "comparison.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    rows = result["rows"]
    if rows:
        with (output / "comparison_curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote matched comparison to {output}")


if __name__ == "__main__":
    main()
