#!/usr/bin/env python3
"""Turn verbose training telemetry into compact curves and bottleneck findings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TASKS = ("classification", "regression")
REGRESSION_COMPONENTS = (
    "mae",
    "rmse",
    "huber",
    "log_rmse",
    "relative_mae",
    "bias",
    "median_ae",
    "quantile",
)


def _mean(section, key):
    value = (section or {}).get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _number(section, key):
    try:
        value = float((section or {}).get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _series(records, section, key, aggregate=True):
    result = []
    for record in records:
        block = record.get(section, {})
        value = _mean(block, key) if aggregate else _number(block, key)
        if value is not None:
            result.append((int(record["epoch"]), value))
    return result


def _learning_phase_summary(series):
    if not series:
        return {}
    values = [value for _, value in series]
    cutoff_index = min(len(values) - 1, max(0, math.ceil(2 * len(values) / 3) - 1))
    best_index = min(range(len(values)), key=values.__getitem__)
    first, cutoff, last, best = (
        values[0],
        values[cutoff_index],
        values[-1],
        values[best_index],
    )
    total_best_improvement = first - best
    early_improvement = first - cutoff
    late_change = cutoff - last
    window = values[-min(3, len(values)) :]
    recent_relative_range = (max(window) - min(window)) / max(abs(min(window)), 1e-12)
    return {
        "first_epoch": series[0][0],
        "first_value": first,
        "two_thirds_epoch": series[cutoff_index][0],
        "two_thirds_value": cutoff,
        "last_epoch": series[-1][0],
        "last_value": last,
        "best_epoch": series[best_index][0],
        "best_value": best,
        "early_improvement": early_improvement,
        "late_change": late_change,
        "fraction_of_best_improvement_reached_by_two_thirds": (
            early_improvement / total_best_improvement
            if total_best_improvement > 1e-12
            else None
        ),
        "recent_three_epoch_relative_range": recent_relative_range,
        "recent_plateau_signal": recent_relative_range < 0.01,
    }


def _optimization_summary(records):
    result = {}
    for name, section, key, aggregate in (
        (
            "gradient_clip_fraction",
            "train",
            "optimization/gradient_clip_fraction",
            True,
        ),
        (
            "gradient_total_preclip",
            "train",
            "optimization/gradient_total_preclip",
            True,
        ),
        ("amp_overflow_fraction", "train", "optimization/amp_overflow", True),
        (
            "step_success_fraction",
            "epoch_metrics",
            "step_success_fraction",
            False,
        ),
    ):
        values = _series(records, section, key, aggregate=aggregate)
        if values:
            result[name] = {
                "first": values[0][1],
                "last": values[-1][1],
                "mean": sum(value for _, value in values) / len(values),
                "max": max(value for _, value in values),
            }
    updates = {}
    for record in records:
        for key, value in record.get("updates", {}).items():
            if key.startswith("optimization/update/") and key.endswith("/relative_l2"):
                updates.setdefault(key, []).append((int(record["epoch"]), float(value)))
    result["relative_parameter_updates"] = {
        key.removeprefix("optimization/update/").removesuffix("/relative_l2"): {
            "first": values[0][1],
            "last": values[-1][1],
            "max": max(value for _, value in values),
        }
        for key, values in sorted(updates.items())
    }
    return result


def _head_summary(records):
    result = {}
    metric_map = {
        "classification": (
            "head/classification/accuracy",
            "head/classification/nll",
            "head/classification/entropy_mean",
            "head/classification/probability_margin_mean",
            "head/classification/selector/log_weight/abs_mean",
            "head/classification/selector/effective_support/mean",
            "head/classification/selector/attention_entropy_mean",
        ),
        "regression": (
            "head/regression/mae_hours",
            "head/regression/rmse_hours",
            "head/regression/median_ae_hours",
            "head/regression/bias_hours",
            "head/regression/branch_weight_entropy_mean",
            "head/regression/selector/log_weight/abs_mean",
            "head/regression/selector/effective_support/mean",
            "head/regression/selector/attention_entropy_mean",
        ),
    }
    for task, metrics in metric_map.items():
        result[task] = {}
        for metric in metrics:
            key = f"task/{task}/{metric}"
            train = _series(records, "train", key)
            validation = _series(records, "validation", key)
            if train or validation:
                result[task][metric] = {
                    "train_first": train[0][1] if train else None,
                    "train_last": train[-1][1] if train else None,
                    "validation_first": validation[0][1] if validation else None,
                    "validation_last": validation[-1][1] if validation else None,
                }
    return result


def _loss_contributions(records):
    if not records:
        return {}
    result = {}
    for task in TASKS:
        task_result = {}
        last_train = records[-1].get("train", {})
        for metric in (
            "loss/primary",
            "loss/confidence_weighted",
            "loss/routing_weighted",
            "loss/regression_gate_aux_weighted",
            "loss/classification_separation_weighted",
            "loss/contrastive_weighted",
            "loss/nca_weighted",
            "loss/variance_weighted",
            "loss/covariance_weighted",
        ):
            value = _mean(last_train, f"task/{task}/{metric}")
            if value is not None:
                task_result[metric] = value
        if task == "regression":
            for component in REGRESSION_COMPONENTS:
                value = _mean(
                    last_train,
                    f"task/regression/loss/regression/{component}_weighted",
                )
                if value is not None:
                    task_result[f"loss/regression/{component}_weighted"] = value
        result[task] = task_result
    return result


def analyze(summary):
    records = summary.get("epochs", [])
    task_summary = {}
    for task in TASKS:
        task_summary[task] = {
            "train_loss": _learning_phase_summary(
                _series(records, "train", f"task/{task}/loss/total")
            ),
            "validation_loss": _learning_phase_summary(
                _series(records, "validation", f"task/{task}/loss/total")
            ),
            "automatic_overfitting": summary.get("generalization", {}).get(
                task, {}
            ),
        }
    optimization = _optimization_summary(records)
    findings = []
    for task, task_result in task_summary.items():
        overfit = task_result["automatic_overfitting"].get("overfitting_signal")
        if overfit:
            findings.append(
                {
                    "severity": "high",
                    "kind": "overfitting",
                    "task": task,
                    "evidence": task_result["automatic_overfitting"],
                }
            )
        validation = task_result["validation_loss"]
        fraction = validation.get(
            "fraction_of_best_improvement_reached_by_two_thirds"
        )
        if fraction is not None and fraction >= 0.9:
            findings.append(
                {
                    "severity": "medium",
                    "kind": "front_loaded_learning",
                    "task": task,
                    "evidence": {
                        "fraction_of_best_improvement_by_two_thirds": fraction,
                        "best_epoch": validation.get("best_epoch"),
                    },
                }
            )
    clipping = optimization.get("gradient_clip_fraction", {})
    if clipping.get("mean", 0.0) > 0.25:
        findings.append(
            {
                "severity": "high",
                "kind": "frequent_gradient_clipping",
                "evidence": clipping,
            }
        )
    if optimization.get("amp_overflow_fraction", {}).get("max", 0.0) > 0.0:
        findings.append(
            {
                "severity": "high",
                "kind": "amp_overflow",
                "evidence": optimization["amp_overflow_fraction"],
            }
        )
    return {
        "schema_version": 1,
        "epochs_analyzed": len(records),
        "tasks": task_summary,
        "optimization": optimization,
        "heads": _head_summary(records),
        "last_epoch_loss_contributions": _loss_contributions(records),
        "findings": findings,
    }


def _curve_rows(summary):
    records = summary.get("epochs", [])
    columns = {
        "classification_train_loss": ("train", "task/classification/loss/total", True),
        "classification_validation_loss": ("validation", "task/classification/loss/total", True),
        "classification_train_accuracy": ("train", "task/classification/head/classification/accuracy", True),
        "classification_validation_accuracy": ("validation", "task/classification/head/classification/accuracy", True),
        "regression_train_loss": ("train", "task/regression/loss/total", True),
        "regression_validation_loss": ("validation", "task/regression/loss/total", True),
        "regression_train_mae_hours": ("train", "task/regression/head/regression/mae_hours", True),
        "regression_validation_mae_hours": ("validation", "task/regression/head/regression/mae_hours", True),
        "gradient_clip_fraction": ("train", "optimization/gradient_clip_fraction", True),
        "gradient_preclip_norm": ("train", "optimization/gradient_total_preclip", True),
        "step_success_fraction": ("epoch_metrics", "step_success_fraction", False),
        "base_lr": ("schedule", "base_lr", False),
    }
    rows = []
    for record in records:
        row = {"epoch": int(record["epoch"])}
        for column, (section, key, aggregate) in columns.items():
            block = record.get(section, {})
            row[column] = _mean(block, key) if aggregate else _number(block, key)
        rows.append(row)
    return rows


def _markdown(analysis):
    lines = [
        "# Training debug analysis",
        "",
        f"Epochs analyzed: {analysis['epochs_analyzed']}.",
        "",
        "## Learning phases",
        "",
    ]
    for task in TASKS:
        result = analysis["tasks"][task]
        validation = result["validation_loss"]
        lines.extend(
            [
                f"### {task.title()}",
                "",
                f"- Best validation epoch: {validation.get('best_epoch')}",
                f"- Best validation loss: {validation.get('best_value')}",
                "- Fraction of best improvement reached by two-thirds: "
                f"{validation.get('fraction_of_best_improvement_reached_by_two_thirds')}",
                "- Automatic overfitting signal: "
                f"{result['automatic_overfitting'].get('overfitting_signal')}",
                "",
            ]
        )
    lines.extend(["## Detected bottlenecks", ""])
    if not analysis["findings"]:
        lines.append("No automatic bottleneck threshold fired.")
    for finding in analysis["findings"]:
        lines.append(
            f"- `{finding['severity']}` `{finding['kind']}`"
            + (f" ({finding['task']})" if finding.get("task") else "")
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    analysis = analyze(summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = _curve_rows(summary)
    if rows:
        with (output / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "analysis.md").write_text(_markdown(analysis), encoding="utf-8")
    print(f"Wrote training analysis to {output}")


if __name__ == "__main__":
    main()
