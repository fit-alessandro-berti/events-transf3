#!/usr/bin/env python3
"""Turn verbose training telemetry into compact curves and bottleneck findings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
LOSS_METRICS = (
    "loss/total",
    "loss/primary",
    "loss/confidence_weighted",
    "loss/routing_weighted",
    "loss/regression_gate_aux_weighted",
    "loss/classification_separation_weighted",
    "loss/contrastive_weighted",
    "loss/nca_weighted",
    "loss/variance_weighted",
    "loss/covariance_weighted",
)
HEAD_METRICS = {
    "classification": (
        "head/classification/accuracy",
        "head/classification/nll",
        "head/classification/true_probability_mean",
        "head/classification/max_probability_mean",
        "head/classification/entropy_mean",
        "head/classification/probability_margin_mean",
        "head/classification/local_class_coverage",
        "head/classification/gate/mean",
        "head/classification/prototype_variances/mean",
        "head/classification/selector/log_weight/abs_mean",
        "head/classification/selector/effective_support/mean",
        "head/classification/selector/attention_entropy_mean",
    ),
    "regression": (
        "head/regression/mae_hours",
        "head/regression/rmse_hours",
        "head/regression/median_ae_hours",
        "head/regression/bias_hours",
        "head/regression/relative_mae",
        "head/regression/branch_weight_entropy_mean",
        "head/regression/selector/log_weight/abs_mean",
        "head/regression/selector/effective_support/mean",
        "head/regression/selector/attention_entropy_mean",
    ),
}


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


def _invariant_overfitting_summary(records, task, configuration):
    metric = (
        "head/classification/nll"
        if task == "classification"
        else "head/regression/mae_hours"
    )
    key = f"task/{task}/{metric}"
    train = dict(_series(records, "train", key))
    validation = _series(records, "validation", key)
    curve = [
        (epoch, train[epoch], value)
        for epoch, value in validation
        if epoch in train
    ]
    if not curve:
        return {}
    patience = max(1, int((configuration or {}).get("overfitting_patience", 3)))
    tolerance = max(
        0.0,
        float(
            (configuration or {}).get("overfitting_relative_tolerance", 0.02)
        ),
    )
    best_index = min(range(len(curve)), key=lambda index: curve[index][2])
    best_epoch, best_train, best_validation = curve[best_index]
    last_epoch, last_train, last_validation = curve[-1]
    degradation = (last_validation - best_validation) / max(
        abs(best_validation), 1e-12
    )
    train_improvement = (best_train - last_train) / max(abs(best_train), 1e-12)
    enough_epochs = len(curve) - 1 - best_index >= patience
    return {
        "metric": metric,
        "best_validation_epoch": best_epoch,
        "best_validation_value": best_validation,
        "last_epoch": last_epoch,
        "last_validation_value": last_validation,
        "relative_validation_degradation": degradation,
        "relative_train_improvement_since_best": train_improvement,
        "overfitting_signal": bool(
            enough_epochs and degradation > tolerance and train_improvement > 0.0
        ),
        "patience_epochs": patience,
        "relative_tolerance": tolerance,
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
    for task, metrics in HEAD_METRICS.items():
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


def _gradient_summary(records):
    result = {}
    pattern = re.compile(
        r"^task/(classification|regression)/optimization/gradient/(.+)/l2_norm$"
    )
    keys = set()
    for record in records:
        for key in record.get("train", {}):
            if pattern.match(key):
                keys.add(key)
    for key in sorted(keys):
        match = pattern.match(key)
        task, group = match.groups()
        values = _series(records, "train", key)
        if not values:
            continue
        result.setdefault(task, {})[group] = {
            "first": values[0][1],
            "last": values[-1][1],
            "mean": sum(value for _, value in values) / len(values),
            "max": max(value for _, value in values),
        }
    return result


def _loss_gradient_summary(records):
    result = {}
    pattern = re.compile(
        r"^task/(classification|regression)/optimization/"
        r"loss_gradient/(.+)/all/l2_norm$"
    )
    keys = set()
    for record in records:
        keys.update(
            key for key in record.get("train", {}) if pattern.match(key)
        )
    for key in sorted(keys):
        task, component = pattern.match(key).groups()
        values = _series(records, "train", key)
        if values:
            result.setdefault(task, {})[component] = {
                "first": values[0][1],
                "last": values[-1][1],
                "mean": sum(value for _, value in values) / len(values),
                "max": max(value for _, value in values),
            }
    return result


def _pool_summary(records, pool_names=None):
    """Report latest held-out behavior per source pool, when instrumented."""
    if not records:
        return {}
    latest = records[-1].get("validation", {})
    names = pool_names or {}
    task_metrics = {
        "classification": (
            "loss/total",
            "head/classification/accuracy",
            "head/classification/nll",
        ),
        "regression": (
            "loss/total",
            "head/regression/mae_hours",
            "head/regression/relative_mae",
            "head/regression/error_p90_hours",
        ),
    }
    result = {}
    for task, metrics in task_metrics.items():
        rows = []
        for key in latest:
            match = re.match(rf"^task/{task}/pool/(\d+)/loss/total$", key)
            if not match:
                continue
            pool = int(match.group(1))
            row = {"pool": pool, "log": names.get(pool, str(pool))}
            for metric in metrics:
                value = _mean(latest, f"task/{task}/pool/{pool}/{metric}")
                if value is not None:
                    row[metric] = value
            rows.append(row)
        if rows:
            result[task] = sorted(rows, key=lambda row: row["pool"])
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


def analyze(summary, pool_names=None):
    records = summary.get("epochs", [])
    diagnostics_configuration = summary.get("configuration", {})
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
            "invariant_overfitting": _invariant_overfitting_summary(
                records, task, diagnostics_configuration
            ),
        }
    optimization = _optimization_summary(records)
    heads = _head_summary(records)
    loss_contributions = _loss_contributions(records)
    findings = []
    for task, task_result in task_summary.items():
        invariant_overfit = task_result["invariant_overfitting"].get(
            "overfitting_signal"
        )
        objective_overfit = task_result["automatic_overfitting"].get(
            "overfitting_signal"
        )
        overfit = invariant_overfit or objective_overfit
        if overfit:
            findings.append(
                {
                    "severity": "high",
                    "kind": "overfitting",
                    "task": task,
                    "evidence": {
                        "invariant": task_result["invariant_overfitting"],
                        "objective": task_result["automatic_overfitting"],
                    },
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
    if len(records) >= 3:
        for task in TASKS:
            total = _series(records, "train", f"task/{task}/loss/total")
            if not total or abs(total[-1][1]) <= 1e-12:
                continue
            for metric, last_value in loss_contributions.get(task, {}).items():
                if metric in {"loss/primary"} or last_value <= 0.0:
                    continue
                values = _series(records, "train", f"task/{task}/{metric}")
                if len(values) < 3:
                    continue
                relative_change = abs(values[-1][1] - values[0][1]) / max(
                    abs(values[0][1]), 1e-12
                )
                contribution = last_value / abs(total[-1][1])
                if contribution >= 0.10 and relative_change < 0.05:
                    findings.append(
                        {
                            "severity": "medium",
                            "kind": "large_stagnant_auxiliary_loss",
                            "task": task,
                            "metric": metric,
                            "evidence": {
                                "last_contribution_fraction": contribution,
                                "relative_change_since_first_epoch": relative_change,
                                "first": values[0][1],
                                "last": values[-1][1],
                            },
                        }
                    )
        last_k = _number(records[-1].get("schedule", {}), "retrieval_k")
        if last_k and last_k > 0:
            for task in TASKS:
                metrics = heads.get(task, {})
                selector = metrics.get(
                    f"head/{task}/selector/log_weight/abs_mean", {}
                )
                support = metrics.get(
                    f"head/{task}/selector/effective_support/mean", {}
                )
                log_weight = selector.get("validation_last")
                effective_support = support.get("validation_last")
                if (
                    log_weight is not None
                    and effective_support is not None
                    and log_weight < 0.05
                    and effective_support / last_k > 0.98
                ):
                    findings.append(
                        {
                            "severity": "medium",
                            "kind": "near_uniform_example_selector",
                            "task": task,
                            "evidence": {
                                "absolute_log_weight": log_weight,
                                "effective_support": effective_support,
                                "retrieval_k": last_k,
                            },
                        }
                    )
        class_metrics = heads.get("classification", {})
        accuracy = class_metrics.get("head/classification/accuracy", {}).get(
            "validation_last"
        )
        max_probability = class_metrics.get(
            "head/classification/max_probability_mean", {}
        ).get("validation_last")
        if (
            accuracy is not None
            and max_probability is not None
            and max_probability - accuracy > 0.10
        ):
            findings.append(
                {
                    "severity": "medium",
                    "kind": "classification_overconfidence",
                    "task": "classification",
                    "evidence": {
                        "validation_accuracy": accuracy,
                        "validation_max_probability": max_probability,
                        "confidence_gap": max_probability - accuracy,
                    },
                }
            )
    return {
        "schema_version": 1,
        "epochs_analyzed": len(records),
        "tasks": task_summary,
        "optimization": optimization,
        "heads": heads,
        "gradients": _gradient_summary(records),
        "loss_gradients": _loss_gradient_summary(records),
        "pools": _pool_summary(records, pool_names),
        "last_epoch_loss_contributions": loss_contributions,
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


def _long_metric_rows(summary, metric_kind):
    records = summary.get("epochs", [])
    rows = []
    for record in records:
        epoch = int(record["epoch"])
        for phase in ("train", "validation"):
            section = record.get(phase, {})
            for task in TASKS:
                if metric_kind == "loss":
                    metrics = list(LOSS_METRICS)
                    if task == "regression":
                        metrics.extend(
                            f"loss/regression/{component}_weighted"
                            for component in REGRESSION_COMPONENTS
                        )
                else:
                    metrics = HEAD_METRICS[task]
                for metric in metrics:
                    value = _mean(section, f"task/{task}/{metric}")
                    if value is not None:
                        rows.append(
                            {
                                "epoch": epoch,
                                "phase": phase,
                                "task": task,
                                "metric": metric,
                                "mean": value,
                            }
                        )
    return rows


def _pool_curve_rows(summary, pool_names=None):
    names = pool_names or {}
    rows = []
    pattern = re.compile(
        r"^task/(classification|regression)/pool/(\d+)/(loss/total|head/.+)$"
    )
    wanted = {
        "loss/total",
        "head/classification/accuracy",
        "head/classification/nll",
        "head/regression/mae_hours",
        "head/regression/relative_mae",
        "head/regression/error_p90_hours",
    }
    for record in summary.get("epochs", []):
        epoch = int(record["epoch"])
        for key in record.get("validation", {}):
            match = pattern.match(key)
            if not match or match.group(3) not in wanted:
                continue
            task, pool_text, metric = match.groups()
            pool = int(pool_text)
            value = _mean(record["validation"], key)
            if value is not None:
                rows.append(
                    {
                        "epoch": epoch,
                        "task": task,
                        "pool": pool,
                        "log": names.get(pool, str(pool)),
                        "metric": metric,
                        "mean": value,
                    }
                )
    return rows


def _loss_gradient_curve_rows(summary):
    rows = []
    pattern = re.compile(
        r"^task/(classification|regression)/optimization/"
        r"loss_gradient/(.+)/all/l2_norm$"
    )
    for record in summary.get("epochs", []):
        for key in record.get("train", {}):
            match = pattern.match(key)
            if not match:
                continue
            value = _mean(record["train"], key)
            if value is not None:
                task, component = match.groups()
                rows.append(
                    {
                        "epoch": int(record["epoch"]),
                        "task": task,
                        "component": component,
                        "gradient_l2": value,
                    }
                )
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
                "- Invariant-metric overfitting signal: "
                f"{result['invariant_overfitting'].get('overfitting_signal')}",
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
    parser.add_argument(
        "--validation_manifest",
        help=(
            "Optional training_validation_split.json. If omitted, the analyzer "
            "uses the file next to --summary when present."
        ),
    )
    args = parser.parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = (
        Path(args.validation_manifest)
        if args.validation_manifest
        else summary_path.with_name("training_validation_split.json")
    )
    pool_names = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pool_names = {
            int(row["pool_index"]): str(row["log"])
            for row in manifest.get("logs", [])
        }
    analysis = analyze(summary, pool_names)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "curves.csv", _curve_rows(summary))
    _write_csv(output / "loss_curves.csv", _long_metric_rows(summary, "loss"))
    _write_csv(output / "head_curves.csv", _long_metric_rows(summary, "head"))
    _write_csv(output / "pool_curves.csv", _pool_curve_rows(summary, pool_names))
    _write_csv(
        output / "loss_gradient_curves.csv", _loss_gradient_curve_rows(summary)
    )
    (output / "analysis.md").write_text(_markdown(analysis), encoding="utf-8")
    print(f"Wrote training analysis to {output}")


if __name__ == "__main__":
    main()
