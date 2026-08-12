"""Metrics for the FM-v3 case-budget protocol."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score


def classification_metrics(
    y_true,
    y_pred,
    probabilities,
    class_universe: Sequence[int],
    confidences,
    case_ids,
    support_counts: Dict[int, int],
    pool_covered,
    retrieval_covered,
):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    pool_covered = np.asarray(pool_covered, dtype=bool)
    retrieval_covered = np.asarray(retrieval_covered, dtype=bool)
    labels = np.asarray(class_universe, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    correct = y_true == y_pred
    class_to_col = {int(label): idx for idx, label in enumerate(labels)}
    true_prob = np.asarray([
        probs[row, class_to_col[int(label)]] if int(label) in class_to_col else 0.0
        for row, label in enumerate(y_true)
    ])
    one_hot = np.zeros_like(probs)
    for row, label in enumerate(y_true):
        if int(label) in class_to_col:
            one_hot[row, class_to_col[int(label)]] = 1.0

    ece = 0.0
    reliability = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if mask.any():
            acc = float(correct[mask].mean())
            conf = float(confidences[mask].mean())
            weight = float(mask.mean())
            ece += weight * abs(acc - conf)
            reliability.append({"lower": float(lower), "upper": float(upper), "n": int(mask.sum()), "accuracy": acc, "confidence": conf})

    order = np.argsort(-confidences)
    cumulative_risk = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    aurc = float(cumulative_risk.mean()) if len(cumulative_risk) else float("nan")
    risk_coverage = []
    for coverage in np.linspace(0.1, 1.0, 10):
        count = max(1, int(math.ceil(len(order) * coverage)))
        risk_coverage.append({"coverage": float(coverage), "risk": float((~correct[order[:count]]).mean())})

    frequency_recall = {}
    bins = {
        "n=0": lambda n: n == 0,
        "n=1": lambda n: n == 1,
        "n=2-5": lambda n: 2 <= n <= 5,
        "n>5": lambda n: n > 5,
    }
    recall_by_label = {int(label): float(value) for label, value in zip(labels, recall)}
    for name, predicate in bins.items():
        members = [recall_by_label[int(label)] for label in labels if predicate(int(support_counts.get(int(label), 0)))]
        frequency_recall[name] = float(np.mean(members)) if members else None

    decomposition = {}
    conditional_pool_recalls, conditional_retrieval_recalls = [], []
    retrieval_given_pool_values = []
    for label in labels:
        class_mask = y_true == label
        class_pool = pool_covered[class_mask]
        class_retrieval = retrieval_covered[class_mask]
        availability = float(class_pool.mean()) if class_mask.any() else 0.0
        retrieval_given_pool = (
            float(class_retrieval[class_pool].mean()) if class_pool.any() else 0.0
        )
        decision_given_retrieval = (
            float(correct[class_mask][class_retrieval].mean()) if class_retrieval.any() else 0.0
        )
        decomposition[str(int(label))] = {
            "pool_availability_A": availability,
            "retrieval_given_pool_R": retrieval_given_pool,
            "decision_given_retrieval_D": decision_given_retrieval,
        }
        if class_pool.any():
            retrieval_given_pool_values.append(retrieval_given_pool)
            conditional_pool_recalls.append(float(correct[class_mask][class_pool].mean()))
        if class_retrieval.any():
            conditional_retrieval_recalls.append(decision_given_retrieval)

    balanced = float(recall.mean())
    chance = 1.0 / max(len(labels), 1)
    return {
        "n_queries": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
        "adjusted_balanced_accuracy": (balanced - chance) / max(1.0 - chance, 1e-12),
        "macro_precision": float(precision.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_recall": {str(int(label)): float(value) for label, value in zip(labels, recall)},
        "per_class_support": {str(int(label)): int(value) for label, value in zip(labels, support)},
        "zero_recall_classes": int((recall == 0).sum()),
        "zero_recall_fraction": float((recall == 0).mean()),
        "recall_p10": float(np.percentile(recall, 10)),
        "support_pool_availability": float(pool_covered.mean()),
        "macro_label_recall_at_k": float(np.mean([
            retrieval_covered[y_true == label].mean() if np.any(y_true == label) else 0.0 for label in labels
        ])),
        "macro_retrieval_given_pool": float(np.mean(retrieval_given_pool_values)) if retrieval_given_pool_values else None,
        "macro_decision_given_retrieval": float(np.mean(conditional_retrieval_recalls)) if conditional_retrieval_recalls else None,
        "conditional_accuracy_pool_covered": float(correct[pool_covered].mean()) if pool_covered.any() else None,
        "conditional_accuracy_retrieval_covered": float(correct[retrieval_covered].mean()) if retrieval_covered.any() else None,
        "conditional_balanced_accuracy_pool_covered": float(np.mean(conditional_pool_recalls)) if conditional_pool_recalls else None,
        "conditional_balanced_accuracy_retrieval_covered": float(np.mean(conditional_retrieval_recalls)) if conditional_retrieval_recalls else None,
        "error_decomposition_per_class": decomposition,
        "frequency_bin_recall": frequency_recall,
        "nll": float(-np.log(np.clip(true_prob, 1e-12, 1.0)).mean()),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_10": float(ece),
        "reliability": reliability,
        "aurc": aurc,
        "risk_coverage": risk_coverage,
    }


def regression_metrics(y_true, y_pred, lower=None, upper=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    median_baseline = np.full_like(y_true, np.median(y_true))
    baseline_mae = float(np.abs(y_true - median_baseline).mean())
    mae = float(errors.mean())
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((y_true - median_baseline) ** 2)))
    result = {
        "n_queries": int(len(y_true)),
        "mae_hours": mae,
        "rmse_hours": rmse,
        "median_absolute_error_hours": float(np.median(errors)),
        "normalized_mae": mae / max(float(np.mean(np.abs(y_true))), 1e-12),
        "mae_skill_vs_median": 1.0 - mae / max(baseline_mae, 1e-12),
        "rmse_skill_vs_median": 1.0 - rmse / max(baseline_rmse, 1e-12),
        "d2_absolute_error": 1.0 - float(errors.sum()) / max(float(np.abs(y_true - np.median(y_true)).sum()), 1e-12),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }
    if lower is not None and upper is not None:
        lower, upper = np.asarray(lower), np.asarray(upper)
        result["interval_coverage"] = float(((y_true >= lower) & (y_true <= upper)).mean())
        result["mean_interval_width_hours"] = float(np.mean(upper - lower))
    return result


def case_bootstrap_interval(values_by_case: Dict[str, tuple], metric_fn, repetitions: int, seed: int):
    """Resample cases (never dependent prefixes) and return a percentile interval."""
    rng = np.random.default_rng(seed)
    cases = list(values_by_case)
    if not cases or repetitions <= 0:
        return None
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        chunks = [values_by_case[str(case)] for case in sampled]
        estimates.append(float(metric_fn(chunks)))
    return {"lower": float(np.percentile(estimates, 2.5)), "upper": float(np.percentile(estimates, 97.5))}
