"""Differentiable, configurable surrogates for deployment metrics.

Evaluation metrics such as accuracy, balanced accuracy, macro-F1, and R2 are
not directly differentiable.  This module keeps their training surrogates and
profile resolution in one place so episodic and retrieval training optimize
the same objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "nll",
    "brier",
)
REGRESSION_METRICS = (
    "mae",
    "rmse",
    "huber",
    "log_rmse",
    "relative_mae",
    "bias",
    "median_ae",
    "quantile",
    "r2",
)

# Equilibrated weights are calibrated against initial per-component gradient
# norms on the fixed source-case smoke protocol. This avoids "equal scalar
# weight" silently meaning 6x NLL pressure or 24x median-AE pressure. Named
# single-metric profiles still reduce exactly to their selected surrogate
# because every blend is divided by its active weight sum.
CLASSIFICATION_EQUILIBRATED_WEIGHTS = {
    "accuracy": 1.0,
    "balanced_accuracy": 1.0,
    "macro_f1": 1.0,
    "nll": 0.18,
    "brier": 1.0,
}
REGRESSION_EQUILIBRATED_WEIGHTS = {
    "mae": 1.0,
    "rmse": 1.0,
    "huber": 1.3,
    "log_rmse": 0.8,
    "relative_mae": 0.15,
    "bias": 0.75,
    "median_ae": 0.08,
    "quantile": 2.0,
    "r2": 1.4,
}

CLASSIFICATION_PROFILES = {
    "equilibrated": CLASSIFICATION_EQUILIBRATED_WEIGHTS,
    "accuracy": {"accuracy": 1.0},
    "balanced_accuracy": {"balanced_accuracy": 1.0},
    "macro_f1": {"macro_f1": 1.0},
    "nll": {"nll": 1.0},
    "brier": {"brier": 1.0},
}
REGRESSION_PROFILES = {
    "equilibrated": REGRESSION_EQUILIBRATED_WEIGHTS,
    "legacy": {"huber": 1.0},
    "mae": {"mae": 1.0},
    "rmse": {"rmse": 1.0},
    "r2": {"r2": 1.0},
}


@dataclass(frozen=True)
class ClassificationObjective:
    loss: torch.Tensor
    components: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    weights: Mapping[str, float]
    profile: str


def _validated_weights(
    weights: Mapping[str, object], valid_names: Sequence[str], context: str
) -> dict[str, float]:
    unknown = sorted(set(weights) - set(valid_names))
    if unknown:
        raise ValueError(
            f"Unknown {context} metric weight(s): {', '.join(unknown)}; "
            f"expected one of {', '.join(valid_names)}"
        )
    parsed = {}
    for name in valid_names:
        value = float(weights.get(name, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{context} metric weight '{name}' must be finite and non-negative"
            )
        parsed[name] = value
    if sum(parsed.values()) <= 0:
        raise ValueError(f"At least one {context} metric weight must be positive")
    return parsed


def resolve_classification_objective(config: Mapping[str, object]):
    """Resolve a named classification profile plus optional weight overrides.

    ``legacy`` exactly preserves the historical label-smoothed cross entropy
    scale. Other profiles blend bounded/normalized metric surrogates.
    ``custom`` starts from zero and therefore requires a positive override.
    """

    objective = config.get("classification_objective", {}) or {}
    if not isinstance(objective, Mapping):
        raise ValueError("classification_objective must be a mapping")
    profile = str(objective.get("profile", "equilibrated")).strip().lower()
    overrides = objective.get("weights", {}) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("classification_objective.weights must be a mapping")
    if profile == "legacy":
        if overrides:
            raise ValueError("legacy classification objective does not accept weights")
        return profile, {}
    if profile == "custom":
        base = {}
    elif profile in CLASSIFICATION_PROFILES:
        base = dict(CLASSIFICATION_PROFILES[profile])
    else:
        raise ValueError(
            "Unknown classification objective profile "
            f"'{profile}'; expected legacy, custom, or one of "
            f"{', '.join(CLASSIFICATION_PROFILES)}"
        )
    base.update(overrides)
    return profile, _validated_weights(
        base, CLASSIFICATION_METRICS, "classification"
    )


def resolve_regression_metric_weights(config: Mapping[str, object]):
    """Resolve regression metric weights while retaining flat-key compatibility."""

    explicit_profile = "regression_objective_profile" in config
    flat_keys = {f"regression_{name}_weight" for name in REGRESSION_METRICS}
    # Programmatic callers that pass historical flat weights without a profile
    # retain their custom objective. YAML roots declare their profile explicitly.
    inferred_custom = not explicit_profile and any(key in config for key in flat_keys)
    profile = str(
        config.get(
            "regression_objective_profile",
            "custom" if inferred_custom else "equilibrated",
        )
    ).strip().lower()
    overrides = config.get("regression_metric_weights", {}) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("regression_metric_weights must be a mapping")
    if profile == "custom":
        base = {
            name: float(
                config.get(
                    f"regression_{name}_weight",
                    0.0 if name == "r2" else {
                        "mae": 0.5,
                        "rmse": 0.5,
                        "huber": 0.15,
                        "log_rmse": 0.15,
                        "relative_mae": 0.05,
                        "bias": 0.05,
                        "median_ae": 0.0,
                        "quantile": 0.0,
                    }[name],
                )
            )
            for name in REGRESSION_METRICS
        }
    elif profile in REGRESSION_PROFILES:
        base = dict(REGRESSION_PROFILES[profile])
    else:
        raise ValueError(
            "Unknown regression objective profile "
            f"'{profile}'; expected custom or one of "
            f"{', '.join(REGRESSION_PROFILES)}"
        )
    base.update(overrides)
    return profile, _validated_weights(base, REGRESSION_METRICS, "regression")


def _row_class_ids(logits: torch.Tensor, class_ids):
    if class_ids is None:
        return torch.arange(logits.numel(), device=logits.device, dtype=torch.long)
    values = torch.as_tensor(class_ids, device=logits.device, dtype=torch.long).reshape(-1)
    if values.numel() != logits.numel():
        raise ValueError("Each class-id row must match its logit row")
    return values


def classification_metric_objective(
    logits_rows: Sequence[torch.Tensor] | torch.Tensor,
    target_indices: Sequence[torch.Tensor | int] | torch.Tensor,
    config: Mapping[str, object],
    *,
    class_id_rows: Sequence[torch.Tensor] | None = None,
    label_smoothing: float = 0.0,
) -> ClassificationObjective:
    """Return a profile-weighted classification loss over a complete episode.

    Accuracy and balanced accuracy use soft correctness/recall. Macro-F1 uses
    soft TP/FP/FN counts in the original class-id space. NLL is divided by the
    uniform-predictor entropy and Brier by its theoretical maximum so equal
    profile weights have comparable scale. Hard metrics are diagnostic only.
    """

    profile, weights = resolve_classification_objective(config)
    if isinstance(logits_rows, torch.Tensor):
        if logits_rows.ndim != 2:
            raise ValueError("classification logits must have shape [query, class]")
        rows = list(logits_rows.unbind(dim=0))
    else:
        rows = [row.reshape(-1) for row in logits_rows]
    if isinstance(target_indices, torch.Tensor):
        targets = list(target_indices.reshape(-1).unbind(dim=0))
    else:
        targets = list(target_indices)
    if len(rows) != len(targets):
        raise ValueError("Classification logits and targets must have equal length")
    if class_id_rows is not None and len(class_id_rows) != len(rows):
        raise ValueError("class_id_rows must have one entry per query")

    valid_rows = []
    for index, (row, target) in enumerate(zip(rows, targets)):
        target_index = int(target.item()) if isinstance(target, torch.Tensor) else int(target)
        if target_index == -100:
            continue
        if target_index < 0 or target_index >= row.numel():
            raise ValueError("Classification target index is outside its logit row")
        ids = _row_class_ids(
            row,
            None if class_id_rows is None else class_id_rows[index],
        )
        valid_rows.append((row.float(), target_index, ids))
    if not valid_rows:
        raise ValueError("Classification objective received no valid targets")

    nll_rows = []
    normalized_nll_rows = []
    probabilities_rows = []
    for logits, target_index, ids in valid_rows:
        target = torch.tensor([target_index], device=logits.device, dtype=torch.long)
        nll = F.cross_entropy(
            logits.unsqueeze(0),
            target,
            label_smoothing=label_smoothing,
        )
        probabilities = F.softmax(logits, dim=0)
        nll_rows.append(nll)
        normalized_nll_rows.append(nll / math.log(max(logits.numel(), 2)))
        probabilities_rows.append(probabilities)

    # Align query-specific class spaces into one differentiable probability
    # matrix. One torch.unique and one scatter per query avoid thousands of
    # scalar GPU synchronizations in a 128-query retrieval episode.
    class_universe = torch.unique(
        torch.cat([ids for _, _, ids in valid_rows]), sorted=True
    )
    aligned_probabilities = []
    target_class_tensors = []
    for probabilities, (_, target_index, ids) in zip(
        probabilities_rows, valid_rows
    ):
        columns = torch.searchsorted(class_universe, ids)
        aligned_probabilities.append(
            probabilities.new_zeros(class_universe.numel()).scatter(
                0, columns, probabilities
            )
        )
        target_class_tensors.append(ids[target_index])
    probability_matrix = torch.stack(aligned_probabilities)
    target_classes = torch.stack(target_class_tensors)
    truth = target_classes[:, None].eq(class_universe[None, :]).float()
    support = truth.sum(dim=0)
    supported = support > 0
    true_positive = (truth * probability_matrix).sum(dim=0)
    false_positive = ((1.0 - truth) * probability_matrix).sum(dim=0)
    false_negative = (truth * (1.0 - probability_matrix)).sum(dim=0)
    true_probability = (truth * probability_matrix).sum(dim=1)

    accuracy_loss = 1.0 - true_probability.mean()
    balanced_accuracy_loss = 1.0 - (
        true_positive[supported] / support[supported]
    ).mean()
    per_class_f1 = (
        2.0 * true_positive
        / (2.0 * true_positive + false_positive + false_negative).clamp_min(1e-8)
    )
    macro_f1_loss = 1.0 - per_class_f1[supported].mean()
    nll_raw = torch.stack(nll_rows).mean()
    components = {
        "accuracy": accuracy_loss,
        "balanced_accuracy": balanced_accuracy_loss,
        "macro_f1": macro_f1_loss,
        "nll": torch.stack(normalized_nll_rows).mean(),
        "brier": 0.5 * (probability_matrix - truth).square().sum(dim=1).mean(),
    }
    if profile == "legacy":
        loss = nll_raw
    else:
        # Named profiles resolve to a complete metric dictionary so callers can
        # inspect every configured weight.  Exclude zero-weight terms from the
        # arithmetic itself: under AMP an unused surrogate can legitimately be
        # infinite, and IEEE ``0 * inf`` would otherwise contaminate the active
        # finite objective with NaN.
        active_weights = {
            name: weight for name, weight in weights.items() if weight > 0.0
        }
        denominator = sum(active_weights.values())
        loss = sum(
            weight * components[name] for name, weight in active_weights.items()
        ) / denominator

    hard_columns = probability_matrix.argmax(dim=1)
    predicted_classes = class_universe[hard_columns]
    hard_prediction = F.one_hot(
        hard_columns, num_classes=class_universe.numel()
    ).float()
    hard_true_positive = (truth * hard_prediction).sum(dim=0)
    hard_false_positive = ((1.0 - truth) * hard_prediction).sum(dim=0)
    hard_false_negative = (truth * (1.0 - hard_prediction)).sum(dim=0)
    hard_accuracy = predicted_classes.eq(target_classes).float().mean()
    hard_balanced_accuracy = (
        hard_true_positive[supported] / support[supported]
    ).mean()
    hard_per_class_f1 = (
        2.0 * hard_true_positive
        / (
            2.0 * hard_true_positive
            + hard_false_positive
            + hard_false_negative
        ).clamp_min(1.0)
    )
    hard_macro_f1 = hard_per_class_f1[supported].mean()

    diagnostics = {
        "head/classification/episode_accuracy": hard_accuracy,
        "head/classification/episode_balanced_accuracy": hard_balanced_accuracy,
        "head/classification/episode_macro_f1": hard_macro_f1,
        "loss/classification/nll_raw": nll_raw,
    }
    for name, component in components.items():
        diagnostics[f"loss/classification/{name}_surrogate_raw"] = component
        diagnostics[f"loss/classification/{name}_surrogate_weighted"] = (
            component.new_tensor(0.0)
            if profile == "legacy" or weights[name] == 0.0
            else weights[name] * component / denominator
        )
        diagnostics[f"objective/classification/{name}_weight"] = component.new_tensor(
            0.0 if profile == "legacy" else weights[name]
        )
    return ClassificationObjective(loss, components, diagnostics, weights, profile)
