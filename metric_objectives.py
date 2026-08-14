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

CLASSIFICATION_PROFILES = {
    "equilibrated": {name: 1.0 for name in CLASSIFICATION_METRICS},
    "accuracy": {"accuracy": 1.0},
    "balanced_accuracy": {"balanced_accuracy": 1.0},
    "macro_f1": {"macro_f1": 1.0},
    "nll": {"nll": 1.0},
    "brier": {"brier": 1.0},
}
REGRESSION_PROFILES = {
    "equilibrated": {name: 1.0 for name in REGRESSION_METRICS},
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
    true_probabilities = []
    brier_rows = []
    probabilities_rows = []
    target_class_ids = []
    predicted_class_ids = []
    for logits, target_index, ids in valid_rows:
        target = torch.tensor([target_index], device=logits.device, dtype=torch.long)
        nll = F.cross_entropy(
            logits.unsqueeze(0),
            target,
            label_smoothing=label_smoothing,
        )
        probabilities = F.softmax(logits, dim=0)
        one_hot = F.one_hot(target, num_classes=logits.numel()).float().squeeze(0)
        nll_rows.append(nll)
        normalized_nll_rows.append(nll / math.log(max(logits.numel(), 2)))
        true_probabilities.append(probabilities[target_index])
        brier_rows.append(0.5 * (probabilities - one_hot).square().sum())
        probabilities_rows.append(probabilities)
        target_class_ids.append(int(ids[target_index].detach().cpu()))
        predicted_class_ids.append(int(ids[probabilities.argmax()].detach().cpu()))

    true_probability = torch.stack(true_probabilities)
    accuracy_loss = 1.0 - true_probability.mean()
    unique_targets = sorted(set(target_class_ids))
    class_universe = sorted({
        int(class_id.detach().cpu())
        for _, _, ids in valid_rows
        for class_id in ids
    })
    per_class_soft_recall = []
    per_class_soft_f1 = []
    for class_id in class_universe:
        truth = torch.tensor(
            [value == class_id for value in target_class_ids],
            device=true_probability.device,
            dtype=true_probability.dtype,
        )
        class_probabilities = []
        for probabilities, (_, _, ids) in zip(probabilities_rows, valid_rows):
            matches = (ids == class_id).nonzero(as_tuple=False).reshape(-1)
            class_probabilities.append(
                probabilities[matches[0]]
                if matches.numel()
                else probabilities.new_tensor(0.0)
            )
        predicted = torch.stack(class_probabilities)
        true_positive = (truth * predicted).sum()
        false_positive = ((1.0 - truth) * predicted).sum()
        false_negative = (truth * (1.0 - predicted)).sum()
        if class_id in unique_targets:
            per_class_soft_recall.append(
                true_positive / truth.sum().clamp_min(1.0)
            )
        per_class_soft_f1.append(
            (2.0 * true_positive)
            / (2.0 * true_positive + false_positive + false_negative).clamp_min(1e-8)
        )

    balanced_accuracy_loss = 1.0 - torch.stack(per_class_soft_recall).mean()
    macro_f1_loss = 1.0 - torch.stack(per_class_soft_f1).mean()
    nll_raw = torch.stack(nll_rows).mean()
    components = {
        "accuracy": accuracy_loss,
        "balanced_accuracy": balanced_accuracy_loss,
        "macro_f1": macro_f1_loss,
        "nll": torch.stack(normalized_nll_rows).mean(),
        "brier": torch.stack(brier_rows).mean(),
    }
    if profile == "legacy":
        loss = nll_raw
    else:
        denominator = sum(weights.values())
        loss = sum(weights[name] * components[name] for name in weights) / denominator

    hard_accuracy = sum(
        prediction == target
        for prediction, target in zip(predicted_class_ids, target_class_ids)
    ) / len(target_class_ids)
    recalls = []
    f1s = []
    for class_id in class_universe:
        tp = sum(
            prediction == class_id and target == class_id
            for prediction, target in zip(predicted_class_ids, target_class_ids)
        )
        fp = sum(
            prediction == class_id and target != class_id
            for prediction, target in zip(predicted_class_ids, target_class_ids)
        )
        fn = sum(
            prediction != class_id and target == class_id
            for prediction, target in zip(predicted_class_ids, target_class_ids)
        )
        if class_id in unique_targets:
            recalls.append(tp / max(tp + fn, 1))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))

    diagnostics = {
        "head/classification/episode_accuracy": loss.new_tensor(hard_accuracy),
        "head/classification/episode_balanced_accuracy": loss.new_tensor(
            sum(recalls) / len(recalls)
        ),
        "head/classification/episode_macro_f1": loss.new_tensor(
            sum(f1s) / len(f1s)
        ),
        "loss/classification/nll_raw": nll_raw,
    }
    for name, component in components.items():
        diagnostics[f"loss/classification/{name}_surrogate_raw"] = component
        diagnostics[f"loss/classification/{name}_surrogate_weighted"] = (
            component.new_tensor(0.0)
            if profile == "legacy"
            else weights[name] * component / sum(weights.values())
        )
        diagnostics[f"objective/classification/{name}_weight"] = component.new_tensor(
            0.0 if profile == "legacy" else weights[name]
        )
    return ClassificationObjective(loss, components, diagnostics, weights, profile)
