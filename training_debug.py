"""Structured, opt-in diagnostics for FM-v3 meta-training.

The helpers in this module deliberately keep tensors out of persisted records.
Step functions return flat dictionaries of finite Python numbers; this module
aggregates them by task/expert/episode, records gradient and parameter movement,
and derives a conservative train/validation overfitting signal.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metric_objectives import (
    resolve_classification_objective,
    resolve_regression_metric_weights,
)


SCHEMA_VERSION = 1


def _finite_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def scalar_metrics(metrics):
    """Drop non-scalar/non-finite values from a step diagnostic mapping."""
    cleaned = {}
    for key, value in (metrics or {}).items():
        number = _finite_float(value)
        if number is not None:
            cleaned[str(key)] = number
    return cleaned


def tensor_distribution(metrics, prefix, values, *, quantiles=True):
    """Append compact distribution statistics for one diagnostic tensor."""
    if values is None or not isinstance(values, torch.Tensor) or not values.numel():
        return metrics
    flat = values.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    metrics[f"{prefix}/finite_fraction"] = float(finite.float().mean().cpu())
    if not finite.any():
        return metrics
    flat = flat[finite]
    metrics[f"{prefix}/mean"] = float(flat.mean().cpu())
    metrics[f"{prefix}/std"] = float(flat.std(correction=0).cpu())
    metrics[f"{prefix}/min"] = float(flat.min().cpu())
    metrics[f"{prefix}/max"] = float(flat.max().cpu())
    metrics[f"{prefix}/abs_mean"] = float(flat.abs().mean().cpu())
    if quantiles:
        quantile_values = torch.quantile(
            flat, flat.new_tensor([0.10, 0.50, 0.90])
        ).cpu().tolist()
        for name, value in zip(("p10", "median", "p90"), quantile_values):
            metrics[f"{prefix}/{name}"] = float(value)
    return metrics


def _normalized_entropy(probabilities, dim=-1):
    count = max(int(probabilities.size(dim)), 2)
    probabilities = probabilities.float().clamp_min(1e-8)
    return -(
        probabilities * torch.log(probabilities)
    ).sum(dim=dim) / math.log(count)


def selection_metrics(diagnostics, prefix):
    metrics = {}
    if not diagnostics:
        return metrics
    logits = diagnostics.get("selection_logits")
    tensor_distribution(metrics, f"{prefix}/log_weight", logits)
    effective = diagnostics.get("selection_effective_count")
    tensor_distribution(
        metrics, f"{prefix}/effective_support", effective, quantiles=True
    )
    trust = diagnostics.get("selection_trust")
    if isinstance(trust, torch.Tensor) and trust.numel():
        metrics[f"{prefix}/trust_max_mean"] = float(
            trust.detach().float().max(dim=-1).values.mean().cpu()
        )
        metrics[f"{prefix}/trust_entropy_mean"] = float(
            _normalized_entropy(trust.detach(), dim=-1).mean().cpu()
        )
    attention = diagnostics.get("selection_attention")
    if isinstance(attention, torch.Tensor) and attention.numel():
        metrics[f"{prefix}/attention_max_mean"] = float(
            attention.detach().float().max(dim=-1).values.mean().cpu()
        )
        metrics[f"{prefix}/attention_entropy_mean"] = float(
            _normalized_entropy(attention.detach(), dim=-1).mean().cpu()
        )
    return metrics


def classification_head_metrics(logits, labels, probabilities, diagnostics=None):
    metrics = {}
    if logits is None or labels is None or probabilities is None:
        return metrics
    logits = logits.detach().float()
    labels = labels.detach().long().reshape(-1)
    probabilities = probabilities.detach().float()
    valid = (labels >= 0) & (labels < probabilities.size(-1))
    metrics["head/classification/query_count"] = float(valid.sum().cpu())
    metrics["head/classification/class_count"] = float(probabilities.size(-1))
    if valid.any():
        selected = probabilities[valid].clamp_min(1e-8)
        targets = labels[valid]
        rows = torch.arange(targets.numel(), device=targets.device)
        true_probability = selected[rows, targets]
        predictions = selected.argmax(dim=-1)
        metrics["head/classification/accuracy"] = float(
            (predictions == targets).float().mean().cpu()
        )
        metrics["head/classification/nll"] = float(
            (-torch.log(true_probability)).mean().cpu()
        )
        metrics["head/classification/true_probability_mean"] = float(
            true_probability.mean().cpu()
        )
        metrics["head/classification/max_probability_mean"] = float(
            selected.max(dim=-1).values.mean().cpu()
        )
        metrics["head/classification/entropy_mean"] = float(
            _normalized_entropy(selected).mean().cpu()
        )
        top = torch.topk(selected, min(2, selected.size(-1)), dim=-1).values
        margin = top[:, 0] - (top[:, 1] if top.size(1) > 1 else 0.0)
        metrics["head/classification/probability_margin_mean"] = float(
            margin.mean().cpu()
        )
    if diagnostics:
        local_counts = diagnostics.get("local_counts")
        pool_counts = diagnostics.get("pool_counts")
        if isinstance(local_counts, torch.Tensor) and local_counts.numel():
            metrics["head/classification/local_class_coverage"] = float(
                (local_counts > 0).float().mean().cpu()
            )
            metrics["head/classification/local_count_max"] = float(
                local_counts.detach().float().max().cpu()
            )
        if isinstance(pool_counts, torch.Tensor) and pool_counts.numel():
            metrics["head/classification/pool_class_coverage"] = float(
                (pool_counts > 0).float().mean().cpu()
            )
        for name in ("gate", "prototype_variances", "local_evidence", "global_evidence"):
            tensor_distribution(
                metrics,
                f"head/classification/{name}",
                diagnostics.get(name),
                quantiles=False,
            )
        metrics.update(selection_metrics(diagnostics, "head/classification/selector"))
    return scalar_metrics(metrics)


def regression_head_metrics(predictions, targets, confidence=None, diagnostics=None):
    metrics = {}
    if predictions is None or targets is None:
        return metrics
    predictions = predictions.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    if predictions.numel() != targets.numel() or not predictions.numel():
        return metrics
    errors = predictions - targets
    absolute = errors.abs()
    target_variance = (targets - targets.mean()).square().mean()
    prediction_mse = errors.square().mean()
    if float(target_variance.cpu()) <= 1e-12:
        r2 = 1.0 if float(prediction_mse.cpu()) <= 1e-12 else 0.0
    else:
        r2 = float((1.0 - prediction_mse / target_variance).cpu())
    metrics.update(
        {
            "head/regression/query_count": float(predictions.numel()),
            "head/regression/mae_hours": float(absolute.mean().cpu()),
            "head/regression/rmse_hours": float(
                torch.sqrt(errors.square().mean()).cpu()
            ),
            "head/regression/median_ae_hours": float(absolute.median().cpu()),
            "head/regression/bias_hours": float(errors.mean().cpu()),
            "head/regression/relative_mae": float(
                (absolute / targets.abs().clamp_min(1.0)).mean().cpu()
            ),
            "head/regression/r2": r2,
            "head/regression/prediction_mean_hours": float(predictions.mean().cpu()),
            "head/regression/target_mean_hours": float(targets.mean().cpu()),
            "head/regression/error_p90_hours": float(
                torch.quantile(absolute, 0.90).cpu()
            ),
        }
    )
    tensor_distribution(
        metrics, "head/regression/prediction", predictions, quantiles=False
    )
    if isinstance(confidence, torch.Tensor) and confidence.numel():
        tensor_distribution(
            metrics, "head/regression/base_confidence", confidence, quantiles=False
        )
    if diagnostics:
        for name in ("std_hours", "std", "similarities"):
            tensor_distribution(
                metrics,
                f"head/regression/{name}",
                diagnostics.get(name),
                quantiles=False,
            )
        branch_predictions = diagnostics.get("branch_predictions_hours")
        aggregation = diagnostics.get("aggregation_weights")
        if isinstance(aggregation, torch.Tensor) and aggregation.numel():
            weights = aggregation.detach().float()
            if weights.ndim == 1:
                weights = weights[:, None]
            metrics["head/regression/branch_weight_entropy_mean"] = float(
                _normalized_entropy(weights.transpose(0, 1)).mean().cpu()
            )
            for index in range(weights.size(0)):
                metrics[f"head/regression/branch_{index}/weight_mean"] = float(
                    weights[index].mean().cpu()
                )
        if isinstance(branch_predictions, torch.Tensor) and branch_predictions.numel():
            branches = branch_predictions.detach().float()
            for index in range(branches.size(0)):
                if branches[index].numel() == targets.numel():
                    branch_error = branches[index].reshape(-1) - targets
                    metrics[f"head/regression/branch_{index}/mae_hours"] = float(
                        branch_error.abs().mean().cpu()
                    )
        metrics.update(selection_metrics(diagnostics, "head/regression/selector"))
    return scalar_metrics(metrics)


def average_metric_dicts(rows):
    values = defaultdict(list)
    for row in rows:
        for key, value in scalar_metrics(row).items():
            values[key].append(value)
    return {key: float(np.mean(items)) for key, items in values.items() if items}


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def add(self, value):
        value = _finite_float(value)
        if value is None:
            return
        self.count += 1
        self.total += value
        self.total_square += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self):
        if not self.count:
            return None
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class MetricAccumulator:
    def __init__(self):
        self.metrics = defaultdict(RunningStats)

    def add(self, metrics, prefixes=()):
        for key, value in scalar_metrics(metrics).items():
            self.metrics[key].add(value)
            for prefix in prefixes:
                self.metrics[f"{prefix}/{key}"].add(value)

    def summary(self):
        return {
            key: summary
            for key in sorted(self.metrics)
            if (summary := self.metrics[key].summary()) is not None
        }


def parameter_group(name):
    rules = (
        ("classification_example_selector", "head/classification_selector"),
        ("regression_example_selector", "head/regression_selector"),
        ("time_transform_bank", "head/time_transform"),
        ("classification_expert_confidence", "head/classification_confidence"),
        ("regression_expert_confidence", "head/regression_confidence"),
        ("task_confidence_head", "routing"),
        ("classification_embedding_adapter", "adapter/classification"),
        ("regression_embedding_adapter", "adapter/regression"),
        ("state_aware_pool", "adapter/prefix_attention"),
        ("temporal_input_encoder", "adapter/temporal_input"),
        ("proto_head", "head/other"),
        ("proj_head", "projection"),
        ("encoder", "encoder"),
        ("embedder", "embedder"),
    )
    for fragment, group in rules:
        if fragment in name:
            return group
    return "other"


def snapshot_trainable_parameters(model):
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def gradient_metrics(model):
    sums = defaultdict(float)
    abs_sums = defaultdict(float)
    counts = defaultdict(int)
    maxima = defaultdict(float)
    finite_counts = defaultdict(int)
    nonzero_counts = defaultdict(int)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        group = parameter_group(name)
        finite = torch.isfinite(gradient)
        values = gradient[finite]
        counts[group] += gradient.numel()
        finite_counts[group] += int(finite.sum().cpu())
        if values.numel():
            sums[group] += float(values.square().sum().cpu())
            abs_sums[group] += float(values.abs().sum().cpu())
            maxima[group] = max(maxima[group], float(values.abs().max().cpu()))
            nonzero_counts[group] += int((values != 0).sum().cpu())
    metrics = {}
    for group in sorted(counts):
        count = max(counts[group], 1)
        metrics[f"optimization/gradient/{group}/l2_norm"] = math.sqrt(sums[group])
        metrics[f"optimization/gradient/{group}/abs_mean"] = abs_sums[group] / count
        metrics[f"optimization/gradient/{group}/abs_max"] = maxima[group]
        metrics[f"optimization/gradient/{group}/finite_fraction"] = finite_counts[group] / count
        metrics[f"optimization/gradient/{group}/nonzero_fraction"] = nonzero_counts[group] / count
    return metrics


def loss_gradient_metrics(model, step_metrics):
    """Attribute sampled gradient energy to each differentiable loss term.

    Total-loss gradients reveal which parameter groups move, but a large scalar
    auxiliary need not have a large gradient. This sampled diagnostic uses
    ``autograd.grad`` without touching ``parameter.grad`` so the subsequent
    optimization backward pass is unchanged.
    """
    selected = (
        "loss/primary",
        "loss/classification_separation_weighted",
        "loss/confidence_weighted",
        "loss/regression_gate_aux_weighted",
        "loss/routing_weighted",
        "loss/regression/mae_weighted",
        "loss/regression/rmse_weighted",
        "loss/regression/huber_weighted",
        "loss/regression/log_rmse_weighted",
        "loss/regression/relative_mae_weighted",
        "loss/regression/bias_weighted",
        "loss/regression/median_ae_weighted",
        "loss/regression/quantile_weighted",
        "loss/regression/r2_weighted",
        "loss/classification/accuracy_surrogate_weighted",
        "loss/classification/balanced_accuracy_surrogate_weighted",
        "loss/classification/macro_f1_surrogate_weighted",
        "loss/classification/nll_surrogate_weighted",
        "loss/classification/brier_surrogate_weighted",
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        return {}
    parameters = [parameter for _, parameter in named_parameters]
    metrics = {}
    for metric_name in selected:
        component = step_metrics.get(metric_name)
        if not isinstance(component, torch.Tensor) or not component.requires_grad:
            continue
        gradients = torch.autograd.grad(
            component,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        group_squares = defaultdict(float)
        total_squares = 0.0
        for (name, _), gradient in zip(named_parameters, gradients):
            if gradient is None:
                continue
            values = gradient.detach().float()
            values = values[torch.isfinite(values)]
            if not values.numel():
                continue
            square_sum = float(values.square().sum().cpu())
            group_squares[parameter_group(name)] += square_sum
            total_squares += square_sum
        component_name = metric_name.removeprefix("loss/")
        metrics[
            f"optimization/loss_gradient/{component_name}/all/l2_norm"
        ] = math.sqrt(total_squares)
        for group, square_sum in sorted(group_squares.items()):
            metrics[
                f"optimization/loss_gradient/{component_name}/{group}/l2_norm"
            ] = math.sqrt(square_sum)
    return metrics


def parameter_update_metrics(model, snapshot):
    update_squares = defaultdict(float)
    parameter_squares = defaultdict(float)
    counts = defaultdict(int)
    for name, parameter in model.named_parameters():
        if name not in snapshot:
            continue
        current = parameter.detach().float().cpu()
        previous = snapshot[name]
        group = parameter_group(name)
        update_squares[group] += float((current - previous).square().sum())
        parameter_squares[group] += float(previous.square().sum())
        counts[group] += current.numel()
    metrics = {}
    for group in sorted(counts):
        update_norm = math.sqrt(update_squares[group])
        parameter_norm = math.sqrt(parameter_squares[group])
        metrics[f"optimization/update/{group}/l2_norm"] = update_norm
        metrics[f"optimization/update/{group}/relative_l2"] = update_norm / max(
            parameter_norm, 1e-12
        )
        metrics[f"optimization/update/{group}/rms"] = math.sqrt(
            update_squares[group] / max(counts[group], 1)
        )
    return metrics


def model_state_metrics(model):
    metrics = {}
    group_squares = defaultdict(float)
    group_counts = defaultdict(int)
    for name, parameter in model.named_parameters():
        values = parameter.detach().float()
        group = parameter_group(name)
        group_squares[group] += float(values.square().sum().cpu())
        group_counts[group] += values.numel()
    for group in sorted(group_counts):
        metrics[f"state/parameter/{group}/l2_norm"] = math.sqrt(group_squares[group])
        metrics[f"state/parameter/{group}/rms"] = math.sqrt(
            group_squares[group] / max(group_counts[group], 1)
        )
    for expert_index, expert in enumerate(getattr(model, "experts", [])):
        head = expert.proto_head
        prefix = f"state/expert_{expert_index}"
        for name in (
            "logit_scale",
            "reg_logit_scale",
            "count_prior",
            "fixed_gate",
            "abstain_bias",
            "abstain_slope",
        ):
            value = getattr(head, name, None)
            number = _finite_float(value)
            if number is not None:
                metrics[f"{prefix}/{name}"] = number
        metrics[f"{prefix}/classification_selector_strength"] = float(
            head.classification_example_selector_strength
        )
        metrics[f"{prefix}/regression_selector_strength"] = float(
            head.regression_example_selector_strength
        )
        bank = getattr(head, "time_transform_bank", None)
        if bank is not None:
            for index, value in enumerate(bank.powers.detach().cpu().tolist()):
                metrics[f"{prefix}/transform_{index}/power"] = float(value)
            for index, value in enumerate(bank.scales.detach().cpu().tolist()):
                metrics[f"{prefix}/transform_{index}/scale_hours"] = float(value)
            for index, value in enumerate(
                bank.aggregation_weights.detach().cpu().tolist()
            ):
                metrics[f"{prefix}/transform_{index}/prior_weight"] = float(value)
    return scalar_metrics(metrics)


def split_training_tasks_by_case(training_tasks, fraction, seed, log_names=None):
    """Create deterministic, task-shared validation cases for every source log."""
    fraction = min(max(float(fraction), 0.0), 0.5)
    if fraction <= 0.0:
        return training_tasks, {"classification": [], "regression": []}, []
    class_pools = training_tasks.get("classification", [])
    reg_pools = training_tasks.get("regression", [])
    count = max(len(class_pools), len(reg_pools))
    train = {"classification": [], "regression": []}
    validation = {"classification": [], "regression": []}
    manifest = []
    for index in range(count):
        class_pool = class_pools[index] if index < len(class_pools) else []
        reg_pool = reg_pools[index] if index < len(reg_pools) else []
        cases = sorted({str(item[2]) for item in class_pool or reg_pool})
        shuffled = cases[:]
        random.Random(int(seed) + 1009 * index).shuffle(shuffled)
        validation_count = max(1, int(round(len(cases) * fraction))) if len(cases) > 1 else 0
        validation_count = min(validation_count, max(0, len(cases) - 1))
        validation_cases = set(shuffled[:validation_count])
        digest = hashlib.sha256(
            "\n".join(sorted(validation_cases)).encode("utf-8")
        ).hexdigest()
        row = {
            "pool_index": index,
            "log": (
                str(log_names[index])
                if log_names is not None and index < len(log_names)
                else str(index)
            ),
            "total_cases": len(cases),
            "validation_cases": len(validation_cases),
            "validation_case_sha256": digest,
            "tasks": {},
        }
        for task, pool in (("classification", class_pool), ("regression", reg_pool)):
            train_pool = [item for item in pool if str(item[2]) not in validation_cases]
            validation_pool = [item for item in pool if str(item[2]) in validation_cases]
            train[task].append(train_pool)
            validation[task].append(validation_pool)
            row["tasks"][task] = {
                "train_prefixes": len(train_pool),
                "validation_prefixes": len(validation_pool),
            }
        manifest.append(row)
    return train, validation, manifest


class TrainingDiagnostics:
    def __init__(self, checkpoint_dir, config):
        self.config = config.get("training_diagnostics", {}) or {}
        classification_profile, classification_weights = (
            resolve_classification_objective(config)
        )
        regression_profile, regression_weights = resolve_regression_metric_weights(
            config.get("fmv3_head", {}) or {}
        )
        self.objective_configuration = {
            "classification": {
                "profile": classification_profile,
                "weights": classification_weights,
            },
            "regression": {
                "profile": regression_profile,
                "weights": regression_weights,
            },
        }
        self.enabled = bool(self.config.get("enabled", False))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.step_interval = max(1, int(self.config.get("step_interval", 25)))
        self.loss_gradient_interval = max(
            0, int(self.config.get("loss_gradient_interval", 0))
        )
        self.epoch_accumulator = MetricAccumulator()
        self.epoch_records = []
        self.steps_path = self.checkpoint_dir / "training_debug_steps.jsonl"
        self.epochs_path = self.checkpoint_dir / "training_debug_epochs.jsonl"
        self.summary_path = self.checkpoint_dir / "training_debug_summary.json"
        if self.enabled and self.epochs_path.exists():
            with self.epochs_path.open("r", encoding="utf-8") as handle:
                self.epoch_records = [
                    json.loads(line) for line in handle if line.strip()
                ]

    def should_record_step(self, step):
        return self.enabled and (int(step) % self.step_interval == 0)

    def should_record_loss_gradients(self, step):
        return (
            self.enabled
            and self.loss_gradient_interval > 0
            and int(step) % self.loss_gradient_interval == 0
        )

    def start_epoch(self):
        self.epoch_accumulator = MetricAccumulator()

    def add_step(
        self,
        epoch,
        step,
        task,
        expert,
        pool,
        episode,
        metrics,
        *,
        sampled=False,
    ):
        if not self.enabled:
            return
        metrics = scalar_metrics(metrics)
        prefixes = (
            f"task/{task}",
            f"expert/{int(expert)}",
            f"pool/{int(pool)}",
            f"episode/{episode}",
            f"task/{task}/expert/{int(expert)}",
            f"task/{task}/pool/{int(pool)}",
        )
        self.epoch_accumulator.add(metrics, prefixes=prefixes)
        if sampled:
            record = {
                "schema_version": SCHEMA_VERSION,
                "epoch": int(epoch),
                "step": int(step),
                "task": str(task),
                "expert": int(expert),
                "pool": int(pool),
                "episode": str(episode),
                "metrics": metrics,
            }
            with self.steps_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _overfitting_summary(self, records):
        patience = max(1, int(self.config.get("overfitting_patience", 3)))
        tolerance = max(float(self.config.get("overfitting_relative_tolerance", 0.02)), 0.0)
        result = {}
        for task in ("classification", "regression"):
            train_key = f"task/{task}/loss/total"
            validation_key = f"task/{task}/loss/total"
            curve = []
            for record in records:
                train = record.get("train", {}).get(train_key, {}).get("mean")
                validation = record.get("validation", {}).get(validation_key, {}).get("mean")
                if train is not None and validation is not None:
                    curve.append((record["epoch"], float(train), float(validation)))
            if not curve:
                continue
            best_index = min(range(len(curve)), key=lambda index: curve[index][2])
            best_epoch, best_train, best_validation = curve[best_index]
            last_epoch, last_train, last_validation = curve[-1]
            degradation = (last_validation - best_validation) / max(abs(best_validation), 1e-12)
            train_improvement = (best_train - last_train) / max(abs(best_train), 1e-12)
            enough_epochs = len(curve) - 1 - best_index >= patience
            result[task] = {
                "best_validation_epoch": int(best_epoch),
                "best_validation_loss": best_validation,
                "last_epoch": int(last_epoch),
                "last_validation_loss": last_validation,
                "relative_validation_degradation": degradation,
                "relative_train_improvement_since_best": train_improvement,
                "overfitting_signal": bool(
                    enough_epochs
                    and degradation > tolerance
                    and train_improvement > 0.0
                ),
                "patience_epochs": patience,
                "relative_tolerance": tolerance,
            }
        return result

    def finish_epoch(
        self,
        epoch,
        validation_accumulator,
        epoch_metrics,
        state_metrics,
        update_metrics,
        schedule,
    ):
        if not self.enabled:
            return
        record = {
            "schema_version": SCHEMA_VERSION,
            "epoch": int(epoch),
            "schedule": scalar_metrics(schedule),
            "train": self.epoch_accumulator.summary(),
            "validation": (
                validation_accumulator.summary()
                if validation_accumulator is not None
                else {}
            ),
            "epoch_metrics": scalar_metrics(epoch_metrics),
            "state": scalar_metrics(state_metrics),
            "updates": scalar_metrics(update_metrics),
        }
        with self.epochs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.epoch_records.append(record)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "configuration": self.config,
            "objective_configuration": self.objective_configuration,
            "epochs": self.epoch_records,
            "generalization": self._overfitting_summary(self.epoch_records),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def save_validation_manifest(checkpoint_dir, manifest, config):
    path = Path(checkpoint_dir) / "training_validation_split.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(config.get("seed", 42)),
        "validation_fraction": float(
            config.get("training_diagnostics", {}).get("validation_fraction", 0.0)
        ),
        "logs": manifest,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
