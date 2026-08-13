import random

import torch
import torch.nn.functional as F

from training_debug import classification_head_metrics, regression_head_metrics
from utils.data_utils import create_episode


def _result(loss, task, diagnostics, return_diagnostics):
    if return_diagnostics:
        return loss, task, diagnostics
    return loss, task


def _regression_component_metrics(head, components):
    metrics = {}
    weights = {
        "mae": head.regression_mae_weight,
        "rmse": head.regression_rmse_weight,
        "huber": head.regression_huber_weight,
        "log_rmse": head.regression_log_rmse_weight,
        "relative_mae": head.regression_relative_mae_weight,
        "bias": head.regression_bias_weight,
        "median_ae": head.regression_median_ae_weight,
        "quantile": head.regression_quantile_weight,
    }
    denominator = head._regression_primary_weight_sum()
    for name, weight in weights.items():
        value = components[name]
        metrics[f"loss/regression/{name}_raw"] = value
        metrics[f"loss/regression/{name}_weighted"] = weight * value / denominator
    metrics["loss/regression/normalizer_hours"] = components["normalizer"]
    return metrics


def run_episodic_step(
    model,
    task_data_pool,
    task_type,
    config,
    should_shuffle_labels,
    return_diagnostics=False,
):
    progress_bar_task = task_type
    diagnostics_out = {
        "data/episode_valid": 0.0,
        "data/pool_prefixes": float(len(task_data_pool)),
    }
    episode = None
    if task_type == "classification":
        episode = create_episode(
            task_data_pool,
            config["num_shots_range"],
            config["num_queries"],
            num_ways_range=(3, 10),
            shuffle_labels=should_shuffle_labels,
        )
    elif len(task_data_pool) >= config["num_shots_range"][1] + config["num_queries"]:
        random.shuffle(task_data_pool)
        num_shots = random.randint(
            config["num_shots_range"][0], config["num_shots_range"][1]
        )
        support_set_raw = task_data_pool[:num_shots]
        query_set_raw = task_data_pool[
            num_shots : num_shots + config["num_queries"]
        ]
        support_set = [(item[0], item[1]) for item in support_set_raw]
        query_set = [(item[0], item[1]) for item in query_set_raw]
        episode = (support_set, query_set)
    if episode is None or not episode[0] or not episode[1]:
        return _result(None, progress_bar_task, diagnostics_out, return_diagnostics)

    support_set, query_set = episode
    diagnostics_out.update(
        {
            "data/episode_valid": 1.0,
            "data/support_count": float(len(support_set)),
            "data/query_count": float(len(query_set)),
        }
    )
    predictions, true_labels, confidence = model(support_set, query_set, task_type)
    if predictions is None:
        return _result(None, progress_bar_task, diagnostics_out, return_diagnostics)

    head_cfg = config.get("fmv3_head", {})
    routing_weight = float(
        head_cfg.get("expert_routing_confidence_loss_weight", 0.0)
    )
    routing_loss = predictions.new_tensor(0.0)
    confidence_loss = predictions.new_tensor(0.0)
    gate_auxiliary = predictions.new_tensor(0.0)

    if task_type == "classification":
        smoothing = min(
            max(float(config.get("classification_label_smoothing", 0.05)), 0.0),
            1.0,
        )
        primary_loss = F.cross_entropy(
            predictions,
            true_labels,
            ignore_index=-100,
            label_smoothing=smoothing,
        )
        loss = primary_loss
        confidence_weight = float(
            head_cfg.get("classification_expert_confidence_loss_weight", 0.0)
        )
        if confidence_weight > 0:
            probabilities = F.softmax(predictions, dim=-1)
            confidence_loss = model.proto_head.classification_expert_confidence_loss(
                probabilities, true_labels
            )
            loss = loss + confidence_weight * confidence_loss
        if routing_weight > 0 and getattr(model, "task_confidence_head", None) is not None:
            valid = true_labels != -100
            if valid.any():
                reliability = (
                    predictions[valid].argmax(dim=-1) == true_labels[valid]
                ).float().mean().detach()
                routing_loss = model.task_confidence_loss(
                    support_set, query_set, task_type, reliability
                )
                loss = loss + routing_weight * routing_loss
                diagnostics_out["routing/target_reliability"] = reliability
        diagnostics_out.update(
            classification_head_metrics(
                predictions,
                true_labels,
                F.softmax(predictions, dim=-1),
                getattr(model, "last_classification_diagnostics", None),
            )
        )
    else:
        head = model.proto_head
        head_diagnostics = getattr(model, "last_regression_diagnostics", None)
        branch_predictions = None
        aggregation_weights = None
        if (
            head_diagnostics is not None
            and head.regression_outputs_hours
            and head.regression_gate_aux_weight > 0
        ):
            branch_predictions = head_diagnostics.get("branch_predictions_hours")
            aggregation_weights = head_diagnostics.get("aggregation_weights")
        components = (
            head.regression_loss_components(
                predictions.squeeze(), true_labels, labels_in_output_space=True
            )
            if head.regression_outputs_hours
            else None
        )
        primary_loss = head.regression_loss(
            predictions.squeeze(),
            true_labels,
            labels_in_output_space=True,
            branch_predictions=None,
            aggregation_weights=None,
        )
        loss = primary_loss
        if components is not None:
            diagnostics_out.update(_regression_component_metrics(head, components))
        if branch_predictions is not None and aggregation_weights is not None:
            gate_auxiliary = head.regression_gate_auxiliary_loss(
                branch_predictions, aggregation_weights, true_labels
            )
            loss = loss + head.regression_gate_aux_weight * gate_auxiliary
        confidence_weight = float(
            head_cfg.get("regression_expert_confidence_loss_weight", 0.0)
        )
        if confidence_weight > 0 and head_diagnostics is not None:
            base_confidence = getattr(model, "last_regression_base_confidence", None)
            if base_confidence is None:
                base_confidence = torch.ones_like(predictions.reshape(-1))
            confidence_loss = head.regression_expert_confidence_loss(
                predictions.squeeze(),
                true_labels,
                base_confidence,
                head_diagnostics,
                labels_in_output_space=True,
            )
            loss = loss + confidence_weight * confidence_loss
        if routing_weight > 0 and getattr(model, "task_confidence_head", None) is not None:
            predictions_flat = predictions.reshape(-1)
            targets_flat = true_labels.reshape(-1)
            reliability = torch.exp(
                -(
                    (predictions_flat.detach() - targets_flat.detach()).abs()
                    / targets_flat.detach().abs().clamp_min(1.0)
                ).clamp(0.0, 20.0)
            ).mean()
            routing_loss = model.task_confidence_loss(
                support_set, query_set, task_type, reliability
            )
            loss = loss + routing_weight * routing_loss
            diagnostics_out["routing/target_reliability"] = reliability
        diagnostics_out.update(
            regression_head_metrics(
                predictions,
                true_labels,
                getattr(model, "last_regression_base_confidence", confidence),
                head_diagnostics,
            )
        )

    diagnostics_out.update(
        {
            "loss/primary": primary_loss,
            "loss/confidence_raw": confidence_loss,
            "loss/confidence_weighted": confidence_loss
            * (
                float(
                    head_cfg.get(
                        "classification_expert_confidence_loss_weight", 0.0
                    )
                )
                if task_type == "classification"
                else float(
                    head_cfg.get("regression_expert_confidence_loss_weight", 0.0)
                )
            ),
            "loss/routing_raw": routing_loss,
            "loss/routing_weighted": routing_weight * routing_loss,
            "loss/regression_gate_aux_raw": gate_auxiliary,
            "loss/regression_gate_aux_weighted": (
                model.proto_head.regression_gate_aux_weight * gate_auxiliary
                if task_type == "regression"
                else 0.0
            ),
            "loss/total": loss,
        }
    )
    return _result(loss, progress_bar_task, diagnostics_out, return_diagnostics)
