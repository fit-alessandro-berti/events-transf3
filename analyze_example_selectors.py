#!/usr/bin/env python3
"""Summarize learned support-selection behavior on real target-log neighborhoods."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pm4py
import torch
import torch.nn.functional as F

from config import CONFIG
from config_utils import apply_experiment_config, deep_merge, load_yaml_config
from evaluation.fmv3_protocol import (
    _fixed_query_indices,
    _route_task_experts,
    _task_indices,
    encode_tasks,
    prepare_case_plan_from_dataframe,
)
from utils.data_utils import get_classification_and_regression_tasks
from utils.model_utils import create_model, init_loader, load_model_weights


FEATURE_NAMES = {
    "classification": [
        "raw_cosine",
        "centered_cosine",
        "neighborhood_zscore",
        "support_centrality",
        "same_class_coherence",
        "normalized_class_support",
    ],
    "regression": [
        "raw_cosine",
        "centered_cosine",
        "neighborhood_zscore",
        "support_centrality",
        "robust_target_deviation",
        "nearest_target_disagreement",
        "signed_target_position",
    ],
}


def _load_resolved_config(checkpoint_dir, eval_config, overrides):
    config = copy.deepcopy(CONFIG)
    yaml_path = checkpoint_dir / "training_config.yaml"
    pth_path = checkpoint_dir / "training_config.pth"
    if yaml_path.exists():
        deep_merge(config, load_yaml_config(str(yaml_path)))
    elif pth_path.exists():
        deep_merge(
            config,
            torch.load(pth_path, map_location="cpu", weights_only=False),
        )
    else:
        raise FileNotFoundError(f"No training configuration in {checkpoint_dir}")
    apply_experiment_config(config, eval_config, overrides)
    return config


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selector_record(task, expert_index, head, features, logits, rng):
    flat_features = features.detach().reshape(-1, features.size(-1))
    flat_logits = logits.detach().reshape(-1)
    selector = getattr(head, f"{task}_example_selector")
    strength = float(getattr(head, f"{task}_example_selector_strength"))
    permutation_delta_sums = []
    for column in range(flat_features.size(1)):
        permutation = torch.as_tensor(
            rng.permutation(flat_features.size(0)), device=flat_features.device
        )
        perturbed = flat_features.clone()
        perturbed[:, column] = flat_features[permutation, column]
        changed = strength * selector(perturbed)
        permutation_delta_sums.append(
            float((changed - flat_logits).abs().sum().cpu())
        )
    trust = F.softmax(logits, dim=1)
    effective_count = trust.square().sum(dim=1).clamp_min(1e-8).reciprocal()
    return {
        "task": task,
        "expert_index": int(expert_index),
        "features": flat_features.cpu().numpy(),
        "logits": flat_logits.cpu().numpy(),
        "effective_count": effective_count.cpu().numpy(),
        "permutation_delta_sums": np.asarray(permutation_delta_sums),
        "strength": strength,
        "queries": int(logits.size(0)),
        "neighbors_per_query": int(logits.size(1)),
    }


@torch.no_grad()
def _collect_task_records(
    task,
    experts,
    expert_indices,
    embeddings_by_expert,
    labels,
    query_indices,
    support_indices,
    retrieval_k,
    rng,
):
    records = []
    if not len(query_indices) or not len(support_indices):
        return records
    device = next(experts[0].parameters()).device
    query_device = torch.as_tensor(query_indices, device=device)
    support_device = torch.as_tensor(support_indices, device=device)
    labels_device = labels.to(device)
    for expert_index, expert, embeddings in zip(
        expert_indices, experts, embeddings_by_expert
    ):
        embeddings = F.normalize(embeddings.to(device), p=2, dim=1)
        queries = embeddings[query_device]
        pool = embeddings[support_device]
        k = min(int(retrieval_k), int(pool.size(0)))
        positions = torch.topk(queries @ pool.t(), k, dim=1).indices
        local = pool[positions]
        local_labels = labels_device[support_device][positions]
        center = local.mean(dim=1, keepdim=True)
        centered_local = F.normalize(local - center, p=2, dim=2)
        centered_query = F.normalize(queries - center.squeeze(1), p=2, dim=1)
        base_similarities = torch.einsum(
            "qd,qkd->qk", centered_query, centered_local
        )
        selector_method = getattr(
            expert.proto_head, f"{task}_selection_logits"
        )
        logits, features = selector_method(
            local,
            local_labels,
            queries,
            base_similarities=base_similarities,
            return_features=True,
        )
        records.append(
            _selector_record(
                task, expert_index, expert.proto_head, features, logits, rng
            )
        )
    return records


def _percentiles(values):
    return {
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _summarize(records, task, expert_index=None):
    selected = [
        record
        for record in records
        if record["task"] == task
        and (expert_index is None or record["expert_index"] == expert_index)
    ]
    features = np.concatenate([record["features"] for record in selected])
    logits = np.concatenate([record["logits"] for record in selected])
    effective = np.concatenate([record["effective_count"] for record in selected])
    delta_sums = np.stack(
        [record["permutation_delta_sums"] for record in selected]
    ).sum(axis=0)
    permutation_change = delta_sums / max(len(logits), 1)
    normalized_importance = permutation_change / max(permutation_change.sum(), 1e-12)
    feature_rows = []
    for column, name in enumerate(FEATURE_NAMES[task]):
        feature_values = features[:, column]
        if np.std(feature_values) <= 1e-12 or np.std(logits) <= 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(feature_values, logits)[0, 1])
        feature_rows.append(
            {
                "name": name,
                "observed": _percentiles(feature_values),
                "logit_correlation": correlation,
                "permutation_mean_absolute_log_weight_change": float(
                    permutation_change[column]
                ),
                "normalized_permutation_importance": float(
                    normalized_importance[column]
                ),
            }
        )
    logit_percentiles = _percentiles(logits)
    return {
        "deployed_strength": float(selected[0]["strength"]),
        "logs_and_experts": len(selected),
        "queries": int(sum(record["queries"] for record in selected)),
        "scored_support_examples": int(len(logits)),
        "neighbors_per_query": int(selected[0]["neighbors_per_query"]),
        "selection_log_weight": {
            **logit_percentiles,
            "mean": float(np.mean(logits)),
            "std": float(np.std(logits)),
            "p90_to_p10_weight_ratio": float(
                np.exp(logit_percentiles["p90"] - logit_percentiles["p10"])
            ),
        },
        "selector_only_effective_support_count": {
            **_percentiles(effective),
            "mean": float(np.mean(effective)),
        },
        "features": feature_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_epoch", type=int, required=True)
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--logs_dir", default="logs_eval")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case_budget", type=int, default=128)
    parser.add_argument("--retrieval_k", type=int, default=20)
    parser.add_argument("--max_queries_per_log", type=int, default=256)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    config = _load_resolved_config(
        checkpoint_dir, args.eval_config, args.overrides
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader = init_loader(config)
    loader.load_training_artifacts(
        str(checkpoint_dir / "training_artifacts.pth")
    )
    model = create_model(config, loader, device)
    load_model_weights(model, str(checkpoint_dir), device, args.checkpoint_epoch)
    model.eval()

    records = []
    sampled_logs = []
    rng = np.random.default_rng(int(config.get("seed", 42)))
    requested = set(args.logs)
    paths = sorted(
        list(Path(args.logs_dir).glob("*.xes"))
        + list(Path(args.logs_dir).glob("*.xes.gz"))
    )
    for log_path in paths:
        log_name = log_path.name.replace(".xes.gz", "").replace(".xes", "")
        if log_name not in requested and log_path.name not in requested:
            continue
        raw_frame = pm4py.read_xes(str(log_path.resolve()))
        case_plan, selected_frame, activity_names = prepare_case_plan_from_dataframe(
            raw_frame, config
        )
        traces = loader.transform_dataframes(
            {log_name: selected_frame},
            activity_names_by_log={log_name: activity_names},
        )[log_name]
        class_tasks, reg_tasks = get_classification_and_regression_tasks(traces)
        tasks_by_type = {
            "classification": class_tasks,
            "regression": reg_tasks,
        }
        used_support_cases = {
            case for order in case_plan["orders"].values() for case in order
        }
        order = case_plan["orders"][(0, "natural")]
        budget = min(int(args.case_budget), len(order))
        selected_cases = set(order[:budget])
        log_summary = {"log": log_name, "case_budget": budget, "tasks": {}}
        for task, tasks in tasks_by_type.items():
            labels = torch.as_tensor(
                [item[1] for item in tasks],
                dtype=torch.long if task == "classification" else torch.float32,
            )
            query_indices = _fixed_query_indices(
                tasks,
                case_plan["test_cases"],
                int(args.max_queries_per_log),
                int(config.get("seed", 42)),
            )
            support_indices = _task_indices(tasks, selected_cases)
            routing_support = [
                item for item in tasks if str(item[2]) in used_support_cases
            ]
            routing_queries = [tasks[int(index)] for index in query_indices]
            experts, routing = _route_task_experts(
                model, list(model.experts), routing_support, routing_queries, task
            )
            expert_indices = (
                routing["selected_expert_indices"]
                if routing is not None
                else list(range(len(experts)))
            )
            embeddings = [
                encode_tasks(
                    expert,
                    tasks,
                    config["fmv3_evaluation"].get("embedding_batch_size", 128),
                    task,
                )
                for expert in experts
            ]
            task_records = _collect_task_records(
                task,
                experts,
                expert_indices,
                embeddings,
                labels,
                query_indices,
                support_indices,
                args.retrieval_k,
                rng,
            )
            records.extend(task_records)
            log_summary["tasks"][task] = {
                "queries": int(len(query_indices)),
                "support_prefixes": int(len(support_indices)),
                "active_experts": list(map(int, expert_indices)),
            }
        sampled_logs.append(log_summary)

    missing = requested - {item["log"] for item in sampled_logs}
    if missing:
        raise FileNotFoundError(f"Requested logs not found: {sorted(missing)}")
    summary = {
        "checkpoint": str(
            Path(args.checkpoint_dir) / f"model_epoch_{args.checkpoint_epoch}.pth"
        ),
        "checkpoint_sha256": _sha256(
            checkpoint_dir / f"model_epoch_{args.checkpoint_epoch}.pth"
        ),
        "case_budget": int(args.case_budget),
        "retrieval_k": int(args.retrieval_k),
        "max_queries_per_log": int(args.max_queries_per_log),
        "logs": sampled_logs,
        "tasks": {},
    }
    for task in FEATURE_NAMES:
        summary["tasks"][task] = {
            "aggregate": _summarize(records, task),
            "experts": {
                str(index): _summarize(records, task, index)
                for index in sorted(
                    {record["expert_index"] for record in records if record["task"] == task}
                )
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote selector diagnostics to {output}")


if __name__ == "__main__":
    main()
