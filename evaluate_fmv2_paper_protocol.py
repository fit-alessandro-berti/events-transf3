#!/usr/bin/env python3
"""Evaluate a checkpoint on the exact public FM-v2 paper splits.

The companion ``experiments-fm`` repository contains one fixed 20% query log
and eight nested support logs for every target log.  This evaluator keeps the
query prefixes disjoint from support cases, retrieves a common neighborhood in
the mean expert embedding, and aggregates the prototypical-head probabilities
or predictions from all experts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error

from config import CONFIG
from utils.data_utils import get_task_data
from utils.model_utils import create_model, init_loader, load_model_weights


FRACTION_CODES = {
    "0.5": "0005",
    "1": "0010",
    "3": "0030",
    "5": "0050",
    "10": "0100",
    "20": "0200",
    "50": "0500",
    "100": "1000",
}
DEFAULT_LOGS = ["billing", "helpdesk", "receipt", "roadtraffic_10000", "sepsis"]


def _fraction_label(value: str | float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _select_queries(tasks, count: int, seed: int):
    if count <= 0 or len(tasks) <= count:
        return list(tasks)
    indices = random.Random(seed).sample(range(len(tasks)), count)
    return [tasks[index] for index in indices]


def _activity_universe(paper_repo: Path, log_name: str):
    candidates = sorted((paper_repo / "base_logs").glob(f"{log_name}.xes*"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one base log for {log_name}, found {candidates}")
    frame = pm4py.read_xes(str(candidates[0]))
    return sorted(frame["concept:name"].dropna().unique().tolist())


def _transform_log(loader, path: Path, key: str, activity_names):
    frame = pm4py.read_xes(str(path))
    return loader.transform_dataframes(
        {key: frame}, activity_names_by_log={key: activity_names}
    )[key]


def _encode(experts, tasks, task_type: str, batch_size: int):
    sequences = [task[0] for task in tasks]
    prepared_batches = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        max_len = max(len(sequence) for sequence in batch)
        padded_frames = []
        masks = []
        for sequence in batch:
            pad_len = max_len - len(sequence)
            frame = pd.DataFrame(sequence)
            if pad_len:
                frame = pd.concat(
                    [
                        frame,
                        pd.DataFrame([experts[0].pad_event] * pad_len),
                    ],
                    ignore_index=True,
                )
            padded_frames.append(frame)
            masks.append([False] * len(sequence) + [True] * pad_len)
        prepared_batches.append(
            (pd.concat(padded_frames, ignore_index=True), masks, len(batch), max_len)
        )

    raw_by_expert = []
    with torch.inference_mode():
        # Preserve the historical expert-major execution order so CUDA
        # numerical behavior and nearest-neighbor tie breaking remain stable.
        # Only the expert-independent DataFrame padding work is cached.
        for expert in experts:
            encoded_batches = []
            for batch_frame, masks, batch_len, max_len in prepared_batches:
                device = next(expert.parameters()).device
                event_embeddings = expert.embedder(
                    batch_frame,
                    use_time_adapter=(task_type == "regression"),
                    time_scale_factor=None,
                )
                event_embeddings = event_embeddings.view(batch_len, max_len, -1)
                mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
                raw = expert.encoder(
                    event_embeddings,
                    src_key_padding_mask=mask,
                    task_type=task_type,
                )
                encoded_batches.append(expert.adapt_task_embeddings(raw, task_type))
            raw_by_expert.append(torch.cat(encoded_batches, dim=0))
    normalized_by_expert = [F.normalize(raw, p=2, dim=1) for raw in raw_by_expert]
    # Match eval_retrieval.py's model-scope protocol: average raw expert
    # embeddings first, then L2-normalize the model representation used to
    # choose the one shared neighbor set.  The proto heads still receive each
    # expert's individually normalized embedding below.
    retrieval_embeddings = F.normalize(
        torch.stack(raw_by_expert).mean(dim=0), p=2, dim=1
    )
    return normalized_by_expert, retrieval_embeddings


def _retrieval_indices(support_embeddings, query_embeddings, max_k: int):
    support_mean = F.normalize(torch.stack(support_embeddings).mean(dim=0), p=2, dim=1)
    query_mean = F.normalize(torch.stack(query_embeddings).mean(dim=0), p=2, dim=1)
    effective_k = min(int(max_k), support_mean.shape[0])
    similarities = query_mean @ support_mean.T
    return torch.topk(similarities, effective_k, dim=1).indices


def _classification_metrics(model, support_embeddings, query_embeddings, support_labels, query_labels, neighbors, k):
    predictions = []
    effective_k = min(int(k), neighbors.shape[1])
    with torch.inference_mode():
        for query_index in range(neighbors.shape[0]):
            selected = neighbors[query_index, :effective_k]
            local_labels = support_labels[selected]
            expert_outputs = []
            classes = None
            truth = query_labels[query_index : query_index + 1]
            for expert_index, expert in enumerate(model.experts):
                logits, expert_classes, probabilities = expert.proto_head.forward_classification(
                    support_embeddings[expert_index][selected],
                    local_labels,
                    query_embeddings[expert_index][query_index : query_index + 1],
                    mode="soft_knn",
                )
                if logits is None:
                    continue
                classes = expert_classes
                learned_logit = (
                    expert.proto_head.classification_expert_confidence_logit(probabilities)
                    if expert.proto_head.classification_expert_confidence_enabled
                    else None
                )
                expert_outputs.append((logits, truth, probabilities, learned_logit))
            if not expert_outputs or classes is None:
                predictions.append(-100)
                continue
            combined, _, _ = model._aggregate_outputs(expert_outputs, "classification", truth)
            predictions.append(int(classes[combined.argmax(dim=1).item()].item()))
    truth_np = query_labels.detach().cpu().numpy()
    return {
        "n_queries": int(len(predictions)),
        "accuracy": float(accuracy_score(truth_np, np.asarray(predictions))),
    }


def _regression_metrics(model, support_embeddings, query_embeddings, support_labels, query_labels, neighbors, k):
    predictions_hours = []
    effective_k = min(int(k), neighbors.shape[1])
    with torch.inference_mode():
        for query_index in range(neighbors.shape[0]):
            selected = neighbors[query_index, :effective_k]
            local_labels = support_labels[selected]
            expert_outputs = []
            truth = query_labels[query_index : query_index + 1]
            for expert_index, expert in enumerate(model.experts):
                prediction, confidence, diagnostics = expert.proto_head.forward_regression(
                    support_embeddings[expert_index][selected],
                    local_labels,
                    query_embeddings[expert_index][query_index : query_index + 1],
                    return_diagnostics=True,
                )
                learned_logit = (
                    expert.proto_head.regression_expert_confidence_logit(
                        prediction, confidence, diagnostics
                    )
                    if expert.proto_head.regression_expert_confidence_enabled
                    else None
                )
                expert_outputs.append((prediction, truth, confidence, learned_logit))
            combined, _, _ = model._aggregate_outputs(expert_outputs, "regression", truth)
            hours = model.experts[0].proto_head.regression_output_to_hours(combined)
            predictions_hours.append(float(hours.item()))
    truth_hours = model.experts[0].proto_head.regression_output_to_hours(query_labels)
    return {
        "n_queries": int(len(predictions_hours)),
        "mae_hours": float(
            mean_absolute_error(truth_hours.detach().cpu().numpy(), predictions_hours)
        ),
    }


def _classification_batch(model, support_embeddings, query_embeddings, support_labels, neighbors):
    local_labels = support_labels[neighbors]
    num_classes = int(max(support_labels.max().item(), local_labels.max().item())) + 1
    expert_probabilities = []
    learned_logits = []
    for expert_index, expert in enumerate(model.experts):
        head = expert.proto_head
        support = support_embeddings[expert_index][neighbors]
        query = query_embeddings[expert_index]
        center = support.mean(dim=1, keepdim=True)
        centered_support = F.normalize(support - center, p=2, dim=2)
        centered_query = F.normalize(query - center.squeeze(1), p=2, dim=1)
        raw_similarities = torch.einsum("qd,qkd->qk", centered_query, centered_support)
        selection_logits = head.classification_selection_logits(
            centered_support, local_labels, centered_query,
            base_similarities=raw_similarities,
        )
        attention = F.softmax(
            raw_similarities * head.logit_scale.clamp(1.0, 20.0) + selection_logits,
            dim=1,
        )
        mass = attention.new_zeros((attention.shape[0], num_classes))
        mass.scatter_add_(1, local_labels, attention)
        counts = attention.new_zeros((attention.shape[0], num_classes))
        counts.scatter_add_(1, local_labels, torch.ones_like(attention))
        logits = torch.log(mass.clamp_min(1e-8)) + head.count_prior * torch.log(
            counts.clamp_min(1.0)
        )
        probabilities = F.softmax(logits, dim=1)
        expert_probabilities.append(probabilities)
        if head.classification_expert_confidence_enabled:
            learned_logits.append(
                head.classification_expert_confidence_logit(probabilities)
            )
    stacked = torch.stack(expert_probabilities)
    if len(learned_logits) == len(expert_probabilities):
        weights = F.softmax(torch.stack(learned_logits), dim=0).unsqueeze(-1)
        combined = (stacked * weights).sum(dim=0)
    else:
        combined = stacked.mean(dim=0)
    return combined.argmax(dim=1)


def _regression_batch(model, support_embeddings, query_embeddings, support_labels, neighbors):
    local_labels = support_labels[neighbors]
    predictions = []
    confidences = []
    learned_logits = []
    for expert_index, expert in enumerate(model.experts):
        head = expert.proto_head
        prediction, confidence, diagnostics = head.forward_regression_batched(
            support_embeddings[expert_index][neighbors],
            local_labels,
            query_embeddings[expert_index],
            return_diagnostics=True,
        )
        predictions.append(prediction)
        confidences.append(confidence)
        if head.regression_expert_confidence_enabled:
            learned_logits.append(
                head.regression_expert_confidence_logit(
                    prediction, confidence, diagnostics
                )
            )
    stacked_predictions = torch.stack(predictions)
    stacked_confidences = torch.stack(confidences)
    if len(learned_logits) == len(predictions):
        weights = F.softmax(torch.stack(learned_logits), dim=0)
    else:
        weights = stacked_confidences / stacked_confidences.sum(dim=0).clamp_min(1e-8)
    combined = (stacked_predictions * weights).sum(dim=0)
    return model.experts[0].proto_head.regression_output_to_hours(combined)


def _vectorized_metrics(
    model, task, support_embeddings, query_embeddings, support_labels, query_labels,
    support_retrieval_embeddings, query_retrieval_embeddings,
    retrieval_k, prediction_batch_size,
):
    outputs = {int(k): [] for k in retrieval_k}
    max_k = min(max(outputs), support_labels.numel())
    with torch.inference_mode():
        for start in range(0, query_labels.numel(), prediction_batch_size):
            stop = min(start + prediction_batch_size, query_labels.numel())
            similarities = (
                query_retrieval_embeddings[start:stop]
                @ support_retrieval_embeddings.T
            )
            neighbors = torch.topk(similarities, max_k, dim=1).indices
            query_batch_embeddings = [values[start:stop] for values in query_embeddings]
            for k in outputs:
                selected = neighbors[:, : min(k, neighbors.shape[1])]
                if task == "classification":
                    prediction = _classification_batch(
                        model, support_embeddings, query_batch_embeddings,
                        support_labels, selected,
                    )
                else:
                    prediction = _regression_batch(
                        model, support_embeddings, query_batch_embeddings,
                        support_labels, selected,
                    )
                outputs[k].append(prediction.detach().cpu())
    truth = query_labels.detach().cpu()
    result = {}
    for k, batches in outputs.items():
        prediction = torch.cat(batches)
        if task == "classification":
            result[k] = {
                "n_queries": int(prediction.numel()),
                "accuracy": float((prediction == truth).float().mean().item()),
            }
        else:
            truth_hours = model.experts[0].proto_head.regression_output_to_hours(
                query_labels
            ).detach().cpu()
            result[k] = {
                "n_queries": int(prediction.numel()),
                "mae_hours": float((prediction - truth_hours).abs().mean().item()),
            }
    return result


def _load_reference(path: Path):
    reference = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["log"], _fraction_label(row["fraction_percent"]))
            reference[key] = {
                "classification": float(row["classification_accuracy"]),
                "regression": float(row["regression_mae_hours"]),
            }
    return reference


def _write_results(output_dir: Path, rows, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fieldnames = [
        "checkpoint", "checkpoint_epoch", "log", "fraction_percent", "task",
        "retrieval_k", "effective_k", "support_prefixes", "n_queries", "accuracy",
        "mae_hours", "fmv2_proto_reference", "candidate_minus_reference", "beats_fmv2",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    selected = []
    group_keys = sorted(
        {(row["log"], row["fraction_percent"], row["task"]) for row in rows}
    )
    for log_name, fraction, task in group_keys:
        candidates = [
            row for row in rows
            if (row["log"], row["fraction_percent"], row["task"])
            == (log_name, fraction, task)
        ]
        if task == "classification":
            chosen = max(candidates, key=lambda row: (row["accuracy"], -row["retrieval_k"]))
        else:
            chosen = min(candidates, key=lambda row: (row["mae_hours"], row["retrieval_k"]))
        selected.append({**chosen, "selection_policy": "best metric; smallest k on ties"})
    selected_fields = fieldnames + ["selection_policy"]
    with (output_dir / "selected_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=selected_fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    parser.add_argument("--paper_repo", default="/tmp/experiments-fm")
    parser.add_argument("--reference", default="paper_docs/fmv2_new2_proto_reference.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logs", nargs="+", default=DEFAULT_LOGS)
    parser.add_argument("--fractions", nargs="+", default=list(FRACTION_CODES))
    parser.add_argument("--retrieval_k", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--prediction_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regression_mode_override",
        choices=("sqrt_knn", "raw_hours_knn"),
        default=None,
        help=(
            "Auditable inference-only ablation of the existing parameter-free "
            "regression aggregation space. The checkpoint default is unchanged "
            "when this option is omitted."
        ),
    )
    parser.add_argument(
        "--allow_legacy_head_missing",
        action="store_true",
        help="Load the public pre-FM-v3 checkpoint whose disabled head extensions are absent.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    saved_config = torch.load(checkpoint_dir / "training_config.pth", weights_only=False)
    CONFIG.clear()
    CONFIG.update(saved_config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader = init_loader(CONFIG)
    loader.load_training_artifacts(checkpoint_dir / "training_artifacts.pth")
    model = create_model(CONFIG, loader, device)
    if args.allow_legacy_head_missing:
        if args.checkpoint_epoch is None:
            raise ValueError("--allow_legacy_head_missing requires --checkpoint_epoch")
        checkpoint_path = checkpoint_dir / f"model_epoch_{args.checkpoint_epoch}.pth"
        incompatible = model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=False
        )
        allowed_fragments = (
            "proto_head._gamma_raw", "proto_head._kappa_raw", "proto_head.abstain_bias",
            "proto_head.abstain_slope", "proto_head.fixed_gate", "proto_head.gate_network.",
        )
        disallowed_missing = [
            key for key in incompatible.missing_keys
            if not any(fragment in key for fragment in allowed_fragments)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unsafe legacy mismatch: missing={disallowed_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        print(f"Loaded legacy checkpoint with {len(incompatible.missing_keys)} disabled head tensors initialized.")
    else:
        checkpoint_path = Path(
            load_model_weights(model, str(checkpoint_dir), device, epoch_num=args.checkpoint_epoch)
        )
    if args.regression_mode_override is not None:
        for expert in model.experts:
            expert.proto_head.regression_mode = args.regression_mode_override
    model.eval()

    paper_repo = Path(args.paper_repo).resolve()
    reference = _load_reference(Path(args.reference))
    fractions = [_fraction_label(value) for value in args.fractions]
    unknown = sorted(set(fractions) - set(FRACTION_CODES))
    if unknown:
        raise ValueError(f"Unknown fractions: {unknown}")
    max_k = max(args.retrieval_k)
    rows = []

    for log_index, log_name in enumerate(args.logs):
        activity_names = _activity_universe(paper_repo, log_name)
        test_path = paper_repo / "logs" / "test" / f"{log_name}.xes.gz"
        test_log = _transform_log(loader, test_path, "test", activity_names)
        all_query_tasks = {
            task: get_task_data(test_log, task, config=CONFIG)
            for task in ("classification", "regression")
        }
        stable_log_index = (
            DEFAULT_LOGS.index(log_name) if log_name in DEFAULT_LOGS else log_index
        )
        query_tasks = {
            task: _select_queries(
                tasks, args.num_queries, args.seed + stable_log_index * 1009
            )
            for task, tasks in all_query_tasks.items()
        }
        query_embeddings = {
            task: _encode(model.experts, tasks, task, args.embedding_batch_size)
            for task, tasks in query_tasks.items()
        }

        for fraction in fractions:
            support_path = (
                paper_repo / "logs" / "training" / FRACTION_CODES[fraction]
                / f"{log_name}.xes.gz"
            )
            support_log = _transform_log(loader, support_path, "support", activity_names)
            for task in ("classification", "regression"):
                support_tasks = get_task_data(support_log, task, config=CONFIG)
                support_embeddings = _encode(
                    model.experts, support_tasks, task, args.embedding_batch_size
                )
                support_labels = torch.as_tensor(
                    [item[1] for item in support_tasks],
                    dtype=torch.long if task == "classification" else torch.float32,
                    device=device,
                )
                selected_queries = query_tasks[task]
                query_labels = torch.as_tensor(
                    [item[1] for item in selected_queries],
                    dtype=torch.long if task == "classification" else torch.float32,
                    device=device,
                )
                metrics_by_k = _vectorized_metrics(
                    model, task, support_embeddings[0], query_embeddings[task][0],
                    support_labels, query_labels,
                    support_embeddings[1], query_embeddings[task][1],
                    args.retrieval_k, args.prediction_batch_size,
                )
                for k in args.retrieval_k:
                    metrics = metrics_by_k[int(k)]
                    if task == "classification":
                        metric_value = metrics["accuracy"]
                        baseline = reference[(log_name, fraction)][task]
                        beats = metric_value > baseline
                    else:
                        metric_value = metrics["mae_hours"]
                        baseline = reference[(log_name, fraction)][task]
                        beats = metric_value < baseline
                    row = {
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_epoch": int(checkpoint_path.stem.rsplit("_", 1)[1]),
                        "log": log_name,
                        "fraction_percent": float(fraction),
                        "task": task,
                        "retrieval_k": int(k),
                        "effective_k": min(int(k), len(support_tasks)),
                        "support_prefixes": len(support_tasks),
                        "fmv2_proto_reference": baseline,
                        "candidate_minus_reference": metric_value - baseline,
                        "beats_fmv2": bool(beats),
                        **metrics,
                    }
                    rows.append(row)
                    metric_name = "accuracy" if task == "classification" else "mae_hours"
                    print(
                        f"{log_name} {fraction}% {task} k={k}: "
                        f"{metric_name}={metric_value:.6g}, FM-v2={baseline:.6g}, beats={beats}"
                    )

    manifest = {
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_path.stem.rsplit("_", 1)[1]),
        "paper_repository": str(paper_repo),
        "paper_repository_commit": "5f4da9d79ac51beb98dda4b6e1fd6f94e3a4abf3",
        "reference": str(Path(args.reference).resolve()),
        "reference_sha256": _sha256(Path(args.reference).resolve()),
        "logs": args.logs,
        "fractions_percent": list(map(float, fractions)),
        "retrieval_k": args.retrieval_k,
        "overview_selection": "best task metric across retrieval_k; smallest k on ties",
        "num_queries_per_task_log": args.num_queries,
        "query_selection": "all held-out prefixes" if args.num_queries <= 0 else "deterministic sampled prefixes",
        "retrieval_representation": "L2(mean(raw expert embedding))",
        "expert_head_representation": "per-expert L2-normalized embedding",
        "neighborhood_sharing": "one model-level neighbor set shared by all experts",
        "checkpoint_regression_mode": (
            saved_config.get("fmv3_head", {}).get("regression_mode", "sqrt_knn")
        ),
        "evaluated_regression_mode": model.experts[0].proto_head.regression_mode,
        "regression_mode_override": args.regression_mode_override,
        "seed": args.seed,
        "model": {
            "num_experts": len(model.experts),
            "d_model": CONFIG["d_model"],
            "n_heads": CONFIG["n_heads"],
            "n_layers": CONFIG["n_layers"],
            "dropout": CONFIG["dropout"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    }
    _write_results(Path(args.output_dir), rows, manifest)
    print(f"Wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
