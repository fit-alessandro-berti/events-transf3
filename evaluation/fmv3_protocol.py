"""Case-level low-data evaluation protocol for FM-v2/FM-v3 checkpoints."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.fmv3_metrics import classification_metrics, regression_metrics
from time_transf import inverse_transform_time


def encode_tasks(expert, tasks, batch_size=128, task_type=None):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            sequences = [item[0] for item in tasks[start : start + batch_size]]
            embeddings.append(F.normalize(
                expert._process_batch(sequences, task_type=task_type), p=2, dim=1
            ).cpu())
    return torch.cat(embeddings) if embeddings else torch.empty((0, expert.d_model))


def fixed_case_split(tasks, seed, test_fraction=0.3, max_test_cases=50):
    cases = sorted({str(item[2]) for item in tasks})
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_test = max(1, int(math.ceil(len(cases) * test_fraction)))
    n_test = min(n_test, int(max_test_cases), max(1, len(cases) - 1))
    return set(cases[:n_test]), set(cases[n_test:])


def _labels_by_case(tasks, allowed_cases):
    result = defaultdict(set)
    for _, label, case_id in tasks:
        case = str(case_id)
        if case in allowed_cases and label is not None and int(label) != -100:
            result[case].add(int(label))
    return result


def support_case_order(tasks, allowed_cases, scenario, seed, max_cases=None):
    rng = random.Random(seed)
    cases = sorted(allowed_cases)
    rng.shuffle(cases)
    if scenario == "natural":
        return cases if max_cases is None else cases[: min(len(cases), int(max_cases))]
    if scenario != "class_aware":
        raise ValueError(f"Unknown support scenario: {scenario}")
    labels_by_case = _labels_by_case(tasks, allowed_cases)
    # Cases with the same label set always have the same greedy score. Grouping
    # them avoids rescanning very large support pools (e.g. ~70k Billing cases)
    # at every one of the first 128 selections.
    cases_by_signature = defaultdict(set)
    for case in cases:
        cases_by_signature[frozenset(labels_by_case.get(case, set()))].add(case)
    selected = []
    counts = Counter()
    target = len(cases) if max_cases is None else min(len(cases), int(max_cases))
    while cases_by_signature and len(selected) < target:
        best_score = None
        best_signatures = []
        for labels in cases_by_signature:
            uncovered = sum(counts[label] == 0 for label in labels)
            balance_gain = sum(1.0 / (1.0 + counts[label]) for label in labels)
            score = (uncovered, balance_gain, len(labels))
            if best_score is None or score > best_score:
                best_score, best_signatures = score, [labels]
            elif score == best_score:
                best_signatures.append(labels)
        best_cases = sorted(
            case for signature in best_signatures for case in cases_by_signature[signature]
        )
        chosen = rng.choice(best_cases)
        selected.append(chosen)
        for label in labels_by_case.get(chosen, set()):
            counts[label] += 1
        signature = frozenset(labels_by_case.get(chosen, set()))
        cases_by_signature[signature].remove(chosen)
        if not cases_by_signature[signature]:
            del cases_by_signature[signature]
    return selected


def _task_indices(tasks, cases):
    return np.asarray([idx for idx, item in enumerate(tasks) if str(item[2]) in cases], dtype=np.int64)


def _fixed_query_indices(tasks, test_cases, max_prefixes, seed):
    indices = _task_indices(tasks, test_cases)
    if len(indices) <= max_prefixes:
        return indices
    rng = np.random.default_rng(seed)
    # Stratify by case so long cases cannot crowd out short cases completely.
    by_case = defaultdict(list)
    for idx in indices:
        by_case[str(tasks[int(idx)][2])].append(int(idx))
    chosen = []
    case_cycle = sorted(by_case)
    while len(chosen) < max_prefixes and case_cycle:
        next_cycle = []
        for case in case_cycle:
            if by_case[case] and len(chosen) < max_prefixes:
                position = int(rng.integers(0, len(by_case[case])))
                chosen.append(by_case[case].pop(position))
            if by_case[case]:
                next_cycle.append(case)
        case_cycle = next_cycle
    return np.asarray(sorted(chosen), dtype=np.int64)


def _build_case_plan(planning_tasks, config):
    eval_cfg = config.get("fmv3_evaluation", {})
    seed = int(config.get("seed", 42))
    test_cases, support_cases = fixed_case_split(
        planning_tasks, seed, eval_cfg.get("test_case_fraction", 0.3), eval_cfg.get("max_test_cases", 50)
    )
    budgets = [int(value) for value in eval_cfg.get("case_budgets", [1, 2, 4, 8, 16, 32, 64, 128])]
    full_cap = int(eval_cfg.get("include_full_budget_max_cases", 1000))
    if eval_cfg.get("include_full_budget", True) and len(support_cases) <= full_cap:
        budgets.append(len(support_cases))
    budgets = sorted(set(min(value, len(support_cases)) for value in budgets if value > 0))
    max_budget = max(budgets)
    orders, used_support_cases = {}, set()
    for repetition in range(int(eval_cfg.get("repetitions", 5))):
        for scenario in eval_cfg.get("support_scenarios", ["natural", "class_aware"]):
            order = support_case_order(
                planning_tasks, support_cases, scenario, seed + repetition * 1009, max_cases=max_budget
            )
            orders[(repetition, scenario)] = order
            used_support_cases.update(order)
    materialized_cases = test_cases | used_support_cases
    plan = {
        "test_cases": test_cases,
        "support_cases": support_cases,
        "budgets": budgets,
        "orders": orders,
    }
    return plan, materialized_cases


def prepare_case_plan(transformed_log, config):
    """Plan splits/subsets from case label sets before materializing prefixes."""
    planning_tasks = []
    traces_by_case = {}
    for trace in transformed_log:
        if len(trace) < 3:
            continue
        case_id = str(trace[0]["case_id"])
        labels = {
            int(trace[index + 1]["activity_id"])
            for index in range(1, len(trace) - 1)
            if trace[index + 1]["activity_id"] is not None
            and int(trace[index + 1]["activity_id"]) != -100
        }
        if not labels:
            continue
        traces_by_case[case_id] = trace
        planning_tasks.extend((None, label, case_id) for label in labels)
    plan, materialized_cases = _build_case_plan(planning_tasks, config)
    return plan, [traces_by_case[case] for case in traces_by_case if case in materialized_cases]


def prepare_case_plan_from_dataframe(
    frame,
    config,
    case_key="case:concept:name",
    activity_key="concept:name",
    timestamp_key="time:timestamp",
):
    """Plan on compact dataframe columns and return only materialized case rows."""
    work = frame.copy()
    work[timestamp_key] = pd.to_datetime(work[timestamp_key], errors="coerce").dt.tz_localize(None)
    work = work.dropna(subset=[timestamp_key])
    activity_names = sorted(work[activity_key].dropna().unique().tolist())
    activity_to_id = {name: index for index, name in enumerate(activity_names)}
    work = work.sort_values([case_key, timestamp_key])
    work["_fmv3_activity_id"] = work[activity_key].map(activity_to_id)
    work["_fmv3_event_position"] = work.groupby(case_key, sort=False).cumcount()
    case_sizes = work.groupby(case_key, sort=False).size()
    eligible_cases = set(case_sizes[case_sizes >= 3].index.tolist())
    label_rows = work[
        (work["_fmv3_event_position"] >= 2)
        & work[case_key].isin(eligible_cases)
        & work["_fmv3_activity_id"].notna()
    ]
    signatures = label_rows.groupby(case_key, sort=False)["_fmv3_activity_id"].agg(
        lambda values: frozenset(map(int, values))
    )
    planning_tasks = []
    for case_id, labels in signatures.items():
        planning_tasks.extend((None, label, str(case_id)) for label in labels)
    plan, materialized_cases = _build_case_plan(planning_tasks, config)
    selected = work[work[case_key].astype(str).isin(materialized_cases)].copy()
    return plan, selected, activity_names


def _effective_k(similarities, initial_k, mode, eval_cfg):
    max_available = similarities.numel()
    if mode != "dynamic_expanded_local":
        return min(int(initial_k), max_available)
    max_k = min(int(eval_cfg.get("dynamic_retrieval_max_k", 200)), max_available)
    threshold = float(eval_cfg.get("dynamic_retrieval_entropy_threshold", 0.65))
    k = min(int(initial_k), max_k)
    while k < max_k:
        values = torch.topk(similarities, k).values
        probs = F.softmax(values, dim=0)
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum() / math.log(max(k, 2))).item())
        if entropy <= threshold:
            break
        k = min(max_k, k * 2)
    return k


def _align_probabilities(classes, probabilities, universe):
    aligned = np.zeros(len(universe), dtype=float)
    positions = {int(label): idx for idx, label in enumerate(universe)}
    for idx, label in enumerate(classes.tolist()):
        if int(label) in positions:
            aligned[positions[int(label)]] = float(probabilities[idx])
    return aligned


def _activity_context(task, max_order):
    """Return the recent log-local activity state for a prefix task."""
    prefix = task[0]
    return tuple(int(event["activity_id"]) for event in prefix[-int(max_order) :])


def _structured_class_probabilities(
    labels,
    contexts,
    query_indices,
    support_indices,
    class_universe,
    max_order=3,
    smoothing=0.5,
):
    """Balanced class-conditional n-gram likelihood with suffix backoff.

    The score estimates P(recent activity context | next activity) under a
    uniform next-activity prior. Longer suffixes are preferred when observed;
    an unseen state backs off to shorter suffixes and ultimately a uniform
    distribution over support-covered labels.
    """
    max_order = max(1, int(max_order))
    smoothing = max(float(smoothing), 1e-8)
    support_indices = np.asarray(support_indices, dtype=np.int64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    classes = list(map(int, class_universe))
    class_position = {label: position for position, label in enumerate(classes)}
    global_counts = Counter(int(labels_np[index]) for index in support_indices)
    tables = {order: defaultdict(Counter) for order in range(1, max_order + 1)}
    for index in support_indices:
        label = int(labels_np[index])
        context = contexts[int(index)]
        for order in tables:
            tables[order][context[-order:]][label] += 1

    probabilities = np.zeros((len(query_indices), len(classes)), dtype=np.float32)
    context_support = np.zeros(len(query_indices), dtype=np.float32)
    selected_orders = np.zeros(len(query_indices), dtype=np.int64)
    for row, index in enumerate(query_indices):
        query_context = contexts[int(index)]
        counts = None
        vocabulary_size = 1
        for order in range(max_order, 0, -1):
            candidate = tables[order].get(query_context[-order:])
            if candidate:
                counts = candidate
                vocabulary_size = max(1, len(tables[order]))
                context_support[row] = float(sum(candidate.values()))
                selected_orders[row] = order
                break

        scores = np.full(len(classes), -1e9, dtype=np.float64)
        for label, position in class_position.items():
            if global_counts.get(label, 0) <= 0:
                continue
            if counts is None:
                scores[position] = 0.0
            else:
                scores[position] = (
                    math.log(counts.get(label, 0) + smoothing)
                    - math.log(global_counts[label] + smoothing * vocabulary_size)
                )
        scores -= scores.max()
        row_probabilities = np.exp(scores)
        row_probabilities /= row_probabilities.sum().clip(min=1e-12)
        probabilities[row] = row_probabilities
    return probabilities, context_support, selected_orders, dict(global_counts)


def _structured_prediction(
    labels,
    contexts,
    query_indices,
    support_indices,
    class_universe,
    max_order,
    smoothing,
):
    probabilities, context_support, selected_orders, support_counts = (
        _structured_class_probabilities(
            labels, contexts, query_indices, support_indices, class_universe,
            max_order=max_order, smoothing=smoothing,
        )
    )
    universe = np.asarray(class_universe, dtype=np.int64)
    predicted = universe[np.argmax(probabilities, axis=1)]
    true = np.asarray([int(labels[int(index)]) for index in query_indices], dtype=np.int64)
    pool_covered = np.asarray([support_counts.get(int(label), 0) > 0 for label in true])
    return {
        "y_true": true.tolist(),
        "y_pred": predicted.tolist(),
        "probabilities": probabilities,
        "confidences": probabilities.max(axis=1).tolist(),
        "pool_covered": pool_covered.tolist(),
        "retrieval_covered": pool_covered.tolist(),
        "support_counts": support_counts,
        "structured_context_support": context_support,
        "structured_selected_order": selected_orders,
    }


def _fuse_structured_prediction(base, structured, class_universe, weight, tau, fusion):
    base_probabilities = np.asarray(base["probabilities"], dtype=np.float64)
    structured_probabilities = np.asarray(structured["probabilities"], dtype=np.float64)
    base_probabilities /= base_probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    structured_probabilities /= structured_probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    support = np.asarray(structured["structured_context_support"], dtype=np.float64)
    tau = max(float(tau), 0.0)
    reliability = np.ones_like(support) if tau == 0.0 else support / (support + tau)
    effective_weight = np.clip(float(weight) * reliability, 0.0, 1.0)[:, None]
    effective_weight[support <= 0] = 0.0
    if fusion == "mixture":
        probabilities = (
            (1.0 - effective_weight) * base_probabilities
            + effective_weight * structured_probabilities
        )
    elif fusion in {"product", "log_linear"}:
        logits = (
            (1.0 - effective_weight) * np.log(base_probabilities.clip(min=1e-12))
            + effective_weight * np.log(structured_probabilities.clip(min=1e-12))
        )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
    else:
        raise ValueError(f"Unknown structured fusion: {fusion}")
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    universe = np.asarray(class_universe, dtype=np.int64)
    fused = dict(base)
    fused["probabilities"] = probabilities
    fused["y_pred"] = universe[np.argmax(probabilities, axis=1)].tolist()
    fused["confidences"] = probabilities.max(axis=1).tolist()
    fused["structured_context_support"] = support.tolist()
    if "structured_selected_order" in structured:
        fused["structured_selected_order"] = np.asarray(
            structured["structured_selected_order"], dtype=np.int64
        ).tolist()
    fused["structured_effective_weight"] = effective_weight[:, 0].tolist()
    return fused


def _batched_head_probabilities(
    head,
    query,
    pool,
    pool_labels,
    local_positions,
    prediction_classes,
    retrieval_mode,
    prior_mode,
    prior_strength,
):
    """Vectorized equivalent of per-query fixed-k classification heads."""
    device = query.device
    classes = torch.as_tensor(prediction_classes, dtype=pool_labels.dtype, device=device)
    num_queries, num_classes = query.size(0), classes.numel()
    local = pool[local_positions]
    local_labels = pool_labels[local_positions]
    class_masks = torch.stack([local_labels == cls for cls in classes], dim=2)
    local_counts = class_masks.sum(dim=1).float()
    pool_counts = torch.stack([(pool_labels == cls).sum() for cls in classes]).float()

    selected_mode = head.classification_mode if retrieval_mode == "configured" else retrieval_mode
    selected_mode = str(selected_mode).lower()
    selected_mode = {"soft_knn": "legacy_soft_knn", "proto": "global", "global_proto": "global"}.get(
        selected_mode, selected_mode
    )
    if selected_mode == "foundation_knn":
        probabilities = local_counts / local_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        return probabilities, probabilities.new_zeros(num_queries)

    query = F.normalize(query, p=2, dim=1)
    pool = F.normalize(pool, p=2, dim=1)
    local = pool[local_positions]
    if selected_mode == "legacy_soft_knn":
        center = local.mean(dim=1, keepdim=True)
        centered_local = F.normalize(local - center, p=2, dim=2)
        centered_query = F.normalize(query - center.squeeze(1), p=2, dim=1)
        similarities = torch.einsum("qd,qkd->qk", centered_query, centered_local)
        attention = F.softmax(similarities * head.logit_scale.clamp(1.0, 20.0), dim=1)
        mass = attention.new_zeros((num_queries, num_classes))
        mass.scatter_add_(1, class_masks.long().argmax(dim=2), attention)
        logits = torch.log(mass.clamp_min(1e-8)) + head.count_prior * torch.log(local_counts.clamp_min(1.0))
        logits = logits.masked_fill(local_counts == 0, -torch.inf)
        return F.softmax(logits, dim=1), logits.new_zeros(num_queries)

    local_query = query
    if head.local_centering:
        local_center = local.mean(dim=1, keepdim=True)
        local = F.normalize(local - local_center, p=2, dim=2)
        local_query = F.normalize(query - local_center.squeeze(1), p=2, dim=1)
    local_similarities = torch.einsum("qd,qkd->qk", local_query, local) * head.local_scale
    local_evidence = query.new_full((num_queries, num_classes), -torch.inf)
    gamma = head.evidence_gamma
    for column in range(num_classes):
        mask = class_masks[:, :, column]
        values = torch.logsumexp(local_similarities.masked_fill(~mask, -torch.inf), dim=1)
        present = local_counts[:, column] > 0
        local_evidence[:, column] = torch.where(
            present,
            values - gamma * torch.log(local_counts[:, column].clamp_min(1.0)),
            values,
        )

    task_prior = pool.mean(dim=0)
    global_query = query
    if head.global_centering:
        pool = F.normalize(pool - task_prior.unsqueeze(0), p=2, dim=1)
        global_query = F.normalize(query - task_prior.unsqueeze(0), p=2, dim=1)
        task_prior = torch.zeros_like(task_prior)
    prototypes = []
    pool_present = pool_counts > 0
    for column, cls in enumerate(classes):
        members = pool[pool_labels == cls]
        if members.numel() == 0:
            prototypes.append(task_prior)
            continue
        mean = members.mean(dim=0)
        if head.shrinkage_mode in {"fixed", "learned"} and not head.global_centering:
            weight = pool_counts[column] / (pool_counts[column] + head.shrinkage_kappa)
            mean = weight * mean + (1.0 - weight) * task_prior
        prototypes.append(mean)
    prototypes = F.normalize(torch.stack(prototypes), p=2, dim=1)
    global_evidence = global_query @ prototypes.t() * head.global_scale
    global_evidence = global_evidence.masked_fill(~pool_present.unsqueeze(0), -torch.inf)

    fallback_has_missing = None
    if selected_mode == "local":
        evidence = local_evidence
    elif selected_mode == "global":
        evidence = global_evidence
    elif selected_mode == "global_local":
        if head.gate_mode == "dynamic":
            safe_local = torch.where(torch.isfinite(local_evidence), local_evidence, torch.zeros_like(local_evidence))
            retrieval_probs = F.softmax(local_similarities, dim=1)
            entropy = -(retrieval_probs * F.log_softmax(local_similarities, dim=1)).sum(dim=1)
            entropy = entropy / math.log(max(local_similarities.size(1), 2))
            features = torch.stack([
                torch.tanh(safe_local),
                torch.log1p(local_counts),
                torch.log1p(pool_counts).unsqueeze(0).expand(num_queries, -1),
                entropy.unsqueeze(1).expand(-1, num_classes),
                torch.tanh(safe_local - global_evidence),
            ], dim=-1)
            gate = torch.sigmoid(head.gate_network(features).squeeze(-1))
            gate = torch.where(local_counts > 0, gate, torch.zeros_like(gate))
        else:
            gate = torch.full_like(local_evidence, float(head.fixed_gate.item()))
            gate = torch.where(local_counts > 0, gate, torch.zeros_like(gate))
        evidence = torch.logaddexp(
            local_evidence + torch.log(gate.clamp_min(1e-8)),
            global_evidence + torch.log1p(-gate.clamp(max=1.0 - 1e-8)),
        )
    elif selected_mode == "coverage_fallback":
        local_present = local_counts > 0
        valid_local = local_evidence.masked_fill(~local_present, -torch.inf)
        best_local = valid_local.max(dim=1, keepdim=True).values
        best_present_global = global_evidence.masked_fill(~local_present, -torch.inf).max(
            dim=1, keepdim=True
        ).values
        missing_logits = (
            best_local
            + global_evidence
            - best_present_global
            - head.coverage_fallback_margin
        )
        missing_candidates = (~local_present) & pool_present.unsqueeze(0)
        fallback_has_missing = missing_candidates.any(dim=1)
        evidence = torch.where(missing_candidates, missing_logits, valid_local)
    else:
        raise ValueError(f"Unsupported batched classification mode: {selected_mode}")

    selected_prior = str(prior_mode or head.prior_mode)
    beta = head.prior_strength if prior_strength is None else float(prior_strength)
    if selected_prior in {"none", "uniform", "balanced"} or beta == 0.0:
        prior_logits = evidence.new_zeros((num_queries, num_classes))
    elif selected_prior in {"natural", "empirical"}:
        candidate_mask = local_counts > 0 if selected_mode == "local" else pool_present.unsqueeze(0).expand(num_queries, -1)
        smoothed = pool_counts.unsqueeze(0).expand(num_queries, -1) + head.prior_smoothing
        denominator = (smoothed * candidate_mask).sum(dim=1, keepdim=True).clamp_min(1e-8)
        prior_logits = beta * torch.log((smoothed / denominator).clamp_min(1e-8))
    else:
        raise ValueError(f"Unknown prior mode: {selected_prior}")
    logits = evidence + prior_logits
    temperature = logits.new_full((num_queries, 1), head.inference_temperature)
    if fallback_has_missing is not None:
        fallback_temperature = temperature.new_full(
            temperature.shape, head.fallback_inference_temperature
        )
        temperature = torch.where(
            fallback_has_missing.unsqueeze(1), fallback_temperature, temperature
        )
    abstain = logits.new_zeros(num_queries)
    if head.enable_abstention:
        abstain_logits = head.abstain_bias - F.softplus(head.abstain_slope) * evidence.max(dim=1).values
        all_logits = torch.cat([logits, abstain_logits.unsqueeze(1)], dim=1)
        all_probabilities = F.softmax(all_logits / temperature, dim=1)
        abstain = all_probabilities[:, -1]
        probabilities = all_probabilities[:, :-1]
    else:
        probabilities = F.softmax(logits / temperature, dim=1)
    return probabilities, abstain


@torch.no_grad()
def _predict_classification_fixed_k(
    experts,
    embeddings_by_expert,
    labels,
    query_indices,
    support_indices,
    class_universe,
    retrieval_k,
    retrieval_mode,
    prior_mode,
    prior_strength,
):
    support_labels_cpu = labels[support_indices]
    support_counts = Counter(int(value) for value in support_labels_cpu.tolist())
    prediction_classes = sorted(set(map(int, class_universe)) | set(support_counts))
    metric_columns = [prediction_classes.index(int(label)) for label in class_universe]
    device = next(experts[0].parameters()).device
    support_indices_device = torch.as_tensor(support_indices, device=device)
    query_indices_device = torch.as_tensor(query_indices, device=device)
    labels_device = labels.to(device)
    embeddings_device = [embedding.to(device) for embedding in embeddings_by_expert]
    k_eff = min(int(retrieval_k), len(support_indices))

    expert_probabilities, abstain_probabilities = [], []
    for expert, embeddings in zip(experts, embeddings_device):
        expert_queries = embeddings[query_indices_device]
        expert_pool = embeddings[support_indices_device]
        local_positions = torch.topk(expert_queries @ expert_pool.t(), k_eff, dim=1).indices
        probabilities, abstain = _batched_head_probabilities(
            expert.proto_head,
            expert_queries,
            expert_pool,
            labels_device[support_indices_device],
            local_positions,
            prediction_classes,
            retrieval_mode,
            prior_mode,
            prior_strength,
        )
        expert_probabilities.append(probabilities)
        abstain_probabilities.append(abstain)
    reference_queries = embeddings_device[0][query_indices_device]
    reference_pool = embeddings_device[0][support_indices_device]
    reference_positions = torch.topk(reference_queries @ reference_pool.t(), k_eff, dim=1).indices
    local_labels = labels_device[support_indices_device][reference_positions]
    mean_probs = torch.stack(expert_probabilities).mean(dim=0)
    mean_abstain = torch.stack(abstain_probabilities).mean(dim=0)
    best_conf, best_idx = mean_probs.max(dim=1)
    predicted = torch.as_tensor(prediction_classes, device=device)[best_idx]
    abstain_mask = mean_abstain > best_conf
    predicted = torch.where(
        abstain_mask,
        predicted.new_full(predicted.shape, int(experts[0].proto_head.abstain_label)),
        predicted,
    )
    confidence = torch.where(abstain_mask, mean_abstain, best_conf)
    true = labels_device[query_indices_device]
    pool_covered = torch.as_tensor(
        [support_counts.get(int(value), 0) > 0 for value in true.tolist()], device=device
    )
    retrieval_covered = (local_labels == true.unsqueeze(1)).any(dim=1)
    return {
        "y_true": true.cpu().tolist(),
        "y_pred": predicted.cpu().tolist(),
        "probabilities": mean_probs[:, metric_columns].cpu().numpy(),
        "confidences": confidence.cpu().tolist(),
        "pool_covered": pool_covered.cpu().tolist(),
        "retrieval_covered": retrieval_covered.cpu().tolist(),
        "support_counts": dict(support_counts),
    }


@torch.no_grad()
def _predict_classification_dynamic_batched(
    experts,
    embeddings_by_expert,
    labels,
    query_indices,
    support_indices,
    class_universe,
    retrieval_k,
    prior_mode,
    prior_strength,
    eval_cfg,
):
    """Preserve per-query entropy expansion, then batch queries sharing k."""
    device = next(experts[0].parameters()).device
    reference = embeddings_by_expert[0].to(device)
    query_device = torch.as_tensor(query_indices, device=device)
    support_device = torch.as_tensor(support_indices, device=device)
    similarities = reference[query_device] @ reference[support_device].t()
    groups = defaultdict(list)
    for position in range(len(query_indices)):
        effective = _effective_k(similarities[position], retrieval_k, "dynamic_expanded_local", eval_cfg)
        groups[int(effective)].append(position)
    merged = {
        "y_true": [None] * len(query_indices),
        "y_pred": [None] * len(query_indices),
        "probabilities": [None] * len(query_indices),
        "confidences": [None] * len(query_indices),
        "pool_covered": [None] * len(query_indices),
        "retrieval_covered": [None] * len(query_indices),
    }
    support_counts = None
    for effective, positions in groups.items():
        subset = np.asarray([query_indices[position] for position in positions], dtype=np.int64)
        result = _predict_classification_fixed_k(
            experts, embeddings_by_expert, labels, subset, support_indices, class_universe,
            effective, "configured", prior_mode, prior_strength,
        )
        support_counts = result["support_counts"]
        for local_position, output_position in enumerate(positions):
            for key in merged:
                merged[key][output_position] = result[key][local_position]
    merged["probabilities"] = np.asarray(merged["probabilities"])
    merged["support_counts"] = support_counts or {}
    return merged


@torch.no_grad()
def predict_classification(
    experts,
    embeddings_by_expert,
    labels,
    query_indices,
    support_indices,
    class_universe,
    retrieval_k,
    retrieval_mode,
    prior_mode,
    prior_strength,
    eval_cfg,
    structured_contexts=None,
):
    structured_modes = {"structured", "fm_structured_mix", "fm_structured_product"}
    if retrieval_mode in structured_modes:
        if structured_contexts is None:
            raise ValueError(f"{retrieval_mode} requires structured prefix contexts")
        max_order = int(eval_cfg.get("structured_max_order", 3))
        smoothing = float(eval_cfg.get("structured_smoothing", 0.5))
        structured = _structured_prediction(
            labels, structured_contexts, query_indices, support_indices,
            class_universe, max_order, smoothing,
        )
        if retrieval_mode == "structured":
            return structured
        base = _predict_classification_fixed_k(
            experts, embeddings_by_expert, labels, query_indices, support_indices,
            class_universe, retrieval_k, "configured", prior_mode, prior_strength,
        )
        fusion = "mixture" if retrieval_mode == "fm_structured_mix" else "product"
        return _fuse_structured_prediction(
            base,
            structured,
            class_universe,
            float(eval_cfg.get("structured_weight", 0.5)),
            float(eval_cfg.get("structured_tau", 2.0)),
            fusion,
        )
    if retrieval_mode == "dynamic_expanded_local":
        return _predict_classification_dynamic_batched(
            experts, embeddings_by_expert, labels, query_indices, support_indices,
            class_universe, retrieval_k, prior_mode, prior_strength, eval_cfg,
        )
    else:
        return _predict_classification_fixed_k(
            experts, embeddings_by_expert, labels, query_indices, support_indices,
            class_universe, retrieval_k, retrieval_mode, prior_mode, prior_strength,
        )
    support_labels_cpu = labels[support_indices]
    support_counts = Counter(int(value) for value in support_labels_cpu.tolist())
    prediction_classes = sorted(set(map(int, class_universe)) | set(support_counts))
    metric_columns = [prediction_classes.index(int(label)) for label in class_universe]
    y_true, y_pred, probs_out, confidence, pool_cov, retrieval_cov = [], [], [], [], [], []
    device = next(experts[0].parameters()).device
    labels_device = labels.to(device)
    support_indices_device = torch.as_tensor(support_indices, device=device)
    embeddings_device = [embedding.to(device) for embedding in embeddings_by_expert]

    for query_idx in query_indices:
        expert_probabilities = []
        reference_embeddings = embeddings_device[0]
        reference_query = reference_embeddings[int(query_idx) : int(query_idx) + 1]
        reference_pool = reference_embeddings[support_indices_device]
        reference_similarities = (reference_query @ reference_pool.t()).view(-1)
        k_eff = _effective_k(reference_similarities, retrieval_k, retrieval_mode, eval_cfg)
        local_positions = torch.topk(reference_similarities, k_eff).indices
        first_local_labels = labels_device[support_indices_device[local_positions]].detach().cpu()
        classes_ref = None
        abstain_probabilities = []
        for expert, embeddings in zip(experts, embeddings_device):
            query = embeddings[int(query_idx) : int(query_idx) + 1]
            pool = embeddings[support_indices_device]
            local_features = pool[local_positions]
            local_labels = labels_device[support_indices_device[local_positions]]

            if retrieval_mode == "foundation_knn":
                classes = torch.unique(local_labels, sorted=True)
                counts = torch.stack([(local_labels == cls).sum() for cls in classes]).float()
                probabilities = counts / counts.sum()
            else:
                override_mode = None if retrieval_mode in {"configured", "dynamic_expanded_local"} else retrieval_mode
                _, classes, all_probs = expert.proto_head.forward_classification(
                    local_features,
                    local_labels,
                    query,
                    mode=override_mode,
                    global_support_features=pool,
                    global_support_labels=labels_device[support_indices_device],
                    prior_mode=prior_mode,
                    prior_strength=prior_strength,
                )
                if classes is None:
                    continue
                probabilities = all_probs[0]
                abstain_match = (classes == expert.proto_head.abstain_label).nonzero(as_tuple=False)
                abstain_probabilities.append(
                    float(probabilities[abstain_match[0, 0]]) if abstain_match.numel() else 0.0
                )
            classes_ref = classes
            expert_probabilities.append(_align_probabilities(classes, probabilities, prediction_classes))

        if not expert_probabilities:
            continue
        mean_probs = np.mean(expert_probabilities, axis=0)
        true_label = int(labels[int(query_idx)])
        abstain_probability = float(np.mean(abstain_probabilities)) if abstain_probabilities else 0.0
        best_idx = int(np.argmax(mean_probs))
        predicted = int(prediction_classes[best_idx])
        best_conf = float(mean_probs[best_idx])
        if abstain_probability > best_conf:
            predicted = int(experts[0].proto_head.abstain_label)
            best_conf = abstain_probability
        y_true.append(true_label)
        y_pred.append(predicted)
        probs_out.append(mean_probs[metric_columns])
        confidence.append(best_conf)
        pool_cov.append(support_counts.get(true_label, 0) > 0)
        retrieval_cov.append(bool(first_local_labels is not None and (first_local_labels == true_label).any()))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probs_out,
        "confidences": confidence,
        "pool_covered": pool_cov,
        "retrieval_covered": retrieval_cov,
        "support_counts": dict(support_counts),
    }


@torch.no_grad()
def _calibrate_time_transform_weights(
    experts, embeddings_device, labels_device, support_indices_device,
    support_case_ids, retrieval_k, eval_cfg,
):
    """Fit a convex branch prior using labeled support prefixes only."""
    num_branches = experts[0].proto_head.time_transform_bank.num_transforms
    default = torch.stack([
        expert.proto_head.time_transform_bank.aggregation_weights
        for expert in experts
    ]).mean(dim=0)
    max_prefixes = int(eval_cfg.get("regression_calibration_max_prefixes", 512))
    if support_indices_device.numel() < 2 or max_prefixes <= 0:
        return default, {"calibration_prefixes": 0}

    count = min(max_prefixes, int(support_indices_device.numel()))
    chosen_positions = torch.linspace(
        0, support_indices_device.numel() - 1, count,
        device=support_indices_device.device,
    ).round().long().unique()
    calibration_indices = support_indices_device[chosen_positions]
    pool_case_ids = np.asarray(support_case_ids, dtype=object)
    sampled_case_ids = pool_case_ids[chosen_positions.cpu().numpy()]
    exclude_same_case = bool(eval_cfg.get("regression_calibration_exclude_same_case", False))
    expert_branches = []
    valid_mask = None
    for expert, embeddings in zip(experts, embeddings_device):
        query = F.normalize(embeddings[calibration_indices], p=2, dim=1)
        pool = F.normalize(embeddings[support_indices_device], p=2, dim=1)
        similarities = query @ pool.t()
        if exclude_same_case:
            mask = torch.as_tensor(
                sampled_case_ids[:, None] == pool_case_ids[None, :],
                device=similarities.device,
            )
        else:
            mask = calibration_indices[:, None] == support_indices_device[None, :]
        similarities = similarities.masked_fill(mask, -torch.inf)
        eligible = (~mask).sum(dim=1)
        current_valid = eligible > 0
        if not current_valid.any():
            return default, {"calibration_prefixes": 0}
        valid_mask = current_valid if valid_mask is None else (valid_mask & current_valid)
        k_eff = min(int(retrieval_k), max(1, int(eligible[current_valid].min().item())))
        positions = torch.topk(similarities, k_eff, dim=1).indices
        local = pool[positions]
        local_targets = labels_device[support_indices_device][positions]
        _, _, diagnostics = expert.proto_head.forward_regression_batched(
            local, local_targets, query, return_diagnostics=True
        )
        expert_branches.append(diagnostics["branch_predictions_hours"])
    if valid_mask is None or not valid_mask.any():
        return default, {"calibration_prefixes": 0}

    branches = torch.stack(expert_branches).mean(dim=0)[:, valid_mask]
    targets = labels_device[calibration_indices][valid_mask].clamp_min(0.0).square()
    errors = branches - targets.unsqueeze(0)
    mae = errors.abs().mean(dim=1)
    rmse = torch.sqrt(errors.square().mean(dim=1).clamp_min(1e-8))
    mae_weight = float(eval_cfg.get("regression_calibration_mae_weight", 0.5))
    rmse_weight = float(eval_cfg.get("regression_calibration_rmse_weight", 0.5))
    denominator = max(mae_weight + rmse_weight, 1e-8)
    relative_score = (
        mae_weight * mae / mae.min().clamp_min(1e-8)
        + rmse_weight * rmse / rmse.min().clamp_min(1e-8)
    ) / denominator
    temperature = float(eval_cfg.get("regression_calibration_temperature", 20.0))
    weights = F.softmax(-temperature * (relative_score - relative_score.min()), dim=0)
    diagnostics = {
        "calibration_prefixes": int(valid_mask.sum().item()),
        "calibration_branch_mae_hours": mae.cpu().tolist(),
        "calibration_branch_rmse_hours": rmse.cpu().tolist(),
        "calibration_weights": weights.cpu().tolist(),
    }
    return weights, diagnostics


@torch.no_grad()
def predict_regression(
    experts, embeddings_by_expert, labels, query_indices, support_indices,
    retrieval_k, support_case_ids=None, eval_cfg=None,
):
    device = next(experts[0].parameters()).device
    labels_device = labels.to(device).float()
    support_indices_device = torch.as_tensor(support_indices, device=device)
    query_indices_device = torch.as_tensor(query_indices, device=device)
    embeddings_device = [embedding.to(device) for embedding in embeddings_by_expert]
    eval_cfg = eval_cfg or {}
    calibrated_weights = None
    calibration_diagnostics = {}
    if (
        experts[0].proto_head.regression_uses_time_transform_bank
        and eval_cfg.get("regression_support_calibration", False)
        and support_case_ids is not None
    ):
        calibrated_weights, calibration_diagnostics = _calibrate_time_transform_weights(
            experts, embeddings_device, labels_device, support_indices_device,
            support_case_ids, retrieval_k, eval_cfg,
        )
    k_eff = min(int(retrieval_k), len(support_indices))
    expert_predictions = []
    expert_stds = []
    expert_branch_predictions = []
    expert_aggregation_weights = []
    for expert, embeddings in zip(experts, embeddings_device):
        query = F.normalize(embeddings[query_indices_device], p=2, dim=1)
        pool = F.normalize(embeddings[support_indices_device], p=2, dim=1)
        positions = torch.topk(query @ pool.t(), k_eff, dim=1).indices
        local = pool[positions]
        local_targets = labels_device[support_indices_device][positions]
        prediction, _, diagnostics = expert.proto_head.forward_regression_batched(
            local, local_targets, query, return_diagnostics=True
        )
        expert_predictions.append(prediction)
        if expert.proto_head.regression_outputs_hours:
            expert_stds.append(diagnostics["std_hours"])
            if "branch_predictions_hours" in diagnostics:
                expert_branch_predictions.append(diagnostics["branch_predictions_hours"])
                expert_aggregation_weights.append(diagnostics["aggregation_weights"])
    reference = F.normalize(embeddings_device[0], p=2, dim=1)
    reference_positions = torch.topk(
        reference[query_indices_device] @ reference[support_indices_device].t(), k_eff, dim=1
    ).indices
    neighbor_targets = labels_device[support_indices_device][reference_positions]
    mean_prediction = torch.stack(expert_predictions).mean(dim=0)
    transformed_truth = labels_device[query_indices_device].cpu().numpy()
    truths = inverse_transform_time(transformed_truth)
    if experts[0].proto_head.regression_outputs_hours:
        if calibrated_weights is not None:
            mean_branches = torch.stack(expert_branch_predictions).mean(dim=0)
            calibrated_prediction = (
                calibrated_weights[:, None] * mean_branches
            ).sum(dim=0)
            calibration_mix = min(
                1.0,
                max(0.0, float(eval_cfg.get("regression_calibration_mix", 1.0))),
            )
            mean_prediction = (
                (1.0 - calibration_mix) * mean_prediction
                + calibration_mix * calibrated_prediction
            )
            calibration_diagnostics["calibration_mix"] = calibration_mix
        predictions = mean_prediction.cpu().numpy()
        stacked_predictions = torch.stack(expert_predictions)
        within_variance = torch.stack(expert_stds).square().mean(dim=0)
        between_variance = stacked_predictions.var(dim=0, correction=0)
        std_hours = torch.sqrt((within_variance + between_variance).clamp_min(1e-8)).cpu().numpy()
        lower = np.maximum(0.0, predictions - 1.645 * std_hours)
        upper = predictions + 1.645 * std_hours
    else:
        transformed_prediction = mean_prediction.cpu().numpy()
        transformed_std = neighbor_targets.std(dim=1, correction=0).cpu().numpy()
        predictions = inverse_transform_time(transformed_prediction)
        lower = inverse_transform_time(np.maximum(0.0, transformed_prediction - 1.645 * transformed_std))
        upper = inverse_transform_time(transformed_prediction + 1.645 * transformed_std)
    branch_diagnostics = None
    if expert_branch_predictions:
        mean_branches = torch.stack(expert_branch_predictions).mean(dim=0).cpu().numpy()
        truth_array = np.asarray(truths, dtype=float)
        branch_diagnostics = {
            "branch_mae_hours": np.mean(
                np.abs(mean_branches - truth_array[None, :]), axis=1
            ).tolist(),
            "branch_rmse_hours": np.sqrt(
                np.mean((mean_branches - truth_array[None, :]) ** 2, axis=1)
            ).tolist(),
            "mean_aggregation_weights": torch.stack(expert_aggregation_weights)
            .mean(dim=(0, 2))
            .cpu()
            .tolist(),
            "oracle_branch_mae_hours": float(
                np.min(np.abs(mean_branches - truth_array[None, :]), axis=0).mean()
            ),
            **calibration_diagnostics,
        }
    return truths.tolist(), predictions.tolist(), lower.tolist(), upper.tolist(), branch_diagnostics


def _bootstrap_balanced_accuracy(prediction_data, tasks, query_indices, universe, repetitions, seed):
    by_case = defaultdict(list)
    for pos, task_idx in enumerate(query_indices[: len(prediction_data["y_true"])]):
        by_case[str(tasks[int(task_idx)][2])].append(pos)
    if not by_case or repetitions <= 0:
        return None
    rng = np.random.default_rng(seed)
    cases = list(by_case)
    estimates = []
    true = np.asarray(prediction_data["y_true"])
    pred = np.asarray(prediction_data["y_pred"])
    for _ in range(repetitions):
        sampled = rng.choice(cases, len(cases), replace=True)
        positions = np.concatenate([np.asarray(by_case[str(case)], dtype=int) for case in sampled])
        recalls = []
        for label in universe:
            mask = true[positions] == label
            recalls.append(float((pred[positions][mask] == label).mean()) if mask.any() else 0.0)
        estimates.append(float(np.mean(recalls)))
    return {"lower": float(np.percentile(estimates, 2.5)), "upper": float(np.percentile(estimates, 97.5))}


def evaluate_log(model, test_tasks, log_name, config, output_jsonl: Path, case_plan=None):
    eval_cfg = config.get("fmv3_evaluation", {})
    requested_tasks = set(eval_cfg.get("tasks", ["classification", "regression"]))
    unknown_tasks = requested_tasks - {"classification", "regression"}
    if unknown_tasks:
        raise ValueError(f"Unknown FM-v3 evaluation tasks: {sorted(unknown_tasks)}")
    seed = int(config.get("seed", 42))
    experts = list(model.experts) if hasattr(model, "experts") else [model]
    all_class_tasks = test_tasks["classification"]
    all_reg_tasks = test_tasks["regression"]
    if case_plan is None:
        test_cases, support_cases = fixed_case_split(
            all_class_tasks, seed, eval_cfg.get("test_case_fraction", 0.3), eval_cfg.get("max_test_cases", 50)
        )
        budgets = [int(value) for value in eval_cfg.get("case_budgets", [1, 2, 4, 8, 16, 32, 64, 128])]
        full_cap = int(eval_cfg.get("include_full_budget_max_cases", 1000))
        if eval_cfg.get("include_full_budget", True) and len(support_cases) <= full_cap:
            budgets.append(len(support_cases))
        budgets = sorted(set(min(value, len(support_cases)) for value in budgets if value > 0))
        max_budget = max(budgets)
        orders = {}
        used_support_cases = set()
        for repetition in range(int(eval_cfg.get("repetitions", 5))):
            for scenario in eval_cfg.get("support_scenarios", ["natural", "class_aware"]):
                order = support_case_order(
                    all_class_tasks, support_cases, scenario, seed + repetition * 1009, max_cases=max_budget
                )
                orders[(repetition, scenario)] = order
                used_support_cases.update(order)
    else:
        test_cases = case_plan["test_cases"]
        support_cases = case_plan["support_cases"]
        budgets = case_plan["budgets"]
        orders = case_plan["orders"]
        used_support_cases = {case for order in orders.values() for case in order}

    # The huge Billing log has 273k prefixes. Embed only fixed queries and cases
    # that can occur in at least one nested support subset.
    materialized_cases = test_cases | used_support_cases
    class_tasks = [task for task in all_class_tasks if str(task[2]) in materialized_cases]
    reg_tasks = [task for task in all_reg_tasks if str(task[2]) in materialized_cases]
    max_queries = int(eval_cfg.get("max_query_prefixes", 1000))
    class_query_indices = _fixed_query_indices(class_tasks, test_cases, max_queries, seed)
    reg_query_indices = _fixed_query_indices(reg_tasks, test_cases, max_queries, seed)
    class_labels = torch.as_tensor([int(item[1]) for item in class_tasks], dtype=torch.long)
    structured_order = int(eval_cfg.get("structured_max_order", 3))
    class_contexts = [_activity_context(task, structured_order) for task in class_tasks]
    reg_labels = torch.as_tensor([float(item[1]) for item in reg_tasks], dtype=torch.float32)
    universe = sorted({int(class_labels[idx]) for idx in class_query_indices})
    selected_counts = []
    class_embeddings = []
    reg_embeddings = []
    if "classification" in requested_tasks:
        selected_counts.append(f"{len(class_tasks)} classification")
        class_embeddings = [
            encode_tasks(expert, class_tasks, eval_cfg.get("embedding_batch_size", 128), "classification")
            for expert in experts
        ]
    if "regression" in requested_tasks:
        selected_counts.append(f"{len(reg_tasks)} regression")
        reg_embeddings = [
            encode_tasks(expert, reg_tasks, eval_cfg.get("embedding_batch_size", 128), "regression")
            for expert in experts
        ]
    print(f"[{log_name}] encoding {' and '.join(selected_counts)} prefixes")

    rows = []
    for repetition in range(int(eval_cfg.get("repetitions", 5))):
        for scenario in eval_cfg.get("support_scenarios", ["natural", "class_aware"]):
            ordered_cases = orders[(repetition, scenario)]
            for budget in budgets:
                selected_cases = set(ordered_cases[:budget])
                class_support_indices = _task_indices(class_tasks, selected_cases)
                reg_support_indices = _task_indices(reg_tasks, selected_cases)
                if not len(class_support_indices):
                    continue
                if "classification" in requested_tasks:
                    profiles = eval_cfg.get("evaluation_profiles") or [{
                        "name": "main",
                        "retrieval_modes": eval_cfg.get("retrieval_modes", ["configured"]),
                        "prior_modes": eval_cfg.get("prior_modes", ["balanced", "natural"]),
                        "prior_strengths": eval_cfg.get("prior_strengths", [1.0]),
                        "retrieval_k": eval_cfg.get("retrieval_k", [5, 20, 50]),
                    }]
                    for profile in profiles:
                        for retrieval_mode in profile.get("retrieval_modes", ["configured"]):
                            for prior_mode in profile.get("prior_modes", ["balanced"]):
                                for prior_strength in profile.get("prior_strengths", [1.0]):
                                    for retrieval_k in profile.get("retrieval_k", [20]):
                                        prediction = predict_classification(
                                            experts, class_embeddings, class_labels, class_query_indices,
                                            class_support_indices, universe, int(retrieval_k), retrieval_mode,
                                            prior_mode, float(prior_strength), eval_cfg,
                                            structured_contexts=class_contexts,
                                        )
                                        metrics = classification_metrics(
                                            prediction["y_true"], prediction["y_pred"], prediction["probabilities"],
                                            universe, prediction["confidences"],
                                            [class_tasks[int(idx)][2] for idx in class_query_indices],
                                            prediction["support_counts"], prediction["pool_covered"],
                                            prediction["retrieval_covered"],
                                        )
                                        metrics["balanced_accuracy_ci"] = _bootstrap_balanced_accuracy(
                                            prediction, class_tasks, class_query_indices, universe,
                                            int(eval_cfg.get("bootstrap_repetitions", 200)),
                                            seed + repetition * 1009 + budget + int(retrieval_k),
                                        )
                                        structured_diagnostics = {}
                                        if "structured_context_support" in prediction:
                                            context_support = np.asarray(
                                                prediction["structured_context_support"], dtype=float
                                            )
                                            selected_order = np.asarray(
                                                prediction["structured_selected_order"], dtype=float
                                            )
                                            structured_diagnostics = {
                                                "structured_context_coverage": float(
                                                    np.mean(context_support > 0)
                                                ),
                                                "structured_mean_context_support": float(
                                                    context_support.mean()
                                                ),
                                                "structured_mean_selected_order": float(
                                                    selected_order.mean()
                                                ),
                                            }
                                            if "structured_effective_weight" in prediction:
                                                structured_diagnostics[
                                                    "structured_mean_effective_weight"
                                                ] = float(
                                                    np.mean(prediction["structured_effective_weight"])
                                                )
                                        row = {
                                            "task": "classification", "log": log_name,
                                            "experiment": config.get("experiment_name", "unnamed"),
                                            "evaluation_profile": profile.get("name", "unnamed"),
                                            "repetition": repetition, "support_scenario": scenario,
                                            "case_budget": budget, "support_prefixes": int(len(class_support_indices)),
                                            "retrieval_mode": retrieval_mode, "prior_mode": prior_mode,
                                            "prior_strength": float(prior_strength),
                                            "retrieval_k": int(retrieval_k), **metrics,
                                            "structured_max_order": structured_order,
                                            "structured_smoothing": float(
                                                eval_cfg.get("structured_smoothing", 0.5)
                                            ),
                                            "structured_weight": float(eval_cfg.get("structured_weight", 0.5)),
                                            "structured_tau": float(eval_cfg.get("structured_tau", 2.0)),
                                            **structured_diagnostics,
                                        }
                                        rows.append(row)
                                        with output_jsonl.open("a", encoding="utf-8") as handle:
                                            handle.write(json.dumps(row, sort_keys=True) + "\n")

                if "regression" in requested_tasks and len(reg_support_indices):
                    regression_k = int(max(eval_cfg.get("retrieval_k", [20])))
                    truth, pred, lower, upper, branch_diagnostics = predict_regression(
                        experts, reg_embeddings, reg_labels, reg_query_indices,
                        reg_support_indices, regression_k,
                        support_case_ids=[reg_tasks[int(idx)][2] for idx in reg_support_indices],
                        eval_cfg=eval_cfg,
                    )
                    row = {
                        "task": "regression", "log": log_name,
                        "experiment": config.get("experiment_name", "unnamed"),
                        "repetition": repetition, "support_scenario": scenario,
                        "case_budget": budget, "support_prefixes": int(len(reg_support_indices)),
                        "retrieval_k": regression_k,
                        "regression_mode": str(
                            experts[0].proto_head.regression_mode
                        ),
                        "regression_num_transforms": int(
                            experts[0].proto_head.time_transform_bank.num_transforms
                            if experts[0].proto_head.time_transform_bank is not None else 1
                        ),
                        "regression_transform_aggregation": str(
                            experts[0].proto_head.time_transform_bank.aggregation
                            if experts[0].proto_head.time_transform_bank is not None else "single"
                        ),
                        **regression_metrics(truth, pred, lower, upper),
                    }
                    if eval_cfg.get("regression_branch_diagnostics", False):
                        row["regression_branch_diagnostics"] = branch_diagnostics
                    rows.append(row)
                    with output_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows
