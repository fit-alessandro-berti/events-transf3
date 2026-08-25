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

from evaluation.fmv3_metrics import (
    classification_metrics,
    regression_metrics,
    prefix_length_metrics,
)
from utils.data_utils import prefix_task_length
from utils.retrieval_utils import class_diverse_topk_indices
from time_transf import inverse_transform_time


def encode_tasks(
    expert, tasks, batch_size=128, task_type=None, representation="decision"
):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            sequences = [item[0] for item in tasks[start : start + batch_size]]
            encoded =expert ._process_batch (sequences ,task_type =task_type )
            if task_type =="classification":
                if representation =="retrieval":
                    encoded =expert .classification_retrieval_features (encoded )
                elif representation =="decision":
                    encoded =expert .classification_decision_features (encoded )
                else :raise ValueError (f"Unknown classification representation: {representation}")
            embeddings.append(F.normalize(encoded, p=2, dim=1).cpu())
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


def _coverage_features_by_case(tasks, allowed_cases):
    features =defaultdict (set )
    for prefix ,label ,case_id in tasks :
        case =str (case_id )
        if case not in allowed_cases or label is None or int (label )==-100 :continue
        target =int (label )
        features [case ].add (("activity",target ))
        if prefix is None :continue
        if isinstance (prefix ,tuple )and all (
        isinstance (value ,(int ,np .integer ))for value in prefix ):
            context =tuple (map (int ,prefix ))
        else :
            context =tuple (
            int (event ["activity_id"])
            for event in prefix
            if event .get ("activity_id")is not None
            and int (event ["activity_id"])!=-100 )
        if context :features [case ].add (("transition",context [-1 ],target ))
        if len (context )>=2 :
            features [case ].add (("rare_suffix",*context [-2 :],target ))
    return features


def support_case_order(tasks, allowed_cases, scenario, seed, max_cases=None):
    rng = random.Random(seed)
    cases = sorted(allowed_cases)
    rng.shuffle(cases)
    if scenario == "natural":
        return cases if max_cases is None else cases[: min(len(cases), int(max_cases))]
    if scenario not in {"class_aware","coverage_aware"}:
        raise ValueError(f"Unknown support scenario: {scenario}")
    labels_by_case = _labels_by_case(tasks, allowed_cases)
    coverage_features =_coverage_features_by_case (tasks ,allowed_cases )
    feature_sets =(
    coverage_features if scenario =="coverage_aware"
    else {case :{("activity",label )for label in labels}
    for case ,labels in labels_by_case .items ()}
    )
    # Cases with the same label set always have the same greedy score. Grouping
    # them avoids rescanning very large support pools (e.g. ~70k Billing cases)
    # at every one of the first 128 selections.
    cases_by_signature = defaultdict(set)
    for case in cases:
        cases_by_signature[frozenset(feature_sets.get(case, set()))].add(case)
    selected = []
    counts = Counter()
    target = len(cases) if max_cases is None else min(len(cases), int(max_cases))
    while cases_by_signature and len(selected) < target:
        best_score = None
        best_signatures = []
        for labels in cases_by_signature:
            weighted_uncovered =sum (
            {"activity":4.0 ,"transition":2.0 ,"rare_suffix":1.0}.get (
            feature [0 ],1.0 )for feature in labels if counts [feature ]==0 )
            balance_gain = sum(1.0 / (1.0 + counts[label]) for label in labels)
            score = (weighted_uncovered, balance_gain, len(labels))
            if best_score is None or score > best_score:
                best_score, best_signatures = score, [labels]
            elif score == best_score:
                best_signatures.append(labels)
        best_cases = sorted(
            case for signature in best_signatures for case in cases_by_signature[signature]
        )
        chosen = rng.choice(best_cases)
        selected.append(chosen)
        for label in feature_sets.get(chosen, set()):
            counts[label] += 1
        signature = frozenset(feature_sets.get(chosen, set()))
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
    minimum_prefix = int(
        (config.get("data", {}) or {}).get("minimum_prefix_length", 1)
    )
    for trace in transformed_log:
        if len(trace) < minimum_prefix + 1:
            continue
        case_id = str(trace[0]["case_id"])
        case_tasks =[]
        for index in range (minimum_prefix -1 ,len (trace )-1 ):
            next_label =trace [index +1 ].get ("activity_id")
            if next_label is None or int (next_label )==-100 :continue
            context =tuple (
            int (event ["activity_id"])
            for event in trace [max (0 ,index -2 ):index +1 ]
            if event .get ("activity_id")is not None
            and int (event ["activity_id"])!=-100 )
            case_tasks .append ((context ,int (next_label ),case_id ))
        if not case_tasks:
            continue
        traces_by_case[case_id] = trace
        planning_tasks .extend (case_tasks )
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
    work[timestamp_key] = pd.to_datetime(
        work[timestamp_key], errors="coerce", utc=True
    ).dt.tz_convert(None)
    work = work.dropna(subset=[timestamp_key])
    activity_names = sorted(work[activity_key].dropna().unique().tolist())
    activity_to_id = {name: index for index, name in enumerate(activity_names)}
    work["_fmv3_original_order"] = np.arange(len(work))
    work = work.sort_values(
        [case_key, timestamp_key, "_fmv3_original_order"], kind="mergesort"
    )
    work["_fmv3_activity_id"] = work[activity_key].map(activity_to_id)
    case_sizes = work.groupby(case_key, sort=False).size()
    minimum_prefix = int(
        (config.get("data", {}) or {}).get("minimum_prefix_length", 1)
    )
    eligible_cases = set(
        case_sizes[case_sizes >= minimum_prefix + 1].index.tolist()
    )
    planning_tasks = []
    for case_id ,case_rows in work [work [case_key ].isin (eligible_cases )].groupby (
    case_key ,sort =False ):
        activity_ids =[
        None if pd .isna (value )else int (value )
        for value in case_rows ["_fmv3_activity_id"].tolist ()]
        for target_position in range (minimum_prefix ,len (activity_ids )):
            target =activity_ids [target_position ]
            if target is None :continue
            context =tuple (
            value for value in activity_ids [max (0 ,target_position -3 ):target_position ]
            if value is not None )
            planning_tasks .append ((context ,target ,str (case_id )))
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


def _fuse_structured_prediction(
    base,
    structured,
    class_universe,
    weight,
    tau,
    fusion,
    low_support_threshold=0,
    low_support_weight=None,
    low_support_tau=None,
    output_temperature=1.0,
):
    base_probabilities = np.asarray(base["probabilities"], dtype=np.float64)
    structured_probabilities = np.asarray(structured["probabilities"], dtype=np.float64)
    base_probabilities /= base_probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    structured_probabilities /= structured_probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    support = np.asarray(structured["structured_context_support"], dtype=np.float64)
    support_counts = structured.get("support_counts") or base.get("support_counts") or {}
    total_support = int(sum(int(count) for count in support_counts.values()))
    selected_weight = float(weight)
    selected_tau = float(tau)
    low_support_threshold = int(low_support_threshold)
    if 0 < low_support_threshold and total_support <= low_support_threshold:
        if low_support_weight is not None:
            selected_weight = float(low_support_weight)
        if low_support_tau is not None:
            selected_tau = float(low_support_tau)
    selected_tau = max(selected_tau, 0.0)
    reliability = (
        np.ones_like(support)
        if selected_tau == 0.0
        else support / (support + selected_tau)
    )
    effective_weight = np.clip(selected_weight * reliability, 0.0, 1.0)[:, None]
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
    output_temperature = float(output_temperature)
    if not math.isfinite(output_temperature) or output_temperature <= 0.0:
        raise ValueError("classification output temperature must be finite and positive")
    if output_temperature != 1.0:
        calibrated_logits = np.log(probabilities.clip(min=1e-12)) / output_temperature
        calibrated_logits -= calibrated_logits.max(axis=1, keepdims=True)
        probabilities = np.exp(calibrated_logits)
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
    fused["structured_total_support"] = total_support
    fused["structured_selected_weight"] = selected_weight
    fused["structured_selected_tau"] = selected_tau
    return fused


def _enforce_support_only_candidates(prediction, class_universe):
    """Make support-absent schema labels illegal after every fusion branch."""
    probabilities =np .asarray (prediction ["probabilities"],dtype =float ).copy ()
    support_counts =prediction .get ("support_counts",{})or {}
    allowed =np .asarray ([
    int (support_counts .get (int (label ),0 ))>0 for label in class_universe ],
    dtype =bool )
    probabilities [:,~allowed ]=0.0
    denominator =probabilities .sum (axis =1 ,keepdims =True )
    if bool ((denominator <=0 ).any ()):
        zero_rows =np .where (denominator [:,0 ]<=0 )[0 ]
        allowed_columns =np .where (allowed )[0 ]
        probabilities [np .ix_ (zero_rows ,allowed_columns )]=1.0 /max (
        int (allowed_columns .size ),1 )
        denominator =probabilities .sum (axis =1 ,keepdims =True )
    probabilities /=denominator .clip (min =1e-12 )
    universe =np .asarray (class_universe ,dtype =int )
    result =dict (prediction )
    result ["probabilities"]=probabilities
    result ["y_pred"]=universe [np .argmax (probabilities ,axis =1 )].tolist ()
    result ["confidences"]=probabilities .max (axis =1 ).tolist ()
    return result


def _expert_confidence_weights(
    logits: torch.Tensor,
    temperature: float = 1.0,
    prior_weights: torch.Tensor | None = None,
):
    """Softmax expert-confidence logits with a configurable temperature."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "Expert-confidence temperature must be finite and positive"
        )
    scaled = logits / temperature
    if prior_weights is not None:
        prior = prior_weights.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-8)
        while prior.ndim < scaled.ndim:
            prior = prior.unsqueeze(-1)
        scaled = scaled + prior.log()
    return F.softmax(scaled, dim=0)


def _weighted_stack_mean(values: torch.Tensor, weights: torch.Tensor):
    weights = weights.to(device=values.device, dtype=values.dtype).clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-8)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (weights * values).sum(dim=0)


def _eval_cfg_value(eval_cfg, task_type, key, default):
    eval_cfg = eval_cfg or {}
    task_key = f"{task_type}_{key}"
    return eval_cfg.get(task_key, eval_cfg.get(key, default))


def _eval_cfg_bool(eval_cfg, task_type, key, default):
    value = _eval_cfg_value(eval_cfg, task_type, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _virtual_expert_row(eval_cfg, task_type):
    return {
        "virtual_expert_replicates": int(
            _eval_cfg_value(eval_cfg, task_type, "virtual_expert_replicates", 1)
        ),
        "virtual_expert_support_fraction": float(
            _eval_cfg_value(eval_cfg, task_type, "virtual_expert_support_fraction", 1.0)
        ),
        "virtual_expert_min_support_prefixes": int(
            _eval_cfg_value(eval_cfg, task_type, "virtual_expert_min_support_prefixes", 1)
        ),
        "virtual_expert_full_support_weight": float(
            _eval_cfg_value(eval_cfg, task_type, "virtual_expert_full_support_weight", 1.0)
        ),
        "virtual_expert_subbag_weight": float(
            _eval_cfg_value(eval_cfg, task_type, "virtual_expert_subbag_weight", 1.0)
        ),
    }


def _stable_hash_scores(indices, seed):
    values = np.asarray(indices, dtype=np.uint64)
    x = values + np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _virtual_support_views(
    support_indices,
    eval_cfg=None,
    task_type="classification",
    labels=None,
    salt=0,
):
    """Return deterministic support sub-bags for test-time virtual experts."""
    base = np.asarray(support_indices, dtype=np.int64)
    if base.size == 0:
        return [base]
    replicas = max(1, int(_eval_cfg_value(eval_cfg, task_type, "virtual_expert_replicates", 1)))
    if replicas <= 1:
        return [base]
    min_prefixes = max(1, int(_eval_cfg_value(eval_cfg, task_type, "virtual_expert_min_support_prefixes", 1)))
    if base.size < min_prefixes:
        return [base]
    fraction = float(_eval_cfg_value(eval_cfg, task_type, "virtual_expert_support_fraction", 1.0))
    fraction = min(1.0, max(0.0, fraction))
    target = int(math.ceil(base.size * fraction))
    target = min(base.size, max(1, target))
    if target >= base.size:
        return [base]

    include_full = _eval_cfg_bool(eval_cfg, task_type, "virtual_expert_include_full_support", True)
    preserve_labels = (
        task_type == "classification"
        and labels is not None
        and _eval_cfg_bool(eval_cfg, task_type, "virtual_expert_preserve_labels", True)
    )
    if preserve_labels:
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        support_labels = labels_np[base]
    else:
        support_labels = None

    seed = int(_eval_cfg_value(eval_cfg, task_type, "virtual_expert_seed", 42))
    seed += int(salt) * 1000003
    views = [base] if include_full else []
    replica = 1
    while len(views) < replicas and replica <= replicas * 4:
        scores = _stable_hash_scores(base, seed + replica * 0x9E3779B1)
        selected = np.zeros(base.size, dtype=bool)
        if preserve_labels and support_labels is not None:
            for label in sorted(set(map(int, support_labels.tolist()))):
                positions = np.flatnonzero(support_labels == label)
                if positions.size:
                    selected[positions[np.argmin(scores[positions])]] = True
        remaining = np.flatnonzero(~selected)
        needed = max(0, target - int(selected.sum()))
        if needed > 0 and remaining.size:
            chosen = remaining[np.argsort(scores[remaining])[:needed]]
            selected[chosen] = True
        candidate = base[selected]
        if candidate.size and not any(np.array_equal(candidate, view) for view in views):
            views.append(candidate)
        replica += 1
    return views or [base]


def _virtual_support_view_specs(
    support_indices,
    eval_cfg=None,
    task_type="classification",
    labels=None,
    salt=0,
):
    base = np.asarray(support_indices, dtype=np.int64)
    full_weight = float(_eval_cfg_value(eval_cfg, task_type, "virtual_expert_full_support_weight", 1.0))
    subbag_weight = float(_eval_cfg_value(eval_cfg, task_type, "virtual_expert_subbag_weight", 1.0))
    specs = []
    for view in _virtual_support_views(support_indices, eval_cfg, task_type, labels=labels, salt=salt):
        weight = full_weight if np.array_equal(view, base) else subbag_weight
        specs.append((view, max(0.0, float(weight))))
    return specs


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
    candidate_features=None,
):
    """Vectorized equivalent of per-query fixed-k classification heads."""
    device = query.device
    classes = torch.as_tensor(prediction_classes, dtype=pool_labels.dtype, device=device)
    num_queries, num_classes = query.size(0), classes.numel()
    local = pool[local_positions]
    local_labels = pool_labels[local_positions]
    if head.semantic_candidate_decoder_enabled and candidate_features is not None:
        result =head .forward_classification_batched (
            local ,local_labels ,query ,
            global_support_features =pool ,
            global_support_labels =pool_labels ,
            candidate_classes =classes ,
            candidate_features =candidate_features ,
            prior_mode =prior_mode ,prior_strength =prior_strength ,
        )
        logits ,output_classes ,probabilities =result
        if logits is None :return query .new_zeros ((num_queries ,num_classes )),query .new_zeros (num_queries )
        abstain =probabilities .new_zeros (num_queries )
        aligned =probabilities .new_zeros ((num_queries ,num_classes ))
        for output_column ,label in enumerate (output_classes ):
            if int (label )==head .abstain_label :
                abstain =probabilities [:,output_column ]
                continue
            match =(classes ==label ).nonzero (as_tuple =False )
            if match .numel ():
                aligned [:,match [0 ,0 ]]=probabilities [:,output_column ]
        return aligned ,abstain
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
    selector_local = local
    selector_query = query
    if selected_mode == "legacy_soft_knn":
        center = local.mean(dim=1, keepdim=True)
        centered_local = F.normalize(local - center, p=2, dim=2)
        centered_query = F.normalize(query - center.squeeze(1), p=2, dim=1)
        similarities = torch.einsum("qd,qkd->qk", centered_query, centered_local)
        selection_logits = head.classification_selection_logits(
            selector_local,
            local_labels,
            selector_query,
            base_similarities=similarities,
        )
        attention = F.softmax(
            similarities * head.logit_scale.clamp(1.0, 20.0) + selection_logits,
            dim=1,
        )
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
    raw_local_similarities = torch.einsum("qd,qkd->qk", local_query, local)
    selection_logits = head.classification_selection_logits(
        selector_local,
        local_labels,
        selector_query,
        base_similarities=raw_local_similarities,
    )
    local_similarities = raw_local_similarities * head.local_scale + selection_logits
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
    eval_cfg=None,
    retrieval_embeddings_by_expert=None,
    candidate_features_by_expert=None,
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
    retrieval_embeddings_device =[
    embedding .to (device )for embedding in (
    retrieval_embeddings_by_expert or embeddings_by_expert )]
    candidate_features_device =[
    None if features is None else features .to (device )
    for features in (candidate_features_by_expert or [None ]*len (experts ))]
    k_eff = min(int(retrieval_k), len(support_indices))

    expert_probabilities, abstain_probabilities, expert_confidence_logits, view_weights = [], [], [], []
    retrieval_coverages_by_expert =defaultdict (list )
    output_expert_ids =[]
    policy =str ((eval_cfg or {}).get (
    "classification_retrieval_policy","nearest" )).lower ()
    for expert_index, (expert, embeddings, retrieval_embeddings) in enumerate(zip(
    experts, embeddings_device, retrieval_embeddings_device)):
        expert_queries = embeddings[query_indices_device]
        retrieval_queries =retrieval_embeddings [query_indices_device ]
        for view_indices, view_weight in _virtual_support_view_specs(
            support_indices, eval_cfg, "classification", labels=labels, salt=expert_index
        ):
            view_device = torch.as_tensor(view_indices, device=device)
            view_k = min(int(retrieval_k), int(view_device.numel()))
            expert_pool = embeddings[view_device]
            retrieval_pool =retrieval_embeddings [view_device ]
            similarities =retrieval_queries @retrieval_pool .t ()
            if policy =="class_diverse":
                semantic_scores =None
                if candidate_features_device [expert_index ]is not None :
                    semantic_query =F .normalize (
                    expert .proto_head .semantic_query_projection (expert_queries ),
                    p =2 ,dim =1 )
                    semantic_candidates =F .normalize (
                    expert .proto_head .semantic_candidate_projection (
                    candidate_features_device [expert_index ]),p =2 ,dim =1 )
                    semantic_scores =semantic_query @semantic_candidates .t ()
                local_positions =class_diverse_topk_indices (
                similarities ,labels_device [view_device ],view_k ,
                classes_per_shortlist =int ((eval_cfg or {}).get (
                "class_diverse_shortlist_classes",10 )),
                examples_per_class =int ((eval_cfg or {}).get (
                "class_diverse_examples_per_class",2 )),
                candidate_classes =torch .as_tensor (
                prediction_classes ,device =device ),
                candidate_scores =semantic_scores ,
                semantic_weight =float ((eval_cfg or {}).get (
                "class_diverse_semantic_weight",1.0 )),
                )
            else :
                local_positions = torch.topk(similarities, view_k, dim=1).indices
            local_labels =labels_device [view_device ][local_positions ]
            true_labels =labels_device [query_indices_device ]
            retrieval_coverages_by_expert [expert_index ].append ((
            local_labels ==true_labels .unsqueeze (1 )).any (dim =1 ))
            probabilities, abstain = _batched_head_probabilities(
                expert.proto_head,
                expert_queries,
                expert_pool,
                labels_device[view_device],
                local_positions,
                prediction_classes,
                retrieval_mode,
                prior_mode,
                prior_strength,
                candidate_features =candidate_features_device [expert_index ],
            )
            expert_probabilities.append(probabilities)
            output_expert_ids .append (expert_index )
            abstain_probabilities.append(abstain)
            view_weights.append(float(view_weight))
            if expert.proto_head.classification_expert_confidence_enabled:
                expert_confidence_logits.append(
                    expert.proto_head.classification_expert_confidence_logit(probabilities)
                )
            else:
                expert_confidence_logits.append(probabilities.new_zeros(probabilities.size(0)))
    stacked_probabilities = torch.stack(expert_probabilities)
    stacked_abstain = torch.stack(abstain_probabilities)
    view_weights_tensor = torch.as_tensor(view_weights, device=device)
    if any(expert.proto_head.classification_expert_confidence_enabled for expert in experts):
        expert_weights = _expert_confidence_weights(
            torch.stack(expert_confidence_logits), prior_weights=view_weights_tensor
        )
        mean_probs = (expert_weights.unsqueeze(-1) * stacked_probabilities).sum(dim=0)
        mean_abstain = (expert_weights * stacked_abstain).sum(dim=0)
    else:
        expert_weights = None
        mean_probs = _weighted_stack_mean(stacked_probabilities, view_weights_tensor)
        mean_abstain = _weighted_stack_mean(stacked_abstain, view_weights_tensor)
    expert_coverage =torch .stack ([
    torch .stack (retrieval_coverages_by_expert [expert_index ]).any (dim =0 )
    for expert_index in range (len (experts ))])
    coverage_any =expert_coverage .any (dim =0 )
    coverage_every =expert_coverage .all (dim =0 )
    output_expert_ids_tensor =torch .as_tensor (
    output_expert_ids ,dtype =torch .long ,device =device )
    if expert_weights is not None :
        expert_total_weight =expert_weights .new_zeros ((
        len (experts ),expert_weights .size (1 )))
        expert_total_weight .index_add_ (0 ,output_expert_ids_tensor ,expert_weights )
    else :
        normalized_views =view_weights_tensor /view_weights_tensor .sum ().clamp_min (1e-8 )
        expert_total_weight =normalized_views .new_zeros (len (experts ))
        expert_total_weight .index_add_ (
        0 ,output_expert_ids_tensor ,normalized_views )
        expert_total_weight =expert_total_weight .unsqueeze (1 ).expand (
        -1 ,expert_coverage .size (1 ))
    highest_expert =expert_total_weight .argmax (dim =0 )
    coverage_highest =expert_coverage .gather (
    0 ,highest_expert .unsqueeze (0 )).squeeze (0 )
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
    result = {
        "y_true": true.cpu().tolist(),
        "y_pred": predicted.cpu().tolist(),
        "probabilities": mean_probs[:, metric_columns].cpu().numpy(),
        "confidences": confidence.cpu().tolist(),
        "pool_covered": pool_covered.cpu().tolist(),
        "retrieval_covered": coverage_any.cpu().tolist(),
        "retrieval_covered_any_expert": coverage_any.cpu().tolist(),
        "retrieval_covered_every_expert": coverage_every.cpu().tolist(),
        "retrieval_covered_highest_weight_expert": coverage_highest.cpu().tolist(),
        "retrieval_covered_union_experts": coverage_any.cpu().tolist(),
        "support_counts": dict(support_counts),
    }
    if expert_weights is not None:
        result["classification_expert_confidence_weights_mean"] = (
            expert_weights.mean(dim=1).cpu().tolist()
        )
    return result


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
            effective, "configured", prior_mode, prior_strength, eval_cfg=eval_cfg,
        )
        support_counts = result["support_counts"]
        for local_position, output_position in enumerate(positions):
            for key in merged:
                merged[key][output_position] = result[key][local_position]
    merged["probabilities"] = np.asarray(merged["probabilities"])
    merged["support_counts"] = support_counts or {}
    return merged


@torch.no_grad ()
def _predict_frozen_embedding_logistic(
    embeddings_by_expert, labels, query_indices, support_indices,
    class_universe, eval_cfg,
):
    """Class-balanced diagnostic on frozen foundation decision embeddings."""
    from sklearn .linear_model import LogisticRegression

    classes =list (map (int ,class_universe ))
    class_to_column ={label :column for column ,label in enumerate (classes )}
    support_y =labels [support_indices ].cpu ().numpy ().astype (int )
    probabilities =[]
    for expert_index ,embeddings in enumerate (embeddings_by_expert ):
        support_x =embeddings [support_indices ].cpu ().numpy ()
        query_x =embeddings [query_indices ].cpu ().numpy ()
        observed =np .unique (support_y )
        aligned =np .zeros ((len (query_indices ),len (classes )),dtype =float )
        if len (observed )==1 :
            if int (observed [0 ])in class_to_column :
                aligned [:,class_to_column [int (observed [0 ])]]=1.0
        else :
            classifier =LogisticRegression (
            C =float ((eval_cfg or {}).get ("foundation_logistic_c",1.0 )),
            class_weight ="balanced",max_iter =int ((eval_cfg or {}).get (
            "foundation_logistic_max_iter",500 )),
            random_state =int ((eval_cfg or {}).get ("virtual_expert_seed",42 ))+expert_index )
            classifier .fit (support_x ,support_y )
            partial =classifier .predict_proba (query_x )
            for source_column ,label in enumerate (classifier .classes_ ):
                if int (label )in class_to_column :
                    aligned [:,class_to_column [int (label )]]=partial [:,source_column ]
        probabilities .append (aligned )
    mean_probs =np .mean (probabilities ,axis =0 )
    predicted =np .asarray (classes ,dtype =int )[np .argmax (mean_probs ,axis =1 )]
    true =labels [query_indices ].cpu ().numpy ().astype (int )
    support_counts =Counter (map (int ,support_y .tolist ()))
    covered =np .asarray ([support_counts .get (int (label ),0 )>0 for label in true ])
    return {
    "y_true":true .tolist (),"y_pred":predicted .tolist (),
    "probabilities":mean_probs ,"confidences":mean_probs .max (axis =1 ).tolist (),
    "pool_covered":covered .tolist (),"retrieval_covered":covered .tolist (),
    "retrieval_covered_any_expert":covered .tolist (),
    "retrieval_covered_every_expert":covered .tolist (),
    "retrieval_covered_highest_weight_expert":covered .tolist (),
    "retrieval_covered_union_experts":covered .tolist (),
    "support_counts":dict (support_counts ),
    }


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
    retrieval_embeddings_by_expert=None,
    candidate_features_by_expert=None,
):
    if retrieval_mode =="foundation_logistic":
        return _predict_frozen_embedding_logistic (
        embeddings_by_expert ,labels ,query_indices ,support_indices ,
        class_universe ,eval_cfg )
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
            eval_cfg=eval_cfg,
            retrieval_embeddings_by_expert=retrieval_embeddings_by_expert,
            candidate_features_by_expert=candidate_features_by_expert,
        )
        fusion = "mixture" if retrieval_mode == "fm_structured_mix" else "product"
        return _fuse_structured_prediction(
            base,
            structured,
            class_universe,
            float(eval_cfg.get("structured_weight", 0.5)),
            float(eval_cfg.get("structured_tau", 2.0)),
            fusion,
            low_support_threshold=int(
                eval_cfg.get("structured_low_support_threshold", 0)
            ),
            low_support_weight=eval_cfg.get("structured_low_support_weight"),
            low_support_tau=eval_cfg.get("structured_low_support_tau"),
            output_temperature=eval_cfg.get("classification_output_temperature", 1.0),
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
            eval_cfg=eval_cfg,
            retrieval_embeddings_by_expert=retrieval_embeddings_by_expert,
            candidate_features_by_expert=candidate_features_by_expert,
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


def _regression_calibration_mix(eval_cfg, case_budget=None):
    """Return the support-calibration blend for a known support-case budget."""
    default = min(
        1.0,
        max(0.0, float(eval_cfg.get("regression_calibration_mix", 1.0))),
    )
    schedule = eval_cfg.get("regression_calibration_mix_by_budget") or {}
    if case_budget is None or not isinstance(schedule, dict):
        return default
    try:
        budget = int(case_budget)
    except (TypeError, ValueError):
        return default
    for key, value in schedule.items():
        try:
            scheduled_budget = int(key)
        except (TypeError, ValueError):
            continue
        if scheduled_budget == budget:
            return min(1.0, max(0.0, float(value)))
    return default


def _regression_interval_std_multiplier(eval_cfg):
    """Return the positive Gaussian-style scale used for time intervals."""
    value = float((eval_cfg or {}).get("regression_interval_std_multiplier", 1.645))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Regression interval std multiplier must be finite and positive")
    return value


@torch.no_grad()
def predict_regression(
    experts, embeddings_by_expert, labels, query_indices, support_indices,
    retrieval_k, support_case_ids=None, eval_cfg=None, case_budget=None,
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
    expert_confidence_logits = []
    view_weights = []
    for expert_index, (expert, embeddings) in enumerate(zip(experts, embeddings_device)):
        query = F.normalize(embeddings[query_indices_device], p=2, dim=1)
        for view_indices, view_weight in _virtual_support_view_specs(
            support_indices, eval_cfg, "regression", salt=expert_index
        ):
            view_device = torch.as_tensor(view_indices, device=device)
            view_k = min(int(retrieval_k), int(view_device.numel()))
            pool = F.normalize(embeddings[view_device], p=2, dim=1)
            positions = torch.topk(query @ pool.t(), view_k, dim=1).indices
            local = pool[positions]
            local_targets = labels_device[view_device][positions]
            prediction, confidence, diagnostics = expert.proto_head.forward_regression_batched(
                local, local_targets, query, return_diagnostics=True
            )
            expert_predictions.append(prediction)
            view_weights.append(float(view_weight))
            if expert.proto_head.regression_expert_confidence_enabled:
                expert_confidence_logits.append(
                    expert.proto_head.regression_expert_confidence_logit(
                        prediction, confidence, diagnostics
                    )
                )
            else:
                expert_confidence_logits.append(prediction.new_zeros(prediction.numel()))
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
    stacked_predictions = torch.stack(expert_predictions)
    view_weights_tensor = torch.as_tensor(view_weights, device=device)
    if any(expert.proto_head.regression_expert_confidence_enabled for expert in experts):
        expert_confidence_temperature = float(
            eval_cfg.get("regression_expert_confidence_temperature", 1.0)
        )
        expert_weights = _expert_confidence_weights(
            torch.stack(expert_confidence_logits),
            temperature=expert_confidence_temperature,
            prior_weights=view_weights_tensor,
        )
        mean_prediction = (expert_weights * stacked_predictions).sum(dim=0)
    else:
        expert_confidence_temperature = None
        expert_weights = None
        mean_prediction = _weighted_stack_mean(stacked_predictions, view_weights_tensor)
    transformed_truth = labels_device[query_indices_device].cpu().numpy()
    truths = inverse_transform_time(transformed_truth)
    if experts[0].proto_head.regression_outputs_hours:
        if calibrated_weights is not None:
            stacked_branches = torch.stack(expert_branch_predictions)
            if expert_weights is None:
                mean_branches = _weighted_stack_mean(stacked_branches, view_weights_tensor)
            else:
                mean_branches = (
                    expert_weights.unsqueeze(1) * stacked_branches
                ).sum(dim=0)
            calibrated_prediction = (
                calibrated_weights[:, None] * mean_branches
            ).sum(dim=0)
            calibration_mix = _regression_calibration_mix(
                eval_cfg, case_budget=case_budget
            )
            mean_prediction = (
                (1.0 - calibration_mix) * mean_prediction
                + calibration_mix * calibrated_prediction
            )
            calibration_diagnostics["calibration_mix"] = calibration_mix
        predictions = mean_prediction.cpu().numpy()
        stacked_stds = torch.stack(expert_stds)
        if expert_weights is None:
            within_variance = _weighted_stack_mean(stacked_stds.square(), view_weights_tensor)
            between_variance = _weighted_stack_mean(
                (stacked_predictions - mean_prediction.unsqueeze(0)).square(),
                view_weights_tensor,
            )
        else:
            within_variance = (expert_weights * stacked_stds.square()).sum(dim=0)
            between_variance = (
                expert_weights * (stacked_predictions - mean_prediction.unsqueeze(0)).square()
            ).sum(dim=0)
        std_hours = torch.sqrt((within_variance + between_variance).clamp_min(1e-8)).cpu().numpy()
        interval_scale = _regression_interval_std_multiplier(eval_cfg)
        lower = np.maximum(0.0, predictions - interval_scale * std_hours)
        upper = predictions + interval_scale * std_hours
    else:
        transformed_prediction = mean_prediction.cpu().numpy()
        transformed_std = neighbor_targets.std(dim=1, correction=0).cpu().numpy()
        predictions = inverse_transform_time(transformed_prediction)
        interval_scale = _regression_interval_std_multiplier(eval_cfg)
        lower = inverse_transform_time(
            np.maximum(0.0, transformed_prediction - interval_scale * transformed_std)
        )
        upper = inverse_transform_time(
            transformed_prediction + interval_scale * transformed_std
        )
    branch_diagnostics = None
    if expert_branch_predictions:
        stacked_branches = torch.stack(expert_branch_predictions)
        if expert_weights is None:
            mean_branch_tensor = _weighted_stack_mean(stacked_branches, view_weights_tensor)
            mean_aggregation_tensor = _weighted_stack_mean(
                torch.stack(expert_aggregation_weights), view_weights_tensor
            )
        else:
            mean_branch_tensor = (expert_weights.unsqueeze(1) * stacked_branches).sum(dim=0)
            mean_aggregation_tensor = (
                expert_weights.unsqueeze(1) * torch.stack(expert_aggregation_weights)
            ).sum(dim=0)
        mean_branches = mean_branch_tensor.cpu().numpy()
        truth_array = np.asarray(truths, dtype=float)
        branch_diagnostics = {
            "branch_mae_hours": np.mean(
                np.abs(mean_branches - truth_array[None, :]), axis=1
            ).tolist(),
            "branch_rmse_hours": np.sqrt(
                np.mean((mean_branches - truth_array[None, :]) ** 2, axis=1)
            ).tolist(),
            "mean_aggregation_weights": mean_aggregation_tensor
            .mean(dim=1)
            .cpu()
            .tolist(),
            "oracle_branch_mae_hours": float(
                np.min(np.abs(mean_branches - truth_array[None, :]), axis=0).mean()
            ),
            **calibration_diagnostics,
        }
        if expert_weights is not None:
            branch_diagnostics["regression_expert_confidence_weights_mean"] = (
                expert_weights.mean(dim=1).cpu().tolist()
            )
            branch_diagnostics["regression_expert_confidence_temperature"] = (
                expert_confidence_temperature
            )
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


@torch.no_grad()
def _route_task_experts(model, experts, support_tasks, query_tasks, task_type):
    """Route before encoding so unselected experts do no task-level work."""
    if not getattr(model, "expert_routing_confidence_enabled", False):
        return list(experts), None
    selected_indices, _ = model.route_experts(
        support_tasks, query_tasks, task_type
    )
    diagnostics = dict(model.last_routing_diagnostics or {})
    diagnostics["inactive_expert_count"] = len(experts) - len(selected_indices)
    selected = [experts[index] for index in selected_indices]
    if not selected:
        raise RuntimeError(f"Expert routing selected no experts for {task_type}")
    return selected, diagnostics


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
    candidate_labels =sorted (
    list (getattr (all_class_tasks ,"candidate_labels",())or ()),
    key =lambda candidate :int (candidate ["label_id"]),
    )
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
    universe =(
    [int (candidate ["label_id"])for candidate in candidate_labels ]
    if candidate_labels else
    sorted ({int (value )for value in class_labels .tolist ()})
    )
    selected_counts = []
    class_embeddings = []
    class_retrieval_embeddings =[]
    candidate_features =[]
    reg_embeddings = []
    routing_support_class = [
        task for task in class_tasks if str(task[2]) in used_support_cases
    ]
    routing_query_class = [
        class_tasks[int(index)] for index in class_query_indices
    ]
    routing_support_reg = [
        task for task in reg_tasks if str(task[2]) in used_support_cases
    ]
    routing_query_reg = [reg_tasks[int(index)] for index in reg_query_indices]
    class_experts, class_routing = _route_task_experts(
        model, experts, routing_support_class, routing_query_class, "classification"
    )
    reg_experts, reg_routing = _route_task_experts(
        model, experts, routing_support_reg, routing_query_reg, "regression"
    )
    if "classification" in requested_tasks:
        selected_counts.append(f"{len(class_tasks)} classification")
        class_embeddings = [
            encode_tasks(
                expert, class_tasks,
                eval_cfg.get("embedding_batch_size", 128), "classification",
                representation="decision",
            )
            for expert in class_experts
        ]
        class_retrieval_embeddings =[
            encode_tasks(
                expert, class_tasks,
                eval_cfg.get("embedding_batch_size", 128), "classification",
                representation="retrieval",
            )
            for expert in class_experts
        ]
        if str (eval_cfg .get ("schema_mode","schema_known")).lower ()=="support_only":
            candidate_features =[None ]*len (class_experts )
        else :
            candidate_features =[
                expert .encode_candidate_labels (candidate_labels )
                for expert in class_experts
            ]
    if "regression" in requested_tasks:
        selected_counts.append(f"{len(reg_tasks)} regression")
        reg_embeddings = [
            encode_tasks(expert, reg_tasks, eval_cfg.get("embedding_batch_size", 128), "regression")
            for expert in reg_experts
        ]
    print(f"[{log_name}] encoding {' and '.join(selected_counts)} prefixes")
    if class_routing is not None and "classification" in requested_tasks:
        print(
            f"[{log_name}] classification experts "
            f"{class_routing['selected_expert_indices']} active; "
            f"{class_routing['inactive_expert_count']} inactive"
        )
    if reg_routing is not None and "regression" in requested_tasks:
        print(
            f"[{log_name}] regression experts "
            f"{reg_routing['selected_expert_indices']} active; "
            f"{reg_routing['inactive_expert_count']} inactive"
        )

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
                                            class_experts, class_embeddings, class_labels, class_query_indices,
                                            class_support_indices, universe, int(retrieval_k), retrieval_mode,
                                            prior_mode, float(prior_strength), eval_cfg,
                                            structured_contexts=class_contexts,
                                            retrieval_embeddings_by_expert=class_retrieval_embeddings,
                                            candidate_features_by_expert=candidate_features,
                                        )
                                        if str (eval_cfg .get (
                                        "schema_mode","schema_known")).lower ()=="support_only":
                                            prediction =_enforce_support_only_candidates (
                                            prediction ,universe )
                                        metrics = classification_metrics(
                                            prediction["y_true"], prediction["y_pred"], prediction["probabilities"],
                                            universe, prediction["confidences"],
                                            [class_tasks[int(idx)][2] for idx in class_query_indices],
                                            prediction["support_counts"], prediction["pool_covered"],
                                            prediction["retrieval_covered"],
                                            retrieval_covered_any_expert=prediction.get(
                                                "retrieval_covered_any_expert"
                                            ),
                                            retrieval_covered_every_expert=prediction.get(
                                                "retrieval_covered_every_expert"
                                            ),
                                            retrieval_covered_highest_weight_expert=prediction.get(
                                                "retrieval_covered_highest_weight_expert"
                                            ),
                                            retrieval_covered_union_experts=prediction.get(
                                                "retrieval_covered_union_experts"
                                            ),
                                        )
                                        metrics["balanced_accuracy_ci"] = _bootstrap_balanced_accuracy(
                                            prediction, class_tasks, class_query_indices, universe,
                                            int(eval_cfg.get("bootstrap_repetitions", 200)),
                                            seed + repetition * 1009 + budget + int(retrieval_k),
                                        )
                                        metrics["prefix_length_metrics"] = prefix_length_metrics(
                                            "classification",
                                            [
                                                prefix_task_length(class_tasks, int(idx))
                                                for idx in class_query_indices
                                            ],
                                            prediction["y_true"],
                                            prediction["y_pred"],
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
                                            "schema_mode":str (eval_cfg .get (
                                            "schema_mode","schema_known")),
                                            "evaluation_split": config.get(
                                                "evaluation_split", "unspecified"
                                            ),
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
                                            **_virtual_expert_row(eval_cfg, "classification"),
                                            **structured_diagnostics,
                                        }
                                        if class_routing is not None:
                                            row["expert_routing"] = class_routing
                                        if (
                                            eval_cfg.get("expert_aggregation_diagnostics", False)
                                            and "classification_expert_confidence_weights_mean" in prediction
                                        ):
                                            row["expert_aggregation_diagnostics"] = {
                                                "classification_expert_confidence_weights_mean": prediction[
                                                    "classification_expert_confidence_weights_mean"
                                                ]
                                            }
                                        rows.append(row)
                                        with output_jsonl.open("a", encoding="utf-8") as handle:
                                            handle.write(json.dumps(row, sort_keys=True) + "\n")

                if "regression" in requested_tasks and len(reg_support_indices):
                    regression_k = int(max(eval_cfg.get("retrieval_k", [20])))
                    truth, pred, lower, upper, branch_diagnostics = predict_regression(
                        reg_experts, reg_embeddings, reg_labels, reg_query_indices,
                        reg_support_indices, regression_k,
                        support_case_ids=[reg_tasks[int(idx)][2] for idx in reg_support_indices],
                        eval_cfg=eval_cfg,
                        case_budget=budget,
                    )
                    process_duration_hours = [
                        float(target)
                        + float(reg_tasks[int(index)][0][-1]["time_from_start"])
                        / 3600.0
                        for target, index in zip(truth, reg_query_indices)
                    ]
                    row = {
                        "task": "regression", "log": log_name,
                        "evaluation_split": config.get(
                            "evaluation_split", "unspecified"
                        ),
                        "experiment": config.get("experiment_name", "unnamed"),
                        "repetition": repetition, "support_scenario": scenario,
                        "case_budget": budget, "support_prefixes": int(len(reg_support_indices)),
                        "retrieval_k": regression_k,
                        "regression_mode": str(
                            reg_experts[0].proto_head.regression_mode
                        ),
                        "regression_num_transforms": int(
                            reg_experts[0].proto_head.time_transform_bank.num_transforms
                            if reg_experts[0].proto_head.time_transform_bank is not None else 1
                        ),
                        "regression_transform_aggregation": str(
                            reg_experts[0].proto_head.time_transform_bank.aggregation
                            if reg_experts[0].proto_head.time_transform_bank is not None else "single"
                        ),
                        **_virtual_expert_row(eval_cfg, "regression"),
                        **regression_metrics(
                            truth,
                            pred,
                            lower,
                            upper,
                            process_duration_hours=process_duration_hours,
                        ),
                        "prefix_length_metrics": prefix_length_metrics(
                            "regression",
                            [
                                prefix_task_length(reg_tasks, int(idx))
                                for idx in reg_query_indices
                            ],
                            truth,
                            pred,
                        ),
                    }
                    if reg_routing is not None:
                        row["expert_routing"] = reg_routing
                    if eval_cfg.get("regression_branch_diagnostics", False):
                        row["regression_branch_diagnostics"] = branch_diagnostics
                    rows.append(row)
                    with output_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows
