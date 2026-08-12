import random
from contextlib import nullcontext
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from utils.retrieval_utils import find_knn_indices


def _encode_case_ids_to_int(case_ids_np: np.ndarray) -> torch.Tensor:
    """case_ids may be strings/objects -> map to contiguous ints for tensor ops."""
    uniq = list(dict.fromkeys(case_ids_np.tolist()))
    mapping = {cid: i for i, cid in enumerate(uniq)}
    return torch.tensor([mapping[cid] for cid in case_ids_np.tolist()], dtype=torch.long)


def _autocast_disabled_for(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _sample_balanced_classification_batch(
    task_pool, batch_size, min_per_class=2, max_classes=None
):
    """
    Ensure selected classes appear at least min_per_class times, ideally across cases.
    Fall back to random sampling when constraints cannot be met.
    """
    by_label = defaultdict(list)
    for seq, label, case_id in task_pool:
        if label is None:
            continue
        if int(label) == -100:
            continue
        by_label[int(label)].append((seq, int(label), case_id))

    eligible = []
    for label, items in by_label.items():
        if len(items) < min_per_class:
            continue
        if len({cid for _, _, cid in items}) < 2:
            continue
        eligible.append(label)

    if not eligible:
        return random.sample(task_pool, batch_size)

    num_classes = min(len(eligible), max(1, batch_size // min_per_class))
    if max_classes is not None:
        num_classes = min(num_classes, max(1, int(max_classes)))
    chosen_labels = random.sample(eligible, num_classes)

    batch = []
    for label in chosen_labels:
        items = by_label[label]
        by_case = defaultdict(list)
        for item in items:
            by_case[item[2]].append(item)
        cases = list(by_case.keys())
        random.shuffle(cases)

        for case_id in cases[:min_per_class]:
            batch.append(random.choice(by_case[case_id]))

        while sum(1 for item in batch if item[1] == label) < min_per_class:
            batch.append(random.choice(items))

    if chosen_labels:
        label_cycle = chosen_labels[:]
        random.shuffle(label_cycle)
        cycle_idx = 0
        while len(batch) < batch_size:
            lbl = label_cycle[cycle_idx % len(label_cycle)]
            batch.append(random.choice(by_label[lbl]))
            cycle_idx += 1

    return batch[:batch_size]


def _valid_classification_items(task_pool):
    return [item for item in task_pool if item[1] is not None and int(item[1]) != -100]


def _weighted_episode_type(config):
    mix = config.get("fmv3_training", {}).get("episode_mix", {"balanced": 1.0})
    names = [name for name, weight in mix.items() if float(weight) > 0]
    weights = [float(mix[name]) for name in names]
    return random.choices(names, weights=weights, k=1)[0] if names else "balanced"


def _sample_classification_batch(task_pool, batch_size, episode_type, config):
    """Sample balanced, natural, long-tail, or random-shot deployment episodes."""
    valid = _valid_classification_items(task_pool)
    train_cfg = config.get("fmv3_training", {})
    case_range = train_cfg.get("cases_per_episode_range")
    if case_range and len(case_range) == 2:
        cases = list({str(item[2]) for item in valid})
        target_cases = random.randint(max(2, int(case_range[0])), max(2, int(case_range[1])))
        chosen_cases = set(random.sample(cases, min(target_cases, len(cases))))
        restricted = [item for item in valid if str(item[2]) in chosen_cases]
        if len(restricted) >= batch_size:
            valid = restricted
    if len(valid) < batch_size:
        return None
    if episode_type in {"balanced", "missing_local_label", "missing_pool_label"}:
        return _sample_balanced_classification_batch(
            valid,
            batch_size,
            min_per_class=int(config.get("retrieval_min_per_class", 2)),
            max_classes=config.get("retrieval_train_max_classes"),
        )
    if episode_type == "natural":
        return random.sample(valid, batch_size)

    by_label = defaultdict(list)
    for item in valid:
        by_label[int(item[1])].append(item)
    labels = list(by_label)
    if not labels:
        return None
    if episode_type in {"long_tail", "rare_path"}:
        labels.sort(key=lambda label: len(by_label[label]), reverse=True)
        power_range = train_cfg.get("long_tail_power_range")
        if power_range and len(power_range) == 2:
            power = random.uniform(float(power_range[0]), float(power_range[1]))
        else:
            power = float(train_cfg.get("long_tail_power", 1.5))
        power = max(power, 0.0)
        if episode_type == "long_tail":
            weights = np.asarray([(rank + 1) ** (-power) for rank in range(len(labels))], dtype=float)
        else:
            weights = np.asarray([len(by_label[label]) ** (-power) for label in labels], dtype=float)
        weights /= weights.sum()
        return [random.choice(by_label[int(np.random.choice(labels, p=weights))]) for _ in range(batch_size)]
    if episode_type == "random_shot":
        random.shuffle(labels)
        low = max(1, int(train_cfg.get("random_shot_min", 1)))
        high = max(low, int(train_cfg.get("random_shot_max", 20)))
        sampled = []
        for label in labels:
            shot = random.randint(low, high)
            items = by_label[label]
            sampled.extend(random.sample(items, min(shot, len(items))))
            if len(sampled) >= batch_size:
                break
        if len(sampled) < batch_size:
            sampled.extend(random.sample(valid, batch_size - len(sampled)))
        random.shuffle(sampled)
        return sampled[:batch_size]
    raise ValueError(f"Unknown FM-v3 episode type: {episode_type}")


def _supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
):
    """
    Supervised contrastive loss.
    Positives are same-label, different-case pairs.
    Same-case pairs are removed from denominator by heavy negative masking.
    """
    device = z.device
    temp = max(float(temperature), 1e-6)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        batch_size = z.size(0)

        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case

        # Use a large finite negative value to avoid fp16 overflow / NaNs in AMP paths.
        logits = logits.masked_fill(ignore, -1e4)

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t()) & (~ignore)
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return None

        log_prob = F.log_softmax(logits, dim=1)
        loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / pos_counts.clamp_min(1).float()
        return loss_per[valid].mean()


def _regression_neighbor_contrastive(
    z: torch.Tensor,
    y: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
    pos_k: int = 2,
):
    """
    Target-neighborhood contrastive objective for regression.
    Positives per anchor are nearest labels in target space (excluding same case).
    """
    device = z.device
    temp = max(float(temperature), 1e-6)
    pos_k = max(int(pos_k), 1)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        y = y.float().view(-1)
        batch_size = z.size(0)
        if batch_size == 0:
            return None

        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case
        logits = logits.masked_fill(ignore, -1e4)

        log_prob = F.log_softmax(logits, dim=1)

        # Vectorized nearest-target positives: |y_i - y_j| with ignored pairs
        # pushed to +inf so they never win the top-k over real candidates.
        diffs = (y.view(-1, 1) - y.view(1, -1)).abs()
        diffs = diffs.masked_fill(ignore, float("inf"))
        valid_counts = (~ignore).sum(dim=1)
        valid = valid_counts > 0
        if not valid.any():
            return None
        k_cap = min(pos_k, batch_size - 1)
        if k_cap <= 0:
            return None
        positive_idx = torch.topk(diffs, k_cap, dim=1, largest=False).indices
        pos_diffs = diffs.gather(1, positive_idx)
        pos_log_prob = log_prob.gather(1, positive_idx)
        pos_valid = torch.isfinite(pos_diffs)
        pos_counts = pos_valid.sum(dim=1).clamp_min(1).float()
        loss_per = -(pos_log_prob * pos_valid.float()).sum(dim=1) / pos_counts
        return loss_per[valid].mean()


def _nca_knn_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
):
    """Supervised NCA-style objective: maximize same-label probability mass."""
    device = z.device
    temp = max(float(temperature), 1e-6)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        batch_size = z.size(0)
        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case
        logits = logits.masked_fill(ignore, -1e4)

        log_prob = F.log_softmax(logits, dim=1)
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t()) & (~ignore)
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return None

        log_p_pos = torch.logsumexp(log_prob.masked_fill(~pos_mask, -1e4), dim=1)
        return (-log_p_pos[valid]).mean()


def _variance_loss(z: torch.Tensor, eps: float = 1e-4, target_std: float = 1.0):
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))


def _covariance_loss(z: torch.Tensor):
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    if n <= 1:
        return torch.tensor(0.0, device=z.device)
    cov = (z.t() @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


def run_retrieval_step(model, task_data_pool, task_type, config):
    progress_bar_task = f"retrieval_{task_type}"
    retrieval_k_train = int(config.get("retrieval_train_k", 5))
    retrieval_batch_size = int(config.get("retrieval_train_batch_size", 64))

    if len(task_data_pool) < retrieval_batch_size:
        return None, progress_bar_task

    episode_type = "regression"
    if task_type == "classification":
        episode_type = _weighted_episode_type(config)
        batch_tasks_raw = _sample_classification_batch(
            task_data_pool, retrieval_batch_size, episode_type, config
        )
        if not batch_tasks_raw:
            return None, f"{progress_bar_task}_empty"
        progress_bar_task = f"{progress_bar_task}_{episode_type}"
    else:
        batch_tasks_raw = random.sample(task_data_pool, retrieval_batch_size)

    batch_prefixes = [t[0] for t in batch_tasks_raw]
    batch_labels = np.array([t[1] for t in batch_tasks_raw])
    batch_case_ids = np.array([t[2] for t in batch_tasks_raw], dtype=object)

    time_scale_factor =None
    if task_type =="regression"and model .proto_head .regression_uses_time_transform_bank :
        time_scale_factor =model .proto_head .time_transform_bank .sample_augmentation_factor (
            next (model .parameters ()))
    all_embeddings = model._process_batch(
        batch_prefixes, task_type=task_type, time_scale_factor=time_scale_factor
    )
    z_ssl = model.proj_head(all_embeddings) if hasattr(model, "proj_head") else all_embeddings
    device = all_embeddings.device

    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
    all_embeddings_norm_detached = all_embeddings_norm.detach()
    cls_pos_k_cfg = int(config.get("retrieval_cls_pos_k", 2))
    neg_pool_factor = max(1, int(config.get("retrieval_neg_pool_factor", 4)))
    neg_random_frac = float(config.get("retrieval_neg_random_frac", 0.25))
    neg_random_frac = min(max(neg_random_frac, 0.0), 1.0)
    pos_use_nearest_cfg = config.get("retrieval_pos_use_nearest", True)
    if isinstance(pos_use_nearest_cfg, str):
        pos_use_nearest = pos_use_nearest_cfg.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        pos_use_nearest = bool(pos_use_nearest_cfg)

    contrastive_w = float(config.get("retrieval_contrastive_weight", 0.2))
    contrastive_temp = float(config.get("retrieval_contrastive_temp", 0.07))
    knn_aux_w = float(config.get("retrieval_knn_aux_weight", 0.0))
    contrastive_loss = None
    nca_loss = None
    labels_t = None
    case_ids_int = None

    if contrastive_w > 0 or (task_type == "classification" and knn_aux_w > 0):
        case_ids_int = _encode_case_ids_to_int(batch_case_ids).to(device)

    if task_type == "classification":
        labels_t = torch.as_tensor(batch_labels, dtype=torch.long, device=device)

    if contrastive_w > 0:
        if task_type == "classification":
            contrastive_loss = _supcon_loss(
                z_ssl, labels_t, case_ids_int, temperature=contrastive_temp
            )
        else:
            y_t = torch.as_tensor(batch_labels, dtype=torch.float32, device=device)
            pos_k = int(config.get("retrieval_regression_pos_k", 2))
            contrastive_loss = _regression_neighbor_contrastive(
                z_ssl, y_t, case_ids_int, temperature=contrastive_temp, pos_k=pos_k
            )

    if task_type == "classification" and knn_aux_w > 0 and labels_t is not None and case_ids_int is not None:
        nca_loss = _nca_knn_loss(
            z_ssl,
            labels_t,
            case_ids_int,
            temperature=contrastive_temp,
        )

    total_loss_for_batch = 0.0
    queries_processed = 0
    use_regression_gate_aux = (
        model.proto_head.regression_outputs_hours
        and model.proto_head.regression_gate_aux_weight > 0
    )

    if task_type == "regression":
        # Self-excluded kNN once for the batch, then head forward in groups of
        # equal neighborhood size (same math as the old per-query loop).
        labels_float = torch.as_tensor(batch_labels, dtype=torch.float32, device=device)
        if case_ids_int is None:
            case_ids_int = _encode_case_ids_to_int(batch_case_ids).to(device)
        with torch.no_grad():
            sims = all_embeddings_norm_detached @ all_embeddings_norm_detached.t()
            same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
            sims = sims.masked_fill(same_case, float("-inf"))
            valid_counts = torch.isfinite(sims).sum(dim=1)

        regression_predictions = []
        regression_targets = []
        regression_branch_predictions = []
        regression_aggregation_weights = []
        # Group queries by k_eff so we preserve per-query neighborhood size
        # while still running the expensive head in batched mode.
        groups = defaultdict(list)
        for query_i in range(retrieval_batch_size):
            n_valid = int(valid_counts[query_i].item())
            if n_valid <= 0:
                continue
            k_eff = min(retrieval_k_train, n_valid)
            groups[k_eff].append(query_i)

        for k_eff, query_ids in groups.items():
            query_idx = torch.as_tensor(query_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                neighbors = torch.topk(sims[query_idx], k_eff, dim=1).indices
            support_embeddings = all_embeddings[neighbors]
            support_labels_tensor = labels_float[neighbors]
            query_embeddings = all_embeddings[query_idx]
            if use_regression_gate_aux:
                prediction, _, diagnostics = model.proto_head.forward_regression_batched(
                    support_embeddings,
                    support_labels_tensor,
                    query_embeddings,
                    return_diagnostics=True,
                    augmentation_factor=time_scale_factor,
                )
                regression_branch_predictions.append(
                    diagnostics["branch_predictions_hours"]
                )
                regression_aggregation_weights.append(
                    diagnostics["aggregation_weights"]
                )
            else:
                prediction, _ = model.proto_head.forward_regression_batched(
                    support_embeddings,
                    support_labels_tensor,
                    query_embeddings,
                    augmentation_factor=time_scale_factor,
                )
            regression_predictions.append(prediction.reshape(-1))
            regression_targets.append(labels_float[query_idx])

        if regression_predictions:
            total_loss_for_batch = model.proto_head.regression_loss(
                torch.cat(regression_predictions, dim=0),
                torch.cat(regression_targets, dim=0),
                branch_predictions=(
                    torch.cat(regression_branch_predictions, dim=1)
                    if regression_branch_predictions else None
                ),
                aggregation_weights=(
                    torch.cat(regression_aggregation_weights, dim=1)
                    if regression_aggregation_weights else None
                ),
            )
            queries_processed = 1
    else:
        for i in range(retrieval_batch_size):
            query_label = batch_labels[i]
            query_case_id = batch_case_ids[i]
            query_embedding = all_embeddings[i : i + 1]

            with torch.no_grad():
                query_embedding_norm = all_embeddings_norm_detached[i : i + 1]

            if int(query_label) == -100:
                continue
            with torch.no_grad():
                eligible = batch_case_ids != query_case_id
                if episode_type == "missing_pool_label":
                    eligible &= batch_labels != query_label
                pool_indices_np = np.where(eligible)[0]
                if pool_indices_np.size == 0:
                    continue
                pool_indices = torch.from_numpy(pool_indices_np).to(device)

                local_eligible = eligible.copy()
                if episode_type == "missing_local_label":
                    local_eligible &= batch_labels != query_label
                local_mask = torch.from_numpy(np.where(~local_eligible)[0]).to(device)

                # Balanced FM-v2 episodes preserve the historical guaranteed-positive
                # behavior. Other episode types use ordinary retrieval and may omit it.
                if episode_type == "balanced":
                    positive_np = np.where(
                        (batch_labels == query_label) & (batch_case_ids != query_case_id)
                    )[0]
                    if positive_np.size == 0:
                        continue
                    sims = (query_embedding_norm @ all_embeddings_norm_detached.t()).squeeze(0)
                    pos_k = min(cls_pos_k_cfg, int(positive_np.size), max(1, retrieval_k_train - 1))
                    positives = torch.from_numpy(positive_np).to(device)
                    if pos_use_nearest:
                        positives = positives[torch.topk(sims[positives], pos_k).indices]
                    else:
                        positives = positives[torch.randperm(positives.numel(), device=device)[:pos_k]]
                    negative_mask = (~local_eligible) | (batch_labels == query_label)
                    negatives = find_knn_indices(
                        query_embedding_norm,
                        all_embeddings_norm_detached,
                        k=max(1, retrieval_k_train - pos_k),
                        indices_to_mask=torch.from_numpy(np.where(negative_mask)[0]).to(device),
                    )
                    support_indices = torch.cat([positives, negatives])[:retrieval_k_train]
                else:
                    support_indices = find_knn_indices(
                        query_embedding_norm,
                        all_embeddings_norm_detached,
                        k=min(retrieval_k_train, int(local_eligible.sum())),
                        indices_to_mask=local_mask,
                    )

            if support_indices.numel() == 0:
                continue
            support_embeddings = all_embeddings[support_indices]
            support_labels_tensor = labels_t[support_indices]
            global_embeddings = all_embeddings[pool_indices]
            global_labels = labels_t[pool_indices]
            logits, proto_classes, _ = model.proto_head.forward_classification(
                support_embeddings,
                support_labels_tensor,
                query_embedding,
                global_support_features=global_embeddings,
                global_support_labels=global_labels,
            )
            if logits is None:
                continue

            label_map = {orig.item(): new for new, orig in enumerate(proto_classes)}
            target_label = (
                int(config.get("fmv3_head", {}).get("abstain_label", -101))
                if episode_type == "missing_pool_label"
                else int(query_label)
            )
            mapped_label = torch.tensor(
                [label_map.get(target_label, -100)], device=device, dtype=torch.long
            )
            if mapped_label.item() == -100:
                continue

            smoothing = min(
                max(float(config.get("classification_label_smoothing", 0.05)), 0.0),
                1.0,
            )
            loss = F.cross_entropy(logits, mapped_label, label_smoothing=smoothing)

            if loss is not None and not torch.isnan(loss):
                total_loss_for_batch = total_loss_for_batch + loss
                queries_processed += 1

    loss_out = None
    if queries_processed > 0:
        loss_out = total_loss_for_batch / queries_processed
        if contrastive_loss is not None:
            loss_out = loss_out + (contrastive_w * contrastive_loss)
    elif contrastive_loss is not None:
        loss_out = contrastive_w * contrastive_loss

    if nca_loss is not None and knn_aux_w > 0:
        if loss_out is None:
            loss_out = knn_aux_w * nca_loss
        else:
            loss_out = loss_out + (knn_aux_w * nca_loss)

    var_w = float(config.get("retrieval_var_weight", 0.0))
    cov_w = float(config.get("retrieval_cov_weight", 0.0))
    if loss_out is not None and (var_w > 0 or cov_w > 0):
        with _autocast_disabled_for(device):
            reg = torch.tensor(0.0, device=device)
            if var_w > 0:
                reg = reg + (var_w * _variance_loss(z_ssl))
            if cov_w > 0:
                reg = reg + (cov_w * _covariance_loss(z_ssl))
        loss_out = loss_out + reg

    return loss_out, progress_bar_task
