import copy
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from training_debug import (
    MetricAccumulator,
    TrainingDiagnostics,
    gradient_metrics,
    loss_gradient_metrics,
    model_state_metrics,
    parameter_group,
    parameter_update_metrics,
    snapshot_trainable_parameters,
    scalar_metrics,
)
from training_strategies.episodic_strategy import run_episodic_step
from training_strategies.retrieval_strategy import run_retrieval_step
from training_strategies.train_utils import evaluate_embedding_quality
from training_log_sets import (
    active_training_log_sets,
    choose_training_log_set,
    training_log_set_weight,
)


def split_params_for_proto(model):
    base_params = []
    proto_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (proto_params if ".proto_head." in name else base_params).append(parameter)
    return base_params, proto_params


def expert_adapter_diversity_loss(model):
    """Penalize collapsed lightweight experts without duplicating the encoder."""
    vectors = []
    for expert in getattr(model, "experts", ()):
        adapter = getattr(expert, "expert_adapter", None)
        if adapter is None:
            continue
        parameters = [
            parameter.reshape(-1)
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ]
        if parameters:
            vectors.append(torch.cat(parameters))
    if len(vectors) < 2:
        reference = next(model.parameters())
        return reference.new_zeros(())
    normalized = [vector / vector.norm().clamp_min(1e-8) for vector in vectors]
    similarities = torch.stack(
        [
            (left * right).sum().square()
            for index, left in enumerate(normalized)
            for right in normalized[index + 1 :]
        ]
    )
    return similarities.mean()


def adaptive_clip_grad_(parameters, factor=0.02, epsilon=1e-3):
    """Unit-wise AGC, excluding biases/norm vectors and zero-init tensors."""
    parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None and parameter.requires_grad
    ]
    if not parameters:
        return 0.0, 0.0
    total_squared = 0.0
    clipped = 0
    eligible_units =0
    factor = max(float(factor), 0.0)
    epsilon = max(float(epsilon), 1e-12)
    for parameter in parameters:
        grad_norm = parameter.grad.detach().float().norm()
        total_squared += float(grad_norm.square().cpu())
        # Biases and normalization scales have no meaningful output-unit axis.
        # Exactly zero-initialized heads must receive an unconstrained first
        # update or an epsilon-scale ceiling erases their learning signal.
        if parameter .ndim <=1 or torch .count_nonzero (parameter .detach ())==0 :
            continue
        unit_dims =tuple (range (1 ,parameter .ndim ))
        parameter_units =parameter .detach ().float ().norm (
        dim =unit_dims ,keepdim =True ).clamp_min (epsilon )
        gradient_units =parameter .grad .detach ().float ().norm (
        dim =unit_dims ,keepdim =True )
        maximum =factor *parameter_units
        should_clip =gradient_units >maximum
        scale =(maximum /gradient_units .clamp_min (1e-12 )).clamp_max (1.0 )
        parameter .grad .mul_ (scale .to (parameter .grad .dtype ))
        clipped +=int (should_clip .sum ().item ())
        eligible_units +=int (should_clip .numel ())
    return total_squared **0.5 ,clipped /max (eligible_units ,1 )


def build_training_state(
    epoch, model, optimizer, scheduler, scaler, log_set_rng, config
):
    """Capture every state required to reproduce the next optimizer step."""
    return {
        "format_version": 2,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "log_set_rng": log_set_rng.getstate(),
        "config": copy.deepcopy(config),
    }


def gpu_utilization_percent():
    """Return a best-effort instantaneous GPU utilization sample."""
    if not torch.cuda.is_available():
        return 0.0
    utilization = getattr(torch.cuda, "utilization", None)
    if utilization is None:
        return None
    try:
        return float(utilization())
    except (OSError, RuntimeError, AttributeError, ImportError):
        return None


def optimizer_parameter_groups(model, base_lr, lr_multipliers=None):
    """Build legacy base/proto groups plus opt-in component LR groups."""
    multipliers = {
        str(group): float(multiplier)
        for group, multiplier in (lr_multipliers or {}).items()
        if float(multiplier) > 0.0 and abs(float(multiplier) - 1.0) > 1e-12
    }
    special_parameters = {group: [] for group in multipliers}
    base_params = []
    proto_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = parameter_group(name)
        if group in special_parameters:
            special_parameters[group].append(parameter)
        elif ".proto_head." in name:
            proto_params.append(parameter)
        else:
            base_params.append(parameter)

    groups = []
    proto_group_index = None
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    elif proto_params:
        groups.append({"params": proto_params, "lr": base_lr})
        proto_params = []
    if proto_params:
        groups.append({"params": proto_params, "lr": base_lr})
        proto_group_index = len(groups) - 1
    special_group_indices = {}
    for group in sorted(special_parameters):
        parameters = special_parameters[group]
        if not parameters:
            continue
        groups.append(
            {
                "params": parameters,
                "lr": base_lr * multipliers[group],
            }
        )
        special_group_indices[group] = len(groups) - 1
    return groups, proto_group_index, special_group_indices, multipliers


def _strategy_step(
    strategy,
    active_expert,
    task_pool,
    task_type,
    config,
    should_shuffle_labels,
    return_diagnostics,
):
    if strategy == "episodic":
        return run_episodic_step(
            active_expert,
            task_pool,
            task_type,
            config,
            should_shuffle_labels,
            return_diagnostics=return_diagnostics,
        )
    if strategy == "retrieval":
        return run_retrieval_step(
            active_expert,
            task_pool,
            task_type,
            config,
            return_diagnostics=return_diagnostics,
        )
    raise ValueError(f"Unknown concrete training strategy: {strategy}")


def _validation_accumulator(
    model,
    validation_tasks,
    config,
    *,
    accumulator=None,
    log_set_name=None,
    schedule_weight=1.0,
):
    diagnostics_cfg = config.get("training_diagnostics", {}) or {}
    if not diagnostics_cfg.get("enabled", False) or not validation_tasks:
        return None
    steps_per_pool = max(
        1, int(diagnostics_cfg.get("validation_steps_per_pool", 1))
    )
    seed = int(config.get("seed", 42)) + int(
        diagnostics_cfg.get("validation_seed_offset", 100000)
    )

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    module_states = [(module, module.training) for module in model.modules()]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic encoder/dropout behavior, but raw training-time classifier
    # logits. The transform bank stays in eval mode so scale augmentation is off.
    model.eval()
    for expert in model.experts:
        expert.proto_head.train(True)
        if expert.proto_head.time_transform_bank is not None:
            expert.proto_head.time_transform_bank.train(False)

    accumulator = accumulator or MetricAccumulator()
    total_step_count =sum (
    len ([pool for pool in validation_tasks .get (task_type,[])if pool ])
    *steps_per_pool for task_type in ('classification','regression'))
    training_strategy = str(config.get("training_strategy", "episodic"))
    with torch.no_grad():
        for task_type in ("classification", "regression"):
            pools = validation_tasks.get(task_type, [])
            task_step_count =max (
            1 ,len ([pool for pool in pools if pool ])*steps_per_pool )
            for pool_index, pool in enumerate(pools):
                if not pool:
                    continue
                for repeat in range(steps_per_pool):
                    expert_index = (pool_index * steps_per_pool + repeat) % model.num_experts
                    active_expert = model.experts[expert_index]
                    concrete_strategy = training_strategy
                    if concrete_strategy == "mixed":
                        concrete_strategy = "retrieval" if repeat % 2 == 0 else "episodic"
                    loss, episode, metrics = _strategy_step(
                        concrete_strategy,
                        active_expert,
                        pool,
                        task_type,
                        config,
                        False,
                        True,
                    )
                    if loss is not None and torch.isfinite(loss):
                        metrics["loss/total"] = loss
                    log_set_prefixes =(
                        (
                            f"log_set/{log_set_name}",
                            f"log_set/{log_set_name}/task/{task_type}",
                            f"log_set/{log_set_name}/pool/{pool_index}",
                        )
                        if log_set_name is not None
                        else ()
                    )
                    weighted_prefixes =(
                        (
                            (
                                "schedule_weighted",
                                float(schedule_weight) / max(total_step_count, 1),
                            ),
                            (
                                f"schedule_weighted/task/{task_type}",
                                float(schedule_weight) / task_step_count,
                            ),
                        )
                        if log_set_name is not None
                        else ()
                    )
                    accumulator.add(
                        metrics,
                        prefixes=(
                            f"task/{task_type}",
                            f"expert/{expert_index}",
                            f"pool/{pool_index}",
                            f"episode/{episode}",
                            f"task/{task_type}/expert/{expert_index}",
                            f"task/{task_type}/pool/{pool_index}",
                            *log_set_prefixes,
                        ),
                        weighted_prefixes=weighted_prefixes,
                    )

    for module, training in module_states:
        module.training = training
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return accumulator


def _normalize_training_task_sets(training_tasks, config):
    """Accept both legacy flat tasks and the epoch-ranged task-set format."""
    if isinstance(training_tasks, dict) and (
        "classification" in training_tasks or "regression" in training_tasks
    ):
        return [
            {
                "name": "default",
                "start_epoch": 1,
                "end_epoch": int(config["epochs"]),
                "tasks": training_tasks,
            }
        ]
    if not isinstance(training_tasks, (list, tuple)) or not training_tasks:
        raise ValueError("Training tasks must contain at least one log set.")

    normalized = []
    for index, task_set in enumerate(training_tasks):
        if not isinstance(task_set, dict):
            raise ValueError(f"Training task set {index + 1} must be a mapping.")
        tasks = task_set.get("tasks", task_set)
        if not isinstance(tasks, dict):
            raise ValueError(
                f"Training task set '{task_set.get('name', index + 1)}' "
                "has invalid tasks."
            )
        normalized.append(
            {
                "name": str(task_set.get("name", f"set_{index + 1}")),
                "start_epoch": int(task_set.get("start_epoch", 1)),
                "end_epoch": int(task_set.get("end_epoch", config["epochs"])),
                "weight_schedule": task_set.get("weight_schedule"),
                "tasks": tasks,
            }
        )
    return normalized


def _pools_for_task_set(task_set):
    tasks = task_set["tasks"]
    classification = [
        (index, pool)
        for index, pool in enumerate(tasks.get("classification", []))
        if pool
    ]
    regression = [
        (index, pool)
        for index, pool in enumerate(tasks.get("regression", []))
        if pool
    ]
    return classification, regression


def _validation_for_task_set(validation_task_sets, selected_name):
    if not validation_task_sets:
        return None
    for task_set in validation_task_sets:
        if task_set["name"] == selected_name:
            return task_set["tasks"]
    return None


def _validation_for_active_task_sets(
    model, validation_task_sets, active_task_sets, config, epoch
):
    if not validation_task_sets:
        return None
    accumulator =MetricAccumulator ()
    evaluated =0
    for task_set in active_task_sets :
        tasks =_validation_for_task_set (
        validation_task_sets ,task_set ['name'])
        if not tasks :continue
        _validation_accumulator (
        model ,tasks ,config ,accumulator =accumulator ,
        log_set_name =task_set ['name'],
        schedule_weight =training_log_set_weight (task_set ,epoch ))
        evaluated +=1
    return accumulator if evaluated else None


def train(
    model,
    training_tasks,
    loader,
    config,
    checkpoint_dir,
    resume_epoch=0,
    stop_after_epoch=None,
    cleanup_checkpoints=False,
    validation_tasks=None,
    resume_state=None,
):
    print("🚀 Starting meta-training...")
    run_config_snapshot = copy.deepcopy(config)
    if resume_epoch > 0:
        print(f"--- Resuming from epoch {resume_epoch + 1} ---")
    (
        optimizer_groups,
        proto_group_index,
        special_group_indices,
        lr_multipliers,
    ) = optimizer_parameter_groups(
        model,
        config["lr"],
        config.get("training_lr_multipliers", {}),
    )
    optimizer = optim.AdamW(
        optimizer_groups,
        lr=config["lr"],
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    base_reference_multiplier = next(
        (
            lr_multipliers[group]
            for group, index in special_group_indices.items()
            if index == 0
        ),
        1.0,
    )
    scheduler_horizon = max(1, int(config["epochs"]) - int(resume_epoch))
    scheduler = CosineAnnealingLR(
        optimizer, T_max=scheduler_horizon, eta_min=1e-6
    )
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"✅ Automatic Mixed Precision (AMP) enabled: {use_amp}")

    training_task_sets = _normalize_training_task_sets(training_tasks, config)
    validation_task_sets = (
        _normalize_training_task_sets(validation_tasks, config)
        if validation_tasks is not None
        else []
    )
    empty_sets = [
        task_set["name"]
        for task_set in training_task_sets
        if not any(_pools_for_task_set(task_set))
    ]
    if empty_sets:
        raise ValueError(
            "Training log sets produced no valid tasks: " + ", ".join(empty_sets)
        )
    print("✅ Training log sets:")
    for task_set in training_task_sets:
        print(
            f"  - {task_set['name']}: epochs "
            f"[{task_set['start_epoch']}, {task_set['end_epoch']}]"
        )
    log_set_rng = random.Random(int(config.get("seed", 42)) + 7919)
    if resume_state is not None:
        if int(resume_state.get("epoch", -1)) != int(resume_epoch):
            raise ValueError("Resume-state epoch does not match resume_epoch")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state.get("scaler", {}))
        random.setstate(resume_state["python_rng"])
        np.random.set_state(resume_state["numpy_rng"])
        torch.random.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        log_set_rng.setstate(resume_state["log_set_rng"])
        print("✅ Restored optimizer, scheduler, AMP scaler, and all RNG states.")
    training_strategy = config.get("training_strategy", "episodic")
    print(f"✅ Training Strategy: '{training_strategy}'")
    if training_strategy in {"retrieval", "mixed"}:
        print(f"  - Retrieval k (train): {config.get('retrieval_train_k', 5)}")
        print(
            "  - Retrieval batch size (train): "
            f"{config.get('retrieval_train_batch_size', 64)}"
        )

    proto_warmup_epochs = int(config.get("proto_head_warmup_epochs", 2))
    proto_lr_multiplier_after = float(
        config.get("proto_head_lr_mult_after_warmup", 0.1)
    )
    base_variance_weight = float(config.get("retrieval_var_weight", 0.0))
    base_covariance_weight = float(config.get("retrieval_cov_weight", 0.0))
    base_contrastive_weight = float(
        config.get("retrieval_contrastive_weight", 0.0)
    )
    regularization_ramp_epochs = max(
        1, int(config.get("retrieval_reg_ramp_epochs", 5))
    )
    retrieval_k_start = int(config.get("retrieval_k_start", 12))
    retrieval_k_end = int(
        config.get("retrieval_k_end", config.get("retrieval_train_k", 20))
    )
    retrieval_k_ramp_epochs = max(
        1, int(config.get("retrieval_k_ramp_epochs", 8))
    )
    negative_random_start = float(
        config.get("retrieval_neg_random_frac_start", 0.60)
    )
    negative_random_end = float(
        config.get("retrieval_neg_random_frac_end", 0.15)
    )
    nearest_positive_epochs = max(
        0, int(config.get("retrieval_pos_nearest_epochs", 2))
    )
    contrastive_ramp_config = config.get("retrieval_contrastive_ramp", False)
    contrastive_ramp = (
        contrastive_ramp_config.strip().lower() in {"1", "true", "yes", "y", "on"}
        if isinstance(contrastive_ramp_config, str)
        else bool(contrastive_ramp_config)
    )
    shuffle_strategy = str(config.get("episodic_label_shuffle", "no")).lower()
    print(f"✅ Episodic Label Shuffle strategy set to: '{shuffle_strategy}'")
    if model.num_experts > 1:
        print(
            "✅ MoE Training enabled: deterministic balanced quotas across "
            f"{model.num_experts} experts."
        )

    diagnostics = TrainingDiagnostics(checkpoint_dir, config)
    if diagnostics.enabled:
        print(
            "🔬 Structured training diagnostics enabled: component losses, "
            "heads, gradients, updates, and case-held-out validation."
        )
    gradient_clip_norm = max(float(config.get("gradient_clip_norm", 1.0)), 0.0)
    gradient_clip_mode = str(config.get("gradient_clip_mode", "global")).lower()
    last_saved_epoch = 0
    for epoch in range(resume_epoch, config["epochs"]):
        epoch_number = epoch + 1
        active_task_sets = active_training_log_sets(
            training_task_sets, epoch_number
        )
        active_names = ", ".join(task_set["name"] for task_set in active_task_sets)
        print(
            f"\n🗂️ Epoch {epoch_number}: mixing training log sets "
            f"[{active_names}] per successful optimizer step"
        )
        model.train()
        epoch_started_at =time .monotonic ()
        if torch .cuda .is_available ():
            torch .cuda .reset_peak_memory_stats ()
        diagnostics.start_epoch()
        parameter_snapshot = (
            snapshot_trainable_parameters(model) if diagnostics.enabled else {}
        )
        total_loss = 0.0
        finite_loss_steps = 0
        optimizer_steps = 0
        amp_overflow_steps = 0
        skipped_steps = 0
        nonfinite_steps = 0
        task_counts = {"classification": 0, "regression": 0}
        expert_counts = {index: 0 for index in range(model.num_experts)}
        log_set_counts ={task_set ["name"]:0 for task_set in active_task_sets }
        examples_processed =0
        base_lr = (
            optimizer.param_groups[0]["lr"] / base_reference_multiplier
        )
        if proto_group_index is not None:
            proto_multiplier = (
                1.0 if epoch < proto_warmup_epochs else proto_lr_multiplier_after
            )
            optimizer.param_groups[proto_group_index]["lr"] = (
                base_lr * proto_multiplier
            )
        for group, group_index in special_group_indices.items():
            optimizer.param_groups[group_index]["lr"] = (
                base_lr * lr_multipliers[group]
            )
        if training_strategy in {"retrieval", "mixed"}:
            ramp = min(
                1.0, float(epoch + 1) / float(regularization_ramp_epochs)
            )
            config["retrieval_var_weight"] = base_variance_weight * ramp
            config["retrieval_cov_weight"] = base_covariance_weight * ramp
            if contrastive_ramp:
                config["retrieval_contrastive_weight"] = (
                    base_contrastive_weight * ramp
                )
            k_ramp = min(
                1.0, float(epoch + 1) / float(retrieval_k_ramp_epochs)
            )
            current_k = int(
                round(
                    retrieval_k_start
                    + (retrieval_k_end - retrieval_k_start) * k_ramp
                )
            )
            max_k_by_batch = max(
                1, int(config.get("retrieval_train_batch_size", 64)) - 1
            )
            config["retrieval_train_k"] = max(
                1, min(current_k, max_k_by_batch)
            )
            negative_random = negative_random_start + (
                negative_random_end - negative_random_start
            ) * k_ramp
            config["retrieval_neg_random_frac"] = min(
                1.0, max(0.0, negative_random)
            )
            config["retrieval_pos_use_nearest"] = epoch < nearest_positive_epochs

        should_shuffle_labels = (
            shuffle_strategy == "yes"
            or (shuffle_strategy == "mixed" and epoch % 2 == 0)
        )
        description = f"Epoch {epoch + 1}/{config['epochs']} [mixed log sets]"
        if shuffle_strategy != "no":
            description += f" (Shuffle: {'ON' if should_shuffle_labels else 'OFF'})"
        target_optimizer_steps = int(config["episodes_per_epoch"])
        classification_probability = min(
            1.0,
            max(0.0, float(config.get("classification_task_probability", 0.5))),
        )
        classification_quota = int(
            round(target_optimizer_steps * classification_probability)
        )
        planned_tasks = (
            ["classification"] * classification_quota
            + ["regression"] * (target_optimizer_steps - classification_quota)
        )
        random.shuffle(planned_tasks)
        planned_experts = [
            index % model.num_experts for index in range(target_optimizer_steps)
        ]
        random.shuffle(planned_experts)
        attempt = 0
        max_attempts = max(target_optimizer_steps * 50, 100)
        progress_bar = tqdm(total=target_optimizer_steps, desc=description)
        while optimizer_steps < target_optimizer_steps:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Could not obtain {target_optimizer_steps} successful optimizer "
                    f"steps after {attempt} attempts in epoch {epoch_number}."
                )
            step = attempt
            attempt += 1
            selected_task_set = choose_training_log_set(
                training_task_sets, epoch_number, rng=log_set_rng
            )
            classification_pools, regression_pools = _pools_for_task_set(
                selected_task_set
            )
            expert_index = planned_experts[optimizer_steps]
            active_expert = model.experts[expert_index]
            concrete_strategy = training_strategy
            if training_strategy == "mixed":
                concrete_strategy = (
                    "retrieval" if optimizer_steps % 2 == 0 else "episodic"
                )
            task_type = planned_tasks[optimizer_steps]
            available_pools = (
                classification_pools
                if task_type == "classification"
                else regression_pools
            )
            if not available_pools:
                skipped_steps += 1
                continue
            pool_index, task_pool = random.choice(available_pools)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                result = _strategy_step(
                    concrete_strategy,
                    active_expert,
                    task_pool,
                    task_type,
                    config,
                    should_shuffle_labels,
                    diagnostics.enabled,
                )
            if diagnostics.enabled:
                loss, progress_task, step_metrics = result
            else:
                loss, progress_task = result
                step_metrics = {}

            sampled_step = diagnostics.should_record_step(step)
            if loss is None:
                skipped_steps += 1
                step_metrics["optimization/step_applied"] = 0.0
            elif not torch.isfinite(loss):
                nonfinite_steps += 1
                step_metrics["optimization/nonfinite_loss"] = 1.0
                step_metrics["optimization/step_applied"] = 0.0
            else:
                diversity_weight =max (0.0 ,float (
                (config .get ("moe_settings",{})or {}).get (
                "expert_diversity_weight",0.0 )))
                if diversity_weight >0.0 :
                    diversity_loss =expert_adapter_diversity_loss (model )
                    loss =loss +diversity_weight *diversity_loss
                    step_metrics ["loss/expert_diversity_raw"]=diversity_loss
                    step_metrics ["loss/expert_diversity_weighted"]=(
                    diversity_weight *diversity_loss )
                if diagnostics.should_record_loss_gradients(step):
                    step_metrics.update(loss_gradient_metrics(model, step_metrics))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if sampled_step:
                    step_metrics.update(gradient_metrics(model))
                if gradient_clip_mode == "adaptive":
                    preclip_norm, clip_fraction = adaptive_clip_grad_(
                        model.parameters(),
                        factor=config.get("adaptive_gradient_clip_factor", 0.02),
                        epsilon=config.get("adaptive_gradient_clip_epsilon", 1e-3),
                    )
                    step_metrics["optimization/gradient_total_preclip"] = preclip_norm
                    step_metrics["optimization/gradient_clip_fraction"] = clip_fraction
                elif gradient_clip_norm > 0:
                    preclip_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    step_metrics["optimization/gradient_total_preclip"] = preclip_norm
                    step_metrics["optimization/gradient_clip_fraction"] = float(
                        float(preclip_norm) > gradient_clip_norm
                    )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                applied = scale_after >= scale_before
                step_metrics["optimization/amp_scale"] = scale_after
                step_metrics["optimization/amp_overflow"] = float(not applied)
                step_metrics["optimization/step_applied"] = float(applied)
                if applied:
                    finite_loss_steps += 1
                    total_loss += float(loss.detach().cpu())
                    optimizer_steps += 1
                    task_counts[task_type] += 1
                    expert_counts[expert_index] += 1
                    log_set_counts [selected_task_set ["name"]]+=1
                    if concrete_strategy =="retrieval":
                        examples_processed +=min (
                        int (config .get ("retrieval_train_batch_size",64 )),
                        len (task_pool ),
                        )
                    else :
                        examples_processed +=int (config .get ("num_queries",0 ))
                    progress_bar.update(1)
                else:
                    amp_overflow_steps += 1
                step_metrics["loss/total"] = loss

            diagnostics.add_step(
                epoch + 1,
                step,
                task_type,
                expert_index,
                pool_index,
                progress_task,
                step_metrics,
                sampled=sampled_step,
            )
            postfix = {
                "loss": f"{loss.item():.4f}" if loss is not None else "N/A",
                "task": progress_task,
                "attempts": attempt,
            }
            if model.num_experts > 1:
                postfix["expert"] = expert_index
            progress_bar.set_postfix(postfix)
        progress_bar.close()

        average_loss = total_loss / max(finite_loss_steps, 1)
        legacy_average_loss = total_loss / max(attempt, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_elapsed_seconds =max (time .monotonic ()-epoch_started_at ,1e-9 )
        message = (
            f"\nEpoch {epoch + 1} finished. Average Loss: {average_loss:.4f} "
            f"({optimizer_steps}/{target_optimizer_steps} optimizer steps "
            f"from {attempt} attempts)"
        )
        if proto_group_index is not None:
            message += (
                f" | Base LR: {current_lr:.6f}"
                f" | Proto LR: {optimizer.param_groups[proto_group_index]['lr']:.6f}"
            )
        else:
            message += f" | Current LR: {current_lr:.6f}"
        if special_group_indices:
            special_lrs = ", ".join(
                f"{group}={optimizer.param_groups[index]['lr']:.6f}"
                for group, index in special_group_indices.items()
            )
            message += f" | Component LRs: {special_lrs}"
        print(message)

        validation = _validation_for_active_task_sets(
            model,
            validation_task_sets,
            active_task_sets,
            config,
            epoch_number,
        )
        update_metrics = (
            parameter_update_metrics(model, parameter_snapshot)
            if diagnostics.enabled
            else {}
        )
        state_metrics = model_state_metrics(model) if diagnostics.enabled else {}
        epoch_metrics = {
            "finite_loss_steps": finite_loss_steps,
            "optimizer_steps_applied": optimizer_steps,
            "amp_overflow_steps": amp_overflow_steps,
            "skipped_steps": skipped_steps,
            "nonfinite_steps": nonfinite_steps,
            "step_success_fraction": optimizer_steps
            / max(attempt, 1),
            "average_loss_successful_steps": average_loss,
            "legacy_average_loss_scheduled_steps": legacy_average_loss,
            "epoch_elapsed_seconds":epoch_elapsed_seconds ,
            "optimizer_steps_per_second":optimizer_steps /epoch_elapsed_seconds ,
            "examples_processed":examples_processed ,
            "examples_per_second":examples_processed /epoch_elapsed_seconds ,
            "gpu_peak_memory_bytes":(
            torch .cuda .max_memory_allocated ()
            if torch .cuda .is_available ()else 0
            ),
            "gpu_utilization_percent":gpu_utilization_percent (),
            **{
                f"task/{task}/steps": count for task, count in task_counts.items()
            },
            **{
                f"expert/{index}/steps": count
                for index, count in expert_counts.items()
            },
            **{
                f"log_set/{name}/steps":count
                for name ,count in log_set_counts .items ()
            },
        }
        schedule = {
            "base_lr": current_lr,
            "proto_lr": (
                optimizer.param_groups[proto_group_index]["lr"]
                if proto_group_index is not None
                else current_lr
            ),
            "retrieval_k": config.get("retrieval_train_k", 0),
            "negative_random_fraction": config.get(
                "retrieval_neg_random_frac", 0.0
            ),
            "contrastive_weight": config.get(
                "retrieval_contrastive_weight", 0.0
            ),
            "variance_weight": config.get("retrieval_var_weight", 0.0),
            "covariance_weight": config.get("retrieval_cov_weight", 0.0),
            **{
                f"component_lr/{group}": optimizer.param_groups[index]["lr"]
                for group, index in special_group_indices.items()
            },
        }
        with open(
            os.path.join(checkpoint_dir, "training_metrics.jsonl"),
            "a",
            encoding="utf-8",
        ) as metrics_handle:
            metrics_handle.write(
                json.dumps(
                    {
                        "epoch": epoch_number,
                        "active_training_log_sets": [
                            task_set["name"] for task_set in active_task_sets
                        ],
                        "epoch_metrics": scalar_metrics(epoch_metrics),
                        "schedule": scalar_metrics(schedule),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        diagnostics.finish_epoch(
            epoch + 1,
            validation,
            epoch_metrics,
            state_metrics,
            update_metrics,
            schedule,
            context={
                "training_log_set": "weighted_mixture",
                "active_training_log_sets": [
                    task_set["name"] for task_set in active_task_sets
                ],
            },
        )
        if diagnostics.enabled and validation is not None:
            validation_summary = validation.summary()
            for task in ("classification", "regression"):
                key = f"task/{task}/loss/total"
                if key in validation_summary:
                    print(
                        f"  - Held-out {task} loss: "
                        f"{validation_summary[key]['mean']:.4f}"
                    )

        evaluate_embedding_quality(model.experts[0], loader)
        scheduler.step()
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_epoch_{epoch + 1}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        training_state_path = os.path.join(
            checkpoint_dir, f"training_state_epoch_{epoch + 1}.pth"
        )
        torch.save(
            build_training_state(
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                log_set_rng,
                run_config_snapshot,
            ),
            training_state_path,
        )
        print(f"💾 Model checkpoint saved to {checkpoint_path}")
        last_saved_epoch = epoch + 1
        if stop_after_epoch is not None and epoch + 1 == stop_after_epoch:
            print(
                f"\n--- 🛑 Stopping training after epoch {stop_after_epoch} "
                "as requested. ---"
            )
            break

    print("✅ Meta-training complete.")
    if cleanup_checkpoints and last_saved_epoch > 0:
        print("--- 🧹 Cleaning up intermediate checkpoints...")
        file_to_keep = f"model_epoch_{last_saved_epoch}.pth"
        print(f"  - Keeping final checkpoint: {file_to_keep}")
        checkpoint_pattern = re.compile(r"^model_epoch_(\d+)\.pth$")
        state_pattern = re.compile(r"^training_state_epoch_(\d+)\.pth$")
        try:
            removed_count = 0
            for filename in os.listdir(checkpoint_dir):
                checkpoint_match = checkpoint_pattern.match(filename)
                state_match = state_pattern.match(filename)
                keep_state = f"training_state_epoch_{last_saved_epoch}.pth"
                if (
                    (checkpoint_match and filename != file_to_keep)
                    or (state_match and filename != keep_state)
                ):
                    os.remove(os.path.join(checkpoint_dir, filename))
                    removed_count += 1
            if removed_count > 0:
                print(f"  - Removed {removed_count} intermediate checkpoint(s).")
            else:
                print("  - No intermediate checkpoints found to remove.")
        except Exception as error:
            print(f"  - ⚠️ Error during checkpoint cleanup: {error}")
    elif cleanup_checkpoints:
        print("--- ⚠️ Skipping checkpoint cleanup: No epoch was saved. ---")
