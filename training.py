import os
import random
import re

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
    parameter_update_metrics,
    snapshot_trainable_parameters,
)
from training_strategies.episodic_strategy import run_episodic_step
from training_strategies.retrieval_strategy import run_retrieval_step
from training_strategies.train_utils import evaluate_embedding_quality


def split_params_for_proto(model):
    base_params = []
    proto_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (proto_params if ".proto_head." in name else base_params).append(parameter)
    return base_params, proto_params


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


def _validation_accumulator(model, validation_tasks, config):
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

    accumulator = MetricAccumulator()
    training_strategy = str(config.get("training_strategy", "episodic"))
    with torch.no_grad():
        for task_type in ("classification", "regression"):
            pools = validation_tasks.get(task_type, [])
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
                    accumulator.add(
                        metrics,
                        prefixes=(
                            f"task/{task_type}",
                            f"expert/{expert_index}",
                            f"pool/{pool_index}",
                            f"episode/{episode}",
                            f"task/{task_type}/expert/{expert_index}",
                            f"task/{task_type}/pool/{pool_index}",
                        ),
                    )

    for module, training in module_states:
        module.training = training
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return accumulator


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
):
    print("🚀 Starting meta-training...")
    if resume_epoch > 0:
        print(f"--- Resuming from epoch {resume_epoch + 1} ---")
    base_params, proto_params = split_params_for_proto(model)
    optimizer_groups = []
    proto_group_index = None
    if base_params:
        optimizer_groups.append({"params": base_params, "lr": config["lr"]})
    elif proto_params:
        optimizer_groups.append({"params": proto_params, "lr": config["lr"]})
        proto_params = []
    if proto_params:
        optimizer_groups.append({"params": proto_params, "lr": config["lr"]})
        proto_group_index = len(optimizer_groups) - 1
    optimizer = optim.AdamW(
        optimizer_groups,
        lr=config["lr"],
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    if resume_epoch > 0:
        scheduler.last_epoch = resume_epoch
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"✅ Automatic Mixed Precision (AMP) enabled: {use_amp}")

    classification_pools = [
        (index, pool)
        for index, pool in enumerate(training_tasks["classification"])
        if pool
    ]
    regression_pools = [
        (index, pool)
        for index, pool in enumerate(training_tasks["regression"])
        if pool
    ]
    if not classification_pools and not regression_pools:
        print("❌ Error: No valid training tasks available. Aborting training.")
        return
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
            "✅ MoE Training enabled: Randomly selecting 1 of "
            f"{model.num_experts} experts per step."
        )

    diagnostics = TrainingDiagnostics(checkpoint_dir, config)
    if diagnostics.enabled:
        print(
            "🔬 Structured training diagnostics enabled: component losses, "
            "heads, gradients, updates, and case-held-out validation."
        )
    gradient_clip_norm = max(float(config.get("gradient_clip_norm", 1.0)), 0.0)
    last_saved_epoch = 0
    for epoch in range(resume_epoch, config["epochs"]):
        model.train()
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
        base_lr = optimizer.param_groups[0]["lr"]
        if proto_group_index is not None:
            proto_multiplier = (
                1.0 if epoch < proto_warmup_epochs else proto_lr_multiplier_after
            )
            optimizer.param_groups[proto_group_index]["lr"] = (
                base_lr * proto_multiplier
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
        description = f"Epoch {epoch + 1}/{config['epochs']}"
        if shuffle_strategy != "no":
            description += f" (Shuffle: {'ON' if should_shuffle_labels else 'OFF'})"
        progress_bar = tqdm(range(config["episodes_per_epoch"]), desc=description)
        for step in progress_bar:
            expert_index = random.randint(0, model.num_experts - 1)
            active_expert = model.experts[expert_index]
            concrete_strategy = training_strategy
            if training_strategy == "mixed":
                concrete_strategy = "retrieval" if step % 2 == 0 else "episodic"
            classification_probability = min(
                1.0,
                max(
                    0.0,
                    float(config.get("classification_task_probability", 0.5)),
                ),
            )
            task_type = (
                "classification"
                if random.random() < classification_probability
                else "regression"
            )
            if task_type == "classification" and classification_pools:
                pool_index, task_pool = random.choice(classification_pools)
            elif task_type == "regression" and regression_pools:
                pool_index, task_pool = random.choice(regression_pools)
            else:
                task_type = "regression" if regression_pools else "classification"
                available_pools = (
                    regression_pools if regression_pools else classification_pools
                )
                pool_index, task_pool = random.choice(available_pools)
            if not task_pool:
                skipped_steps += 1
                continue

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
                finite_loss_steps += 1
                if diagnostics.should_record_loss_gradients(step):
                    step_metrics.update(loss_gradient_metrics(model, step_metrics))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if sampled_step:
                    step_metrics.update(gradient_metrics(model))
                if gradient_clip_norm > 0:
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
                step_metrics["optimization/amp_scale"] = scale_after
                step_metrics["optimization/amp_overflow"] = float(
                    scale_after < scale_before
                )
                step_metrics["optimization/step_applied"] = float(
                    scale_after >= scale_before
                )
                total_loss += float(loss.detach().cpu())
                if scale_after >= scale_before:
                    optimizer_steps += 1
                    task_counts[task_type] += 1
                    expert_counts[expert_index] += 1
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
            }
            if model.num_experts > 1:
                postfix["expert"] = expert_index
            progress_bar.set_postfix(postfix)

        average_loss = total_loss / max(finite_loss_steps, 1)
        legacy_average_loss = total_loss / max(config["episodes_per_epoch"], 1)
        current_lr = optimizer.param_groups[0]["lr"]
        message = (
            f"\nEpoch {epoch + 1} finished. Average Loss: {average_loss:.4f} "
            f"({optimizer_steps}/{config['episodes_per_epoch']} optimizer steps; "
            f"{finite_loss_steps} finite losses)"
        )
        if proto_group_index is not None:
            message += (
                f" | Base LR: {current_lr:.6f}"
                f" | Proto LR: {optimizer.param_groups[proto_group_index]['lr']:.6f}"
            )
        else:
            message += f" | Current LR: {current_lr:.6f}"
        print(message)

        validation = _validation_accumulator(model, validation_tasks, config)
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
            / max(config["episodes_per_epoch"], 1),
            "average_loss_successful_steps": average_loss,
            "legacy_average_loss_scheduled_steps": legacy_average_loss,
            **{
                f"task/{task}/steps": count for task, count in task_counts.items()
            },
            **{
                f"expert/{index}/steps": count
                for index, count in expert_counts.items()
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
        }
        diagnostics.finish_epoch(
            epoch + 1,
            validation,
            epoch_metrics,
            state_metrics,
            update_metrics,
            schedule,
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
        try:
            removed_count = 0
            for filename in os.listdir(checkpoint_dir):
                if checkpoint_pattern.match(filename) and filename != file_to_keep:
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
