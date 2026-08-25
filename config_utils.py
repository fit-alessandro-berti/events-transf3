"""YAML experiment configuration and dotted command-line overrides."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


DELETE_VALUE = "__delete__"


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if value == DELETE_VALUE:
            base.pop(key, None)
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    extends = data.pop("extends", None)
    if extends:
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        parent = load_yaml_config(str(parent_path.resolve()))
        return deep_merge(parent, data)
    return data


def parse_override(raw: str):
    if "=" not in raw:
        raise ValueError(f"Override must have KEY=VALUE form: {raw}")
    key, value = raw.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def apply_experiment_config(config: Dict[str, Any], path: str | None, overrides: Iterable[str] = ()):
    deep_merge(config, load_yaml_config(path))
    for raw in overrides:
        key, value = parse_override(raw)
        set_dotted(config, key, value)
    return config


def stable_config_hash(value: Any) -> str:
    """Return a deterministic SHA-256 for JSON/YAML-compatible metadata."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_file_hash(relative_path: str) -> str:
    return hashlib.sha256(
        (Path(__file__).resolve().parent / relative_path).read_bytes()
    ).hexdigest()


def preprocessing_config_hash(config: Dict[str, Any]) -> str:
    """Hash every setting that changes fitted loader/preprocessing semantics."""
    return stable_config_hash(
        {
            "embedding_strategy": config.get("embedding_strategy"),
            "pretrained_settings": config.get("pretrained_settings", {}),
            "learned_settings": config.get("learned_settings", {}),
            "data": {
                key: (config.get("data", {}) or {}).get(key)
                for key in (
                    "max_generic_attributes",
                    "attribute_hash_buckets",
                )
            },
            "implementation_sha256": {
                path: _source_file_hash(path)
                for path in ("data_generator.py", "utils/data_utils.py")
            },
        }
    )


def model_architecture_config_hash(config: Dict[str, Any]) -> str:
    """Hash graph-defining model settings while excluding training schedules."""
    return stable_config_hash(
        {
            "embedding_strategy": config.get("embedding_strategy"),
            "pretrained_settings": config.get("pretrained_settings", {}),
            "learned_settings": config.get("learned_settings", {}),
            "data": config.get("data", {}),
            "moe_settings": config.get("moe_settings", {}),
            "d_model": config.get("d_model"),
            "n_heads": config.get("n_heads"),
            "n_layers": config.get("n_layers"),
            "dropout": config.get("dropout"),
            "num_numerical_features": config.get("num_numerical_features"),
            "fmv3_head": config.get("fmv3_head", {}),
            "implementation_sha256": {
                path: _source_file_hash(path)
                for path in (
                    "components/attribute_encoder.py",
                    "components/char_cnn_embedder.py",
                    "components/event_encoder.py",
                    "components/learned_event_embedder.py",
                    "components/meta_learner.py",
                    "components/moe_model.py",
                    "components/pretrained_event_embedder.py",
                    "components/prototypical_head.py",
                    "components/task_confidence.py",
                    "components/temporal_adapter.py",
                )
            },
        }
    )


def save_yaml_config(config: Dict[str, Any], path: str) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def validate_run_configuration(
    config: Dict[str, Any], *, check_checkpoint_paths: bool = False
) -> str:
    """Validate explicit checkpoint semantics before data/model initialization."""
    mode = str(config.get("run_mode", "train")).strip().lower()
    mode = {"exact_resume": "resume", "init": "initialize"}.get(mode, mode)
    if mode not in {"train", "resume", "initialize", "assemble"}:
        raise ValueError(
            "run_mode must be one of train, resume, initialize, or assemble"
        )
    training_enabled = bool(config.get("training_enabled", True))
    scope = str(config.get("trainable_scope", "all")).strip().lower()
    initialize_path = config.get("initialize_from_checkpoint")
    experiment_name = str(config.get("experiment_name", "")).strip().lower()
    if "retrain" in experiment_name and mode != "train":
        raise ValueError(
            f"Retraining experiment '{experiment_name}' must use run_mode: train"
        )
    if "retrain" in experiment_name and not training_enabled:
        raise ValueError(
            f"Retraining experiment '{experiment_name}' requires training_enabled: true"
        )
    if mode != "assemble" and not training_enabled:
        raise ValueError(
            f"run_mode {mode} requires training_enabled: true; use an evaluation "
            "entry point for checkpoint-only configurations"
        )
    clip_mode = str(config.get("gradient_clip_mode", "global")).strip().lower()
    if clip_mode not in {"adaptive", "global", "none"}:
        raise ValueError(
            "gradient_clip_mode must be one of adaptive, global, or none"
        )
    if clip_mode == "global" and float(config.get("gradient_clip_norm", 0.0)) <= 0:
        raise ValueError("Global gradient clipping requires gradient_clip_norm > 0")
    if "clip" in experiment_name and clip_mode != "global":
        raise ValueError(
            f"Clipping experiment '{experiment_name}' must explicitly use "
            "gradient_clip_mode: global"
        )
    if str(config.get("embedding_strategy", "learned")).lower() == "pretrained":
        revision = (config.get("pretrained_settings", {}) or {}).get("revision")
        if not revision or str(revision).strip().lower() in {"main", "master", "latest"}:
            raise ValueError(
                "Pretrained runs require pretrained_settings.revision pinned to an "
                "immutable model revision"
            )
    if training_enabled and scope != "all" and mode not in {"resume", "initialize"}:
        raise ValueError(
            f"trainable_scope='{scope}' requires run_mode resume or initialize; "
            "a frozen random backbone is not allowed"
        )
    if mode == "initialize":
        if not initialize_path:
            raise ValueError(
                "run_mode initialize requires initialize_from_checkpoint"
            )
        artifacts_path = config.get("initialize_from_artifacts")
        if not artifacts_path:
            raise ValueError(
                "run_mode initialize requires initialize_from_artifacts; loader "
                "artifacts must never be silently refitted"
            )
        if "source_epoch" not in config or "additional_epochs" not in config:
            raise ValueError(
                "run_mode initialize requires source_epoch and additional_epochs"
            )
        source_epoch = int(config["source_epoch"])
        additional_epochs = int(config["additional_epochs"])
        if source_epoch < 0 or additional_epochs < 1:
            raise ValueError("source_epoch must be >= 0 and additional_epochs >= 1")
        final_epoch = source_epoch + additional_epochs
        if int(config.get("epochs", final_epoch)) != final_epoch:
            raise ValueError(
                "initialize epoch semantics are inconsistent: epochs must equal "
                "source_epoch + additional_epochs"
            )
        if not bool(config.get("reset_optimizer", True)):
            raise ValueError(
                "Weights-only initialization cannot preserve optimizer state; use "
                "run_mode: resume for exact continuation"
            )
        if not bool(config.get("reset_scheduler", True)):
            raise ValueError(
                "Weights-only initialization cannot preserve scheduler state; use "
                "run_mode: resume for exact continuation"
            )
        if check_checkpoint_paths and not Path(initialize_path).is_file():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {initialize_path}"
            )
        if check_checkpoint_paths and not Path(artifacts_path).is_file():
            raise FileNotFoundError(
                f"Initialization artifacts not found: {artifacts_path}"
            )
    if mode == "assemble":
        if training_enabled:
            raise ValueError("run_mode assemble requires training_enabled: false")
        assembly = config.get("assembly", {}) or {}
        required = {
            "base_checkpoint",
            "classification_checkpoint",
            "regression_checkpoint",
            "output_checkpoint",
        }
        missing = sorted(required - set(assembly))
        if missing:
            raise ValueError(
                "Assembly configuration is missing: " + ", ".join(missing)
            )
        if check_checkpoint_paths:
            for key in required - {"output_checkpoint"}:
                if not Path(assembly[key]).is_file():
                    raise FileNotFoundError(
                        f"Assembly checkpoint not found for {key}: {assembly[key]}"
                    )
    return mode


def validate_exact_resume_config(
    current: Dict[str, Any], checkpoint_config: Dict[str, Any]
) -> None:
    """Reject an exact resume when optimization-affecting config changed."""
    current_copy = copy.deepcopy(current)
    checkpoint_copy = copy.deepcopy(checkpoint_config)
    # The mode necessarily changes from the original train/initialize command
    # to resume. Operational stop/cleanup flags are CLI-only and not persisted.
    current_copy.pop("run_mode", None)
    checkpoint_copy.pop("run_mode", None)
    if current_copy != checkpoint_copy:
        current_keys = set(current_copy)
        checkpoint_keys = set(checkpoint_copy)
        changed = sorted(
            (current_keys ^ checkpoint_keys)
            | {
                key
                for key in current_keys & checkpoint_keys
                if current_copy[key] != checkpoint_copy[key]
            }
        )
        raise ValueError(
            "Exact resume configuration differs from the saved run at: "
            + ", ".join(changed[:20])
            + ("..." if len(changed) > 20 else "")
            + ". Use --initialize_from for an intentional configuration change."
        )
