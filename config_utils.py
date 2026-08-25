"""YAML experiment configuration and dotted command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
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
        if check_checkpoint_paths and not Path(initialize_path).is_file():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {initialize_path}"
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
