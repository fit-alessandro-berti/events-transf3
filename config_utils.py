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
