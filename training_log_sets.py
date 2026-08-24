"""Resolve and select epoch-ranged sets of training event logs."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_LOG_PATTERNS = ("*.xes", "*.xes.gz")


def _epoch_range(spec: Mapping[str, Any], total_epochs: int) -> tuple[int, int]:
    epoch_range = spec.get("epochs", (1, total_epochs))
    if (
        not isinstance(epoch_range, (list, tuple))
        or len(epoch_range) != 2
    ):
        raise ValueError(
            "Each training log set needs 'epochs': (first_epoch, last_epoch)."
        )
    first_epoch, last_epoch = (int(value) for value in epoch_range)
    if first_epoch < 1 or last_epoch < first_epoch:
        raise ValueError(
            f"Invalid training log set epoch range: {tuple(epoch_range)}"
        )
    return first_epoch, last_epoch


def _path_key(path: Path) -> str:
    name = path.name
    if name.endswith(".xes.gz"):
        return name[:-7]
    if name.endswith(".xes"):
        return name[:-4]
    return path.stem


def _directory_log_paths(spec: Mapping[str, Any]) -> dict[str, str]:
    directory = spec.get("directory")
    if directory is None:
        return {}
    directory_path = Path(os.fspath(directory))
    if not directory_path.is_dir():
        raise ValueError(
            f"Training log set directory does not exist: {directory_path}"
        )
    patterns = spec.get("patterns", DEFAULT_LOG_PATTERNS)
    if isinstance(patterns, str):
        patterns = (patterns,)
    discovered: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(directory_path.glob(str(pattern))):
            if path.is_file():
                discovered[_path_key(path)] = os.fspath(path)
    return discovered


def resolve_training_log_sets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve configured paths and validate inclusive epoch coverage.

    A set can contain an explicit ``log_paths`` mapping, a ``directory`` plus
    optional glob ``patterns``, or both. The legacy ``log_paths.training``
    mapping remains supported when ``training_log_sets`` is absent.
    """
    total_epochs = int(config.get("epochs", 1))
    raw_sets = config.get("training_log_sets")
    if raw_sets is None:
        raw_sets = [
            {
                "name": "default",
                "epochs": (1, total_epochs),
                "log_paths": config.get("log_paths", {}).get("training", {}),
            }
        ]
    if not isinstance(raw_sets, (list, tuple)) or not raw_sets:
        raise ValueError("'training_log_sets' must be a non-empty list.")

    resolved = []
    names = set()
    for index, raw_spec in enumerate(raw_sets):
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Training log set {index + 1} must be a mapping.")
        if not raw_spec.get("enabled", True):
            continue
        name = str(raw_spec.get("name", f"set_{index + 1}"))
        if name in names:
            raise ValueError(f"Duplicate training log set name: {name}")
        names.add(name)
        first_epoch, last_epoch = _epoch_range(raw_spec, total_epochs)

        explicit = raw_spec.get("log_paths", {}) or {}
        if not isinstance(explicit, Mapping):
            raise ValueError(
                f"Training log set '{name}' has a non-mapping 'log_paths'."
            )
        log_paths = {str(key): os.fspath(path) for key, path in explicit.items()}
        for key, path in _directory_log_paths(raw_spec).items():
            if key in log_paths and Path(log_paths[key]) != Path(path):
                raise ValueError(
                    f"Training log set '{name}' contains duplicate log key '{key}'."
                )
            log_paths[key] = path
        if not log_paths:
            raise ValueError(f"Training log set '{name}' contains no event logs.")
        missing = [path for path in log_paths.values() if not Path(path).is_file()]
        if missing:
            raise ValueError(
                f"Training log set '{name}' contains missing files: {missing}"
            )
        resolved.append(
            {
                "name": name,
                "start_epoch": first_epoch,
                "end_epoch": last_epoch,
                "log_paths": log_paths,
            }
        )

    if not resolved:
        raise ValueError("No enabled training log sets are configured.")
    uncovered = [
        epoch
        for epoch in range(1, total_epochs + 1)
        if not active_training_log_sets(resolved, epoch)
    ]
    if uncovered:
        preview = ", ".join(str(epoch) for epoch in uncovered[:10])
        suffix = "..." if len(uncovered) > 10 else ""
        raise ValueError(
            "No training log set is enabled for configured epoch(s): "
            f"{preview}{suffix}"
        )
    return resolved


def combined_training_log_paths(
    log_sets: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Return all scheduled logs with namespaced keys for loader fitting."""
    combined = {}
    seen_paths = set()
    for log_set in log_sets:
        for log_name, path in log_set["log_paths"].items():
            normalized_path = Path(path).resolve()
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            combined[f"{log_set['name']}/{log_name}"] = path
    return combined


def active_training_log_sets(
    log_sets: Sequence[Mapping[str, Any]], epoch: int
) -> list[Mapping[str, Any]]:
    """Return all log sets whose inclusive range contains ``epoch``."""
    return [
        log_set
        for log_set in log_sets
        if int(log_set["start_epoch"]) <= int(epoch) <= int(log_set["end_epoch"])
    ]


def choose_training_log_set(
    log_sets: Sequence[Mapping[str, Any]], epoch: int, rng=random
) -> Mapping[str, Any]:
    """Uniformly choose one of the log sets active in ``epoch``."""
    active = active_training_log_sets(log_sets, epoch)
    if not active:
        raise ValueError(f"No training log set is enabled for epoch {epoch}.")
    if len(active) == 1:
        return active[0]
    return rng.choice(active)
