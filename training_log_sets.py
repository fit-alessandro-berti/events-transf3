"""Resolve and select epoch-ranged sets of training event logs."""

from __future__ import annotations

import os
import random
import hashlib
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
    first_raw, last_raw = epoch_range
    first_epoch = 1 if first_raw is None else int(first_raw)
    last_epoch = total_epochs if last_raw is None else int(last_raw)
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
    exclusions = spec.get("exclude_paths", ()) or ()
    if isinstance(exclusions, (str, os.PathLike)):
        exclusions = (exclusions,)
    excluded_paths = {Path(os.fspath(path)).resolve() for path in exclusions}
    discovered: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(directory_path.glob(str(pattern))):
            if path.is_file() and path.resolve() not in excluded_paths:
                discovered[_path_key(path)] = os.fspath(path)
    return discovered


def _weight_schedule(
    spec: Mapping[str, Any], total_epochs: int, first_epoch: int, last_epoch: int
) -> list[tuple[int, int, float]]:
    raw_schedule = spec.get("weight_schedule")
    if raw_schedule is None:
        weight = float(spec.get("sampling_weight", 1.0))
        raw_schedule = ((first_epoch, last_epoch, weight),)
    if not isinstance(raw_schedule, (list, tuple)) or not raw_schedule:
        raise ValueError("weight_schedule must be a non-empty sequence.")
    schedule = []
    for entry in raw_schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(
                "Each weight_schedule entry must be (first_epoch, last_epoch, weight)."
            )
        start = first_epoch if entry[0] is None else int(entry[0])
        end = total_epochs if entry[1] is None else int(entry[1])
        weight = float(entry[2])
        if start > last_epoch or end < first_epoch:
            continue
        start = max(start, first_epoch)
        end = min(end, last_epoch)
        if end < start:
            continue
        if weight < 0.0:
            raise ValueError("Training log-set sampling weights must be non-negative.")
        schedule.append((start, end, weight))
    if not schedule:
        raise ValueError("weight_schedule does not overlap the configured epochs.")
    return schedule


def resolve_training_log_sets(
    config: Mapping[str, Any], *, validate_epoch_coverage: bool = True
) -> list[dict[str, Any]]:
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
        weight_schedule = _weight_schedule(
            raw_spec, total_epochs, first_epoch, last_epoch
        )

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
                "weight_schedule": weight_schedule,
                "log_paths": log_paths,
            }
        )

    if not resolved:
        raise ValueError("No enabled training log sets are configured.")
    if validate_epoch_coverage:
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
        and training_log_set_weight(log_set, epoch) > 0.0
    ]


def training_log_set_weight(log_set: Mapping[str, Any], epoch: int) -> float:
    schedule = log_set.get("weight_schedule")
    if not schedule:
        return float(log_set.get("sampling_weight", 1.0))
    return sum(
        float(weight)
        for start, end, weight in schedule
        if int(start) <= int(epoch) <= int(end)
    )


def choose_training_log_set(
    log_sets: Sequence[Mapping[str, Any]], epoch: int, rng=random
) -> Mapping[str, Any]:
    """Choose an active log set using its epoch-specific sampling weight."""
    active = active_training_log_sets(log_sets, epoch)
    if not active:
        raise ValueError(f"No training log set is enabled for epoch {epoch}.")
    if len(active) == 1:
        return active[0]
    weights = [training_log_set_weight(log_set, epoch) for log_set in active]
    if hasattr(rng, "choices"):
        return rng.choices(active, weights=weights, k=1)[0]
    threshold = rng.random() * sum(weights)
    for log_set, weight in zip(active, weights):
        threshold -= weight
        if threshold <= 0:
            return log_set
    return active[-1]


def configured_evaluation_log_paths(config: Mapping[str, Any]) -> dict[str, str]:
    paths = config.get("log_paths", {}).get("testing", {}) or {}
    if not isinstance(paths, Mapping):
        raise ValueError("log_paths.testing must be a mapping.")
    combined = {str(name): os.fspath(path) for name, path in paths.items()}
    split_sets = config.get("evaluation_log_sets", {}) or {}
    if not isinstance(split_sets, Mapping):
        raise ValueError("evaluation_log_sets must be a mapping.")
    for split, split_paths in split_sets.items():
        if not isinstance(split_paths, Mapping):
            raise ValueError(
                f"evaluation_log_sets.{split} must be a mapping."
            )
        for name, path in split_paths.items():
            combined[f"{split}/{name}"] = os.fspath(path)
    return combined


def validate_evaluation_split(
    config: Mapping[str, Any],
    evaluation_paths: Mapping[str, Any],
    split: str,
) -> None:
    """Prevent architecture screens from silently consuming the meta-test set."""
    split = str(split).strip().lower()
    if split == "external":
        return
    configured = (config.get("evaluation_log_sets", {}) or {}).get(split)
    if not isinstance(configured, Mapping) or not configured:
        raise ValueError(f"Unknown or empty evaluation split: {split}")
    allowed = {Path(path).resolve() for path in configured.values()}
    unexpected = [
        str(Path(path).resolve())
        for path in evaluation_paths.values()
        if Path(path).resolve() not in allowed
    ]
    if unexpected:
        raise ValueError(
            f"Evaluation path(s) are not in the '{split}' split: {unexpected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_evaluation_disjointness(
    config: Mapping[str, Any],
    log_sets: Sequence[Mapping[str, Any]],
    evaluation_paths: Mapping[str, Any] | None = None,
) -> None:
    """Reject evaluation logs present in training by path or file content."""
    training = {
        Path(path).resolve(): f"{log_set['name']}/{name}"
        for log_set in log_sets
        for name, path in log_set["log_paths"].items()
    }
    selected_evaluation_paths = (
        configured_evaluation_log_paths(config)
        if evaluation_paths is None
        else evaluation_paths
    )
    evaluation = {
        Path(path).resolve(): name
        for name, path in selected_evaluation_paths.items()
    }
    missing = [str(path) for path in evaluation if not path.is_file()]
    if missing:
        raise ValueError(f"Configured evaluation logs are missing: {missing}")
    path_overlap = sorted(set(training) & set(evaluation))
    if path_overlap:
        raise ValueError(
            "Training/evaluation path overlap: "
            + ", ".join(str(path) for path in path_overlap)
        )

    training_hashes: dict[str, list[Path]] = {}
    for path in training:
        training_hashes.setdefault(_sha256(path), []).append(path)
    duplicate_content = []
    for path in evaluation:
        matches = training_hashes.get(_sha256(path), ())
        duplicate_content.extend((path, match) for match in matches)
    if duplicate_content:
        details = ", ".join(
            f"{evaluation[test]}={test} duplicates {training[train]}={train}"
            for test, train in duplicate_content
        )
        raise ValueError(f"Training/evaluation content overlap: {details}")
