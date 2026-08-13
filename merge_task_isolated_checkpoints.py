#!/usr/bin/env python3
"""Audit and merge independently trained classification/regression adapters.

The classification run is allowed to differ from the base only in its
classification embedding adapter and example selector. The regression run is
allowed to differ only in its regression embedding adapter, example selector,
time-transform bank, and regression expert-confidence head. The script refuses
the merge if either run changed a shared, routing, or out-of-scope tensor.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from config_utils import load_yaml_config, save_yaml_config


CLASSIFICATION_KEYS = (
    ".classification_embedding_adapter.",
    ".proto_head.classification_example_selector.",
)
REGRESSION_KEYS = (
    ".regression_embedding_adapter.",
    ".proto_head.time_transform_bank.",
    ".proto_head.regression_expert_confidence.",
    ".proto_head.regression_example_selector.",
)


def _changed_keys(base, candidate):
    shared = set(base) & set(candidate)
    return {
        key
        for key in shared
        if base[key].shape != candidate[key].shape
        or not torch.equal(base[key].cpu(), candidate[key].cpu())
    }


def _assert_scoped(name, changed, allowed_fragments):
    unexpected = sorted(
        key for key in changed if not any(fragment in key for fragment in allowed_fragments)
    )
    if unexpected:
        raise RuntimeError(f"{name} changed out-of-scope tensors: {unexpected}")


def merge_task_isolated_state_dicts(base, classification, regression):
    classification_changed = _changed_keys(base, classification)
    regression_changed = _changed_keys(base, regression)
    _assert_scoped("classification checkpoint", classification_changed, CLASSIFICATION_KEYS)
    _assert_scoped("regression checkpoint", regression_changed, REGRESSION_KEYS)

    new_classification_keys = set(classification) - set(base)
    _assert_scoped(
        "classification checkpoint new parameters",
        new_classification_keys,
        CLASSIFICATION_KEYS,
    )
    new_regression_keys = set(regression) - set(base)
    _assert_scoped(
        "regression checkpoint new parameters",
        new_regression_keys,
        REGRESSION_KEYS,
    )

    merged = dict(base)
    for key in classification_changed | new_classification_keys:
        merged[key] = classification[key]
    for key in regression_changed | new_regression_keys:
        merged[key] = regression[key]
    return (
        merged,
        classification_changed | new_classification_keys,
        regression_changed | new_regression_keys,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--regression", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        help="Resolved YAML entry point to save beside the merged checkpoint.",
    )
    parser.add_argument(
        "--artifacts",
        help="Training artifacts to copy beside the merged checkpoint.",
    )
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    classification = torch.load(
        args.classification, map_location="cpu", weights_only=True
    )
    regression = torch.load(args.regression, map_location="cpu", weights_only=True)
    merged, classification_keys, regression_keys = merge_task_isolated_state_dicts(
        base, classification, regression
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    if args.config:
        resolved = load_yaml_config(args.config)
        torch.save(resolved, output.parent / "training_config.pth")
        save_yaml_config(resolved, str(output.parent / "training_config.yaml"))
    if args.artifacts:
        shutil.copy2(args.artifacts, output.parent / "training_artifacts.pth")
    print(
        f"Merged {len(classification_keys)} classification and "
        f"{len(regression_keys)} regression tensors into {output}"
    )


if __name__ == "__main__":
    main()
