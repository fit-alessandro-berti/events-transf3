#!/usr/bin/env python3
"""Run per-log LSTM and class-weighted linear low-data baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from config import CONFIG
from config_utils import parse_override, set_dotted
from evaluation.fmv3_metrics import classification_metrics
from evaluation.fmv3_protocol import (
    _fixed_query_indices,
    _task_indices,
    fixed_case_split,
    support_case_order,
)
from evaluation.low_data_baselines import (
    load_classification_tasks,
    predict_classical,
    predict_lstm,
    predict_tabpfn,
    predict_weighted_linear,
    train_lstm,
)
from training_log_sets import validate_evaluation_split


def _metric_row(name, probabilities, tasks, query_indices, support_indices, universe, metadata):
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    truth = np.asarray([tasks[int(index)][1] for index in query_indices], dtype=int)
    support_counts = Counter(int(tasks[int(index)][1]) for index in support_indices)
    pool_covered = [support_counts[int(label)] > 0 for label in truth]
    metrics = classification_metrics(
        truth, predictions, probabilities[:, universe], universe, confidence,
        [tasks[int(index)][2] for index in query_indices], dict(support_counts),
        pool_covered, pool_covered,
    )
    return {"task": "classification", "experiment": name, **metadata, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fmv3/baselines.yaml")
    parser.add_argument("--logs_dir", default="logs_eval")
    parser.add_argument("--logs", nargs="+", default=None)
    parser.add_argument("--output_dir", default="evaluation_results/fmv3/baselines")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluation_split",
        choices=["screening", "meta_test", "external"],
        default="screening",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for raw in args.overrides:
        key, value = parse_override(raw)
        set_dotted(config, key, value)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = int(config.get("seed", 42))
    paths = sorted(list(Path(args.logs_dir).glob("*.xes")) + list(Path(args.logs_dir).glob("*.xes.gz")))
    if args.logs:
        requested = set(args.logs)
        paths = [path for path in paths if path.name in requested or path.name.replace(".xes.gz", "").replace(".xes", "") in requested]
    validate_evaluation_split(
        CONFIG, {path.name: str(path.resolve()) for path in paths}, args.evaluation_split
    )

    for path in paths:
        log_name = path.name.replace(".xes.gz", "").replace(".xes", "")
        output_path, done_path = output_dir / f"{log_name}.jsonl", output_dir / f"{log_name}.complete"
        if args.resume and done_path.exists():
            continue
        if output_path.exists():
            output_path.unlink()
        data_config = config.get("data", {}) or {}
        tasks, activities = load_classification_tasks(
            str(path.resolve()),
            max_seq_len=int(data_config.get("max_sequence_length", 32)),
            minimum_prefix_length=int(data_config.get("minimum_prefix_length", 1)),
        )
        num_classes = len(activities)
        test_cases, support_cases = fixed_case_split(
            tasks, seed, config.get("test_case_fraction", 0.3), config.get("max_test_cases", 50)
        )
        query_indices = _fixed_query_indices(tasks, test_cases, int(config.get("max_query_prefixes", 1000)), seed)
        universe = sorted({int(tasks[int(index)][1]) for index in query_indices})
        budgets = list(map(int, config.get("case_budgets", [1, 2, 4, 8, 16, 32, 64, 128])))
        if config.get("include_full_budget", True) and len(support_cases) <= int(config.get("include_full_budget_max_cases", 1000)):
            budgets.append(len(support_cases))
        budgets = sorted(set(min(value, len(support_cases)) for value in budgets if value > 0))
        for repetition in range(int(config.get("repetitions", 5))):
            for scenario in config.get("support_scenarios", ["natural", "class_aware"]):
                order = support_case_order(tasks, support_cases, scenario, seed + repetition * 1009, max_cases=max(budgets))
                for budget in budgets:
                    support_indices = _task_indices(tasks, set(order[:budget]))
                    if not len(support_indices):
                        continue
                    metadata = {
                        "log": log_name, "repetition": repetition, "support_scenario": scenario,
                        "case_budget": budget, "support_prefixes": int(len(support_indices)),
                        "retrieval_mode": "not_applicable", "prior_mode": "model_specific", "retrieval_k": None,
                    }
                    rows = []
                    natural_model, counts = train_lstm(
                        tasks, support_indices, num_classes, "natural_ce", config["sequence"], device,
                        seed + repetition * 1009 + budget,
                    )
                    natural_probabilities = predict_lstm(
                        natural_model, tasks, query_indices, counts, "natural_ce", config["sequence"], device
                    )
                    rows.append(_metric_row("baseline_lstm_natural_ce", natural_probabilities, tasks, query_indices, support_indices, universe, metadata))
                    adjusted = predict_lstm(
                        natural_model, tasks, query_indices, counts, "logit_adjustment", config["sequence"], device
                    )
                    rows.append(_metric_row("baseline_lstm_logit_adjustment", adjusted, tasks, query_indices, support_indices, universe, metadata))
                    for mode in ["class_weighted_ce", "balanced_softmax"]:
                        model, mode_counts = train_lstm(
                            tasks, support_indices, num_classes, mode, config["sequence"], device,
                            seed + repetition * 1009 + budget,
                        )
                        probabilities = predict_lstm(
                            model, tasks, query_indices, mode_counts, mode, config["sequence"], device
                        )
                        rows.append(_metric_row(f"baseline_lstm_{mode}", probabilities, tasks, query_indices, support_indices, universe, metadata))
                    linear = predict_weighted_linear(tasks, support_indices, query_indices, num_classes, seed)
                    rows.append(_metric_row("baseline_weighted_logistic", linear, tasks, query_indices, support_indices, universe, metadata))
                    for classical_mode in config.get("classical_modes", ["random_forest", "gaussian_nb"]):
                        classical = predict_classical(
                            tasks, support_indices, query_indices, num_classes,
                            seed + repetition * 1009 + budget, classical_mode,
                        )
                        rows.append(_metric_row(
                            f"baseline_{classical_mode}", classical, tasks, query_indices,
                            support_indices, universe, metadata,
                        ))
                    if config.get("tabpfn", {}).get("enabled", False):
                        tabpfn_probabilities = predict_tabpfn(
                            tasks, support_indices, query_indices, num_classes,
                            config["tabpfn"], device, seed + repetition * 1009 + budget,
                        )
                        rows.append(_metric_row(
                            "baseline_tabpfn", tabpfn_probabilities, tasks, query_indices,
                            support_indices, universe, metadata,
                        ))
                    with output_path.open("a", encoding="utf-8") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
        done_path.write_text("complete\n", encoding="utf-8")

    rows = []
    for path in output_dir.glob("*.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    flat_keys = [
        "task", "experiment", "log", "repetition", "support_scenario", "case_budget",
        "support_prefixes", "n_queries", "accuracy", "balanced_accuracy", "adjusted_balanced_accuracy", "macro_precision",
        "macro_f1", "zero_recall_fraction", "support_pool_availability", "nll",
        "macro_retrieval_given_pool", "macro_decision_given_retrieval",
        "multiclass_brier", "ece_10", "aurc",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} baseline rows to {output_dir}")


if __name__ == "__main__":
    main()
