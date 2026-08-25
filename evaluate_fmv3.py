#!/usr/bin/env python3
"""Evaluate one checkpoint with the fixed-query FM-v3 low-data protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pm4py
import torch

from config import CONFIG
from config_utils import apply_experiment_config, deep_merge, load_yaml_config
from evaluation.fmv3_protocol import evaluate_log, prepare_case_plan_from_dataframe
from utils.data_utils import get_classification_and_regression_tasks
from training_log_sets import (
    resolve_training_log_sets,
    validate_training_evaluation_disjointness,
    validate_evaluation_split,
)
from utils.model_utils import create_model, init_loader, load_model_weights


def _load_checkpoint_config(checkpoint_dir: Path):
    yaml_path = checkpoint_dir / "training_config.yaml"
    torch_path = checkpoint_dir / "training_config.pth"
    if yaml_path.exists():
        deep_merge(CONFIG, load_yaml_config(str(yaml_path)))
    elif torch_path.exists():
        deep_merge(CONFIG, torch.load(torch_path, map_location="cpu", weights_only=False))
    else:
        raise FileNotFoundError(f"No training config in {checkpoint_dir}")


def _write_summary(rows, output_dir: Path):
    columns = [
        "task", "experiment", "evaluation_split", "evaluation_profile", "log", "repetition", "support_scenario", "case_budget",
        "support_prefixes", "retrieval_mode", "prior_mode", "prior_strength", "retrieval_k", "n_queries",
        "structured_max_order", "structured_smoothing", "structured_weight", "structured_tau",
        "structured_context_coverage", "structured_mean_context_support",
        "structured_mean_selected_order", "structured_mean_effective_weight",
        "virtual_expert_replicates", "virtual_expert_support_fraction",
        "virtual_expert_min_support_prefixes", "virtual_expert_full_support_weight",
        "virtual_expert_subbag_weight",
        "accuracy", "balanced_accuracy", "adjusted_balanced_accuracy", "macro_precision", "macro_f1", "zero_recall_fraction",
        "support_pool_availability", "macro_label_recall_at_k", "macro_retrieval_given_pool",
        "macro_decision_given_retrieval", "nll", "multiclass_brier",
        "ece_10", "aurc", "mae_hours", "rmse_hours", "log_mae", "median_absolute_error_hours", "normalized_mae",
        "mae_skill_vs_median", "rmse_skill_vs_median", "d2_absolute_error", "r2", "interval_coverage",
        "mean_interval_width_hours",
        "regression_mode", "regression_num_transforms", "regression_transform_aggregation",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    curves = []
    class_rows = [row for row in rows if row["task"] == "classification"]
    group_keys = sorted({(
        row["experiment"], row["log"], row["support_scenario"],
        row["retrieval_mode"], row["prior_mode"], row["retrieval_k"]
    ) for row in class_rows})
    for key in group_keys:
        selected = [row for row in class_rows if (
            row["experiment"], row["log"], row["support_scenario"],
            row["retrieval_mode"], row["prior_mode"], row["retrieval_k"]
        ) == key]
        by_budget = {}
        for budget in sorted({int(row["case_budget"]) for row in selected}):
            values = [float(row["balanced_accuracy"]) for row in selected if int(row["case_budget"]) == budget]
            by_budget[budget] = float(np.mean(values))
        budgets = np.asarray(sorted(by_budget), dtype=float)
        values = np.asarray([by_budget[int(budget)] for budget in budgets])
        x = np.log2(budgets)
        auc = float(np.trapezoid(values, x) / max(float(x[-1] - x[0]), 1e-12)) if len(x) > 1 else float(values[0])
        target = float(CONFIG.get("fmv3_evaluation", {}).get("threshold_fraction", 0.9)) * float(values[-1])
        reached = [int(budget) for budget, value in zip(budgets, values) if value >= target]
        curves.append({
            "experiment": key[0], "log": key[1], "support_scenario": key[2],
            "retrieval_mode": key[3], "prior_mode": key[4], "retrieval_k": key[5],
            "balanced_accuracy_by_budget": {str(k): v for k, v in by_budget.items()},
            "log2_case_budget_auc": auc, "cases_to_90pct_max_budget": min(reached) if reached else None,
        })
    with (output_dir / "learning_curves.json").open("w", encoding="utf-8") as handle:
        json.dump(curves, handle, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--logs_dir", default="logs_eval")
    parser.add_argument("--logs", nargs="+", default=None, help="Optional filename/stem filter.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    parser.add_argument("--eval_config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluation_split",
        choices=["screening", "meta_test", "external"],
        default="screening",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    _load_checkpoint_config(checkpoint_dir)
    apply_experiment_config(CONFIG, args.eval_config, args.overrides)
    experiment = str(CONFIG.get("experiment_name", checkpoint_dir.name))
    output_dir = Path(args.output_dir or f"evaluation_results/{experiment}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loader = init_loader(CONFIG)
    loader.load_training_artifacts(str(checkpoint_dir / "training_artifacts.pth"))
    model = create_model(CONFIG, loader, device)
    load_model_weights(model, str(checkpoint_dir), device, args.checkpoint_epoch)
    model.eval()

    log_paths = sorted(list(Path(args.logs_dir).glob("*.xes")) + list(Path(args.logs_dir).glob("*.xes.gz")))
    if args.logs:
        requested = set(args.logs)
        log_paths = [path for path in log_paths if path.name in requested or path.name.replace(".xes.gz", "").replace(".xes", "") in requested]
    if not log_paths:
        raise FileNotFoundError(f"No XES logs found in {args.logs_dir}")
    validate_training_evaluation_disjointness(
        CONFIG,
        resolve_training_log_sets(CONFIG, validate_epoch_coverage=False),
        {path.name: str(path) for path in log_paths},
    )
    validate_evaluation_split(
        CONFIG,
        {path.name: str(path) for path in log_paths},
        args.evaluation_split,
    )
    CONFIG["evaluation_split"] = args.evaluation_split
    for log_path in log_paths:
        log_name = log_path.name.replace(".xes.gz", "").replace(".xes", "")
        result_path = output_dir / f"{log_name}.jsonl"
        done_path = output_dir / f"{log_name}.complete"
        if args.resume and done_path.exists():
            print(f"[{experiment}] skip completed log {log_name}")
            continue
        if result_path.exists():
            result_path.unlink()
        raw_frame = pm4py.read_xes(str(log_path.resolve()))
        case_plan, selected_frame, activity_names = prepare_case_plan_from_dataframe(raw_frame, CONFIG)
        materialized_traces = loader.transform_dataframes(
            {log_name: selected_frame}, activity_names_by_log={log_name: activity_names}
        )[log_name]
        classification_tasks, regression_tasks = get_classification_and_regression_tasks(
            materialized_traces, config=CONFIG
        )
        tasks = {"classification": classification_tasks, "regression": regression_tasks}
        evaluate_log(model, tasks, log_name, CONFIG, result_path, case_plan=case_plan)
        done_path.write_text("complete\n", encoding="utf-8")

    rows = []
    for path in sorted(output_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    _write_summary(rows, output_dir)
    print(f"Wrote {len(rows)} result rows to {output_dir}")


if __name__ == "__main__":
    main()
