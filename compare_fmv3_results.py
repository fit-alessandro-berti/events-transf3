#!/usr/bin/env python3
"""Compare two paired FM-v3 results.csv files on deployment-facing metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PAIR_FIELDS = (
    "task",
    "evaluation_profile",
    "log",
    "repetition",
    "support_scenario",
    "case_budget",
    "retrieval_mode",
    "prior_mode",
    "prior_strength",
    "retrieval_k",
)
METRICS = {
    "classification": (
        ("balanced_accuracy", "max"),
        ("accuracy", "max"),
        ("macro_f1", "max"),
        ("nll", "min"),
        ("multiclass_brier", "min"),
        ("ece_10", "min"),
    ),
    "regression": (
        ("mae_hours", "min"),
        ("rmse_hours", "min"),
        ("median_absolute_error_hours", "min"),
        ("normalized_mae", "min"),
        ("mae_skill_vs_median", "max"),
        ("rmse_skill_vs_median", "max"),
    ),
}


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in PAIR_FIELDS)
        if key in result:
            raise ValueError(f"Duplicate paired row in {path}: {key}")
        result[key] = row
    return result


def _finite(row, metric):
    try:
        value = float(row.get(metric, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def compare(reference_path: Path, candidate_path: Path):
    reference = _read(reference_path)
    candidate = _read(candidate_path)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(
            "Results are not paired: "
            f"candidate is missing {len(missing)} and adds {len(extra)} rows"
        )
    output = {"pair_fields": PAIR_FIELDS, "tasks": {}}
    for task, metrics in METRICS.items():
        keys = [key for key in reference if reference[key].get("task") == task]
        task_result = {"paired_rows": len(keys), "metrics": {}}
        for metric, direction in metrics:
            pairs = []
            for key in keys:
                left = _finite(reference[key], metric)
                right = _finite(candidate[key], metric)
                if left is not None and right is not None:
                    pairs.append((left, right))
            if not pairs:
                continue
            deltas = [right - left for left, right in pairs]
            wins = sum(
                right > left if direction == "max" else right < left
                for left, right in pairs
            )
            ties = sum(right == left for left, right in pairs)
            task_result["metrics"][metric] = {
                "direction": direction,
                "paired_rows": len(pairs),
                "reference_mean": statistics.fmean(left for left, _ in pairs),
                "candidate_mean": statistics.fmean(right for _, right in pairs),
                "candidate_minus_reference": statistics.fmean(deltas),
                "candidate_wins": wins,
                "ties": ties,
                "candidate_losses": len(pairs) - wins - ties,
            }
        output["tasks"][task] = task_result
    return output


def _markdown(result, reference_label, candidate_label):
    lines = [
        "# Paired FM-v3 result comparison",
        "",
        f"Reference: `{reference_label}`. Candidate: `{candidate_label}`.",
        "",
    ]
    for task, task_result in result["tasks"].items():
        lines.extend(
            [
                f"## {task.title()} ({task_result['paired_rows']} paired rows)",
                "",
                "| Metric | Direction | Reference | Candidate | Delta | Wins/ties/losses |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric, values in task_result["metrics"].items():
            lines.append(
                f"| {metric} | {values['direction']} | "
                f"{values['reference_mean']:.6g} | "
                f"{values['candidate_mean']:.6g} | "
                f"{values['candidate_minus_reference']:+.6g} | "
                f"{values['candidate_wins']}/{values['ties']}/"
                f"{values['candidate_losses']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference_label", default="reference")
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = compare(Path(args.reference), Path(args.candidate))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "comparison.md").write_text(
        _markdown(result, args.reference_label, args.candidate_label),
        encoding="utf-8",
    )
    print(f"Wrote paired comparison to {output}")


if __name__ == "__main__":
    main()
