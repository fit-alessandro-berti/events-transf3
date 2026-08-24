import json
import tempfile
import unittest
from pathlib import Path

import torch

from training import optimizer_parameter_groups
from training_debug import (
    MetricAccumulator,
    TrainingDiagnostics,
    classification_head_metrics,
    loss_gradient_metrics,
    regression_head_metrics,
    split_training_tasks_by_case,
)


class TrainingDebugTests(unittest.TestCase):
    def test_summary_persists_resolved_objective_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = TrainingDiagnostics(
                directory,
                {
                    "classification_objective": {"profile": "accuracy"},
                    "fmv3_head": {"regression_objective_profile": "r2"},
                    "training_diagnostics": {"enabled": True},
                },
            )
            diagnostics.start_epoch()
            diagnostics.epoch_accumulator.add(
                {"loss/total": 1.0}, prefixes=("task/classification",)
            )
            diagnostics.finish_epoch(1, None, {}, {}, {}, {})
            summary = json.loads(
                (Path(directory) / "training_debug_summary.json").read_text()
            )
            objectives = summary["objective_configuration"]
            self.assertEqual(objectives["classification"]["profile"], "accuracy")
            self.assertEqual(objectives["classification"]["weights"]["accuracy"], 1.0)
            self.assertEqual(objectives["regression"]["profile"], "r2")
            self.assertEqual(objectives["regression"]["weights"]["r2"], 1.0)

    def test_epoch_context_preserves_selected_training_log_set(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = TrainingDiagnostics(
                directory, {"training_diagnostics": {"enabled": True}}
            )
            diagnostics.start_epoch()
            diagnostics.finish_epoch(
                1,
                None,
                {},
                {},
                {},
                {},
                context={
                    "training_log_set": "generated",
                    "active_training_log_sets": ["default", "generated"],
                },
            )
            summary = json.loads(
                (Path(directory) / "training_debug_summary.json").read_text()
            )
            self.assertEqual(
                summary["epochs"][0]["context"]["training_log_set"],
                "generated",
            )

    def test_component_lr_groups_are_disjoint_and_scaled(self):
        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proto_head = torch.nn.Module()
                self.proto_head.classification_example_selector = torch.nn.Linear(
                    2, 2
                )
                self.proto_head.other = torch.nn.Linear(2, 2)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.experts = torch.nn.ModuleList([Expert()])

        model = ToyModel()
        groups, proto_index, special_indices, multipliers = (
            optimizer_parameter_groups(
                model, 1e-4, {"head/classification_selector": 20.0}
            )
        )
        self.assertIsNotNone(proto_index)
        selector_index = special_indices["head/classification_selector"]
        self.assertEqual(groups[selector_index]["lr"], 0.002)
        self.assertEqual(multipliers["head/classification_selector"], 20.0)
        parameter_ids = [
            id(parameter) for group in groups for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(
            set(parameter_ids),
            {id(parameter) for parameter in model.parameters()},
        )

    def test_loss_gradient_attribution_separates_parameter_groups(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 1, bias=False)
                self.task_confidence_head = torch.nn.Linear(2, 1, bias=False)

        model = ToyModel()
        values = torch.tensor([[1.0, -2.0]])
        components = {
            "loss/primary": model.encoder(values).square().mean(),
            "loss/routing_weighted": (
                model.task_confidence_head(values).square().mean()
            ),
        }
        metrics = loss_gradient_metrics(model, components)
        self.assertGreater(
            metrics["optimization/loss_gradient/primary/encoder/l2_norm"], 0.0
        )
        self.assertGreater(
            metrics[
                "optimization/loss_gradient/routing_weighted/routing/l2_norm"
            ],
            0.0,
        )
        self.assertNotIn(
            "optimization/loss_gradient/primary/routing/l2_norm", metrics
        )

    def test_case_split_is_deterministic_disjoint_and_shared_across_tasks(self):
        tasks = {
            "classification": [[([{"x": 1}], index % 2, f"case-{index}") for index in range(10)]],
            "regression": [[([{"x": 1}], float(index), f"case-{index}") for index in range(10)]],
        }
        train_a, validation_a, manifest_a = split_training_tasks_by_case(
            tasks, 0.2, 42, ["log"]
        )
        train_b, validation_b, manifest_b = split_training_tasks_by_case(
            tasks, 0.2, 42, ["log"]
        )
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(validation_a, validation_b)
        for task in ("classification", "regression"):
            train_cases = {item[2] for item in train_a[task][0]}
            validation_cases = {item[2] for item in validation_a[task][0]}
            self.assertFalse(train_cases & validation_cases)
            self.assertEqual(len(validation_cases), 2)
        self.assertEqual(
            {item[2] for item in validation_a["classification"][0]},
            {item[2] for item in validation_a["regression"][0]},
        )

    def test_head_metrics_include_predictions_selector_and_branch_behavior(self):
        class_metrics = classification_head_metrics(
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            torch.tensor([0, 1]),
            torch.softmax(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), dim=-1),
            {
                "selection_logits": torch.tensor([[0.2, -0.2], [0.1, -0.1]]),
                "selection_trust": torch.softmax(
                    torch.tensor([[0.2, -0.2], [0.1, -0.1]]), dim=-1
                ),
                "selection_effective_count": torch.tensor([1.9, 2.0]),
                "selection_attention": torch.tensor([[0.8, 0.2], [0.6, 0.4]]),
            },
        )
        self.assertEqual(class_metrics["head/classification/accuracy"], 1.0)
        self.assertIn(
            "head/classification/selector/log_weight/p90", class_metrics
        )

        regression_metrics = regression_head_metrics(
            torch.tensor([10.0, 20.0]),
            torch.tensor([12.0, 18.0]),
            torch.tensor([0.5, 0.6]),
            {
                "branch_predictions_hours": torch.tensor(
                    [[10.0, 20.0], [11.0, 19.0]]
                ),
                "aggregation_weights": torch.tensor(
                    [[0.25, 0.50], [0.75, 0.50]]
                ),
            },
        )
        self.assertEqual(regression_metrics["head/regression/mae_hours"], 2.0)
        self.assertIn("head/regression/branch_1/weight_mean", regression_metrics)

    def test_step_metrics_preserve_task_pool_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = TrainingDiagnostics(
                directory, {"training_diagnostics": {"enabled": True}}
            )
            diagnostics.start_epoch()
            diagnostics.add_step(
                1,
                0,
                "regression",
                3,
                7,
                "retrieval_regression",
                {"loss/total": 2.5},
                sampled=True,
            )
            summary = diagnostics.epoch_accumulator.summary()
            self.assertEqual(
                summary["task/regression/pool/7/loss/total"]["mean"], 2.5
            )
            record = json.loads(
                (Path(directory) / "training_debug_steps.jsonl").read_text()
            )
            self.assertEqual(record["pool"], 7)

    def test_epoch_summary_detects_validation_degradation_after_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = TrainingDiagnostics(
                directory,
                {
                    "training_diagnostics": {
                        "enabled": True,
                        "overfitting_patience": 2,
                        "overfitting_relative_tolerance": 0.02,
                    }
                },
            )
            for epoch, train_loss, validation_loss in (
                (1, 1.0, 1.0),
                (2, 0.9, 0.9),
                (3, 0.8, 1.0),
                (4, 0.7, 1.1),
            ):
                diagnostics.start_epoch()
                diagnostics.epoch_accumulator.add(
                    {"loss/total": train_loss}, prefixes=("task/classification",)
                )
                validation = MetricAccumulator()
                validation.add(
                    {"loss/total": validation_loss},
                    prefixes=("task/classification",),
                )
                diagnostics.finish_epoch(
                    epoch, validation, {}, {}, {}, {}
                )
            summary = json.loads(
                (Path(directory) / "training_debug_summary.json").read_text()
            )
            result = summary["generalization"]["classification"]
            self.assertEqual(result["best_validation_epoch"], 2)
            self.assertTrue(result["overfitting_signal"])


if __name__ == "__main__":
    unittest.main()
