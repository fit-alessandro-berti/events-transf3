import math
import unittest

import torch
import torch.nn.functional as F

from components.prototypical_head import PrototypicalHead
from config_utils import load_yaml_config
from metric_objectives import (
    classification_metric_objective,
    resolve_classification_objective,
    resolve_regression_metric_weights,
)


class MetricObjectiveTests(unittest.TestCase):
    def test_matched_experiment_configs_change_only_requested_profile(self):
        expected = {
            "training_metric_equilibrated_retrain.yaml": (
                "equilibrated", "equilibrated"
            ),
            "training_metric_accuracy_retrain.yaml": (
                "accuracy", "equilibrated"
            ),
            "training_metric_balanced_accuracy_retrain.yaml": (
                "balanced_accuracy", "equilibrated"
            ),
            "training_metric_mae_retrain.yaml": ("equilibrated", "mae"),
            "training_metric_r2_retrain.yaml": ("equilibrated", "r2"),
        }
        reference = load_yaml_config(
            "configs/fmv3/training_metric_equilibrated_retrain.yaml"
        )
        for filename, profiles in expected.items():
            config = load_yaml_config(f"configs/fmv3/{filename}")
            self.assertEqual(
                config["classification_objective"]["profile"], profiles[0]
            )
            self.assertEqual(
                config["fmv3_head"]["regression_objective_profile"], profiles[1]
            )
            for invariant in (
                "seed", "epochs", "episodes_per_epoch", "lr", "weight_decay",
                "classification_task_probability", "training_lr_multipliers",
            ):
                self.assertEqual(config[invariant], reference[invariant])

    def test_legacy_classification_is_exact_smoothed_cross_entropy(self):
        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
        labels = torch.tensor([0, 1])
        result = classification_metric_objective(
            logits,
            labels,
            {"classification_objective": {"profile": "legacy"}},
            label_smoothing=0.1,
        )
        expected = F.cross_entropy(logits, labels, label_smoothing=0.1)
        torch.testing.assert_close(result.loss, expected)
        result.loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_equilibrated_classification_is_equal_normalized_blend(self):
        logits = torch.tensor(
            [[2.0, 0.0, -1.0], [0.0, 1.5, -0.5], [0.2, -0.1, 0.7]],
            requires_grad=True,
        )
        result = classification_metric_objective(
            logits,
            torch.tensor([0, 1, 2]),
            {"classification_objective": {"profile": "equilibrated"}},
        )
        self.assertEqual(set(result.weights), {
            "accuracy", "balanced_accuracy", "macro_f1", "nll", "brier"
        })
        expected = torch.stack(list(result.components.values())).mean()
        torch.testing.assert_close(result.loss, expected)
        self.assertIn(
            "head/classification/episode_balanced_accuracy", result.diagnostics
        )
        result.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_balanced_accuracy_upweights_the_minority_class(self):
        # Three majority examples are easy; the one minority example is wrong.
        logits = torch.tensor(
            [[4.0, -2.0], [3.0, -1.0], [3.5, -1.5], [2.0, -0.5]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 0, 1])
        accuracy = classification_metric_objective(
            logits,
            labels,
            {"classification_objective": {"profile": "accuracy"}},
        )
        balanced = classification_metric_objective(
            logits,
            labels,
            {"classification_objective": {"profile": "balanced_accuracy"}},
        )
        self.assertGreater(
            float(balanced.loss.detach()), float(accuracy.loss.detach())
        )
        balanced.loss.backward()
        self.assertGreater(float(logits.grad[3].abs().sum()), 0.0)

    def test_macro_f1_supports_query_specific_class_spaces(self):
        first = torch.tensor([2.0, 0.0], requires_grad=True)
        second = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        result = classification_metric_objective(
            [first, second],
            [0, 1],
            {"classification_objective": {"profile": "macro_f1"}},
            class_id_rows=[torch.tensor([10, 20]), torch.tensor([30, 20, 10])],
        )
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertGreater(float(first.grad.abs().sum()), 0.0)
        self.assertGreater(float(second.grad.abs().sum()), 0.0)

    def test_objective_validation_rejects_unknown_and_zero_weights(self):
        with self.assertRaisesRegex(ValueError, "Unknown classification"):
            resolve_classification_objective(
                {"classification_objective": {
                    "profile": "custom", "weights": {"recall_at_10": 1.0}
                }}
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_regression_metric_weights({
                "regression_objective_profile": "custom",
                "regression_metric_weights": {"mae": 0.0},
                **{
                    f"regression_{name}_weight": 0.0
                    for name in (
                        "mae", "rmse", "huber", "log_rmse", "relative_mae",
                        "bias", "median_ae", "quantile", "r2",
                    )
                },
            })

    def test_mae_and_r2_profiles_select_exact_components(self):
        predictions = torch.tensor([4.0, 14.0, 31.0], requires_grad=True)
        targets = torch.tensor([1.0, 10.0, 30.0])
        mae_head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_objective_profile="mae",
        )
        r2_head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_objective_profile="r2",
        )
        mae_components = mae_head.regression_loss_components(
            predictions, targets, labels_in_output_space=True
        )
        r2_components = r2_head.regression_loss_components(
            predictions, targets, labels_in_output_space=True
        )
        torch.testing.assert_close(
            mae_head.regression_loss(
                predictions, targets, labels_in_output_space=True
            ),
            mae_components["mae"],
        )
        torch.testing.assert_close(
            r2_head.regression_loss(
                predictions, targets, labels_in_output_space=True
            ),
            r2_components["r2"],
        )
        ratio = float(r2_components["r2_error_ratio"].detach())
        self.assertAlmostEqual(
            float(r2_components["r2"].detach()), math.log1p(ratio), places=6
        )
        r2_components["r2"].backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_r2_surrogate_is_shift_and_scale_invariant(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_objective_profile="r2",
        )
        targets = torch.tensor([2.0, 5.0, 11.0])
        predictions = torch.tensor([3.0, 4.0, 9.0])
        base = head.regression_loss_components(
            predictions, targets, labels_in_output_space=True
        )["r2"]
        transformed = head.regression_loss_components(
            predictions * 7.0 + 100.0,
            targets * 7.0 + 100.0,
            labels_in_output_space=True,
        )["r2"]
        torch.testing.assert_close(base, transformed, atol=1e-6, rtol=1e-6)

    def test_constant_target_r2_surrogate_remains_finite(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_objective_profile="r2",
        )
        components = head.regression_loss_components(
            torch.tensor([9.0, 11.0, 10.0]),
            torch.tensor([10.0, 10.0, 10.0]),
            labels_in_output_space=True,
        )
        self.assertTrue(torch.isfinite(components["r2"]))
        self.assertGreater(float(components["r2"]), 0.0)


if __name__ == "__main__":
    unittest.main()
