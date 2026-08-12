import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from components.pretrained_event_embedder import PretrainedEventEmbedder
from components.prototypical_head import LearnedTimeTransformBank, PrototypicalHead
from components.temporal_adapter import (
    IndependentTemporalInputEncoder,
    LearnedTemporalInputAdapter,
)
from config_utils import load_yaml_config
from utils.parameter_utils import configure_trainable_scope


class FMV3HeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_promoted_multimetric_defaults_and_historical_pin(self):
        head = PrototypicalHead(regression_mode="learned_transform_ensemble")
        self.assertEqual(head.regression_huber_weight, 0.15)
        self.assertEqual(head.regression_log_rmse_weight, 0.15)
        self.assertEqual(head.regression_relative_mae_weight, 0.05)
        self.assertEqual(head.regression_bias_weight, 0.05)

        base = load_yaml_config("configs/fmv3/base.yaml")["fmv3_head"]
        historical = load_yaml_config("configs/fmv3/00_fmv2.yaml")["fmv3_head"]
        selected = load_yaml_config(
            "configs/fmv3/loss_multimetric_gate_aux_005.yaml"
        )
        selected_alias = load_yaml_config("configs/fmv3/selected.yaml")
        for key, expected in {
            "regression_huber_weight": 0.15,
            "regression_log_rmse_weight": 0.15,
            "regression_relative_mae_weight": 0.05,
            "regression_bias_weight": 0.05,
        }.items():
            self.assertEqual(base[key], expected)
            self.assertEqual(historical[key], 0.0)
            self.assertEqual(selected["fmv3_head"][key], expected)
            self.assertEqual(selected_alias["fmv3_head"][key], expected)
        self.assertEqual(selected["selected_checkpoint_epoch"], 38)
        self.assertEqual(selected_alias["selected_checkpoint_epoch"], 38)

    def test_count_neutral_local_evidence_removes_duplicate_bias(self):
        head = PrototypicalHead(
            classification_mode="local",
            count_normalization="fixed",
            count_normalization_gamma=1.0,
            prior_mode="balanced",
        )
        support = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        labels = torch.tensor([0, 0, 0, 1])
        query = torch.tensor([[1.0, 0.0]])
        logits, classes, _ = head.forward_classification(support, labels, query)
        self.assertEqual(classes.tolist(), [0, 1])
        self.assertAlmostEqual(logits[0, 0].item(), logits[0, 1].item(), places=5)

    def test_global_memory_keeps_locally_missing_label_predictable(self):
        head = PrototypicalHead(
            classification_mode="global_local",
            count_normalization_gamma=1.0,
            prior_mode="balanced",
            local_gate=0.8,
        )
        local = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
        local_labels = torch.tensor([0, 0])
        pool = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]])
        pool_labels = torch.tensor([0, 1, 1])
        query = torch.tensor([[0.0, 1.0]])
        logits, classes, _, diagnostics = head.forward_classification(
            local,
            local_labels,
            query,
            global_support_features=pool,
            global_support_labels=pool_labels,
            return_diagnostics=True,
        )
        self.assertEqual(classes.tolist(), [0, 1])
        self.assertEqual(torch.argmax(logits, dim=1).item(), 1)
        self.assertEqual(diagnostics["local_counts"].tolist(), [2.0, 0.0])
        self.assertEqual(diagnostics["gate"][0, 1].item(), 0.0)

    def test_natural_prior_is_explicit_and_controllable(self):
        head = PrototypicalHead(
            classification_mode="global",
            prior_mode="natural",
            prior_strength=1.0,
            prior_smoothing=0.0,
        )
        support = torch.tensor([[1.0, 0.0]] * 3 + [[1.0, 0.0]])
        labels = torch.tensor([0, 0, 0, 1])
        query = torch.tensor([[1.0, 0.0]])
        natural, _, _ = head.forward_classification(support, labels, query)
        balanced, _, _ = head.forward_classification(support, labels, query, prior_mode="balanced")
        self.assertGreater((natural[0, 0] - natural[0, 1]).item(), 1.0)
        self.assertAlmostEqual(balanced[0, 0].item(), balanced[0, 1].item(), places=5)

    def test_count_dependent_shrinkage_is_strongest_for_one_shot(self):
        head = PrototypicalHead(
            classification_mode="global",
            shrinkage_mode="fixed",
            shrinkage_kappa=3.0,
        )
        support = torch.tensor([[0.0, 1.0]] + [[1.0, 0.0]] * 6)
        labels = torch.tensor([0] + [1] * 6)
        query = torch.tensor([[0.0, 1.0]])
        _, _, _, diag = head.forward_classification(
            support, labels, query, return_diagnostics=True
        )
        raw_one = support[labels == 0].mean(0)
        raw_many = support[labels == 1].mean(0)
        one_shift = torch.norm(diag["prototypes"][0] - raw_one)
        many_shift = torch.norm(diag["prototypes"][1] - raw_many)
        self.assertGreater(one_shift.item(), many_shift.item())

    def test_uncovered_class_can_map_to_abstention(self):
        head = PrototypicalHead(
            classification_mode="global_local",
            enable_abstention=True,
            abstain_label=-101,
        )
        support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        logits, classes, probs = head.forward_classification(support, labels, torch.tensor([[-1.0, 0.0]]))
        self.assertEqual(classes[-1].item(), -101)
        self.assertEqual(logits.shape, probs.shape)

    def test_learned_temperature_is_used_by_fmv3_evidence(self):
        head = PrototypicalHead(
            classification_mode="local",
            learn_temperature=True,
            count_normalization_gamma=1.0,
        )
        support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        query = torch.tensor([[1.0, 0.0]])
        logits, _, _ = head.forward_classification(support, labels, query)
        logits.sum().backward()
        self.assertIsNotNone(head.logit_scale.grad)

    def test_coverage_fallback_preserves_local_order_without_global_override(self):
        head = PrototypicalHead(
            classification_mode="coverage_fallback",
            count_normalization_gamma=0.0,
            local_centering=True,
            global_centering=True,
            coverage_fallback_margin=1.0,
        )
        local = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        local_labels = torch.tensor([0, 0, 1])
        pool = torch.cat([local, torch.tensor([[-1.0, 0.0]])])
        pool_labels = torch.tensor([0, 0, 1, 2])
        query = torch.tensor([[1.0, 0.0]])
        legacy = PrototypicalHead(classification_mode="legacy_soft_knn")
        legacy_logits, legacy_classes, _ = legacy.forward_classification(local, local_labels, query)
        logits, classes, _ = head.forward_classification(
            local,
            local_labels,
            query,
            global_support_features=pool,
            global_support_labels=pool_labels,
        )
        local_columns = [classes.tolist().index(label) for label in legacy_classes.tolist()]
        self.assertEqual(
            torch.argmax(logits[:, local_columns], dim=1).item(),
            torch.argmax(legacy_logits, dim=1).item(),
        )

    def test_coverage_fallback_can_recover_locally_missing_class(self):
        head = PrototypicalHead(
            classification_mode="coverage_fallback",
            count_normalization_gamma=0.0,
            global_centering=False,
            coverage_fallback_margin=0.0,
        )
        local = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
        local_labels = torch.tensor([0, 0])
        pool = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.0, 0.9]])
        pool_labels = torch.tensor([0, 0, 1, 1])
        logits, classes, _ = head.forward_classification(
            local,
            local_labels,
            torch.tensor([[0.0, 1.0]]),
            global_support_features=pool,
            global_support_labels=pool_labels,
        )
        self.assertEqual(classes[torch.argmax(logits, dim=1)].item(), 1)

    def test_learned_time_transforms_are_invertible_in_raw_hours(self):
        for branches in (4, 8):
            bank = LearnedTimeTransformBank(num_transforms=branches)
            hours = torch.tensor([0.0, 0.02, 1.0, 24.0, 10_000.0, 1_000_000.0])
            reconstructed = bank.inverse(bank.transform(hours))
            torch.testing.assert_close(
                reconstructed,
                hours.unsqueeze(0).expand(branches, -1),
                atol=2e-2,
                rtol=2e-4,
            )

    def test_time_transform_ensemble_predicts_hours_and_receives_gradients(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=4,
            regression_scale_augmentation=False,
        )
        support = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.1, 0.9], [-0.8, 0.2]]
        )
        # Regression labels are stored as sqrt(hours) for checkpoint/data compatibility.
        labels = torch.sqrt(torch.tensor([1.0, 10.0, 100.0, 10_000.0]))
        query = torch.tensor([[0.7, 0.3], [0.0, 1.0]])
        predictions, _, diagnostics = head.forward_regression(
            support, labels, query, return_diagnostics=True
        )
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(torch.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        self.assertEqual(diagnostics["branch_predictions_hours"].shape, (4, 2))
        torch.testing.assert_close(
            diagnostics["aggregation_weights"].sum(dim=0), torch.ones(2)
        )

        # Targets are stored sqrt(hours); default conversion squares them to hours.
        loss = head.regression_loss(
            predictions, torch.sqrt(torch.tensor([2.0, 80.0]))
        )
        loss.backward()
        bank = head.time_transform_bank
        for parameter in (
            bank.power_logits,
            bank.log_scales,
            bank.branch_logit_scales,
            bank.aggregation_logits,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_raw_hours_regression_ablation_uses_direct_weighted_mean(self):
        head = PrototypicalHead(regression_mode="raw_hours_knn")
        with torch.no_grad():
            head.reg_logit_scale.fill_(1.0)
        support = torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32
        )
        labels = torch.sqrt(torch.tensor([1.0, 25.0, 400.0]))
        query = torch.tensor([[0.6, 0.4], [0.0, 1.0]], dtype=torch.float32)
        predictions, _, diagnostics = head.forward_regression(
            support, labels, query, return_diagnostics=True
        )
        self.assertTrue(head.regression_outputs_hours)
        self.assertFalse(head.regression_uses_time_transform_bank)
        self.assertIsNone(head.time_transform_bank)
        self.assertNotIn("branch_predictions_hours", diagnostics)
        targets = labels.square().unsqueeze(0).expand(query.size(0), -1)
        torch.testing.assert_close(
            predictions,
            (diagnostics["attention"] * targets).sum(dim=1),
        )
        loss = head.regression_loss(
            predictions, torch.sqrt(torch.tensor([10.0, 300.0]))
        )
        self.assertTrue(torch.isfinite(loss))

    def test_scale_augmentation_range_and_eval_identity(self):
        bank = LearnedTimeTransformBank(
            num_transforms=4,
            regression_scale_augmentation_min=0.02,
            regression_scale_augmentation_max=50.0,
        )
        reference = torch.ones(1)
        bank.train()
        samples = torch.stack([bank.sample_augmentation_factor(reference) for _ in range(200)])
        self.assertGreaterEqual(samples.min().item(), 0.02)
        self.assertLessEqual(samples.max().item(), 50.0)
        bank.eval()
        self.assertEqual(bank.sample_augmentation_factor(reference).item(), 1.0)

    def test_dynamic_time_gate_is_query_specific_and_scale_robust(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=4,
            regression_transform_aggregation="dynamic",
            regression_scale_augmentation=False,
        )
        support = torch.tensor(
            [
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            ]
        )
        labels = torch.sqrt(
            torch.tensor([[1.0, 4.0, 100.0], [10.0, 100.0, 10_000.0]])
        )
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        predictions, _, diagnostics = head.forward_regression_batched(
            support, labels, query, return_diagnostics=True
        )
        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(diagnostics["aggregation_weights"].shape, (4, 2))
        torch.testing.assert_close(
            diagnostics["aggregation_weights"].sum(dim=0), torch.ones(2)
        )
        head.regression_loss(
            predictions, torch.sqrt(torch.tensor([2.0, 80.0]))
        ).backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in head.time_transform_bank.dynamic_gate.parameters()
        ))

    def test_regression_gate_auxiliary_prefers_the_best_branch_per_query(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=3,
            regression_gate_aux_weight=0.1,
            regression_gate_target_temperature=0.1,
        )
        branch_predictions = torch.tensor(
            [[10.0, 80.0], [40.0, 20.0], [100.0, 100.0]],
            requires_grad=True,
        )
        targets = torch.tensor([10.0, 20.0])
        aligned_logits = torch.tensor(
            [[5.0, -5.0], [-5.0, 5.0], [-5.0, -5.0]],
            requires_grad=True,
        )
        misaligned_logits = -aligned_logits.detach()
        aligned = head.regression_gate_auxiliary_loss(
            branch_predictions,
            torch.softmax(aligned_logits, dim=0),
            targets,
        )
        misaligned = head.regression_gate_auxiliary_loss(
            branch_predictions,
            torch.softmax(misaligned_logits, dim=0),
            targets,
        )
        self.assertLess(float(aligned.detach()), float(misaligned.detach()))
        aligned.backward()
        self.assertIsNotNone(aligned_logits.grad)
        self.assertGreater(float(aligned_logits.grad.abs().sum()), 0.0)
        self.assertIsNone(branch_predictions.grad)

    @staticmethod
    def _legacy_two_term_weights():
        """Pin complementary metrics off so only MAE+RMSE remain (historical)."""
        return dict(
            regression_huber_weight=0.0,
            regression_log_rmse_weight=0.0,
            regression_relative_mae_weight=0.0,
            regression_bias_weight=0.0,
            regression_quantile_weight=0.0,
        )

    def test_regression_loss_scale_power_preserves_historical_default(self):
        labels = torch.sqrt(torch.tensor([10.0, 100.0]))
        predictions = torch.tensor([20.0, 110.0])
        historical = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=0.5,
            regression_rmse_weight=0.5,
            **self._legacy_two_term_weights(),
        )
        fixed_reference = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=0.5,
            regression_rmse_weight=0.5,
            regression_loss_scale_power=0.0,
            regression_loss_reference_hours=100.0,
            **self._legacy_two_term_weights(),
        )
        self.assertAlmostEqual(
            float(historical.regression_loss(predictions, labels)), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(fixed_reference.regression_loss(predictions, labels)), 0.1, places=5
        )

    def test_regression_loss_rejects_zero_metric_weights(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PrototypicalHead(
                regression_mode="learned_transform_ensemble",
                regression_mae_weight=0.0,
                regression_rmse_weight=0.0,
                regression_huber_weight=0.0,
                regression_log_rmse_weight=0.0,
                regression_relative_mae_weight=0.0,
                regression_bias_weight=0.0,
                regression_quantile_weight=0.0,
            )

    def test_regression_loss_respects_label_space_flag(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=1.0,
            regression_rmse_weight=0.0,
            regression_loss_scale_power=1.0,
            **self._legacy_two_term_weights(),
        )
        predictions = torch.tensor([20.0, 110.0])
        hours = torch.tensor([10.0, 100.0])
        sqrt_labels = torch.sqrt(hours)
        from_sqrt = float(head.regression_loss(predictions, sqrt_labels))
        from_hours = float(
            head.regression_loss(
                predictions, hours, labels_in_output_space=True
            )
        )
        self.assertAlmostEqual(from_sqrt, from_hours, places=5)
        # Passing already-hours labels without the flag would square them again.
        wrongly_squared = float(head.regression_loss(predictions, hours))
        self.assertNotAlmostEqual(from_hours, wrongly_squared, places=3)

    def test_regression_loss_components_are_finite_and_weighted_in_primary_loss(self):
        """Shipped multi-metric mix: each term finite; MAE and RMSE both contribute."""
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=0.5,
            regression_rmse_weight=0.5,
            regression_huber_weight=0.15,
            regression_log_rmse_weight=0.15,
            regression_relative_mae_weight=0.05,
            regression_bias_weight=0.05,
            regression_quantile_weight=0.1,
            regression_quantile_level=0.5,
            regression_loss_scale_power=1.0,
        )
        # Asymmetric errors: large positive residual + smaller negative.
        predictions = torch.tensor([30.0, 80.0, 5.0], requires_grad=True)
        labels = torch.sqrt(torch.tensor([10.0, 100.0, 4.0]))
        components = head.regression_loss_components(predictions, labels)
        for key in (
            "mae", "rmse", "huber", "log_rmse", "relative_mae", "bias", "quantile"
        ):
            self.assertIn(key, components)
            self.assertTrue(torch.isfinite(components[key]))
            self.assertGreaterEqual(float(components[key].detach()), 0.0)

        full = head.regression_loss(predictions, labels)
        mae_only = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=1.0,
            regression_rmse_weight=0.0,
            **self._legacy_two_term_weights(),
        ).regression_loss(predictions.detach(), labels)
        rmse_only = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=0.0,
            regression_rmse_weight=1.0,
            **self._legacy_two_term_weights(),
        ).regression_loss(predictions.detach(), labels)
        # Full mix is a convex combination of positive terms, so it sits
        # between pure MAE and pure RMSE only when those dominate; with extra
        # terms it must remain finite and strictly positive for nonzero errors.
        self.assertTrue(torch.isfinite(full))
        self.assertGreater(float(full.detach()), 0.0)
        self.assertGreater(float(mae_only.detach()), 0.0)
        self.assertGreater(float(rmse_only.detach()), 0.0)
        # Turning off all extras recovers the classical MAE+RMSE blend.
        classical = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=0.5,
            regression_rmse_weight=0.5,
            **self._legacy_two_term_weights(),
        ).regression_loss(predictions.detach(), labels)
        self.assertAlmostEqual(
            float(classical.detach()),
            0.5 * float(mae_only.detach()) + 0.5 * float(rmse_only.detach()),
            places=5,
        )
        full.backward()
        self.assertIsNotNone(predictions.grad)
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertGreater(float(predictions.grad.abs().sum()), 0.0)

    def test_regression_loss_extra_terms_change_loss_and_keep_grads_on_bank(self):
        head_rich = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=4,
            regression_scale_augmentation=False,
            regression_mae_weight=0.5,
            regression_rmse_weight=0.5,
            regression_huber_weight=0.2,
            regression_log_rmse_weight=0.2,
            regression_relative_mae_weight=0.1,
            regression_bias_weight=0.1,
        )
        head_legacy = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=4,
            regression_scale_augmentation=False,
            regression_mae_weight=0.5,
            regression_rmse_weight=0.5,
            **self._legacy_two_term_weights(),
        )
        support = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        labels = torch.sqrt(torch.tensor([1.0, 10.0, 100.0, 1000.0]))
        query = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        # Shared predictions from one forward; compare losses with same tensors.
        with torch.no_grad():
            predictions, _ = head_legacy.forward_regression(support, labels, query)
        targets = predictions.detach() + torch.tensor([5.0, -20.0])
        loss_legacy = head_legacy.regression_loss(
            predictions, targets, labels_in_output_space=True
        )
        loss_rich = head_rich.regression_loss(
            predictions, targets, labels_in_output_space=True
        )
        self.assertTrue(torch.isfinite(loss_rich))
        self.assertTrue(torch.isfinite(loss_legacy))
        # Richer mix must not collapse to the two-term value for nonzero errors.
        self.assertNotAlmostEqual(
            float(loss_rich.detach()), float(loss_legacy.detach()), places=4
        )
        # Gradients reach transform-bank parameters through the real path.
        head_rich.zero_grad(set_to_none=True)
        pred_live, _ = head_rich.forward_regression(support, labels, query)
        head_rich.regression_loss(
            pred_live, targets, labels_in_output_space=True
        ).backward()
        bank = head_rich.time_transform_bank
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in (
                bank.power_logits,
                bank.log_scales,
                bank.branch_logit_scales,
                bank.aggregation_logits,
            )
        ))

    def test_regression_loss_hours_and_sqrt_paths_match_for_multi_metric(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_mae_weight=0.4,
            regression_rmse_weight=0.4,
            regression_huber_weight=0.1,
            regression_log_rmse_weight=0.1,
            regression_relative_mae_weight=0.05,
            regression_bias_weight=0.05,
        )
        predictions = torch.tensor([12.0, 50.0, 200.0])
        hours = torch.tensor([9.0, 64.0, 121.0])
        a = float(head.regression_loss(predictions, torch.sqrt(hours)))
        b = float(
            head.regression_loss(predictions, hours, labels_in_output_space=True)
        )
        self.assertAlmostEqual(a, b, places=5)

    def test_regression_loss_gate_aux_grads_flow_to_gate_not_detached_branches(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=3,
            regression_transform_aggregation="dynamic",
            regression_gate_aux_weight=0.2,
            regression_scale_augmentation=False,
        )
        support = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        labels = torch.sqrt(torch.tensor([1.0, 4.0, 25.0, 100.0]))
        query = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        predictions, _, diagnostics = head.forward_regression(
            support, labels, query, return_diagnostics=True
        )
        # Hour targets near the predictions keep the primary term finite.
        targets_hours = predictions.detach() + torch.tensor([1.0, -3.0])
        loss = head.regression_loss(
            predictions,
            targets_hours,
            labels_in_output_space=True,
            branch_predictions=diagnostics["branch_predictions_hours"],
            aggregation_weights=diagnostics["aggregation_weights"],
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gate_grads = [
            parameter.grad
            for parameter in head.time_transform_bank.dynamic_gate.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gate_grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in gate_grads))
        self.assertIsNotNone(head.time_transform_bank.aggregation_logits.grad)
        self.assertGreater(
            float(head.time_transform_bank.aggregation_logits.grad.abs().sum()), 0.0
        )

    def test_regression_gate_aux_stays_peaked_on_short_horizon_errors(self):
        """Sub-hour branch errors must not be flattened by a 1-hour error floor."""
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=3,
            regression_gate_aux_weight=1.0,
            regression_gate_target_temperature=0.1,
        )
        # Three branches; branch 0 is clearly best on a 0.1h scale.
        branch_predictions = torch.tensor(
            [[0.10, 0.20], [0.40, 0.50], [0.80, 0.90]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0.10, 0.20])
        aligned = torch.softmax(
            torch.tensor([[5.0, 5.0], [-5.0, -5.0], [-5.0, -5.0]]), dim=0
        )
        misaligned = torch.softmax(
            torch.tensor([[-5.0, -5.0], [5.0, 5.0], [-5.0, -5.0]]), dim=0
        )
        aligned_loss = float(
            head.regression_gate_auxiliary_loss(branch_predictions, aligned, targets)
        )
        misaligned_loss = float(
            head.regression_gate_auxiliary_loss(
                branch_predictions, misaligned, targets
            )
        )
        self.assertLess(aligned_loss, misaligned_loss)
        # Soft targets should put majority mass on the best branch.
        errors = (branch_predictions - targets.view(1, -1)).abs()
        scale = errors.mean(dim=0, keepdim=True).clamp_min(1e-4)
        soft = torch.softmax(
            -errors / (scale * head.regression_gate_target_temperature), dim=0
        )
        self.assertGreater(float(soft[0].min()), 0.5)

    def test_legacy_huber_regression_loss_finite(self):
        head = PrototypicalHead(regression_mode="sqrt_knn")
        predictions = torch.tensor([1.5, 2.5], requires_grad=True)
        labels = torch.tensor([1.0, 3.0])
        loss = head.regression_loss(predictions, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(predictions.grad)
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_retrieval_contrastive_helpers_finite_and_grad(self):
        from training_strategies.retrieval_strategy import (
            _covariance_loss,
            _nca_knn_loss,
            _regression_neighbor_contrastive,
            _supcon_loss,
            _variance_loss,
        )

        torch.manual_seed(0)
        z = torch.randn(12, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
        cases = torch.arange(12)
        # Distinct cases so same-label pairs remain valid positives.
        sc = _supcon_loss(z, labels, cases, temperature=0.07)
        self.assertIsNotNone(sc)
        self.assertTrue(torch.isfinite(sc))
        sc.backward(retain_graph=True)
        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.abs().sum()), 0.0)

        z.grad = None
        nca = _nca_knn_loss(z, labels, cases, temperature=0.07)
        self.assertIsNotNone(nca)
        self.assertTrue(torch.isfinite(nca))
        nca.backward(retain_graph=True)
        self.assertGreater(float(z.grad.abs().sum()), 0.0)

        z.grad = None
        y = torch.linspace(0.0, 5.0, 12)
        reg_c = _regression_neighbor_contrastive(
            z, y, cases, temperature=0.07, pos_k=2
        )
        self.assertIsNotNone(reg_c)
        self.assertTrue(torch.isfinite(reg_c))
        reg_c.backward()
        self.assertGreater(float(z.grad.abs().sum()), 0.0)

        z_det = z.detach()
        self.assertTrue(torch.isfinite(_variance_loss(z_det)))
        self.assertTrue(torch.isfinite(_covariance_loss(z_det)))

    def test_vectorized_regression_contrastive_matches_loop_reference(self):
        """Guard the vectorized neighbor-contrastive against a direct loop oracle.

        Target values are uniquely spaced so nearest-k ties cannot diverge
        between subset-topk and full-matrix topk.
        """
        from training_strategies.retrieval_strategy import (
            _regression_neighbor_contrastive,
        )

        torch.manual_seed(3)
        z = torch.nn.functional.normalize(torch.randn(10, 6), dim=-1)
        # Random targets avoid equal-distance ties that make top-k index choice
        # implementation-defined across subset vs full-matrix selection.
        y = torch.randn(10) * 10.0
        cases = torch.tensor([0, 1, 0, 2, 3, 4, 5, 1, 6, 7])
        vectorized = _regression_neighbor_contrastive(z, y, cases, pos_k=2)

        # Independent loop reference (same definition as the original helper).
        z_n = torch.nn.functional.normalize(z.float(), dim=1)
        logits = (z_n @ z_n.t()) / 0.07
        ignore = torch.eye(10, dtype=torch.bool) | cases.view(-1, 1).eq(cases.view(1, -1))
        logits = logits.masked_fill(ignore, -1e4)
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        losses = []
        for i in range(10):
            candidates = (~ignore[i]).nonzero(as_tuple=False).squeeze(1)
            if candidates.numel() == 0:
                continue
            diffs = (y[candidates] - y[i]).abs()
            k_eff = min(2, int(candidates.numel()))
            positives = candidates[torch.topk(diffs, k_eff, largest=False).indices]
            losses.append(-log_prob[i, positives].mean())
        reference = torch.stack(losses).mean()
        self.assertAlmostEqual(float(vectorized), float(reference), places=5)

    def test_batched_regression_neighbors_match_per_query_forward(self):
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=4,
            regression_transform_aggregation="dynamic",
            regression_scale_augmentation=False,
            regression_gate_aux_weight=0.05,
        )
        B, K, D = 6, 4, 8
        support = torch.randn(B, K, D)
        labels = torch.sqrt(torch.rand(B, K) * 50 + 0.1)
        query = torch.randn(B, D)
        pred_b, _, diag_b = head.forward_regression_batched(
            support, labels, query, return_diagnostics=True
        )
        preds = []
        branches = []
        weights = []
        for i in range(B):
            p, _, d = head.forward_regression(
                support[i], labels[i], query[i : i + 1], return_diagnostics=True
            )
            preds.append(p.reshape(-1)[0])
            branches.append(d["branch_predictions_hours"])
            weights.append(d["aggregation_weights"])
        pred_loop = torch.stack(preds)
        torch.testing.assert_close(pred_b, pred_loop, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            diag_b["branch_predictions_hours"],
            torch.cat(branches, dim=1),
            atol=1e-5,
            rtol=1e-5,
        )
        loss = head.regression_loss(
            pred_b,
            labels[:, 0],
            branch_predictions=diag_b["branch_predictions_hours"],
            aggregation_weights=diag_b["aggregation_weights"],
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in head.time_transform_bank.dynamic_gate.parameters()
        ))

    def test_new_regression_bank_cannot_change_classification_logits(self):
        common = {
            "classification_mode": "coverage_fallback",
            "count_normalization_gamma": 0.0,
            "local_centering": True,
            "coverage_fallback_margin": 1.0,
        }
        legacy = PrototypicalHead(regression_mode="sqrt_knn", **common)
        learned = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=8,
            **common,
        )
        shared = {
            key: value
            for key, value in legacy.state_dict().items()
            if key in learned.state_dict()
        }
        learned.load_state_dict(shared, strict=False)
        local = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])
        pool = torch.cat([local, torch.tensor([[-1.0, 0.0]])])
        pool_labels = torch.tensor([0, 0, 1, 2])
        query = torch.tensor([[1.0, 0.0]])
        old_logits, old_classes, _ = legacy.forward_classification(
            local, labels, query,
            global_support_features=pool,
            global_support_labels=pool_labels,
        )
        new_logits, new_classes, _ = learned.forward_classification(
            local, labels, query,
            global_support_features=pool,
            global_support_labels=pool_labels,
        )
        torch.testing.assert_close(new_logits, old_logits, atol=0.0, rtol=0.0)
        torch.testing.assert_close(new_classes, old_classes, atol=0.0, rtol=0.0)

    def test_time_transform_scope_freezes_every_other_parameter(self):
        class TinyExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.proto_head = PrototypicalHead(
                    regression_mode="learned_transform_ensemble",
                    regression_num_transforms=4,
                )

        model = nn.ModuleDict({"expert": TinyExpert()})
        trainable = configure_trainable_scope(model, "time_transform")
        self.assertTrue(trainable)
        self.assertTrue(all(
            "proto_head.time_transform_bank." in name
            or "embedder.time_input_adapter." in name
            for name in trainable
        ))
        self.assertTrue(all(
            parameter.requires_grad == (name in trainable)
            for name, parameter in model.named_parameters()
        ))

    def test_regression_gate_scope_trains_only_selector_parameters(self):
        class TinyExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.proto_head = PrototypicalHead(
                    regression_mode="learned_transform_ensemble",
                    regression_num_transforms=4,
                    regression_transform_aggregation="dynamic",
                )

        model = nn.ModuleDict({"expert": TinyExpert()})
        trainable = configure_trainable_scope(model, "regression_gate")
        self.assertTrue(trainable)
        self.assertTrue(any("dynamic_gate" in name for name in trainable))
        self.assertTrue(any(name.endswith("aggregation_logits") for name in trainable))
        self.assertTrue(all(
            "proto_head.time_transform_bank.dynamic_gate." in name
            or name.endswith("proto_head.time_transform_bank.aggregation_logits")
            for name in trainable
        ))
        self.assertTrue(all(
            parameter.requires_grad == (name in trainable)
            for name, parameter in model.named_parameters()
        ))

    def test_prefix_times_use_k_learned_transforms_in_hours(self):
        for branches in (4, 8):
            adapter = LearnedTemporalInputAdapter(16, num_transforms=branches)
            seconds = torch.tensor([[3600.0, 60.0], [7200.0, 1800.0]])
            transformed = adapter.transformed_features(seconds)
            self.assertEqual(transformed.shape, (2, 2, branches))
            self.assertTrue(torch.isfinite(transformed).all())
            self.assertTrue(((transformed >= 0) & (transformed < 1)).all())
            output = adapter(seconds, augmentation_factor=torch.tensor(50.0))
            self.assertEqual(output.shape, (2, 16))
            output.sum().backward()
            self.assertIsNotNone(adapter.power_logits.grad)
            self.assertIsNotNone(adapter.log_scales.grad)

    def test_temporal_adapter_is_bypassed_exactly_for_classification(self):
        legacy = PretrainedEventEmbedder(4, 3, 16, dropout=0.0)
        temporal = PretrainedEventEmbedder(
            4,
            3,
            16,
            dropout=0.0,
            time_input_config={
                "regression_time_input_transforms": True,
                "regression_time_replace_legacy": True,
                "regression_num_transforms": 8,
            },
        )
        temporal.load_state_dict(legacy.state_dict(), strict=False)
        events = pd.DataFrame(
            {
                "activity_embedding": [
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                ],
                "resource_embedding": [
                    np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ],
                "cost": [10.0, 20.0],
                "time_from_start": [3600.0, 7200.0],
                "time_from_previous": [60.0, 1800.0],
            }
        )
        legacy.eval()
        temporal.eval()
        expected = legacy(events, use_time_adapter=False)
        classification = temporal(events, use_time_adapter=False)
        regression = temporal(events, use_time_adapter=True)
        torch.testing.assert_close(classification, expected, atol=0.0, rtol=0.0)
        self.assertFalse(torch.equal(regression, expected))

    def test_independent_clock_encoders_have_disjoint_parameters_and_shapes(self):
        encoder = IndependentTemporalInputEncoder(
            16,
            temporal_start_num_transforms=3,
            temporal_previous_num_transforms=5,
        )
        self.assertEqual(encoder.start_time_encoder.num_transforms, 3)
        self.assertEqual(encoder.previous_time_encoder.num_transforms, 5)
        start_parameters = {
            parameter.data_ptr()
            for parameter in encoder.start_time_encoder.parameters()
        }
        previous_parameters = {
            parameter.data_ptr()
            for parameter in encoder.previous_time_encoder.parameters()
        }
        self.assertTrue(start_parameters)
        self.assertTrue(previous_parameters)
        self.assertTrue(start_parameters.isdisjoint(previous_parameters))

        seconds = torch.tensor([[3600.0, 60.0], [7200.0, 1800.0]])
        before = encoder.transformed_features(seconds)
        with torch.no_grad():
            encoder.start_time_encoder.power_logits.add_(0.5)
        after = encoder.transformed_features(seconds)
        self.assertFalse(torch.equal(before["time_from_start"], after["time_from_start"]))
        torch.testing.assert_close(
            before["time_from_previous"],
            after["time_from_previous"],
            atol=0.0,
            rtol=0.0,
        )

    def test_independent_temporal_features_feed_both_tasks(self):
        legacy = PretrainedEventEmbedder(4, 3, 16, dropout=0.0)
        temporal = PretrainedEventEmbedder(
            4,
            3,
            16,
            dropout=0.0,
            time_input_config={
                "temporal_input_transforms": True,
                "temporal_start_num_transforms": 3,
                "temporal_previous_num_transforms": 5,
            },
        )
        temporal.load_state_dict(legacy.state_dict(), strict=False)
        events = pd.DataFrame(
            {
                "activity_embedding": [
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                ],
                "resource_embedding": [
                    np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ],
                "cost": [10.0, 20.0],
                "time_from_start": [3600.0, 7200.0],
                "time_from_previous": [60.0, 1800.0],
            }
        )
        legacy.eval()
        temporal.eval()
        fixed = legacy(events, use_time_adapter=False)
        classification = temporal(events, use_time_adapter=False)
        regression = temporal(
            events, use_time_adapter=True, time_scale_factor=torch.tensor(1.0)
        )
        self.assertFalse(torch.equal(classification, fixed))
        torch.testing.assert_close(classification, regression, atol=0.0, rtol=0.0)

        classification.sum().backward()
        self.assertIsNotNone(
            temporal.temporal_input_encoder.start_time_encoder.power_logits.grad
        )
        self.assertIsNotNone(
            temporal.temporal_input_encoder.previous_time_encoder.power_logits.grad
        )

    def test_input_clocks_and_output_target_use_three_independent_banks(self):
        encoder = IndependentTemporalInputEncoder(
            16,
            temporal_start_num_transforms=3,
            temporal_previous_num_transforms=5,
        )
        head = PrototypicalHead(
            regression_mode="learned_transform_ensemble",
            regression_num_transforms=7,
        )
        self.assertEqual(encoder.start_time_encoder.power_logits.numel(), 3)
        self.assertEqual(encoder.previous_time_encoder.power_logits.numel(), 5)
        self.assertEqual(head.time_transform_bank.power_logits.numel(), 7)
        parameter_sets = [
            {parameter.data_ptr() for parameter in module.parameters()}
            for module in (
                encoder.start_time_encoder,
                encoder.previous_time_encoder,
                head.time_transform_bank,
            )
        ]
        self.assertTrue(parameter_sets[0].isdisjoint(parameter_sets[1]))
        self.assertTrue(parameter_sets[0].isdisjoint(parameter_sets[2]))
        self.assertTrue(parameter_sets[1].isdisjoint(parameter_sets[2]))

    def test_temporal_joint_scope_trains_both_clocks_and_output_bank_only(self):
        class TinyExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.embedder = nn.Module()
                self.embedder.temporal_input_encoder = IndependentTemporalInputEncoder(
                    8,
                    temporal_start_num_transforms=3,
                    temporal_previous_num_transforms=5,
                )
                self.proto_head = PrototypicalHead(
                    regression_mode="learned_transform_ensemble",
                    regression_num_transforms=7,
                )

        model = nn.ModuleDict({"expert": TinyExpert()})
        trainable = configure_trainable_scope(model, "temporal_joint")
        self.assertTrue(any("start_time_encoder" in name for name in trainable))
        self.assertTrue(any("previous_time_encoder" in name for name in trainable))
        self.assertTrue(any("time_transform_bank" in name for name in trainable))
        self.assertTrue(all(
            "embedder.temporal_input_encoder." in name
            or "proto_head.time_transform_bank." in name
            for name in trainable
        ))
        self.assertTrue(all(
            parameter.requires_grad == (name in trainable)
            for name, parameter in model.named_parameters()
        ))


if __name__ == "__main__":
    unittest.main()
