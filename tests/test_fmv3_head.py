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
from utils.parameter_utils import configure_trainable_scope


class FMV3HeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

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

        loss = head.regression_loss(predictions, torch.tensor([2.0, 80.0]))
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
        head.regression_loss(predictions, torch.tensor([2.0, 80.0])).backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in head.time_transform_bank.dynamic_gate.parameters()
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
