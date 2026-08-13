import unittest

import numpy as np
import torch

from components.prototypical_head import PrototypicalHead
from evaluation.fmv3_metrics import classification_metrics, regression_metrics
from evaluation.fmv3_protocol import (
    _batched_head_probabilities,
    _expert_confidence_weights,
    _fuse_structured_prediction,
    _structured_class_probabilities,
    _weighted_stack_mean,
    _virtual_support_views,
    fixed_case_split,
    support_case_order,
)


def _tasks():
    tasks = []
    for case in range(10):
        for label in ({0, 1, 2} if case == 9 else {0, case % 2}):
            tasks.append(([{"dummy": True}], label, f"c{case}"))
    return tasks


class FMV3ProtocolTests(unittest.TestCase):
    def test_fixed_split_is_reproducible_and_disjoint(self):
        left = fixed_case_split(_tasks(), 42, 0.3, 4)
        right = fixed_case_split(_tasks(), 42, 0.3, 4)
        self.assertEqual(left, right)
        self.assertFalse(left[0] & left[1])

    def test_class_aware_order_covers_rare_label_first(self):
        tasks = _tasks()
        allowed = {f"c{i}" for i in range(10)}
        order = support_case_order(tasks, allowed, "class_aware", 42)
        first_labels = {item[1] for item in tasks if item[2] == order[0]}
        self.assertIn(2, first_labels)

    def test_balanced_accuracy_retains_zero_recall_class(self):
        result = classification_metrics(
            y_true=[0, 0, 1, 1, 2],
            y_pred=[0, 0, 0, 0, 0],
            probabilities=np.asarray([[1, 0, 0]] * 5, dtype=float),
            class_universe=[0, 1, 2],
            confidences=[1] * 5,
            case_ids=["a", "a", "b", "b", "c"],
            support_counts={0: 3, 1: 1},
            pool_covered=[True, True, True, True, False],
            retrieval_covered=[True, True, False, False, False],
        )
        self.assertAlmostEqual(result["balanced_accuracy"], 1 / 3)
        self.assertEqual(result["zero_recall_classes"], 2)
        self.assertEqual(result["frequency_bin_recall"]["n=0"], 0.0)
        self.assertAlmostEqual(result["macro_label_recall_at_k"], 1 / 3)
        self.assertAlmostEqual(result["macro_retrieval_given_pool"], 0.5)
        self.assertAlmostEqual(result["macro_decision_given_retrieval"], 1.0)

    def test_remaining_time_suite_has_interpretable_zero_point(self):
        result = regression_metrics([1, 2, 100], [2, 2, 2])
        self.assertIn("d2_absolute_error", result)
        self.assertAlmostEqual(result["mae_skill_vs_median"], 0.0)
        self.assertAlmostEqual(result["mae_hours"], 99.0 / 3.0)
        self.assertAlmostEqual(result["rmse_hours"], np.sqrt((1.0 + 0.0 + 98.0 ** 2) / 3.0))

    def test_expert_confidence_temperature_controls_sharpness(self):
        logits = torch.tensor([[0.0], [1.0], [2.0]])
        default = _expert_confidence_weights(logits, temperature=1.0)
        sharp = _expert_confidence_weights(logits, temperature=0.5)
        flat = _expert_confidence_weights(logits, temperature=2.0)
        self.assertGreater(sharp[-1, 0].item(), default[-1, 0].item())
        self.assertLess(flat[-1, 0].item(), default[-1, 0].item())
        torch.testing.assert_close(default.sum(dim=0), torch.ones(1))
        prior = _expert_confidence_weights(
            torch.zeros(3, 1), temperature=1.0, prior_weights=torch.tensor([1.0, 2.0, 1.0])
        )
        torch.testing.assert_close(prior[:, 0], torch.tensor([0.25, 0.5, 0.25]))
        with self.assertRaises(ValueError):
            _expert_confidence_weights(logits, temperature=0.0)

    def test_weighted_stack_mean_preserves_neutral_average(self):
        values = torch.tensor([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])
        torch.testing.assert_close(
            _weighted_stack_mean(values, torch.ones(3)),
            values.mean(dim=0),
        )
        torch.testing.assert_close(
            _weighted_stack_mean(values, torch.tensor([1.0, 0.0, 1.0])),
            torch.tensor([3.0, 7.0]),
        )
        torch.testing.assert_close(
            _weighted_stack_mean(values, torch.zeros(3)),
            values.mean(dim=0),
        )

    def test_virtual_support_views_preserve_default_and_class_labels(self):
        support = np.arange(6)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        self.assertEqual(len(_virtual_support_views(support)), 1)

        views = _virtual_support_views(
            support,
            {
                "virtual_expert_replicates": 3,
                "virtual_expert_support_fraction": 0.4,
                "virtual_expert_min_support_prefixes": 1,
                "virtual_expert_seed": 7,
            },
            "classification",
            labels=labels,
        )
        self.assertEqual(len(views), 3)
        np.testing.assert_array_equal(views[0], support)
        for view in views[1:]:
            self.assertLess(len(view), len(support))
            self.assertEqual(set(labels[view].tolist()), {0, 1, 2})

    def test_batched_coverage_fallback_matches_head(self):
        head = PrototypicalHead(
            classification_mode="coverage_fallback",
            count_normalization_gamma=0.0,
            local_centering=True,
            global_centering=False,
            coverage_fallback_margin=0.7,
            fallback_inference_temperature=0.4,
            prior_mode="balanced",
            classification_example_selector_enabled=True,
        )
        with torch.no_grad():
            head.classification_example_selector.network[-1].weight.normal_(
                mean=0.0, std=0.2
            )
        head.eval()
        pool = torch.tensor([
            [1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]
        ])
        labels = torch.tensor([0, 0, 1, 1, 2])
        query = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
        local_positions = torch.tensor([[0, 1], [0, 4]])
        batched, _ = _batched_head_probabilities(
            head, query, pool, labels, local_positions, [0, 1, 2],
            "configured", "balanced", 1.0,
        )
        expected = []
        for row in range(query.size(0)):
            _, classes, probabilities = head.forward_classification(
                pool[local_positions[row]],
                labels[local_positions[row]],
                query[row : row + 1],
                global_support_features=pool,
                global_support_labels=labels,
                candidate_classes=torch.tensor([0, 1, 2]),
                prior_mode="balanced",
                prior_strength=1.0,
            )
            aligned = torch.zeros(3)
            aligned[classes.long()] = probabilities[0]
            expected.append(aligned)
        torch.testing.assert_close(batched, torch.stack(expected), atol=1e-6, rtol=1e-6)

    def test_structured_probabilities_back_off_to_seen_suffix(self):
        labels = torch.tensor([0, 1, 0, 1, 0])
        contexts = [(4, 7), (5, 7), (4, 8), (5, 8), (9, 7)]
        probabilities, support, orders, counts = _structured_class_probabilities(
            labels,
            contexts,
            query_indices=np.asarray([4]),
            support_indices=np.asarray([0, 1, 2, 3]),
            class_universe=[0, 1],
            max_order=2,
            smoothing=0.5,
        )
        self.assertEqual(counts, {0: 2, 1: 2})
        self.assertEqual(orders.tolist(), [1])
        self.assertEqual(support.tolist(), [2.0])
        np.testing.assert_allclose(probabilities, [[0.5, 0.5]], atol=1e-7)

    def test_structured_likelihood_balances_frequent_and_rare_classes(self):
        labels = torch.tensor([0, 0, 0, 0, 1, 0])
        contexts = [(7,), (1,), (2,), (3,), (7,), (7,)]
        probabilities, _, _, _ = _structured_class_probabilities(
            labels,
            contexts,
            query_indices=np.asarray([5]),
            support_indices=np.asarray([0, 1, 2, 3, 4]),
            class_universe=[0, 1],
            max_order=1,
            smoothing=0.5,
        )
        # Both classes produced state 7 once. Class-conditional normalization
        # makes that observation stronger evidence for the globally rare class.
        self.assertGreater(probabilities[0, 1], probabilities[0, 0])

    def test_structured_fusion_shrinks_weight_by_context_support(self):
        base = {
            "y_true": [0, 1],
            "y_pred": [0, 0],
            "probabilities": np.asarray([[0.8, 0.2], [0.8, 0.2]]),
            "confidences": [0.8, 0.8],
            "support_counts": {0: 2, 1: 2},
        }
        structured = {
            "probabilities": np.asarray([[0.2, 0.8], [0.2, 0.8]]),
            "structured_context_support": [0.0, 2.0],
        }
        fused = _fuse_structured_prediction(
            base, structured, [0, 1], weight=1.0, tau=2.0, fusion="mixture"
        )
        np.testing.assert_allclose(fused["probabilities"][0], [0.8, 0.2])
        np.testing.assert_allclose(fused["probabilities"][1], [0.5, 0.5])
        self.assertEqual(fused["structured_effective_weight"], [0.0, 0.5])

    def test_structured_fusion_can_override_low_support_weight(self):
        base = {
            "y_true": [1],
            "y_pred": [0],
            "probabilities": np.asarray([[0.8, 0.2]]),
            "confidences": [0.8],
            "support_counts": {0: 2, 1: 1},
        }
        structured = {
            "probabilities": np.asarray([[0.2, 0.8]]),
            "structured_context_support": [1.0],
            "support_counts": {0: 2, 1: 1},
        }
        base_fused = _fuse_structured_prediction(
            base,
            structured,
            [0, 1],
            weight=0.75,
            tau=0.5,
            fusion="mixture",
            low_support_threshold=0,
            low_support_weight=1.0,
            low_support_tau=0.25,
        )
        low_support_fused = _fuse_structured_prediction(
            base,
            structured,
            [0, 1],
            weight=0.75,
            tau=0.5,
            fusion="mixture",
            low_support_threshold=8,
            low_support_weight=1.0,
            low_support_tau=0.25,
        )
        self.assertAlmostEqual(base_fused["structured_effective_weight"][0], 0.5)
        self.assertAlmostEqual(low_support_fused["structured_effective_weight"][0], 0.8)
        self.assertEqual(low_support_fused["structured_total_support"], 3)
        self.assertEqual(low_support_fused["structured_selected_weight"], 1.0)
        self.assertEqual(low_support_fused["structured_selected_tau"], 0.25)

    def test_structured_output_temperature_calibrates_without_changing_decision(self):
        base = {
            "y_true": [0],
            "y_pred": [0],
            "probabilities": np.asarray([[0.7, 0.2, 0.1]]),
            "confidences": [0.7],
            "support_counts": {0: 1, 1: 1, 2: 1},
        }
        structured = {
            "probabilities": np.asarray([[0.7, 0.2, 0.1]]),
            "structured_context_support": [1.0],
        }
        default = _fuse_structured_prediction(
            base, structured, [0, 1, 2], weight=0.5, tau=1.0, fusion="mixture"
        )
        sharpened = _fuse_structured_prediction(
            base,
            structured,
            [0, 1, 2],
            weight=0.5,
            tau=1.0,
            fusion="mixture",
            output_temperature=0.5,
        )
        self.assertEqual(default["y_pred"], sharpened["y_pred"])
        self.assertGreater(
            sharpened["probabilities"][0, 0], default["probabilities"][0, 0]
        )
        self.assertAlmostEqual(float(sharpened["probabilities"].sum()), 1.0)
        with self.assertRaises(ValueError):
            _fuse_structured_prediction(
                base,
                structured,
                [0, 1, 2],
                weight=0.5,
                tau=1.0,
                fusion="mixture",
                output_temperature=0.0,
            )


if __name__ == "__main__":
    unittest.main()
