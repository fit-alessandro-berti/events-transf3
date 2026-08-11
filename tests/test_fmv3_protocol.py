import unittest

import numpy as np
import torch

from components.prototypical_head import PrototypicalHead
from evaluation.fmv3_metrics import classification_metrics, regression_metrics
from evaluation.fmv3_protocol import (
    _batched_head_probabilities,
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

    def test_batched_coverage_fallback_matches_head(self):
        head = PrototypicalHead(
            classification_mode="coverage_fallback",
            count_normalization_gamma=0.0,
            local_centering=True,
            global_centering=False,
            coverage_fallback_margin=0.7,
            fallback_inference_temperature=0.4,
            prior_mode="balanced",
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


if __name__ == "__main__":
    unittest.main()
