import unittest

import torch

from components.prototypical_head import PrototypicalHead


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


if __name__ == "__main__":
    unittest.main()
