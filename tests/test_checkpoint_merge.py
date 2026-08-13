import unittest

import torch

from merge_task_isolated_checkpoints import merge_task_isolated_state_dicts


class CheckpointMergeTests(unittest.TestCase):
    def test_merge_accepts_only_disjoint_task_scopes(self):
        base = {
            "experts.0.shared.weight": torch.tensor([1.0]),
            "experts.0.proto_head.time_transform_bank.weight": torch.tensor([2.0]),
        }
        classification = {
            **base,
            "experts.0.classification_embedding_adapter.weight": torch.tensor([3.0]),
            "experts.0.proto_head.classification_example_selector.weight": torch.tensor([6.0]),
        }
        regression = {
            **base,
            "experts.0.proto_head.time_transform_bank.weight": torch.tensor([4.0]),
            "experts.0.regression_embedding_adapter.weight": torch.tensor([5.0]),
            "experts.0.proto_head.regression_example_selector.weight": torch.tensor([7.0]),
        }
        merged, classification_keys, regression_keys = (
            merge_task_isolated_state_dicts(base, classification, regression)
        )
        self.assertEqual(
            classification_keys,
            {
                "experts.0.classification_embedding_adapter.weight",
                "experts.0.proto_head.classification_example_selector.weight",
            },
        )
        self.assertEqual(
            regression_keys,
            {
                "experts.0.proto_head.time_transform_bank.weight",
                "experts.0.regression_embedding_adapter.weight",
                "experts.0.proto_head.regression_example_selector.weight",
            },
        )
        self.assertEqual(float(merged["experts.0.shared.weight"]), 1.0)
        self.assertEqual(
            float(merged["experts.0.classification_embedding_adapter.weight"]), 3.0
        )
        self.assertEqual(
            float(merged["experts.0.proto_head.time_transform_bank.weight"]), 4.0
        )
        self.assertEqual(
            float(merged["experts.0.regression_embedding_adapter.weight"]), 5.0
        )
        self.assertEqual(
            float(merged["experts.0.proto_head.classification_example_selector.weight"]),
            6.0,
        )
        self.assertEqual(
            float(merged["experts.0.proto_head.regression_example_selector.weight"]),
            7.0,
        )

    def test_merge_rejects_shared_tensor_changes(self):
        base = {"experts.0.shared.weight": torch.tensor([1.0])}
        classification = {"experts.0.shared.weight": torch.tensor([2.0])}
        with self.assertRaises(RuntimeError):
            merge_task_isolated_state_dicts(base, classification, base)

        regression = {
            **base,
            "experts.0.embedder.temporal_input_encoder.weight": torch.tensor([2.0]),
        }
        with self.assertRaises(RuntimeError):
            merge_task_isolated_state_dicts(base, base, regression)


if __name__ == "__main__":
    unittest.main()
