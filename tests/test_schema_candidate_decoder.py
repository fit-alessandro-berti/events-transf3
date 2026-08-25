import unittest

import torch

from components.prototypical_head import PrototypicalHead
from data_generator import TransformedLog
from evaluation.fmv3_protocol import (
    _enforce_support_only_candidates,
    _predict_classification_fixed_k,
    _predict_frozen_embedding_logistic,
)
from training_strategies.retrieval_strategy import (
    _sample_deployment_classification_episode,
)
from training_debug import split_training_tasks_by_case
from utils.data_utils import get_classification_and_regression_tasks
from utils.retrieval_utils import class_diverse_topk_indices


class SchemaCandidateDecoderTests(unittest.TestCase):
    def test_schema_metadata_survives_lazy_task_construction(self):
        candidates = (
            {"label_id": 0, "activity_name": "A", "activity_char_ids": (2,)},
            {"label_id": 1, "activity_name": "B", "activity_char_ids": (3,)},
            {"label_id": 2, "activity_name": "SCHEMA ONLY", "activity_char_ids": (4,)},
        )
        trace = [
            {
                "activity_id": 0,
                "activity_name": "A",
                "timestamp": 0.0,
                "case_id": "case-1",
            },
            {
                "activity_id": 1,
                "activity_name": "B",
                "timestamp": 3600.0,
                "case_id": "case-1",
            },
        ]
        tasks, _ = get_classification_and_regression_tasks(
            TransformedLog([trace], candidate_labels=candidates), max_seq_len=4
        )
        self.assertEqual(tasks.label_name_by_id[2], "SCHEMA ONLY")
        self.assertEqual(tasks.records[0].classification_target_name, "B")
        self.assertEqual(len(tasks.candidate_labels), 3)

    def test_absent_candidate_has_finite_probability_and_zero_support_gate(self):
        head = PrototypicalHead(
            feature_dim=2,
            classification_mode="global_local",
            semantic_candidate_decoder_enabled=True,
            support_gate_kappa=2.0,
        )
        support = torch.tensor([[1.0, 0.0]])
        query = torch.tensor([[0.0, 1.0]])
        candidate_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits, classes, probabilities, diagnostics = head.forward_classification(
            support,
            torch.tensor([0]),
            query,
            global_support_features=support,
            global_support_labels=torch.tensor([0]),
            candidate_classes=torch.tensor([0, 1]),
            candidate_features=candidate_features,
            return_diagnostics=True,
        )
        self.assertEqual(classes.tolist(), [0, 1])
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(probabilities[0, 1].item(), 0.0)
        self.assertEqual(diagnostics["support_gate"][0, 1].item(), 0.0)

    def test_support_fitted_residual_is_exactly_zero_for_absent_class(self):
        head = PrototypicalHead(
            feature_dim=2,
            classification_mode="global",
            semantic_candidate_decoder_enabled=True,
            support_fitted_classifier_enabled=True,
            support_fitted_regularization=1.0,
        )
        support = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
        _, _, _, diagnostics = head.forward_classification(
            support,
            torch.tensor([0, 0]),
            torch.tensor([[0.0, 1.0]]),
            candidate_classes=torch.tensor([0, 1]),
            candidate_features=torch.eye(2),
            return_diagnostics=True,
        )
        residual = diagnostics["support_fitted_evidence"]
        self.assertEqual(residual[0, 1].item(), 0.0)

    def test_class_diverse_retrieval_guarantees_breadth_when_classes_fit(self):
        similarities = torch.tensor([0.99, 0.98, 0.97, 0.50, 0.40])
        labels = torch.tensor([0, 0, 0, 1, 2])
        selected = class_diverse_topk_indices(
            similarities, labels, 3, classes_per_shortlist=3, examples_per_class=2
        )
        self.assertEqual(set(labels[selected].tolist()), {0, 1, 2})

    def test_deployment_episode_uses_disjoint_cases_and_all_support_prefixes(self):
        tasks = []
        for case in range(8):
            for prefix in range(3):
                tasks.append(([{"prefix": prefix}], prefix % 2, f"case-{case}"))
        config = {
            "num_queries": 4,
            "fmv3_training": {
                "support_case_budgets": [2],
                "deployment_query_case_fraction": 0.25,
                "deployment_queries_per_episode": 4,
                "deployment_support_max_prefixes": 0,
            },
        }
        support, query = _sample_deployment_classification_episode(
            tasks, "natural", config
        )
        support_cases = {item[2] for item in support}
        query_cases = {item[2] for item in query}
        self.assertFalse(support_cases & query_cases)
        self.assertEqual(len(support), 2 * 3)

    def test_case_validation_split_preserves_candidate_schema(self):
        candidates =(
        {"label_id":0 ,"activity_name":"A"},
        {"label_id":1 ,"activity_name":"B"},
        )
        traces =[]
        for case in range (4 ):
            traces .append ([
            {"activity_id":0 ,"activity_name":"A","timestamp":0.0 ,"case_id":case },
            {"activity_id":1 ,"activity_name":"B","timestamp":1.0 ,"case_id":case },
            ])
        classification ,regression =get_classification_and_regression_tasks (
        TransformedLog (traces ,candidate_labels =candidates ),max_seq_len =2 )
        train ,validation ,_=split_training_tasks_by_case ({
        "classification":[classification ],"regression":[regression ]
        },0.25 ,7 )
        self .assertEqual (len (train ["classification"][0 ].candidate_labels ),2 )
        self .assertEqual (len (validation ["classification"][0 ].candidate_labels ),2 )

    def test_evaluator_can_predict_schema_class_absent_from_support_pool(self):
        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.proto_head = PrototypicalHead(
                    feature_dim=2,
                    classification_mode="global_local",
                    semantic_candidate_decoder_enabled=True,
                )

        expert = Expert()
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=torch.float32
        )
        prediction = _predict_classification_fixed_k(
            [expert],
            [embeddings],
            torch.tensor([0, 0, 1]),
            query_indices=torch.tensor([2]).numpy(),
            support_indices=torch.tensor([0, 1]).numpy(),
            class_universe=[0, 1],
            retrieval_k=2,
            retrieval_mode="configured",
            prior_mode="balanced",
            prior_strength=1.0,
            eval_cfg={"classification_retrieval_policy": "class_diverse"},
            retrieval_embeddings_by_expert=[embeddings],
            candidate_features_by_expert=[torch.eye(2)],
        )
        self.assertEqual(prediction["y_pred"], [1])
        self.assertEqual(prediction["pool_covered"], [False])
        self.assertEqual(prediction["retrieval_covered"], [False])
        self.assertGreater(prediction["probabilities"][0, 1], 0.0)

    def test_frozen_embedding_logistic_diagnostic_uses_foundation_geometry(self):
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8], [0.9, 0.0], [0.0, 0.9]]
        )
        prediction = _predict_frozen_embedding_logistic(
            [embeddings],
            torch.tensor([0, 0, 1, 1, 0, 1]),
            query_indices=torch.tensor([4, 5]).numpy(),
            support_indices=torch.tensor([0, 1, 2, 3]).numpy(),
            class_universe=[0, 1],
            eval_cfg={"foundation_logistic_c": 10.0},
        )
        self.assertEqual(prediction["y_pred"], [0, 1])

    def test_support_only_contract_masks_absent_labels_after_fusion(self):
        prediction = _enforce_support_only_candidates(
            {
                "probabilities": [[0.1, 0.9]],
                "support_counts": {0: 3},
                "y_pred": [1],
                "confidences": [0.9],
            },
            [0, 1],
        )
        self.assertEqual(prediction["probabilities"][0, 1], 0.0)
        self.assertEqual(prediction["y_pred"], [0])


if __name__ == "__main__":
    unittest.main()
