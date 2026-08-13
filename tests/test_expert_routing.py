import unittest

import torch
import torch.nn as nn

from components.moe_model import MoEModel
from components.task_confidence import (
    TASK_DESCRIPTOR_DIM,
    TaskConfidenceHead,
    build_task_descriptor,
)
from utils.parameter_utils import configure_trainable_scope


def _episode(label=0):
    sequence = [
        {
            "activity_id": 1,
            "time_from_start": 12.0,
            "time_from_previous": 2.0,
            "cost": 3.0,
        }
    ]
    return [(sequence, label)]


class _DummyExpert(nn.Module):
    def __init__(self, score):
        super().__init__()
        self.score = nn.Parameter(torch.tensor(float(score)))
        self.forward_calls = 0
        self.last_expert_confidence_logit = None

    def task_confidence_logit(self, support_set, query_set, task_type):
        return self.score

    def forward(self, support_set, query_set, task_type):
        self.forward_calls += 1
        count = len(query_set)
        labels = torch.zeros(count, dtype=torch.long, device=self.score.device)
        probabilities = torch.tensor(
            [[0.8, 0.2]], device=self.score.device
        ).expand(count, -1)
        return probabilities.log(), labels, probabilities


class ExpertRoutingTests(unittest.TestCase):
    def test_descriptor_is_finite_and_does_not_read_query_labels(self):
        support = _episode(0) + _episode(1)
        query_a = _episode(0)
        query_b = _episode(999)
        descriptor_a = build_task_descriptor(
            support, query_a, "classification"
        )
        descriptor_b = build_task_descriptor(
            support, query_b, "classification"
        )
        self.assertEqual(descriptor_a.shape, (TASK_DESCRIPTOR_DIM,))
        self.assertTrue(torch.isfinite(descriptor_a).all())
        torch.testing.assert_close(descriptor_a, descriptor_b)

    def test_all_router_architectures_learn(self):
        descriptor = build_task_descriptor(
            _episode(1), _episode(0), "classification"
        )
        for architecture in sorted(TaskConfidenceHead.ARCHITECTURES):
            head = TaskConfidenceHead(architecture, hidden_dim=8)
            initial = head(descriptor)
            self.assertEqual(initial.ndim, 0)
            loss = head.reliability_loss(descriptor, torch.tensor(1.0))
            loss.backward()
            self.assertTrue(any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            ))

    def test_eval_executes_exactly_top_half(self):
        model = MoEModel(
            num_experts=0,
            strategy="learned",
            proto_head_config={
                "expert_routing_confidence_enabled": True,
                "expert_active_fraction": 0.5,
            },
        )
        model.experts = nn.ModuleList([
            _DummyExpert(0.1),
            _DummyExpert(0.9),
            _DummyExpert(0.8),
            _DummyExpert(-0.2),
        ])
        model.num_experts = 4
        model.eval()
        predictions, labels, confidence = model(
            _episode(0), _episode(0), "classification"
        )
        self.assertEqual(model.active_expert_count, 2)
        self.assertEqual(
            model.last_routing_diagnostics["selected_expert_indices"], [1, 2]
        )
        self.assertEqual(
            [expert.forward_calls for expert in model.experts], [0, 1, 1, 0]
        )
        self.assertEqual(predictions.shape, (1, 2))
        self.assertEqual(labels.tolist(), [0])
        self.assertEqual(confidence.shape, (1,))

    def test_routing_scope_freezes_everything_else(self):
        class TinyExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.task_confidence_head = TaskConfidenceHead("mlp", hidden_dim=8)

        model = nn.ModuleDict({"expert": TinyExpert()})
        trainable = configure_trainable_scope(
            model, "expert_routing_confidence"
        )
        self.assertTrue(trainable)
        self.assertTrue(all("task_confidence_head." in name for name in trainable))
        self.assertTrue(all(
            parameter.requires_grad == (name in trainable)
            for name, parameter in model.named_parameters()
        ))


if __name__ == "__main__":
    unittest.main()
