import math
import unittest

import torch
import torch.nn as nn

from components.event_encoder import EventEncoder, StateAwarePrefixProjection
from utils.model_utils import load_state_dict_compatible
from utils.parameter_utils import configure_trainable_scope


class StateAwareEventEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_disabled_configuration_matches_historical_projection_exactly(self):
        encoder = EventEncoder(8, 2, 1, dropout=0.0).eval()
        src = torch.randn(3, 4, 8)
        mask = torch.tensor(
            [[False, False, True, True], [False] * 4, [False, True, True, True]]
        )
        actual = encoder(src, src_key_padding_mask=mask)

        batch, _, width = src.shape
        cls = encoder.cls_token.expand(batch, 1, width)
        manual_src = torch.cat([cls, src], dim=1)
        manual_mask = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool), mask], dim=1
        )
        manual_src = encoder.pos_encoder(manual_src * math.sqrt(encoder.d_model))
        tokens = encoder.transformer_encoder(
            manual_src, src_key_padding_mask=manual_mask
        )
        cls_out = tokens[:, 0]
        event_tokens = tokens[:, 1:]
        pooled, _ = encoder.mha_pool(
            query=encoder.pool_query.expand(batch, -1, -1),
            key=event_tokens,
            value=event_tokens,
            key_padding_mask=mask,
        )
        expected = encoder.out_norm(
            encoder.final_projection(torch.cat([cls_out, pooled[:, 0]], dim=-1))
        )
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_state_attention_masks_padding_and_is_task_conditioned(self):
        encoder = EventEncoder(
            8,
            2,
            1,
            dropout=0.0,
            prefix_config={
                "state_aware_prefix_attention": True,
                "prefix_attention_hidden_dim": 8,
            },
        ).eval()
        src = torch.randn(2, 4, 8)
        mask = torch.tensor(
            [[False, False, True, True], [False, False, False, False]]
        )
        classification, diagnostics = encoder(
            src,
            src_key_padding_mask=mask,
            task_type="classification",
            return_attention=True,
        )
        regression = encoder(
            src, src_key_padding_mask=mask, task_type="regression"
        )
        self.assertFalse(torch.equal(classification, regression))
        weights = diagnostics["state_attention"]
        self.assertEqual(tuple(weights.shape), (2, 2, 1, 4))
        torch.testing.assert_close(
            weights[0, :, :, 2:], torch.zeros_like(weights[0, :, :, 2:])
        )
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1))
        )
        self.assertAlmostEqual(float(diagnostics["state_gate"]), torch.sigmoid(torch.tensor(-3.0)).item())
        self.assertAlmostEqual(float(diagnostics["recency_strength"]), 0.25, places=5)

    def test_padding_values_cannot_change_prefix_projection(self):
        encoder = EventEncoder(
            8,
            2,
            1,
            dropout=0.0,
            prefix_config={"state_aware_prefix_attention": True},
        ).eval()
        valid = torch.randn(1, 2, 8)
        first = torch.cat([valid, torch.randn(1, 2, 8)], dim=1)
        second = torch.cat([valid, torch.randn(1, 2, 8) * 1000], dim=1)
        mask = torch.tensor([[False, False, True, True]])
        out_first = encoder(
            first, src_key_padding_mask=mask, task_type="classification"
        )
        out_second = encoder(
            second, src_key_padding_mask=mask, task_type="classification"
        )
        torch.testing.assert_close(out_first, out_second, atol=1e-6, rtol=1e-6)

    def test_recency_bias_is_monotone_and_masks_padding(self):
        pool = StateAwarePrefixProjection(
            8, 2, dropout=0.0, prefix_attention_initial_recency=0.5
        )
        mask = torch.tensor(
            [[False, False, False, True], [False, False, False, False]]
        )
        bias = pool.recency_attention_bias(
            mask, 2, 4, "classification", torch.device("cpu"), torch.float32
        ).reshape(2, 2, 1, 4)
        self.assertTrue(torch.isneginf(bias[0, :, :, 3]).all())
        self.assertTrue((bias[0, :, :, 0] < bias[0, :, :, 1]).all())
        self.assertTrue((bias[0, :, :, 1] < bias[0, :, :, 2]).all())
        torch.testing.assert_close(
            bias[1, :, :, -1], torch.zeros_like(bias[1, :, :, -1])
        )

    def test_classification_gradient_reaches_only_its_task_controls(self):
        encoder = EventEncoder(
            8,
            2,
            1,
            dropout=0.0,
            prefix_config={"state_aware_prefix_attention": True},
        )
        src = torch.randn(2, 4, 8)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        output = encoder(
            src, src_key_padding_mask=mask, task_type="classification"
        )
        output[:, 0].sum().backward()
        pool = encoder.state_aware_pool
        self.assertGreater(float(pool.task_queries.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(pool.task_queries.grad[1].abs().sum()), 0.0)
        self.assertGreater(float(pool.gate_logits.grad[0].abs()), 0.0)
        self.assertEqual(float(pool.gate_logits.grad[1].abs()), 0.0)
        self.assertGreater(float(pool.recency_logits.grad[0].abs()), 0.0)
        self.assertEqual(float(pool.recency_logits.grad[1].abs()), 0.0)

    def test_checkpoint_migration_and_constrained_scopes(self):
        class TinyModel(nn.Module):
            def __init__(self, enabled):
                super().__init__()
                self.encoder = EventEncoder(
                    8,
                    2,
                    1,
                    dropout=0.0,
                    prefix_config={"state_aware_prefix_attention": enabled},
                )
                self.embedder = nn.Module()
                self.embedder.temporal_input_encoder = nn.Linear(2, 2)
                self.proto_head = nn.Module()
                self.proto_head.time_transform_bank = nn.Linear(2, 2)

        old = TinyModel(False)
        migrated = TinyModel(True)
        incompatible = load_state_dict_compatible(migrated, old.state_dict())
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all("encoder.state_aware_pool." in key for key in incompatible.missing_keys)
        )

        prefix_names = configure_trainable_scope(migrated, "prefix_attention")
        self.assertTrue(prefix_names)
        self.assertTrue(
            all("encoder.state_aware_pool." in name for name in prefix_names)
        )
        joint_names = configure_trainable_scope(migrated, "temporal_prefix_joint")
        self.assertTrue(any("encoder.state_aware_pool." in name for name in joint_names))
        self.assertTrue(any("temporal_input_encoder." in name for name in joint_names))
        self.assertTrue(any("time_transform_bank." in name for name in joint_names))
        self.assertTrue(
            all(
                "encoder.state_aware_pool." in name
                or "temporal_input_encoder." in name
                or "time_transform_bank." in name
                for name in joint_names
            )
        )


if __name__ == "__main__":
    unittest.main()

