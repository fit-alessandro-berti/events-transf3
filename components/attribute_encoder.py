"""Schema-agnostic encoder for sparse typed XES event/case attributes."""

from __future__ import annotations

import torch
import torch.nn as nn
import math


class GenericAttributeEncoder(nn.Module):
    def __init__(self, d_model, hash_buckets=4096, attribute_dim=32):
        super().__init__()
        buckets = max(32, int(hash_buckets))
        dim = max(8, int(attribute_dim))
        self.name_embedding = nn.Embedding(buckets, dim, padding_idx=0)
        self.value_embedding = nn.Embedding(buckets, dim, padding_idx=0)
        self.type_embedding = nn.Embedding(4, dim, padding_idx=0)
        self.attribute_projection = nn.Sequential(
            nn.LayerNorm(3 * dim + 2),
            nn.Linear(3 * dim + 2, dim),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(dim, int(d_model), bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, events, device):
        count = len(events)
        max_attributes = max(
            (len(event.get("generic_attributes", ())) for event in events),
            default=0,
        )
        if count == 0 or max_attributes == 0:
            return self.output_projection.weight.new_zeros((count, self.output_projection.out_features))

        rows = []
        for event in events:
            attributes = [
                (
                    int(name_id),
                    int(type_id),
                    int(value_id),
                    math.copysign(math.log1p(abs(float(numeric))), float(numeric)),
                    float(is_missing),
                    1.0,
                )
                for name_id, type_id, value_id, numeric, is_missing
                in event.get("generic_attributes", ())
            ]
            attributes.extend(
                [(0, 0, 0, 0.0, 1.0, 0.0)]
                * (max_attributes - len(attributes))
            )
            rows.append(attributes)
        packed = torch.as_tensor(rows, dtype=torch.float32, device=device)
        names = packed[..., 0].long()
        types = packed[..., 1].long()
        values = packed[..., 2].long()
        numerics = packed[..., 3]
        missing = packed[..., 4]
        present = packed[..., 5]

        encoded = torch.cat(
            [
                self.name_embedding(names),
                self.type_embedding(types),
                self.value_embedding(values),
                numerics.unsqueeze(-1),
                missing.unsqueeze(-1),
            ],
            dim=-1,
        )
        encoded = self.attribute_projection(encoded) * present.unsqueeze(-1)
        pooled = encoded.sum(dim=1) / present.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.output_projection(pooled)
