"""Lightweight task-level confidence used to route MoE experts before execution.

The existing prototype-head confidence estimators depend on an expert's output.
They are useful for calibrating an already computed ensemble, but cannot make an
expert inactive.  This module deliberately uses only cheap, expert-independent
episode statistics so every expert can declare confidence before its encoder is
run.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_DESCRIPTOR_DIM = 16


def _safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _signed_log1p(value: float) -> float:
    value = _safe_float(value)
    return math.copysign(math.log1p(abs(value)), value)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(correction=0))


def _unpack_item(item):
    if isinstance(item, (tuple, list)):
        sequence = item[0] if item else []
        label = item[1] if len(item) > 1 else None
    else:
        sequence, label = item, None
    return sequence or [], label


def build_task_descriptor(
    support_set: Iterable,
    query_set: Iterable | None,
    task_type: str,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Summarize one classification/regression task without running an expert.

    Query labels are intentionally ignored.  Support labels may be used because
    they are available at inference.  The feature ranges are softly bounded so
    the same router can handle tiny episodes and very large process logs.
    """

    task_type = str(task_type).lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unknown task type: {task_type}")
    support = list(support_set or [])
    query = list(query_set or [])
    unpacked_support = [_unpack_item(item) for item in support]
    unpacked_query = [_unpack_item(item) for item in query]
    sequences = [sequence for sequence, _ in unpacked_support + unpacked_query]

    lengths = [math.log1p(len(sequence)) for sequence in sequences]
    length_mean, length_std = _mean_std(lengths)
    length_max = max(lengths, default=0.0)

    elapsed, previous, costs = [], [], []
    for sequence in sequences:
        if not sequence:
            elapsed.append(0.0)
            previous.append(0.0)
            costs.append(0.0)
            continue
        event = sequence[-1]
        if hasattr(event, "get"):
            elapsed.append(_signed_log1p(event.get("time_from_start", 0.0)))
            previous.append(_signed_log1p(event.get("time_from_previous", 0.0)))
            costs.append(_signed_log1p(event.get("cost", 0.0)))
        else:
            elapsed.append(0.0)
            previous.append(0.0)
            costs.append(0.0)
    elapsed_mean, elapsed_std = _mean_std(elapsed)
    previous_mean, previous_std = _mean_std(previous)
    cost_mean, _ = _mean_std(costs)

    support_labels = [label for _, label in unpacked_support if label is not None]
    if task_type == "classification":
        counts = Counter(int(label) for label in support_labels if int(label) != -100)
        total = sum(counts.values())
        if total:
            probabilities = [count / total for count in counts.values()]
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probabilities)
            entropy /= math.log(max(len(counts), 2))
            dominant = max(probabilities)
        else:
            entropy, dominant = 0.0, 0.0
        label_center = entropy
        label_spread = math.tanh(math.log1p(len(counts)) / 4.0)
        label_shape = dominant
    else:
        transformed = [_signed_log1p(label) for label in support_labels]
        target_mean, target_std = _mean_std(transformed)
        label_center = math.tanh(target_mean / 10.0)
        label_spread = math.tanh(target_std / 5.0)
        raw_targets = [_safe_float(label) for label in support_labels]
        raw_mean, raw_std = _mean_std(raw_targets)
        label_shape = math.tanh(raw_std / (abs(raw_mean) + 1e-6)) if raw_targets else 0.0

    descriptor = [
        1.0 if task_type == "classification" else 0.0,
        1.0 if task_type == "regression" else 0.0,
        math.tanh(math.log1p(len(support)) / 5.0),
        math.tanh(math.log1p(len(query)) / 5.0),
        math.tanh(math.log((len(support) + 1.0) / (len(query) + 1.0))),
        math.tanh(length_mean / 4.0),
        math.tanh(length_std / 2.0),
        math.tanh(length_max / 8.0),
        math.tanh(elapsed_mean / 10.0),
        math.tanh(elapsed_std / 5.0),
        math.tanh(previous_mean / 10.0),
        math.tanh(previous_std / 5.0),
        label_center,
        label_spread,
        label_shape,
        math.tanh(cost_mean / 10.0),
    ]
    return torch.tensor(descriptor, device=device, dtype=dtype)


class TaskConfidenceHead(nn.Module):
    """Predict one comparable reliability logit for an expert and a task."""

    ARCHITECTURES = {"task_bias", "linear", "mlp"}

    def __init__(
        self,
        architecture: str = "mlp",
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.architecture = str(architecture).lower()
        if self.architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"Unknown expert routing architecture: {self.architecture}; "
                f"expected one of {sorted(self.ARCHITECTURES)}"
            )
        hidden_dim = max(2, int(hidden_dim))
        if self.architecture == "task_bias":
            self.network = nn.Embedding(2, 1)
            nn.init.zeros_(self.network.weight)
        elif self.architecture == "linear":
            self.network = nn.Linear(TASK_DESCRIPTOR_DIM, 1)
            nn.init.zeros_(self.network.weight)
            nn.init.zeros_(self.network.bias)
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(TASK_DESCRIPTOR_DIM),
                nn.Linear(TASK_DESCRIPTOR_DIM, hidden_dim),
                nn.GELU(),
                nn.Dropout(max(0.0, float(dropout))),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        single = descriptor.ndim == 1
        descriptor = descriptor.reshape(-1, TASK_DESCRIPTOR_DIM)
        if self.architecture == "task_bias":
            # Descriptor positions 0/1 are a classification/regression one-hot.
            task_index = descriptor[:, 1].round().long().clamp(0, 1)
            logits = self.network(task_index).squeeze(-1)
        else:
            logits = self.network(descriptor).squeeze(-1)
        return logits.squeeze(0) if single else logits

    def reliability_loss(self, descriptor: torch.Tensor, target) -> torch.Tensor:
        target_tensor = torch.as_tensor(
            target, device=descriptor.device, dtype=descriptor.dtype
        ).detach().clamp(0.0, 1.0)
        return F.binary_cross_entropy_with_logits(
            self(descriptor).reshape(-1), target_tensor.reshape(-1)
        )
