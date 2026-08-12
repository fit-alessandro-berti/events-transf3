"""Learned multi-resolution encodings for event-prefix timing covariates."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedTemporalInputAdapter(nn.Module):
    """Encode elapsed/inter-event time with several learned monotone maps.

    Input columns are seconds since case start and seconds since the previous
    event. They are converted to hours, transformed independently by K learned
    power/scale branches, bounded, and projected to an event-level residual.
    """

    def __init__(self, d_model, num_transforms=8, **config):
        super().__init__()
        self.num_features = 2
        self.num_transforms = max(1, int(num_transforms))
        self.power_min = max(float(config.get("regression_power_min", 0.05)), 1e-3)
        self.power_max = max(
            float(config.get("regression_power_max", 1.5)), self.power_min + 1e-3
        )
        initial_powers = torch.linspace(
            self.power_min, self.power_max, self.num_transforms
        )
        fractions = (
            (initial_powers - self.power_min) / (self.power_max - self.power_min)
        ).clamp(0.02, 0.98)
        initial_power_logits = torch.logit(fractions).repeat(self.num_features, 1)
        self.power_logits = nn.Parameter(initial_power_logits)

        scale_low = max(
            float(config.get("regression_input_scale_min_hours", 1.0 / 3600.0)),
            1e-6,
        )
        scale_high = max(
            float(config.get("regression_input_scale_max_hours", 10000.0)), scale_low
        )
        scales = torch.logspace(
            math.log10(scale_low), math.log10(scale_high), self.num_transforms
        ).repeat(self.num_features, 1)
        self.log_scales = nn.Parameter(scales.log())
        width = self.num_features * self.num_transforms
        self.projector = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, max(32, min(128, width * 4))),
            nn.GELU(),
            nn.Linear(max(32, min(128, width * 4)), d_model),
            nn.LayerNorm(d_model),
        )
        self.residual_logit = nn.Parameter(
            torch.tensor(float(config.get("regression_input_residual_logit", -2.0)))
        )

    @property
    def powers(self):
        return self.power_min + (self.power_max - self.power_min) * torch.sigmoid(
            self.power_logits
        )

    @property
    def scales(self):
        return self.log_scales.clamp(math.log(1e-6), math.log(1e8)).exp()

    def transformed_features(self, raw_seconds, augmentation_factor=None):
        hours = raw_seconds.float().clamp_min(0.0) / 3600.0
        if augmentation_factor is not None:
            hours = hours * augmentation_factor
        normalized = hours.unsqueeze(-1) / self.scales.unsqueeze(0)
        powers = self.powers.unsqueeze(0)
        values = torch.expm1(powers * torch.log1p(normalized)) / powers
        # A monotone rational bound avoids exploding input magnitude while
        # preserving branch order and learned characteristic time scales.
        return values / (1.0 + values)

    def forward(self, raw_seconds, augmentation_factor=None):
        features = self.transformed_features(raw_seconds, augmentation_factor)
        residual = self.projector(features.flatten(start_dim=1))
        return torch.sigmoid(self.residual_logit) * residual
