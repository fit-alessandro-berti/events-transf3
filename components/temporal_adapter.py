"""Learned multi-resolution encodings for event-prefix timing covariates.

The legacy two-column adapter is retained only so committed checkpoints remain
loadable. New experiments use :class:`IndependentTemporalInputEncoder`, whose
elapsed and inter-event clocks have no learned parameters in common.
"""

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


class LearnedScalarTimeEncoder(nn.Module):
    """Encode one scalar clock with its own transform bank and projector.

    ``config_prefix`` selects feature-specific configuration keys, for example
    ``temporal_start_num_transforms``. There is intentionally no shared power,
    scale, projection, or residual-gate parameter between scalar encoders.
    """

    def __init__(self, d_model, config_prefix, **config):
        super().__init__()
        self.config_prefix = str(config_prefix)
        self.num_transforms = max(
            1, int(config.get(f"{self.config_prefix}_num_transforms", 4))
        )
        self.power_min = max(
            float(config.get(f"{self.config_prefix}_power_min", 0.05)), 1e-3
        )
        self.power_max = max(
            float(config.get(f"{self.config_prefix}_power_max", 1.5)),
            self.power_min + 1e-3,
        )
        initial_powers = torch.linspace(
            self.power_min, self.power_max, self.num_transforms
        )
        fractions = (
            (initial_powers - self.power_min) / (self.power_max - self.power_min)
        ).clamp(0.02, 0.98)
        self.power_logits = nn.Parameter(torch.logit(fractions))

        default_scale_min = 1.0 / 3600.0
        default_scale_max = 10_000.0
        scale_low = max(
            float(
                config.get(
                    f"{self.config_prefix}_scale_min_hours", default_scale_min
                )
            ),
            1e-6,
        )
        scale_high = max(
            float(
                config.get(
                    f"{self.config_prefix}_scale_max_hours", default_scale_max
                )
            ),
            scale_low,
        )
        initial_scales = torch.logspace(
            math.log10(scale_low), math.log10(scale_high), self.num_transforms
        )
        self.log_scales = nn.Parameter(initial_scales.log())

        hidden = max(16, min(64, self.num_transforms * 8))
        # Bias-free layers make a zero-duration input an exact zero residual.
        # This is useful for padding and avoids learning an event-independent
        # offset through either temporal component.
        self.projector = nn.Sequential(
            nn.Linear(self.num_transforms, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
            nn.LayerNorm(d_model, elementwise_affine=False),
        )
        self.residual_logit = nn.Parameter(
            torch.tensor(
                float(config.get(f"{self.config_prefix}_residual_logit", -3.0))
            )
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
        seconds = raw_seconds.float().reshape(-1).clamp_min(0.0)
        hours = seconds / 3600.0
        if augmentation_factor is not None:
            hours = hours * augmentation_factor
        normalized = hours.unsqueeze(-1) / self.scales.unsqueeze(0)
        powers = self.powers.unsqueeze(0)
        values = torch.expm1(powers * torch.log1p(normalized)) / powers
        return values / (1.0 + values)

    def forward(self, raw_seconds, augmentation_factor=None):
        features = self.transformed_features(raw_seconds, augmentation_factor)
        return torch.sigmoid(self.residual_logit) * self.projector(features)


class IndependentTemporalInputEncoder(nn.Module):
    """Fuse two fully independent clock encoders by residual addition.

    Column 0 is elapsed time from case start. Column 1 is time since the
    previous event. The two submodules intentionally expose different names
    and configuration namespaces so checkpoint inspection can prove that the
    learned components are parameter-disjoint.
    """

    def __init__(self, d_model, **config):
        super().__init__()
        self.start_time_encoder = LearnedScalarTimeEncoder(
            d_model, "temporal_start", **config
        )
        self.previous_time_encoder = LearnedScalarTimeEncoder(
            d_model, "temporal_previous", **config
        )

    def transformed_features(self, raw_seconds, augmentation_factor=None):
        return {
            "time_from_start": self.start_time_encoder.transformed_features(
                raw_seconds[:, 0], augmentation_factor
            ),
            "time_from_previous": self.previous_time_encoder.transformed_features(
                raw_seconds[:, 1], augmentation_factor
            ),
        }

    def forward(self, raw_seconds, augmentation_factor=None):
        start = self.start_time_encoder(raw_seconds[:, 0], augmentation_factor)
        previous = self.previous_time_encoder(
            raw_seconds[:, 1], augmentation_factor
        )
        return start + previous
