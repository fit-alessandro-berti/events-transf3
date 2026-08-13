"""Configurable retrieval/prototype heads used by FM-v2 and FM-v3.

The legacy ``soft_knn`` path is intentionally retained so that a newly trained
FM-v2 checkpoint can be evaluated with exactly the old candidate restriction.
All FM-v3 modes separate local evidence, global label coverage, and the class
prior.  Labels are log-local IDs; no global semantic meaning is assumed.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class LearnedExampleSelector(nn.Module):
    """Bounded, zero-initialized log-weights for support-example trust.

    The selector consumes only interpretable scalar features computed from the
    query/support geometry and observed support labels. Its output is added to
    the head's existing similarity logits, so positive values retain an
    example and negative values suppress it. Zero initialization makes an
    enabled selector exactly reproduce the historical head before training.
    """

    def __init__(self, feature_dim, hidden_dim=16, max_log_weight=2.0):
        super().__init__()
        self.max_log_weight = max(float(max_log_weight), 0.0)
        self.network = nn.Sequential(
            nn.Linear(int(feature_dim), max(int(hidden_dim), 1)),
            nn.GELU(),
            nn.Linear(max(int(hidden_dim), 1), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        raw = self.network(features).squeeze(-1)
        return self.max_log_weight * torch.tanh(raw)


class LearnedTimeTransformBank(nn.Module):
    """Learned monotone target transforms for scale-robust time regression.

    Each branch uses a positive Box--Cox-like transform on raw hours, performs
    its own soft neighbor regression, inverts back to hours, and contributes to
    a learned convex aggregation.  Training-time log-uniform rescaling changes
    the numerical regime seen by the transforms without changing the returned
    unit: predictions are divided by the sampled factor before loss/evaluation.
    """

    def __init__(self, num_transforms=8, init_logit_scale=5.0, **config):
        super().__init__()
        self.num_transforms = max(1, int(num_transforms))
        self.power_min = max(float(config.get("regression_power_min", 0.05)), 1e-3)
        self.power_max = max(
            float(config.get("regression_power_max", 1.50)), self.power_min + 1e-3
        )
        initial_powers = torch.linspace(
            self.power_min, self.power_max, self.num_transforms
        )
        fractions = (
            (initial_powers - self.power_min) / (self.power_max - self.power_min)
        ).clamp(0.02, 1.0 - 0.02)
        self.power_logits = nn.Parameter(torch.logit(fractions))

        scale_low = max(float(config.get("regression_transform_scale_min_hours", 1.0)), 1e-4)
        scale_high = max(
            float(config.get("regression_transform_scale_max_hours", 10000.0)),
            scale_low,
        )
        initial_scales = torch.logspace(
            math.log10(scale_low), math.log10(scale_high), self.num_transforms
        )
        self.log_scales = nn.Parameter(initial_scales.log())
        self.branch_logit_scales = nn.Parameter(
            torch.full((self.num_transforms,), float(init_logit_scale))
        )
        self.aggregation_logits = nn.Parameter(torch.zeros(self.num_transforms))
        self.aggregation = str(config.get("regression_transform_aggregation", "learned"))
        self.dynamic_gate = None
        if self.aggregation == "dynamic":
            # Shared branch scorer: the same rule can be used for four or eight
            # transforms. Inputs are dimensionless to preserve scale transfer.
            self.dynamic_gate = nn.Sequential(
                nn.Linear(10, 32),
                nn.GELU(),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            nn.init.normal_(self.dynamic_gate[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.dynamic_gate[-1].bias)
        self.augmentation_enabled = _as_bool(
            config.get("regression_scale_augmentation", True)
        )
        self.augmentation_min = max(
            float(config.get("regression_scale_augmentation_min", 0.02)), 1e-6
        )
        self.augmentation_max = max(
            float(config.get("regression_scale_augmentation_max", 50.0)),
            self.augmentation_min,
        )

    @property
    def powers(self):
        return self.power_min + (self.power_max - self.power_min) * torch.sigmoid(
            self.power_logits
        )

    @property
    def scales(self):
        return self.log_scales.clamp(math.log(1e-4), math.log(1e8)).exp()

    @property
    def aggregation_weights(self):
        if self.aggregation == "mean":
            return torch.full_like(self.aggregation_logits, 1.0 / self.num_transforms)
        if self.aggregation not in {"learned", "dynamic"}:
            raise ValueError(f"Unknown time-transform aggregation: {self.aggregation}")
        return F.softmax(self.aggregation_logits, dim=0)

    def _dynamic_aggregation_weights(
        self, similarities, support_hours, attention, branch_predictions, raw_mean,
        raw_std,
    ):
        """Return query-specific convex weights with shape [branch, query]."""
        query_scale = support_hours.mean(dim=-1).clamp_min(1e-3)
        geometric_center = torch.exp(
            torch.log(branch_predictions.clamp_min(1e-6)).mean(dim=0)
        ).clamp_min(1e-6)
        entropy = -(
            attention * torch.log(attention.clamp_min(1e-8))
        ).sum(dim=-1) / math.log(max(attention.size(-1), 2))
        max_attention = attention.max(dim=-1).values
        branch_shape = (self.num_transforms, 1)
        power_position = (
            (self.powers - self.power_min) / (self.power_max - self.power_min)
        ).view(branch_shape).expand_as(branch_predictions)
        relative_transform_scale = torch.log(
            self.scales.view(branch_shape) / query_scale.unsqueeze(0)
        ).clamp(-12.0, 12.0) / 12.0
        similarity_max = similarities.max(dim=-1).values.unsqueeze(0).expand_as(
            branch_predictions
        )
        similarity_std = similarities.std(dim=-1, correction=0).unsqueeze(0).expand_as(
            branch_predictions
        )
        features = torch.stack(
            [
                torch.log1p(branch_predictions / query_scale.unsqueeze(0)),
                torch.log1p(raw_mean / query_scale.unsqueeze(0)),
                torch.log1p(raw_std / query_scale.unsqueeze(0)),
                entropy,
                max_attention,
                power_position,
                relative_transform_scale,
                torch.log(branch_predictions.clamp_min(1e-6) / geometric_center.unsqueeze(0)).clamp(-8.0, 8.0) / 8.0,
                similarity_max,
                similarity_std,
            ],
            dim=-1,
        )
        scores = self.dynamic_gate(features).squeeze(-1)
        scores = scores + self.aggregation_logits[:, None]
        return F.softmax(scores, dim=0)

    def sample_augmentation_factor(self, reference):
        if not self.training or not self.augmentation_enabled:
            return reference.new_tensor(1.0)
        log_low = math.log(self.augmentation_min)
        log_high = math.log(self.augmentation_max)
        return torch.exp(reference.new_empty(()).uniform_(log_low, log_high))

    def transform(self, hours):
        """Transform ``[..., support]`` raw hours into ``[branch, ...]``."""
        hours = hours.float().clamp_min(0.0)
        view_shape = (self.num_transforms,) + (1,) * hours.ndim
        powers = self.powers.view(view_shape)
        scales = self.scales.view(view_shape)
        normalized = hours.unsqueeze(0) / scales
        return torch.expm1(powers * torch.log1p(normalized)) / powers

    def inverse(self, values):
        """Invert a tensor whose leading dimension indexes transform branches."""
        view_shape = (self.num_transforms,) + (1,) * (values.ndim - 1)
        powers = self.powers.view(view_shape)
        scales = self.scales.view(view_shape)
        base = (1.0 + powers * values).clamp_min(1e-8)
        return scales * torch.expm1(torch.log(base) / powers)

    def predict(
        self,
        similarities,
        support_hours,
        augmentation_factor=None,
        selection_logits=None,
    ):
        """Predict raw hours from similarities and query-specific support labels.

        ``similarities`` is ``[query, support]``. ``support_hours`` may be a
        shared ``[support]`` vector or query-specific ``[query, support]``.
        """
        if support_hours.ndim == 1:
            support_hours = support_hours.unsqueeze(0).expand(similarities.size(0), -1)
        factor = (
            self.sample_augmentation_factor(support_hours)
            if augmentation_factor is None else augmentation_factor
        )
        augmented_hours = support_hours * factor
        transformed = self.transform(augmented_hours)  # [branch, query, support]
        branch_scales = self.branch_logit_scales.clamp(0.1, 100.0)
        attention_logits = similarities.unsqueeze(0) * branch_scales[:, None, None]
        if selection_logits is not None:
            attention_logits = attention_logits + selection_logits.unsqueeze(0)
        attention = F.softmax(attention_logits, dim=-1)
        transformed_predictions = (attention * transformed).sum(dim=-1)
        branch_predictions = self.inverse(transformed_predictions) / factor
        raw_mean = (attention * support_hours.unsqueeze(0)).sum(dim=-1)
        raw_variance = (
            attention
            * (support_hours.unsqueeze(0) - raw_mean.unsqueeze(-1)).square()
        ).sum(dim=-1)
        raw_std = torch.sqrt(raw_variance.clamp_min(1e-8))
        if self.aggregation == "dynamic":
            aggregation_weights = self._dynamic_aggregation_weights(
                similarities,
                support_hours,
                attention,
                branch_predictions,
                raw_mean,
                raw_std,
            )
        else:
            aggregation_weights = self.aggregation_weights[:, None].expand_as(
                branch_predictions
            )
        prediction = (aggregation_weights * branch_predictions).sum(dim=0)
        total_variance = (
            aggregation_weights
            * (raw_variance + (branch_predictions - prediction.unsqueeze(0)).square())
        ).sum(dim=0)
        diagnostics = {
            "branch_predictions_hours": branch_predictions,
            "branch_attention": attention,
            "aggregation_weights": aggregation_weights,
            "powers": self.powers,
            "scales_hours": self.scales,
            "augmentation_factor": factor,
            "std_hours": torch.sqrt(total_variance.clamp_min(1e-8)),
        }
        return prediction, diagnostics


class PrototypicalHead(nn.Module):
    """Classification and remaining-time heads with config-selected behavior."""

    def __init__(self, init_logit_scale: float = 5.0, **config):
        super().__init__()
        self.config = dict(config)
        self.classification_mode = str(config.get("classification_mode", "legacy_soft_knn"))
        self.local_temperature = max(float(config.get("local_temperature", 0.2)), 1e-4)
        self.global_temperature = max(float(config.get("global_temperature", 0.2)), 1e-4)
        self.learn_temperature = _as_bool(config.get("learn_temperature", False))
        self.local_centering = _as_bool(config.get("local_centering", False))
        self.global_centering = _as_bool(config.get("global_centering", False))
        self.coverage_fallback_margin = float(config.get("coverage_fallback_margin", 0.5))
        self.inference_temperature = max(float(config.get("inference_temperature", 1.0)), 1e-4)
        self.fallback_inference_temperature = max(
            float(config.get("fallback_inference_temperature", self.inference_temperature)), 1e-4
        )
        self._temperature_scale_reference = max(float(init_logit_scale), 1e-4)
        self.prior_mode = str(config.get("prior_mode", "none"))
        self.prior_smoothing = max(float(config.get("prior_smoothing", 1.0)), 0.0)
        self.prior_strength = float(config.get("prior_strength", 0.0))
        self.gamma_mode = str(config.get("count_normalization", "fixed"))
        gamma_init = float(config.get("count_normalization_gamma", 1.0))
        gamma_init = min(max(gamma_init / 1.5, 1e-4), 1.0 - 1e-4)
        self._gamma_raw = nn.Parameter(torch.tensor(math.log(gamma_init / (1.0 - gamma_init))))
        self._gamma_raw.requires_grad_(self.gamma_mode == "learned")

        self.shrinkage_mode = str(config.get("shrinkage_mode", "none"))
        kappa_init = max(float(config.get("shrinkage_kappa", 2.0)), 1e-4)
        self._kappa_raw = nn.Parameter(torch.tensor(math.log(math.expm1(kappa_init))))
        self._kappa_raw.requires_grad_(self.shrinkage_mode == "learned")

        self.gate_mode = str(config.get("gate_mode", "fixed"))
        gate_init = min(max(float(config.get("local_gate", 0.5)), 1e-4), 1.0 - 1e-4)
        self.register_buffer("fixed_gate", torch.tensor(gate_init))
        self.gate_network = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, math.log(gate_init / (1.0 - gate_init)))
        for parameter in self.gate_network.parameters():
            parameter.requires_grad_(self.gate_mode == "dynamic")

        self.enable_abstention = _as_bool(config.get("enable_abstention", False))
        self.abstain_label = int(config.get("abstain_label", -101))
        self.abstain_bias = nn.Parameter(torch.tensor(float(config.get("abstain_bias", 0.0))))
        self.abstain_slope = nn.Parameter(torch.tensor(float(config.get("abstain_slope", 2.0))))
        self.abstain_bias.requires_grad_(self.enable_abstention)
        self.abstain_slope.requires_grad_(self.enable_abstention)

        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.logit_scale.requires_grad_(self.learn_temperature)
        self.reg_logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        selector_hidden = max(int(config.get("example_selector_hidden_dim", 16)), 1)
        self.classification_example_selector_enabled = _as_bool(
            config.get("classification_example_selector_enabled", False)
        )
        self.regression_example_selector_enabled = _as_bool(
            config.get("regression_example_selector_enabled", False)
        )
        self.classification_example_selector_strength = max(
            float(config.get("classification_example_selector_strength", 1.0)), 0.0
        )
        self.regression_example_selector_strength = max(
            float(config.get("regression_example_selector_strength", 1.0)), 0.0
        )
        self.classification_example_selector = None
        self.regression_example_selector = None
        if self.classification_example_selector_enabled:
            self.classification_example_selector = LearnedExampleSelector(
                feature_dim=6,
                hidden_dim=int(
                    config.get("classification_example_selector_hidden_dim", selector_hidden)
                ),
                max_log_weight=float(
                    config.get("classification_example_selector_max_log_weight", 2.0)
                ),
            )
        if self.regression_example_selector_enabled:
            self.regression_example_selector = LearnedExampleSelector(
                feature_dim=7,
                hidden_dim=int(
                    config.get("regression_example_selector_hidden_dim", selector_hidden)
                ),
                max_log_weight=float(
                    config.get("regression_example_selector_max_log_weight", 2.0)
                ),
            )
        self.regression_mode = str(config.get("regression_mode", "sqrt_knn")).lower()
        self.regression_mode = {
            "raw_knn": "raw_hours_knn",
            "raw_prediction": "raw_hours_knn",
        }.get(self.regression_mode, self.regression_mode)
        self.time_transform_bank = None
        if self.regression_mode == "learned_transform_ensemble":
            self.time_transform_bank = LearnedTimeTransformBank(
                num_transforms=int(config.get("regression_num_transforms", 8)),
                init_logit_scale=init_logit_scale,
                **config,
            )
        self.regression_mae_weight = max(
            float(config.get("regression_mae_weight", 0.5)), 0.0
        )
        self.regression_rmse_weight = max(
            float(config.get("regression_rmse_weight", 0.5)), 0.0
        )
        # Selected complementary metrics in raw hours after unit conversion.
        # Historical two-term behavior remains available by setting these
        # four weights to zero explicitly.
        self.regression_huber_weight = max(
            float(config.get("regression_huber_weight", 0.15)), 0.0
        )
        self.regression_huber_delta = max(
            float(config.get("regression_huber_delta", 1.0)), 1e-4
        )
        self.regression_log_rmse_weight = max(
            float(config.get("regression_log_rmse_weight", 0.15)), 0.0
        )
        self.regression_relative_mae_weight = max(
            float(config.get("regression_relative_mae_weight", 0.05)), 0.0
        )
        self.regression_bias_weight = max(
            float(config.get("regression_bias_weight", 0.05)), 0.0
        )
        self.regression_median_ae_weight = max(
            float(config.get("regression_median_ae_weight", 0.0)), 0.0
        )
        self.regression_quantile_weight = max(
            float(config.get("regression_quantile_weight", 0.0)), 0.0
        )
        self.regression_quantile_level = min(
            max(float(config.get("regression_quantile_level", 0.5)), 1e-3), 1.0 - 1e-3
        )
        if self._regression_primary_weight_sum() <= 0:
            raise ValueError(
                "At least one regression primary metric weight must be positive "
                "(mae, rmse, huber, log_rmse, relative_mae, bias, "
                "median_ae, quantile)"
            )
        self.regression_loss_scale_power = min(
            max(float(config.get("regression_loss_scale_power", 1.0)), 0.0), 1.0
        )
        self.regression_loss_reference_hours = max(
            float(config.get("regression_loss_reference_hours", 100.0)), 1e-4
        )
        self.regression_gate_aux_weight = max(
            float(config.get("regression_gate_aux_weight", 0.0)), 0.0
        )
        self.regression_gate_target_temperature = max(
            float(config.get("regression_gate_target_temperature", 0.1)), 1e-4
        )
        confidence_hidden = max(int(config.get("expert_confidence_hidden_dim", 16)), 1)
        self.classification_expert_confidence_enabled = _as_bool(
            config.get("classification_expert_confidence_enabled", False)
        )
        self.regression_expert_confidence_enabled = _as_bool(
            config.get("regression_expert_confidence_enabled", False)
        )
        self.classification_expert_confidence = None
        self.regression_expert_confidence = None
        if self.classification_expert_confidence_enabled:
            self.classification_expert_confidence = nn.Sequential(
                nn.Linear(6, confidence_hidden),
                nn.GELU(),
                nn.Linear(confidence_hidden, 1),
            )
            nn.init.zeros_(self.classification_expert_confidence[-1].weight)
            nn.init.zeros_(self.classification_expert_confidence[-1].bias)
        if self.regression_expert_confidence_enabled:
            self.regression_expert_confidence = nn.Sequential(
                nn.Linear(6, confidence_hidden),
                nn.GELU(),
                nn.Linear(confidence_hidden, 1),
            )
            nn.init.zeros_(self.regression_expert_confidence[-1].weight)
            nn.init.zeros_(self.regression_expert_confidence[-1].bias)

        # Kept for strict compatibility with historical FM-v2 checkpoints.
        self._proto_shrink = nn.Parameter(torch.tensor(-2.0))
        self.count_prior = nn.Parameter(torch.tensor(0.0))

    @property
    def evidence_gamma(self) -> torch.Tensor:
        if self.gamma_mode == "learned":
            return 1.5 * torch.sigmoid(self._gamma_raw)
        return self._gamma_raw.new_tensor(float(self.config.get("count_normalization_gamma", 1.0)))

    @property
    def shrinkage_kappa(self) -> torch.Tensor:
        if self.shrinkage_mode == "learned":
            return F.softplus(self._kappa_raw).clamp_min(1e-4)
        return self._kappa_raw.new_tensor(max(float(self.config.get("shrinkage_kappa", 2.0)), 1e-4))

    def _center_and_renorm(self, support: torch.Tensor, query: torch.Tensor):
        mu = support.mean(dim=0, keepdim=True)
        return _l2_normalize(support - mu), _l2_normalize(query - mu)

    @property
    def local_scale(self) -> torch.Tensor:
        if self.learn_temperature:
            multiplier = self.logit_scale.clamp(1.0, 20.0) / self._temperature_scale_reference
            return multiplier / self.local_temperature
        return self.logit_scale.new_tensor(1.0 / self.local_temperature)

    @property
    def global_scale(self) -> torch.Tensor:
        if self.learn_temperature:
            multiplier = self.logit_scale.clamp(1.0, 20.0) / self._temperature_scale_reference
            return multiplier / self.global_temperature
        return self.logit_scale.new_tensor(1.0 / self.global_temperature)

    @staticmethod
    def _class_counts(labels: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        return torch.stack([(labels == cls).sum() for cls in classes]).float()

    @staticmethod
    def _expand_local_support(support, labels, query):
        """Normalize local inputs to query-specific ``[query, support, ...]``."""
        if support.ndim == 2:
            support = support.unsqueeze(0).expand(query.size(0), -1, -1)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0).expand(query.size(0), -1)
        if support.ndim != 3 or labels.ndim != 2:
            raise ValueError("Local support must be [support, dim] or [query, support, dim]")
        if support.shape[:2] != labels.shape or support.size(0) != query.size(0):
            raise ValueError("Support features, labels, and queries must have matching shapes")
        return support, labels

    def classification_selection_logits(
        self, support, labels, query, base_similarities=None, return_features=False
    ):
        """Score retrieved classification examples without using query labels.

        Features, in order, are raw cosine similarity, the head's centered
        cosine similarity, its within-neighborhood z-score, support centrality,
        leave-one-out same-class coherence, and normalized class support.
        """
        support, labels = self._expand_local_support(support, labels, query)
        if self.classification_example_selector is None:
            zeros = support.new_zeros(support.shape[:2])
            return (zeros, None) if return_features else zeros
        support = _l2_normalize(support)
        query = _l2_normalize(query)
        raw_similarities = torch.einsum("qd,qkd->qk", query, support)
        if base_similarities is None:
            base_similarities = raw_similarities
        relative_similarity = torch.tanh(
            (base_similarities - base_similarities.mean(dim=1, keepdim=True))
            / base_similarities.std(dim=1, keepdim=True, correction=0).clamp_min(1e-4)
        )
        neighborhood_center = _l2_normalize(support.mean(dim=1))
        centrality = torch.einsum("qkd,qd->qk", support, neighborhood_center)

        same_class = labels.unsqueeze(2).eq(labels.unsqueeze(1))
        eye = torch.eye(labels.size(1), device=labels.device, dtype=torch.bool)
        same_class = same_class & ~eye.unsqueeze(0)
        same_counts = same_class.sum(dim=2)
        same_sum = torch.einsum("qkj,qjd->qkd", same_class.float(), support)
        same_prototype = _l2_normalize(
            same_sum / same_counts.clamp_min(1).unsqueeze(2)
        )
        same_class_coherence = torch.einsum("qkd,qkd->qk", support, same_prototype)
        same_class_coherence = torch.where(
            same_counts > 0, same_class_coherence, torch.zeros_like(same_class_coherence)
        )
        class_counts = same_counts + 1
        normalized_class_support = torch.log1p(class_counts.float()) / math.log1p(
            max(labels.size(1), 1)
        )
        features = torch.stack(
            [
                raw_similarities,
                base_similarities,
                relative_similarity,
                centrality,
                same_class_coherence,
                normalized_class_support,
            ],
            dim=-1,
        )
        logits = self.classification_example_selector_strength * (
            self.classification_example_selector(features)
        )
        return (logits, features) if return_features else logits

    def regression_selection_logits(
        self, support, labels, query, base_similarities=None, return_features=False
    ):
        """Score retrieved regression examples from geometry and target agreement.

        The selector uses no query target. Its last three features describe how
        far each observed support target lies from the neighborhood median,
        whether its nearest support neighbor agrees, and on which side of the
        median it lies. Log targets and a robust MAD scale make these features
        comparable across logs measured in very different time units.
        """
        support, labels = self._expand_local_support(support, labels, query)
        if self.regression_example_selector is None:
            zeros = support.new_zeros(support.shape[:2])
            return (zeros, None) if return_features else zeros
        support = _l2_normalize(support)
        query = _l2_normalize(query)
        raw_similarities = torch.einsum("qd,qkd->qk", query, support)
        if base_similarities is None:
            base_similarities = raw_similarities
        relative_similarity = torch.tanh(
            (base_similarities - base_similarities.mean(dim=1, keepdim=True))
            / base_similarities.std(dim=1, keepdim=True, correction=0).clamp_min(1e-4)
        )
        neighborhood_center = _l2_normalize(support.mean(dim=1))
        centrality = torch.einsum("qkd,qd->qk", support, neighborhood_center)

        target_values = self.regression_labels_to_output(labels).clamp_min(0.0)
        log_targets = torch.log1p(target_values)
        target_median = log_targets.median(dim=1, keepdim=True).values
        mad = (log_targets - target_median).abs().median(dim=1, keepdim=True).values
        robust_scale = (1.4826 * mad).clamp_min(0.10)
        signed_target_position = torch.tanh(
            (log_targets - target_median) / robust_scale
        )
        target_deviation = signed_target_position.abs()

        if support.size(1) > 1:
            pair_similarity = torch.einsum("qkd,qjd->qkj", support, support)
            eye = torch.eye(
                support.size(1), device=support.device, dtype=torch.bool
            )
            pair_similarity = pair_similarity.masked_fill(eye.unsqueeze(0), -torch.inf)
            nearest = pair_similarity.argmax(dim=2)
            nearest_targets = torch.gather(log_targets, 1, nearest)
            nearest_target_disagreement = torch.tanh(
                (log_targets - nearest_targets).abs() / robust_scale
            )
        else:
            nearest_target_disagreement = torch.zeros_like(log_targets)
        features = torch.stack(
            [
                raw_similarities,
                base_similarities,
                relative_similarity,
                centrality,
                target_deviation,
                nearest_target_disagreement,
                signed_target_position,
            ],
            dim=-1,
        )
        logits = self.regression_example_selector_strength * (
            self.regression_example_selector(features)
        )
        return (logits, features) if return_features else logits

    @staticmethod
    def _selection_diagnostics(selection_logits, attention):
        trust = F.softmax(selection_logits, dim=-1)
        effective_count = trust.square().sum(dim=-1).clamp_min(1e-8).reciprocal()
        return {
            "selection_logits": selection_logits,
            "selection_trust": trust,
            "selection_effective_count": effective_count,
            "selection_attention": attention,
        }

    def _legacy_soft_knn(self, support, labels, query):
        unique_classes, inv = torch.unique(labels, sorted=True, return_inverse=True)
        raw_sims = query @ support.t()
        selection_logits = self.classification_selection_logits(
            support, labels, query, base_similarities=raw_sims
        )
        sims = raw_sims * self.logit_scale.clamp(1.0, 20.0) + selection_logits
        attn = F.softmax(sims, dim=1)
        mass = torch.zeros(query.size(0), unique_classes.size(0), device=query.device)
        mass.scatter_add_(1, inv.unsqueeze(0).expand(query.size(0), -1), attn)
        logits = torch.log(mass.clamp_min(1e-8))
        counts = torch.bincount(inv, minlength=unique_classes.numel()).float().clamp_min(1.0)
        logits = logits + self.count_prior * torch.log(counts).unsqueeze(0)
        diagnostics = {
            "local_counts": counts,
            "pool_counts": counts,
            "mode": "legacy_soft_knn",
            **self._selection_diagnostics(selection_logits, attn),
        }
        return logits, unique_classes, F.softmax(logits, dim=-1), diagnostics

    def _local_evidence(self, support, labels, query, classes):
        selector_support, selector_query = support, query
        if self.local_centering:
            support, query = self._center_and_renorm(support, query)
        raw_sims = query @ support.t()
        selection_logits = self.classification_selection_logits(
            selector_support,
            labels,
            selector_query,
            base_similarities=raw_sims,
        )
        sims = raw_sims * self.local_scale + selection_logits
        evidence = torch.full((query.size(0), classes.numel()), -1e4, device=query.device)
        counts = self._class_counts(labels, classes).to(query.device)
        gamma = self.evidence_gamma
        for idx, cls in enumerate(classes):
            mask = labels == cls
            if mask.any():
                evidence[:, idx] = torch.logsumexp(sims[:, mask], dim=1) - gamma * torch.log(counts[idx])
        return evidence, counts, sims, selection_logits

    def _global_evidence(self, support, labels, query, classes):
        counts = self._class_counts(labels, classes).to(query.device)
        task_prior = support.mean(dim=0)
        if self.global_centering:
            support = _l2_normalize(support - task_prior.unsqueeze(0))
            query = _l2_normalize(query - task_prior.unsqueeze(0))
            # A shared centroid prior becomes the zero vector after task
            # centering. Do not apply the old origin-dependent interpolation.
            task_prior = torch.zeros_like(task_prior)
        prototypes = []
        variances = []
        for idx, cls in enumerate(classes):
            members = support[labels == cls]
            if members.numel() == 0:
                prototypes.append(task_prior)
                variances.append(query.new_tensor(0.0))
                continue
            mean = members.mean(dim=0)
            if self.shrinkage_mode in {"fixed", "learned"} and not self.global_centering:
                weight = counts[idx] / (counts[idx] + self.shrinkage_kappa)
                mean = weight * mean + (1.0 - weight) * task_prior
            prototypes.append(mean)
            variances.append(((members - members.mean(dim=0)) ** 2).mean())
        prototypes = _l2_normalize(torch.stack(prototypes))
        evidence = (query @ prototypes.t()) * self.global_scale
        return evidence, counts, prototypes, torch.stack(variances)

    def _coverage_fallback_evidence(self, local, global_evidence, local_counts, pool_counts):
        """Keep the local decision and admit only confident missing candidates.

        Local logits retain their exact ordering.  A class missing from the
        retrieved neighbourhood can win only when its global prototype exceeds
        the best locally present class prototype by ``coverage_fallback_margin``.
        This turns global memory into a candidate-coverage fallback instead of
        allowing noisy prototypes to perturb every locally supported class.
        """
        local_present = local_counts > 0
        pool_present = pool_counts > 0
        valid_local = local.masked_fill(~local_present.unsqueeze(0), -torch.inf)
        best_local = valid_local.max(dim=1, keepdim=True).values
        best_present_global = global_evidence.masked_fill(
            ~local_present.unsqueeze(0), -torch.inf
        ).max(dim=1, keepdim=True).values
        missing_logits = (
            best_local
            + global_evidence
            - best_present_global
            - self.coverage_fallback_margin
        )
        missing_candidates = (~local_present & pool_present).unsqueeze(0)
        return torch.where(missing_candidates, missing_logits, valid_local)

    def _prior_logits(self, counts: torch.Tensor, num_queries: int, mode: Optional[str] = None, strength: Optional[float] = None):
        selected = str(mode or self.prior_mode)
        beta = self.prior_strength if strength is None else float(strength)
        if selected in {"none", "uniform", "balanced"} or beta == 0.0:
            return counts.new_zeros((num_queries, counts.numel()))
        if selected not in {"natural", "empirical"}:
            raise ValueError(f"Unknown prior mode: {selected}")
        smooth = self.prior_smoothing
        probs = (counts + smooth) / (counts.sum() + smooth * counts.numel()).clamp_min(1e-8)
        return beta * torch.log(probs.clamp_min(1e-8)).unsqueeze(0).expand(num_queries, -1)

    def _dynamic_gate(self, local, global_evidence, local_counts, pool_counts, local_sims):
        local_present = local_counts > 0
        max_local = torch.where(local > -1e3, local, torch.zeros_like(local))
        if local_sims.numel():
            retrieval_entropy = -(F.softmax(local_sims, dim=1) * F.log_softmax(local_sims, dim=1)).sum(dim=1)
            retrieval_entropy = retrieval_entropy / math.log(max(local_sims.size(1), 2))
        else:
            retrieval_entropy = local.new_ones(local.size(0))
        agreement = torch.tanh(max_local - global_evidence)
        features = torch.stack([
            torch.tanh(max_local),
            torch.log1p(local_counts).unsqueeze(0).expand(local.size(0), -1),
            torch.log1p(pool_counts).unsqueeze(0).expand(local.size(0), -1),
            retrieval_entropy.unsqueeze(1).expand(-1, local.size(1)),
            agreement,
        ], dim=-1)
        gate = torch.sigmoid(self.gate_network(features).squeeze(-1))
        return torch.where(local_present.unsqueeze(0), gate, torch.zeros_like(gate))

    def forward_classification(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        mode: Optional[str] = None,
        *,
        global_support_features: Optional[torch.Tensor] = None,
        global_support_labels: Optional[torch.Tensor] = None,
        candidate_classes: Optional[torch.Tensor] = None,
        prior_counts: Optional[torch.Tensor] = None,
        prior_mode: Optional[str] = None,
        prior_strength: Optional[float] = None,
        return_diagnostics: bool = False,
    ):
        if support_features.numel() == 0 or query_features.numel() == 0:
            empty = (None, None, None, {}) if return_diagnostics else (None, None, None)
            return empty

        selected_mode = str(mode or self.classification_mode).lower()
        aliases = {"soft_knn": "legacy_soft_knn", "proto": "global", "global_proto": "global"}
        selected_mode = aliases.get(selected_mode, selected_mode)

        local_support = _l2_normalize(support_features)
        query = _l2_normalize(query_features)
        if selected_mode == "legacy_soft_knn":
            local_support, local_query = self._center_and_renorm(local_support, query)
            result = self._legacy_soft_knn(local_support, support_labels, local_query)
            return result if return_diagnostics else result[:3]

        pool_features = global_support_features if global_support_features is not None else support_features
        pool_labels = global_support_labels if global_support_labels is not None else support_labels
        pool_features = _l2_normalize(pool_features)
        if candidate_classes is None:
            classes = torch.unique(
                pool_labels
                if selected_mode in {"global", "global_local", "coverage_fallback"}
                else support_labels,
                sorted=True,
            )
        else:
            classes = torch.unique(candidate_classes.to(support_labels.device), sorted=True)
        classes = classes[classes != self.abstain_label]
        if classes.numel() == 0:
            return (None, None, None, {}) if return_diagnostics else (None, None, None)

        local, local_counts, local_sims, selection_logits = self._local_evidence(
            local_support, support_labels, query, classes
        )
        global_evidence, pool_counts, prototypes, prototype_variances = self._global_evidence(
            pool_features, pool_labels, query, classes
        )
        global_evidence = global_evidence.masked_fill(pool_counts.unsqueeze(0) <= 0, -torch.inf)
        if prior_counts is not None:
            pool_counts = prior_counts.to(query.device).float()

        if selected_mode == "local":
            evidence = local
            gate = torch.ones_like(local)
        elif selected_mode == "global":
            evidence = global_evidence
            gate = torch.zeros_like(local)
        elif selected_mode == "global_local":
            if self.gate_mode == "dynamic":
                gate = self._dynamic_gate(local, global_evidence, local_counts, pool_counts, local_sims)
            else:
                gate = torch.full_like(local, float(self.fixed_gate.item()))
                gate = torch.where(local_counts.unsqueeze(0) > 0, gate, torch.zeros_like(gate))
            evidence = torch.logaddexp(
                local + torch.log(gate.clamp_min(1e-8)),
                global_evidence + torch.log1p(-gate.clamp(max=1.0 - 1e-8)),
            )
        elif selected_mode == "coverage_fallback":
            evidence = self._coverage_fallback_evidence(
                local, global_evidence, local_counts, pool_counts
            )
            gate = (local_counts > 0).unsqueeze(0).expand_as(local).to(local.dtype)
        else:
            raise ValueError(f"Unknown classification mode: {selected_mode}")

        logits = evidence + self._prior_logits(pool_counts, query.size(0), prior_mode, prior_strength)
        if self.enable_abstention:
            best_evidence = evidence.max(dim=1).values
            abstain_logits = self.abstain_bias - F.softplus(self.abstain_slope) * best_evidence
            logits = torch.cat([logits, abstain_logits.unsqueeze(1)], dim=1)
            classes = torch.cat([classes, classes.new_tensor([self.abstain_label])])

        output_logits = logits
        if not self.training:
            temperature = logits.new_full((query.size(0), 1), self.inference_temperature)
            if selected_mode == "coverage_fallback":
                has_missing_candidate = ((local_counts <= 0) & (pool_counts > 0)).any()
                if bool(has_missing_candidate):
                    temperature.fill_(self.fallback_inference_temperature)
            output_logits = logits / temperature
        confidence = F.softmax(output_logits, dim=-1)
        diagnostics: Dict[str, torch.Tensor | str] = {
            "mode": selected_mode,
            "local_counts": local_counts,
            "pool_counts": pool_counts,
            "local_evidence": local,
            "global_evidence": global_evidence,
            "gate": gate,
            "prototypes": prototypes,
            "prototype_variances": prototype_variances,
            **self._selection_diagnostics(
                selection_logits, F.softmax(local_sims, dim=1)
            ),
        }
        result = (output_logits, classes, confidence, diagnostics)
        return result if return_diagnostics else result[:3]

    def classification_expert_confidence_logit(self, probabilities):
        """Return a learned per-query expert log weight for class aggregation."""
        if self.classification_expert_confidence is None:
            return probabilities.new_zeros(probabilities.size(0))
        probs = probabilities.float().detach().clamp_min(1e-8)
        num_classes = max(int(probs.size(-1)), 1)
        top_values = torch.topk(probs, min(2, num_classes), dim=-1).values
        top1 = top_values[:, 0]
        top2 = top_values[:, 1] if num_classes > 1 else top1.new_zeros(top1.shape)
        margin = top1 - top2
        entropy = -(probs * torch.log(probs)).sum(dim=-1) / math.log(max(num_classes, 2))
        class_scale = top1.new_full(
            top1.shape, math.log1p(float(num_classes)) / math.log(128.0)
        ).clamp(0.0, 1.0)
        features = torch.stack(
            [
                top1,
                margin,
                entropy,
                1.0 - entropy,
                class_scale,
                top1 * margin,
            ],
            dim=-1,
        )
        return self.classification_expert_confidence(features).squeeze(-1)

    def classification_expert_confidence_loss(self, probabilities, labels):
        if self.classification_expert_confidence is None:
            return probabilities.new_tensor(0.0)
        valid = labels != -100
        if not valid.any():
            return probabilities.new_tensor(0.0)
        selected = probabilities[valid]
        targets = labels[valid]
        logit = self.classification_expert_confidence_logit(selected)
        correctness = (selected.argmax(dim=-1) == targets).float()
        return F.binary_cross_entropy_with_logits(logit, correctness)

    @property
    def regression_outputs_hours(self):
        return self.regression_mode in {"learned_transform_ensemble", "raw_hours_knn"}

    @property
    def regression_uses_time_transform_bank(self):
        return self.time_transform_bank is not None

    def regression_labels_to_output(self, labels):
        labels = labels.float()
        if self.regression_outputs_hours:
            return labels.clamp_min(0.0).square()
        return labels

    def _regression_primary_weight_sum(self) -> float:
        return (
            self.regression_mae_weight
            + self.regression_rmse_weight
            + self.regression_huber_weight
            + self.regression_log_rmse_weight
            + self.regression_relative_mae_weight
            + self.regression_bias_weight
            + self.regression_median_ae_weight
            + self.regression_quantile_weight
        )

    def regression_loss_components(
        self,
        predictions,
        labels,
        labels_in_output_space=False,
    ):
        """Return scale-normalized primary metric terms used by ``regression_loss``.

        All terms are finite scalars in a dimensionless (scale-normalized) space
        except ``log_rmse``, which is computed in ``log1p(hours)`` residual
        space. Callers that need the blended training objective should use
        :meth:`regression_loss` so gate-aux wiring stays consistent.
        """
        targets = (
            labels.float()
            if labels_in_output_space
            else self.regression_labels_to_output(labels)
        )
        predictions_f = predictions.float().reshape(-1)
        targets_f = targets.float().reshape(-1)
        errors = predictions_f - targets_f
        batch_scale = targets_f.detach().median().clamp_min(1.0)
        power = self.regression_loss_scale_power
        normalizer = batch_scale.pow(power) * (
            self.regression_loss_reference_hours ** (1.0 - power)
        )
        normalized = errors / normalizer
        abs_norm = normalized.abs()
        mae = abs_norm.mean()
        rmse = torch.sqrt(normalized.square().mean() + 1e-8)
        # Huber on normalized residuals: quadratic near zero (RMSE-like),
        # linear in the tails (MAE-like).
        delta = self.regression_huber_delta
        huber = torch.where(
            abs_norm <= delta,
            0.5 * normalized.square() / delta,
            abs_norm - 0.5 * delta,
        ).mean()
        # Multi-scale tail pressure without raw-hour domination.
        log_errors = torch.log1p(predictions_f.clamp_min(0.0)) - torch.log1p(
            targets_f.clamp_min(0.0)
        )
        log_rmse = torch.sqrt(log_errors.square().mean() + 1e-8)
        relative_mae = (
            errors.abs() / targets_f.detach().abs().clamp_min(1.0)
        ).mean()
        bias = errors.mean().abs() / normalizer
        median_ae = abs_norm.quantile(0.5)
        # Pinball / quantile residual (level 0.5 is proportional to MAE).
        level = self.regression_quantile_level
        quantile = torch.where(
            errors >= 0,
            level * errors,
            (level - 1.0) * errors,
        ).mean() / normalizer
        return {
            "mae": mae,
            "rmse": rmse,
            "huber": huber,
            "log_rmse": log_rmse,
            "relative_mae": relative_mae,
            "bias": bias,
            "median_ae": median_ae,
            "quantile": quantile,
            "normalizer": normalizer,
            "errors": errors,
            "targets": targets_f,
            "predictions": predictions_f,
        }

    def regression_gate_auxiliary_loss(
        self, branch_predictions, aggregation_weights, targets
    ):
        """Teach the dynamic gate which transform branch fits each query.

        Raw-hour branch errors define a detached soft target. Detaching keeps
        this auxiliary objective from moving branch predictions merely to make
        branch selection easier; the primary MAE/RMSE loss remains authoritative
        for the transform branches. Softmax temperature is
        ``mean_branch_error * regression_gate_target_temperature`` with a tiny
        floor (``1e-4`` h) so short-horizon queries stay peaked.
        """
        branch_predictions = branch_predictions.float()
        aggregation_weights = aggregation_weights.float()
        targets = targets.float().reshape(1, -1)
        if branch_predictions.shape != aggregation_weights.shape:
            raise ValueError(
                "branch_predictions and aggregation_weights must have the same shape"
            )
        if branch_predictions.ndim != 2 or branch_predictions.size(1) != targets.size(1):
            raise ValueError(
                "Branch diagnostics must have shape [branch, query] matching targets"
            )
        branch_errors = (branch_predictions.detach() - targets).abs()
        # Scale temperature by mean absolute branch error so short-horizon
        # queries (sub-hour remaining times) still get peaked soft targets.
        # A 1-hour floor made helpdesk-scale errors nearly uniform.
        error_scale = branch_errors.mean(dim=0, keepdim=True).clamp_min(1e-4)
        target_weights = F.softmax(
            -branch_errors
            / (error_scale * self.regression_gate_target_temperature),
            dim=0,
        )
        return -(
            target_weights
            * torch.log(aggregation_weights.clamp_min(1e-8))
        ).sum(dim=0).mean()

    def regression_expert_confidence_logit(self, predictions, confidence, diagnostics):
        """Return a learned per-query expert log weight for time aggregation."""
        if self.regression_expert_confidence is None:
            return predictions.new_zeros(predictions.reshape(-1).shape)
        predictions = predictions.float().reshape(-1).detach().clamp_min(0.0)
        confidence = confidence.float().reshape(-1).detach().clamp(0.0, 1.0)
        std = diagnostics.get("std_hours", diagnostics.get("std"))
        if std is None:
            std = torch.zeros_like(predictions)
        else:
            std = std.float().reshape(-1).detach().clamp_min(0.0)
        similarities = diagnostics.get("similarities")
        if similarities is None or similarities.numel() == 0:
            max_similarity = torch.zeros_like(predictions)
            similarity_std = torch.zeros_like(predictions)
            entropy = torch.ones_like(predictions)
        else:
            sims = similarities.float().detach()
            max_similarity = sims.max(dim=1).values.clamp(-1.0, 1.0)
            similarity_std = sims.std(dim=1, correction=0).clamp_min(0.0)
            weights = F.softmax(sims, dim=1)
            entropy = -(
                weights * torch.log(weights.clamp_min(1e-8))
            ).sum(dim=1) / math.log(max(sims.size(1), 2))
        features = torch.stack(
            [
                torch.log1p(predictions).clamp(0.0, 20.0) / 20.0,
                torch.log1p(std).clamp(0.0, 20.0) / 20.0,
                confidence,
                max_similarity,
                entropy,
                torch.tanh(similarity_std),
            ],
            dim=-1,
        )
        return self.regression_expert_confidence(features).squeeze(-1)

    def regression_expert_confidence_loss(
        self,
        predictions,
        labels,
        confidence,
        diagnostics,
        labels_in_output_space=False,
    ):
        if self.regression_expert_confidence is None:
            return predictions.new_tensor(0.0)
        targets = (
            labels.float()
            if labels_in_output_space
            else self.regression_labels_to_output(labels)
        ).reshape(-1)
        predictions_f = predictions.float().reshape(-1)
        logit = self.regression_expert_confidence_logit(
            predictions_f, confidence, diagnostics
        )
        scale = targets.detach().abs().clamp_min(1.0)
        reliability = torch.exp(
            -((predictions_f.detach() - targets.detach()).abs() / scale).clamp(0.0, 20.0)
        )
        return F.binary_cross_entropy_with_logits(logit, reliability)

    def regression_loss(
        self,
        predictions,
        labels,
        labels_in_output_space=False,
        branch_predictions=None,
        aggregation_weights=None,
    ):
        """Optimize a multi-metric raw-hour objective and optional gate selection.

        Primary terms (config-weighted, then renormalized) after unit conversion
        and batch-median scale normalization:

        * ``mae`` / ``rmse`` — explicit absolute and squared-error pressure
        * ``huber`` — smooth bridge (quadratic near zero, linear tails)
        * ``log_rmse`` — multi-scale tail pressure in ``log1p(hours)``
        * ``relative_mae`` — per-query absolute error relative to target hours
        * ``bias`` — absolute mean residual (systematic shift control)
        * ``median_ae`` — direct typical-case absolute-error pressure
        * ``quantile`` — pinball residual (optional; level defaults to 0.5)

        Label units:
        - ``labels_in_output_space=False`` (default): ``labels`` are the stored
          task values (``sqrt(hours)`` for remaining time). They are converted
          with :meth:`regression_labels_to_output` before the metric.
        - ``labels_in_output_space=True``: ``labels`` are already in the head's
          output unit (raw hours for raw-hour regression modes).

        Predictions must always be in the head's output unit. Under AMP the
        primary metric is computed in float32 for stable RMSE/median scaling.
        """
        targets = labels.float() if labels_in_output_space else self.regression_labels_to_output(labels)
        if not self.regression_outputs_hours:
            return F.huber_loss(predictions.squeeze(), targets.squeeze())
        components = self.regression_loss_components(
            predictions, labels, labels_in_output_space=labels_in_output_space
        )
        weighted = (
            self.regression_mae_weight * components["mae"]
            + self.regression_rmse_weight * components["rmse"]
            + self.regression_huber_weight * components["huber"]
            + self.regression_log_rmse_weight * components["log_rmse"]
            + self.regression_relative_mae_weight * components["relative_mae"]
            + self.regression_bias_weight * components["bias"]
            + self.regression_median_ae_weight * components["median_ae"]
            + self.regression_quantile_weight * components["quantile"]
        )
        denominator = self._regression_primary_weight_sum()
        loss = weighted / denominator
        has_branch_predictions = branch_predictions is not None
        has_aggregation_weights = aggregation_weights is not None
        if has_branch_predictions != has_aggregation_weights:
            raise ValueError(
                "branch_predictions and aggregation_weights must be provided together"
            )
        if has_branch_predictions and self.regression_gate_aux_weight > 0:
            loss = loss + self.regression_gate_aux_weight * (
                self.regression_gate_auxiliary_loss(
                    branch_predictions,
                    aggregation_weights,
                    components["targets"],
                )
            )
        return loss

    def _regression_from_local(
        self, local_support, support_labels, query, augmentation_factor=None
    ):
        """Shared implementation for common and query-specific neighborhoods."""
        if local_support.ndim == 2:
            local_support = local_support.unsqueeze(0).expand(query.size(0), -1, -1)
        support = _l2_normalize(local_support)
        query = _l2_normalize(query)
        center = support.mean(dim=1, keepdim=True)
        centered_support = _l2_normalize(support - center)
        centered_query = _l2_normalize(query - center.squeeze(1))
        similarities = torch.einsum("qd,qkd->qk", centered_query, centered_support)
        selection_logits = self.regression_selection_logits(
            support,
            support_labels,
            query,
            base_similarities=similarities,
        )
        if self.regression_uses_time_transform_bank:
            support_hours = self.regression_labels_to_output(support_labels)
            prediction, diagnostics = self.time_transform_bank.predict(
                similarities,
                support_hours,
                augmentation_factor=augmentation_factor,
                selection_logits=selection_logits,
            )
            std = diagnostics["std_hours"]
            selection_attention = diagnostics["branch_attention"].mean(dim=0)
        elif self.regression_outputs_hours:
            scale = self.reg_logit_scale.clamp(1.0, 100.0)
            weights = F.softmax(similarities * scale + selection_logits, dim=1)
            targets = self.regression_labels_to_output(support_labels)
            if targets.ndim == 1:
                targets = targets.unsqueeze(0).expand(query.size(0), -1)
            prediction = (weights * targets).sum(dim=1)
            variance = (weights * (targets - prediction.unsqueeze(1)).square()).sum(dim=1)
            std = torch.sqrt(variance + 1e-8)
            diagnostics = {"attention": weights, "std_hours": std}
            selection_attention = weights
        else:
            scale = self.reg_logit_scale.clamp(1.0, 100.0)
            weights = F.softmax(similarities * scale + selection_logits, dim=1)
            targets = support_labels.float()
            if targets.ndim == 1:
                targets = targets.unsqueeze(0).expand(query.size(0), -1)
            prediction = (weights * targets).sum(dim=1)
            variance = (weights * (targets - prediction.unsqueeze(1)).square()).sum(dim=1)
            std = torch.sqrt(variance + 1e-8)
            diagnostics = {"attention": weights, "std": std}
            selection_attention = weights
        confidence = (1.0 / (1.0 + std)) * (
            (similarities.max(dim=1).values + 1.0) / 2.0
        ).clamp(0.0, 1.0)
        diagnostics["similarities"] = similarities
        diagnostics.update(
            self._selection_diagnostics(selection_logits, selection_attention)
        )
        return prediction, confidence.clamp(0.0, 1.0), diagnostics

    def forward_regression(
        self, support_features, support_labels, query_features, return_diagnostics=False,
        augmentation_factor=None,
    ):
        if support_features.numel() == 0 or query_features.numel() == 0:
            device = query_features.device
            result = (
                torch.zeros(query_features.size(0), device=device),
                torch.zeros(query_features.size(0), device=device),
                {},
            )
        else:
            result = self._regression_from_local(
                support_features, support_labels, query_features,
                augmentation_factor=augmentation_factor,
            )
        return result if return_diagnostics else result[:2]

    def forward_regression_batched(
        self, local_support_features, local_support_labels, query_features,
        return_diagnostics=False, augmentation_factor=None,
    ):
        result = self._regression_from_local(
            local_support_features, local_support_labels, query_features,
            augmentation_factor=augmentation_factor,
        )
        return result if return_diagnostics else result[:2]
