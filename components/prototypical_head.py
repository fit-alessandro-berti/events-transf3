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

    def predict(self, similarities, support_hours, augmentation_factor=None):
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
        attention = F.softmax(
            similarities.unsqueeze(0) * branch_scales[:, None, None], dim=-1
        )
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
        self.regression_mode = str(config.get("regression_mode", "sqrt_knn"))
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
        if self.regression_mae_weight + self.regression_rmse_weight <= 0:
            raise ValueError(
                "At least one of regression_mae_weight and "
                "regression_rmse_weight must be positive"
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

    def _legacy_soft_knn(self, support, labels, query):
        unique_classes, inv = torch.unique(labels, sorted=True, return_inverse=True)
        sims = (query @ support.t()) * self.logit_scale.clamp(1.0, 20.0)
        attn = F.softmax(sims, dim=1)
        mass = torch.zeros(query.size(0), unique_classes.size(0), device=query.device)
        mass.scatter_add_(1, inv.unsqueeze(0).expand(query.size(0), -1), attn)
        logits = torch.log(mass.clamp_min(1e-8))
        counts = torch.bincount(inv, minlength=unique_classes.numel()).float().clamp_min(1.0)
        logits = logits + self.count_prior * torch.log(counts).unsqueeze(0)
        return logits, unique_classes, F.softmax(logits, dim=-1), {
            "local_counts": counts, "pool_counts": counts, "mode": "legacy_soft_knn"
        }

    def _local_evidence(self, support, labels, query, classes):
        if self.local_centering:
            support, query = self._center_and_renorm(support, query)
        sims = (query @ support.t()) * self.local_scale
        evidence = torch.full((query.size(0), classes.numel()), -1e4, device=query.device)
        counts = self._class_counts(labels, classes).to(query.device)
        gamma = self.evidence_gamma
        for idx, cls in enumerate(classes):
            mask = labels == cls
            if mask.any():
                evidence[:, idx] = torch.logsumexp(sims[:, mask], dim=1) - gamma * torch.log(counts[idx])
        return evidence, counts, sims

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

        local, local_counts, local_sims = self._local_evidence(local_support, support_labels, query, classes)
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
        }
        result = (output_logits, classes, confidence, diagnostics)
        return result if return_diagnostics else result[:3]

    @property
    def regression_outputs_hours(self):
        return self.regression_mode == "learned_transform_ensemble"

    def regression_labels_to_output(self, labels):
        labels = labels.float()
        if self.regression_outputs_hours:
            return labels.clamp_min(0.0).square()
        return labels

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

    def regression_loss(
        self,
        predictions,
        labels,
        labels_in_output_space=False,
        branch_predictions=None,
        aggregation_weights=None,
    ):
        """Optimize raw-hour MAE/RMSE and optional transform-gate selection.

        Label units:
        - ``labels_in_output_space=False`` (default): ``labels`` are the stored
          task values (``sqrt(hours)`` for remaining time). They are converted
          with :meth:`regression_labels_to_output` before the metric.
        - ``labels_in_output_space=True``: ``labels`` are already in the head's
          output unit (raw hours for ``learned_transform_ensemble``).

        Predictions must always be in the head's output unit. Under AMP the
        primary metric is computed in float32 for stable RMSE/median scaling.
        """
        targets = labels.float() if labels_in_output_space else self.regression_labels_to_output(labels)
        if not self.regression_outputs_hours:
            return F.huber_loss(predictions.squeeze(), targets.squeeze())
        # Keep reduction math in fp32: AMP half can under/overflow large hour
        # ranges when squaring residuals and taking batch medians.
        predictions_f = predictions.float().reshape(-1)
        targets_f = targets.float().reshape(-1)
        errors = predictions_f - targets_f
        batch_scale = targets_f.detach().median().clamp_min(1.0)
        power = self.regression_loss_scale_power
        normalizer = batch_scale.pow(power) * (
            self.regression_loss_reference_hours ** (1.0 - power)
        )
        normalized = errors / normalizer
        mae = normalized.abs().mean()
        rmse = torch.sqrt(normalized.square().mean() + 1e-8)
        denominator = self.regression_mae_weight + self.regression_rmse_weight
        loss = (
            self.regression_mae_weight * mae + self.regression_rmse_weight * rmse
        ) / denominator
        has_branch_predictions = branch_predictions is not None
        has_aggregation_weights = aggregation_weights is not None
        if has_branch_predictions != has_aggregation_weights:
            raise ValueError(
                "branch_predictions and aggregation_weights must be provided together"
            )
        if has_branch_predictions and self.regression_gate_aux_weight > 0:
            loss = loss + self.regression_gate_aux_weight * (
                self.regression_gate_auxiliary_loss(
                    branch_predictions, aggregation_weights, targets_f
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
        if self.regression_outputs_hours:
            support_hours = self.regression_labels_to_output(support_labels)
            prediction, diagnostics = self.time_transform_bank.predict(
                similarities, support_hours, augmentation_factor=augmentation_factor
            )
            std = diagnostics["std_hours"]
        else:
            scale = self.reg_logit_scale.clamp(1.0, 100.0)
            weights = F.softmax(similarities * scale, dim=1)
            targets = support_labels.float()
            if targets.ndim == 1:
                targets = targets.unsqueeze(0).expand(query.size(0), -1)
            prediction = (weights * targets).sum(dim=1)
            variance = (weights * (targets - prediction.unsqueeze(1)).square()).sum(dim=1)
            std = torch.sqrt(variance + 1e-8)
            diagnostics = {"attention": weights, "std": std}
        confidence = (1.0 / (1.0 + std)) * (
            (similarities.max(dim=1).values + 1.0) / 2.0
        ).clamp(0.0, 1.0)
        diagnostics["similarities"] = similarities
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
