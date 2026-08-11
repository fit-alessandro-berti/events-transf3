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

    def forward_regression(self, support_features, support_labels, query_features):
        if support_features.numel() == 0 or query_features.numel() == 0:
            device = query_features.device
            return torch.zeros(query_features.size(0), device=device), torch.zeros(query_features.size(0), device=device)
        support = _l2_normalize(support_features)
        query = _l2_normalize(query_features)
        support, query = self._center_and_renorm(support, query)
        scale = self.reg_logit_scale.clamp(1.0, 100.0)
        sims_raw = query @ support.t()
        weights = F.softmax(sims_raw * scale, dim=1)
        targets = support_labels.view(-1).float()
        prediction = weights @ targets
        variance = (weights * (targets.view(1, -1) - prediction.view(-1, 1)) ** 2).sum(dim=1)
        std = torch.sqrt(variance + 1e-8)
        confidence = (1.0 / (1.0 + std)) * ((sims_raw.max(dim=1).values + 1.0) / 2.0).clamp(0.0, 1.0)
        return prediction, confidence.clamp(0.0, 1.0)
