# -*- coding: utf-8 -*-
"""
util/attention3_fixed.py

This module supports an optional weak texture-priority term, but STRICT_no_rank training disables it with lam_texture=0.
The retained texture preference comes only from the fixed biological prior.

Method retained from the submitted manuscript:
1. Independent S/T/C branch gates.
2. Fixed avian-inspired texture/shape prior.
3. Temperature-scaled bounded gating.
4. Shape-subordination and texture-color cooperation regularization.
5. Weak texture-over-color ranking at the batch-average level.
6. Ablation-safe support for STC, ST, SC and TC.

Important fixes:
- Sparse gate regularization now preserves gradients.
- Priors, learned gate biases and branch scales are indexed by the actual
  branch names, so SC/TC ablations cannot use the wrong S/T/C parameters.
- Lazy modules rebuild when either dimensions or branch identities change.
- Feature/branch dropout defaults are zero because they are not core AFGM
  components in the submitted method.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


_BRANCH_INDEX = {"S": 0, "T": 1, "C": 2}


class _GateMLP(nn.Module):
    """Independent MLP used to estimate one branch's sample-wise gate."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=True),
        )

        # Start every independent gate from a neutral response.
        # This prevents randomly initialized gate logits from immediately
        # saturating at gate_min or gate_max.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class attention(nn.Module):
    """
    Avian-inspired Feature Gating Model (AFGM).

    Inputs:
        latent_shape, latent_texture, latent_color:
            Feature maps [B, C, H, W] or pooled features [B, C].
            Any one branch may be None for retraining-style ablation.

    Output:
        Raw logits [B, class_num], suitable for nn.CrossEntropyLoss.
    """

    def __init__(
        self,
        channel: int = 3,
        class_num: int = 200,
        multi_layer: bool = True,
        hidden_dim: int = 512,
        first_bias: bool = False,
        last_bias: bool = False,
        temperature: float = 0.90,
        gate_hidden: int = 128,
        gate_dropout: float = 0.05,
        classifier_dropout: float = 0.10,
        color_feat_dropout: float = 0.0,
        branch_drop_p: float = 0.0,
        branch_dropout_p: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        feature_l2_norm: bool = False,
        gate_min: float = 0.85,
        gate_max: float = 1.15,
        init_gate: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        init_branch_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()

        if channel not in (2, 3):
            raise ValueError("channel must be 2 or 3.")
        if gate_max <= gate_min:
            raise ValueError("gate_max must be greater than gate_min.")

        self.channel = int(channel)
        self.class_num = int(class_num)
        self.multi_layer = bool(multi_layer)
        self.hidden_dim = int(hidden_dim)
        self.first_bias = bool(first_bias)
        self.last_bias = bool(last_bias)

        self.temperature = float(temperature)
        self.gate_hidden = int(gate_hidden)
        self.gate_dropout = float(gate_dropout)
        self.classifier_dropout = float(classifier_dropout)
        self.color_feat_dropout = float(color_feat_dropout)
        self.branch_drop_p = float(branch_drop_p)
        self.branch_dropout_tuple = tuple(float(x) for x in branch_dropout_p)
        self.feature_l2_norm = bool(feature_l2_norm)

        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)

        # Neutral global offsets retained in the state dict for compatibility,
        # but kept fixed. The submitted method uses the independent gate MLP
        # and the biologically defined fixed prior, not an extra free global bias.
        self.gate_logits = nn.Parameter(
            torch.zeros(3),
            requires_grad=False,
        )

        # Fixed biological prior, configured by the training program.
        self.gate_prior = nn.Parameter(torch.zeros(3), requires_grad=False)

        # Fixed neutral scales. A second learnable scaling variable is
        # redundant with the dynamic gate and caused scale non-identifiability.
        self.branch_scale = nn.Parameter(
            torch.ones(3, dtype=torch.float32),
            requires_grad=False,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn_dict = nn.ModuleDict()
        self.mlp_dict = nn.ModuleDict()
        self.classifier: Optional[nn.Module] = None

        self._built_in_features: Optional[int] = None
        self._built_names: Optional[List[str]] = None

        # One copy retains gradients for the regularizer; one is detached
        # for logging/plotting.
        self._last_g_for_loss: Optional[torch.Tensor] = None
        self._last_g: Optional[torch.Tensor] = None
        self._last_names: Optional[List[str]] = None

        self._init_gate_logits(init_gate)

    def _init_gate_logits(self, init_gate: Sequence[float]) -> None:
        """Map desired initial gate amplitudes into learnable logits."""
        with torch.no_grad():
            initial = torch.tensor(init_gate, dtype=torch.float32)
            initial = torch.clamp(initial, self.gate_min, self.gate_max)
            alpha = (initial - self.gate_min) / (
                self.gate_max - self.gate_min
            )
            alpha = torch.clamp(alpha, 1e-6, 1.0 - 1e-6)
            self.gate_logits.copy_(torch.log(alpha) - torch.log1p(-alpha))

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1).unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(
                f"Expected a 2D or 4D feature tensor, got shape {tuple(x.shape)}."
            )
        return x

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _reset_gate_output_layers(self) -> None:
        """Force every gate MLP to start from a neutral, input-independent output."""
        for gate_mlp in self.mlp_dict.values():
            nn.init.zeros_(gate_mlp.net[-1].weight)
            nn.init.zeros_(gate_mlp.net[-1].bias)

    def set_gate_trainable(self, trainable: bool) -> None:
        """
        Freeze/unfreeze gate parameters and their BatchNorm/dropout behaviour.

        Calling requires_grad_(False) alone does not freeze BatchNorm running
        statistics. This method also switches gate submodules to eval mode
        during classifier-only warm-up.
        """
        trainable = bool(trainable)
        for parameter in self.bn_dict.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.mlp_dict.parameters():
            parameter.requires_grad_(trainable)

        if trainable and self.training:
            self.bn_dict.train()
            self.mlp_dict.train()
        else:
            self.bn_dict.eval()
            self.mlp_dict.eval()

    def _branch_dropout_probability(self, name: str) -> float:
        if self.branch_drop_p > 0:
            return self.branch_drop_p
        return self.branch_dropout_tuple[_BRANCH_INDEX[name]]

    def _inverted_branch_dropout(
        self,
        x: torch.Tensor,
        probability: float,
    ) -> torch.Tensor:
        if probability <= 0 or not self.training:
            return x
        keep = max(1e-6, 1.0 - probability)
        mask = torch.empty_like(x[:, :1, :1, :1]).bernoulli_(keep)
        return x * mask / keep

    def _build_lazy(
        self,
        names: List[str],
        channel_map: Dict[str, int],
        device: torch.device,
    ) -> None:
        self.bn_dict = nn.ModuleDict()
        self.mlp_dict = nn.ModuleDict()

        fused_dim = 0
        for name in names:
            dim = int(channel_map[name])
            self.bn_dict[name] = nn.BatchNorm1d(dim)
            self.mlp_dict[name] = _GateMLP(
                dim,
                hidden=self.gate_hidden,
                dropout=self.gate_dropout,
            )
            fused_dim += dim

        if self.multi_layer:
            self.classifier = nn.Sequential(
                nn.Linear(
                    fused_dim,
                    self.hidden_dim,
                    bias=self.first_bias,
                ),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.classifier_dropout),
                nn.Linear(
                    self.hidden_dim,
                    self.class_num,
                    bias=self.last_bias,
                ),
            )
        else:
            self.classifier = nn.Linear(
                fused_dim,
                self.class_num,
                bias=self.last_bias,
            )

        self.apply(self._initialize_module)

        # _GateMLP zero-initialises its final layer in __init__, but the
        # model-wide initialiser above would overwrite it. Reset it here,
        # after self.apply(...), so all gates truly start from neutral output.
        self._reset_gate_output_layers()

        self.to(device)

        self._built_in_features = fused_dim
        self._built_names = list(names)

    def _prepare_features(
        self,
        latent_shape: Optional[torch.Tensor],
        latent_texture: Optional[torch.Tensor],
        latent_color: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = self.gate_logits.device
        output: Dict[str, torch.Tensor] = {}

        for name, value in (
            ("S", latent_shape),
            ("T", latent_texture),
            ("C", latent_color),
        ):
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            output[name] = self._to_4d(value)

        if len(output) < 2:
            raise RuntimeError("At least two branches among S/T/C are required.")

        return output

    def forward(
        self,
        latent_shape: Optional[torch.Tensor] = None,
        latent_texture: Optional[torch.Tensor] = None,
        latent_color: Optional[torch.Tensor] = None,
        require_grad: bool = False,
    ):
        features = self._prepare_features(
            latent_shape,
            latent_texture,
            latent_color,
        )
        names = list(features.keys())  # fixed semantic order: S, T, C

        channel_map = {
            name: int(features[name].size(1))
            for name in names
        }
        fused_dim = sum(channel_map.values())

        needs_rebuild = (
            self.classifier is None
            or self._built_in_features != fused_dim
            or self._built_names != names
        )
        if needs_rebuild:
            self._build_lazy(names, channel_map, self.gate_logits.device)

        # Optional color-feature dropout. Disabled by default.
        if (
            "C" in features
            and self.training
            and self.color_feat_dropout > 0
        ):
            keep = max(1e-6, 1.0 - self.color_feat_dropout)
            color = features["C"]
            mask = torch.empty_like(
                color[:, :1, :1, :1]
            ).bernoulli_(keep)
            features["C"] = color * mask / keep

        if self.feature_l2_norm:
            for name in names:
                features[name] = self._l2_normalize(features[name])

        # Independent sample-wise gating.
        raw_gate_responses = []
        for name in names:
            pooled = self.avgpool(features[name]).flatten(1)
            pooled = self.bn_dict[name](pooled)
            raw_gate_responses.append(self.mlp_dict[name](pooled))

        raw_gate = torch.stack(raw_gate_responses, dim=1)

        branch_indices = torch.tensor(
            [_BRANCH_INDEX[name] for name in names],
            device=raw_gate.device,
            dtype=torch.long,
        )
        fixed_prior = self.gate_prior.index_select(0, branch_indices)
        fixed_offset = self.gate_logits.index_select(0, branch_indices)

        # Independent gate response + temperature scaling + fixed avian prior.
        # gate_logits is fixed (not learnable) and only preserves the optional
        # init_gate interface; with init_gate=(1,1,1), it is exactly zero.
        normalized_gate = torch.sigmoid(
            raw_gate / max(self.temperature, 1e-6)
            + fixed_offset.unsqueeze(0)
            + fixed_prior.unsqueeze(0)
        )
        gate = self.gate_min + normalized_gate * (
            self.gate_max - self.gate_min
        )

        self._last_g_for_loss = gate
        self._last_g = gate.detach()
        self._last_names = list(names)

        weighted_features = []
        batch_size = next(iter(features.values())).size(0)

        for position, name in enumerate(names):
            dropped = self._inverted_branch_dropout(
                features[name],
                self._branch_dropout_probability(name),
            )
            branch_scale = self.branch_scale[_BRANCH_INDEX[name]]
            weight = (
                gate[:, position] * branch_scale
            ).view(batch_size, 1, 1, 1)
            weighted_features.append(weight * dropped)

        fused = torch.cat(weighted_features, dim=1)
        fused = self.avgpool(fused).flatten(1)
        logits = self.classifier(fused)

        if require_grad:
            with torch.no_grad():
                predicted_class = logits.argmax(dim=1)
            selected = logits[
                torch.arange(batch_size, device=logits.device),
                predicted_class,
            ]
            for parameter in self.parameters():
                if parameter.grad is not None:
                    parameter.grad.zero_()
            selected.sum().backward()

            grad_shape = (
                latent_shape.grad
                if latent_shape is not None and hasattr(latent_shape, "grad")
                else None
            )
            grad_texture = (
                latent_texture.grad
                if latent_texture is not None
                and hasattr(latent_texture, "grad")
                else None
            )
            grad_color = (
                latent_color.grad
                if latent_color is not None and hasattr(latent_color, "grad")
                else None
            )
            return grad_shape, grad_texture, grad_color

        return logits

    def gate_penalty(
        self,
        margin: float = 0.04,
        tolerance: float = 0.08,
        lam_shape: float = 1.0,
        lam_tc: float = 1.0,
        texture_margin: float = 0.005,
        lam_texture: float = 2.0,
        lam_budget: float = 0.0,
        budget_target: float = 1.0,
    ) -> torch.Tensor:
        """
        Biologically motivated asymmetric gate regularization.

        1. Shape subordination:
           g_S should be lower than both g_T and g_C by approximately margin.
        2. Texture-color cooperation:
           |g_T - g_C| should remain within tolerance.
        3. Weak texture priority:
           the batch-average texture gate should be slightly higher than the
           batch-average color gate by approximately texture_margin.

        The texture-priority term is intentionally weak and operates on the
        batch average, allowing individual images to rely more strongly on
        color when that improves classification.

        The loss uses the non-detached gate tensor and therefore contributes
        gradients to the independent gate networks.
        """
        if (
            self._last_g_for_loss is None
            or self._last_names is None
            or set(self._last_names) != {"S", "T", "C"}
        ):
            return next(self.mlp_dict.parameters()).sum() * 0.0

        position = {
            name: index
            for index, name in enumerate(self._last_names)
        }
        # Apply the biological constraints sample by sample. Penalising only
        # the batch mean can hide opposite violations from different samples.
        gate_s = self._last_g_for_loss[:, position["S"]]
        gate_t = self._last_g_for_loss[:, position["T"]]
        gate_c = self._last_g_for_loss[:, position["C"]]

        loss_shape = torch.relu(
            gate_s - torch.minimum(gate_t, gate_c) + margin
        ).mean()
        loss_tc = torch.relu(
            torch.abs(gate_t - gate_c) - tolerance
        ).mean()

        # Weak batch-average ranking:
        # mean(g_T) >= mean(g_C) + texture_margin.
        # This is softer than a per-sample hard ranking and therefore less
        # likely to damage classification accuracy.
        loss_texture_priority = torch.relu(
            gate_c.mean() - gate_t.mean() + float(texture_margin)
        )

        # Optional mean-gate budget. A small value prevents all three branches
        # from drifting upward together while preserving their relative order.
        mean_amplitude = self._last_g_for_loss.mean(dim=1)
        loss_budget = (
            mean_amplitude - float(budget_target)
        ).pow(2).mean()

        return (
            lam_shape * loss_shape
            + lam_tc * loss_tc
            + lam_texture * loss_texture_priority
            + lam_budget * loss_budget
        )

    @torch.no_grad()
    def get_gate_batch(self) -> Optional[torch.Tensor]:
        """Return the most recent batch gates as a detached CPU tensor."""
        if self._last_g is None:
            return None
        return self._last_g.detach().cpu()

    @torch.no_grad()
    def get_gate_values(self) -> List[float]:
        """Return mean gate values in the fixed S/T/C order."""
        values = {"S": float("nan"), "T": float("nan"), "C": float("nan")}
        if self._last_g is not None and self._last_names is not None:
            means = self._last_g.mean(dim=0)
            for index, name in enumerate(self._last_names):
                values[name] = float(means[index].item())
        return [values["S"], values["T"], values["C"]]

    @torch.no_grad()
    def get_gate_dict(self) -> Dict[str, float]:
        values = self.get_gate_values()
        return {"S": values[0], "T": values[1], "C": values[2]}
