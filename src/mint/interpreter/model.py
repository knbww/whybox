"""Interpreter A: a permutation-equivariant reader of another model's internals.

Design constraints that make the transfer claim testable:

* **Set-structured, not sequence-structured.**  No positional encoding over
  units: the interpreter cannot rely on unit ordering or on layer identity,
  only on the depth *value* carried in each descriptor.  Swap two units and the
  outputs swap with them (see `tests/test_interpreter.py::test_unit_permutation_equivariance`).
* **Fixed-width input.**  Any target model, any architecture, any domain enters
  through the same `D_UNIT`-dimensional descriptor, so weights transfer
  unchanged across domains.
* **Split heads.**  The *procedure* heads (which internal variable is causally
  load-bearing, how much of a factor it mediates, whether it is a correlational
  decoy) are shared and are what we test for transfer.  The legacy *closed
  vocabulary* head is per-domain, discarded at transfer, and exists so we can
  measure the two kinds of transfer separately.
* **An open vocabulary.**  The naming head takes the candidate explanations as
  fixed-width signatures at inference time and cross-attends from each candidate
  to B's units, so it works for any number of candidates in any domain.  This is
  what turns a score over units into an *explanation*: a named cause, the units
  that carry it, and a signed prediction of what happens if it is removed --
  every part of which is checkable by executing an intervention on B.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding import D_FACTOR, D_GLOBAL, D_UNIT, UNIT_FEATURES

# Unit descriptors summed against a candidate's locus to ask "is this candidate
# engaged right now?".  All state-conditioned, all ontology-free.
ENGAGEMENT_FEATURES = ("state_act_z", "abs_state_act_z", "state_gradact_scaled",
                       "first_order_ablation", "abs_first_order_ablation_rank",
                       "state_is_active")
ENGAGEMENT_IDX = [UNIT_FEATURES.index(f) for f in ENGAGEMENT_FEATURES]

# Descriptors that describe B's *internals*.  Blanking them yields the
# output-only baseline: same capacity, same training, no access to the model.
INTERNAL_FEATURES = tuple(
    f for f in UNIT_FEATURES
    if f not in ("depth", "is_mlp_neuron", "is_attn_head", "log_site_width")
)


# Descriptor families, for asking *which* part of the model view transfers.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "gradients": ("log_grad_absmean_z", "gradact_mean_z", "log_gradact_absmean_z",
                  "gradact_absmean_rank", "state_grad_scaled", "state_gradact_scaled",
                  "abs_state_gradact_rank", "first_order_ablation",
                  "abs_first_order_ablation_rank"),
    "correlation": ("corr_out", "abs_corr_out", "abs_corr_out_rank"),
    "weights": ("in_norm_z", "out_norm_z", "out_norm_rank"),
    "activation_stats": ("act_mean_z", "log_act_std_z", "act_p10_z", "act_p50_z",
                         "act_p90_z", "frac_active", "redundancy"),
    "state": ("state_act_z", "abs_state_act_z", "state_act_pctile", "state_is_active"),
}


def feature_mask(drop: tuple[str, ...] = ()) -> torch.Tensor:
    m = torch.ones(D_UNIT)
    for name in drop:
        m[UNIT_FEATURES.index(name)] = 0.0
    return m


class Interpreter(nn.Module):
    def __init__(self, d_model: int = 96, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.1, factor_heads: dict[str, int] | None = None,
                 drop_features: tuple[str, ...] = ()):
        super().__init__()
        self.register_buffer("feat_mask", feature_mask(drop_features))
        self.drop_features = drop_features
        self.inp = nn.Sequential(
            nn.Linear(D_UNIT + D_GLOBAL, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.causal = nn.Linear(d_model, 1)
        self.mediation = nn.Linear(d_model, 1)
        self.decoy = nn.Linear(d_model, 1)
        self.effect_mass = nn.Linear(d_model, 1)
        self.factor_heads = nn.ModuleDict(  # legacy closed vocabulary, per domain
            {k: nn.Linear(d_model, v) for k, v in (factor_heads or {}).items()}
        )
        # open vocabulary: candidate explanations arrive as signatures, and each
        # one cross-attends to the units to find its own carriers
        self.d_model = d_model
        self.factor_in = nn.Sequential(
            nn.Linear(D_FACTOR, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        n_eng = len(ENGAGEMENT_IDX)
        self.locus_gain = nn.Parameter(torch.tensor(1.0))
        # Retrieval over reference states beats the learned head on its own, so
        # the head is given it and asked to rerank rather than to compete.
        self.retrieval_gain = nn.Parameter(torch.tensor(1.0))
        self.eng_in = nn.Sequential(nn.Linear(2 * n_eng + 2, d_model), nn.GELU())
        self.name_out = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))
        self.effect_out = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, 1))

    def encode(self, unit_x: torch.Tensor, global_x: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        ux = unit_x * self.feat_mask
        g = global_x[:, None, :].expand(-1, ux.shape[1], -1)
        h = self.inp(torch.cat([ux, g], dim=-1))
        h = self.blocks(h, src_key_padding_mask=~mask)
        return self.norm(h)

    def name_candidates(self, h: torch.Tensor, mask: torch.Tensor,
                        factor_x: torch.Tensor, factor_mask: torch.Tensor,
                        factor_locus: torch.Tensor | None = None,
                        unit_x: torch.Tensor | None = None,
                        retrieval: torch.Tensor | None = None) -> dict:
        """Match each candidate explanation against B's units at this state.

        Each candidate signature becomes a query that attends over the units, so
        the head works for any number of candidates and returns, for free, the
        per-candidate attribution over units that the explanation needs.
        """
        g = self.factor_in(factor_x)  # (B, K, d)
        att = self.q(g) @ self.k(h).transpose(-2, -1) / math.sqrt(self.d_model)  # (B, K, U)
        eng = torch.zeros(g.shape[0], g.shape[1], 2 * len(ENGAGEMENT_IDX) + 2, device=g.device)
        if retrieval is not None:
            eng[..., -2] = retrieval
            eng[..., -1] = torch.log(retrieval.clamp(min=1e-4))
        if factor_locus is not None:
            p = factor_locus * mask[:, None, :]
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
            lz = torch.log1p(p * mask.sum(1)[:, None, None].clamp(min=1))
            att = att + self.locus_gain * lz
            if unit_x is not None:
                f = (unit_x * self.feat_mask)[..., ENGAGEMENT_IDX]  # (B, U, E)
                eng = torch.cat([p @ f, p @ f.abs(), eng[..., -2:]], dim=-1)
        neg = torch.finfo(att.dtype).min
        att = att.masked_fill(~mask[:, None, :], neg)
        z = torch.cat([torch.softmax(att, dim=-1) @ self.v(h), g, self.eng_in(eng)], dim=-1)
        name = self.name_out(z).squeeze(-1)
        if retrieval is not None:
            name = name + self.retrieval_gain * torch.log(retrieval.clamp(min=1e-4))
        return {
            "name": name.masked_fill(~factor_mask, neg),
            "factor_effect": self.effect_out(z).squeeze(-1),
            "carrier": att,
        }

    def forward(self, unit_x, global_x, mask, domain: str | None = None,
                factor_x: torch.Tensor | None = None,
                factor_mask: torch.Tensor | None = None,
                factor_locus: torch.Tensor | None = None,
                retrieval: torch.Tensor | None = None) -> dict:
        h = self.encode(unit_x, global_x, mask)
        neg = torch.finfo(h.dtype).min
        out = {
            "causal": self.causal(h).squeeze(-1).masked_fill(~mask, neg),
            "mediation": self.mediation(h).squeeze(-1).masked_fill(~mask, neg),
            "decoy": self.decoy(h).squeeze(-1),
        }
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        out["effect_mass"] = self.effect_mass(pooled).squeeze(-1)
        if factor_x is not None:
            out.update(self.name_candidates(h, mask, factor_x, factor_mask,
                                            factor_locus, unit_x, retrieval))
        if domain is not None and domain in self.factor_heads:
            out["factor"] = self.factor_heads[domain](pooled)
        return out

    def add_factor_head(self, domain: str, n_factors: int) -> None:
        d = self.causal.in_features
        self.factor_heads[domain] = nn.Linear(d, n_factors)

    def procedure_parameters(self):
        """Everything except the domain-specific closed-vocabulary heads."""
        vocab = {id(p) for p in self.factor_heads.parameters()}
        return [p for p in self.parameters() if id(p) not in vocab]


def listwise_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  tau: float = 0.15, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Soft ranking loss: cross-entropy between attention over units."""
    neg = torch.finfo(pred.dtype).min
    t = torch.softmax(target.masked_fill(~mask, neg) / tau, dim=1)
    logp = torch.log_softmax(pred.masked_fill(~mask, neg), dim=1)
    per = -(t * logp).sum(1)
    if weight is None:
        return per.mean()
    return (per * weight).sum() / weight.sum().clamp(min=1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    p = torch.sigmoid(pred.masked_fill(~mask, 0.0))
    m = mask.float() if weight is None else mask.float() * weight[:, None]
    return (((p - target) ** 2) * m).sum() / m.sum().clamp(min=1)
