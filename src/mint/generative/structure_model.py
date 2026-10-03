"""Reading the *kind* of causal structure out of a model's internal state.

This is the first component of the structured explanation that a derivative
cannot supply. `cause_type` is a property of a neighbourhood -- whether two
positions are interchangeable, separately required, or cancelling -- and a
first-order report at a point cannot express it. Measured on both
interaction-bearing domains, a heuristic applying the classification rules to
the first-order report lands at or below the majority class.

The model reads two streams, both architecture-normalised and free of domain
content: the causal pattern over the target's internal units, and the same over
its input positions. **The absolute position index is deliberately absent** -- on
a fixed-layout domain it lets a model memorise the rule instead of reading the
target, which is exactly how an earlier control came to beat the full model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..causal.structure import TYPES
from ..semantic.encoding import (D_GLOBAL, D_UNIT, GLOBAL_FEATURES,
                                 GLOBAL_PATTERN_FEATURES)
from .model import D_POS

N_TYPES = len(TYPES)


def _encoder(d_in: int, d_model: int, n_layers: int, n_heads: int, dropout: float):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return (nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Linear(d_model, d_model)),
            nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False))


class StructureReader(nn.Module):
    def __init__(self, d_model: int = 96, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, use_units: bool = True, use_positions: bool = True):
        super().__init__()
        self.use_units, self.use_positions = use_units, use_positions
        self.unit_in, self.unit_enc = _encoder(D_UNIT, d_model, n_layers, n_heads, dropout)
        self.pos_in, self.pos_enc = _encoder(D_POS, d_model, n_layers, n_heads, dropout)
        self.head = nn.Sequential(nn.Linear(2 * d_model + D_GLOBAL, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, N_TYPES))
        # Ablating the unit stream must also remove the *summary* of that stream
        # carried in `global_x`. `log_fo_mass` is a scalar over the target's
        # internal-unit first-order pattern, so an "internals-blind" control that
        # keeps it is not blind -- the exact defect `semantic/model.py` already
        # guards against with GLOBAL_PATTERN_FEATURES.
        gmask = torch.ones(D_GLOBAL)
        if not use_units:
            for f in GLOBAL_PATTERN_FEATURES:
                gmask[GLOBAL_FEATURES.index(f)] = 0.0
        self.register_buffer("global_mask", gmask)

    def _pool(self, inp, enc, x, m, on: bool) -> torch.Tensor:
        d = enc.layers[0].linear2.out_features
        if not on:  # the stream is ablated: contribute a constant, not noise
            return x.new_zeros((x.shape[0], d))
        h = enc(inp(x), src_key_padding_mask=~m)
        return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)

    def forward(self, unit_x, unit_mask, pos_x, pos_mask, global_x) -> torch.Tensor:
        u = self._pool(self.unit_in, self.unit_enc, unit_x, unit_mask, self.use_units)
        p = self._pool(self.pos_in, self.pos_enc, pos_x, pos_mask, self.use_positions)
        return self.head(torch.cat([u, p, global_x * self.global_mask], dim=-1))


class CompositeGenerator(nn.Module):
    """Generates the whole claim: kind, support, direction, magnitude.

    The support head emits one logit per input position, so the model *points*
    rather than choosing from a list of named causes -- there is no list. The
    other three heads draw on inventories that are shared across domains, which
    is the only sense in which anything is given to it.
    """

    def __init__(self, d_model: int = 96, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, use_units: bool = True, use_positions: bool = True):
        super().__init__()
        self.trunk = StructureReader(d_model, n_layers, n_heads, dropout,
                                     use_units=use_units, use_positions=use_positions)
        self.trunk.head = nn.Identity()          # keep the two encoders, drop its classifier
        d_pooled = 2 * d_model + D_GLOBAL
        self.kind = nn.Sequential(nn.Linear(d_pooled, d_model), nn.GELU(),
                                  nn.Linear(d_model, N_TYPES))
        self.sign = nn.Sequential(nn.Linear(d_pooled, d_model), nn.GELU(),
                                  nn.Linear(d_model, 2))
        self.strength = nn.Sequential(nn.Linear(d_pooled, d_model), nn.GELU(),
                                      nn.Linear(d_model, 3))
        self.support = nn.Sequential(nn.Linear(D_POS + d_pooled, d_model), nn.GELU(),
                                     nn.Linear(d_model, 1))

    def forward(self, unit_x, unit_mask, pos_x, pos_mask, global_x, cand) -> dict:
        pooled = self.trunk(unit_x, unit_mask, pos_x, pos_mask, global_x)
        ctx = pooled[:, None, :].expand(-1, pos_x.shape[1], -1)
        sup = self.support(torch.cat([pos_x, ctx], -1)).squeeze(-1)
        return {"kind": self.kind(pooled), "sign": self.sign(pooled),
                "strength": self.strength(pooled),
                "support": sup.masked_fill(~cand, torch.finfo(sup.dtype).min)}


class JointGenerator(nn.Module):
    """Four heads that have to agree with each other.

    In the independent version the heads share a trunk and nothing else, so the
    model can be right about the kind on states where it is wrong about the
    carrier and the two never meet. Here the direction and magnitude heads are
    conditioned on the kind and the carrier the model is *committing to*, so the
    question they answer is the one that matters: given this carrier and this
    kind, what follows?

    Training runs them both teacher-forced (on the true kind and carrier) and
    free (on the model's own), and the free pass is the joint term.
    """

    def __init__(self, d_model: int = 96, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, use_units: bool = True, use_positions: bool = True):
        super().__init__()
        self.trunk = StructureReader(d_model, n_layers, n_heads, dropout,
                                     use_units=use_units, use_positions=use_positions)
        self.trunk.head = nn.Identity()
        d_pooled = 2 * d_model + D_GLOBAL
        self.kind = nn.Sequential(nn.Linear(d_pooled, d_model), nn.GELU(),
                                  nn.Linear(d_model, N_TYPES))
        self.support = nn.Sequential(nn.Linear(D_POS + d_pooled, d_model), nn.GELU(),
                                     nn.Linear(d_model, 1))
        self.kind_emb = nn.Linear(N_TYPES, d_model)          # over the kind *distribution*
        self.carrier_emb = nn.Linear(D_POS + 1, d_model)     # over the carrier it commits to
        cond = d_pooled + 2 * d_model
        self.sign = nn.Sequential(nn.Linear(cond, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.strength = nn.Sequential(nn.Linear(cond, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def statement(self, pooled, pos_x, cand, kind_dist, carrier_w):
        """Embed the claim the model is making, so the rest can follow from it."""
        w = (carrier_w * cand).unsqueeze(-1)
        summ = (torch.cat([pos_x, w], -1) * w).sum(1) / w.sum(1).clamp(min=1e-6)
        return torch.cat([pooled, self.kind_emb(kind_dist), self.carrier_emb(summ)], -1)

    def forward(self, unit_x, unit_mask, pos_x, pos_mask, global_x, cand,
                force_kind=None, force_carrier=None) -> dict:
        pooled = self.trunk(unit_x, unit_mask, pos_x, pos_mask, global_x)
        ctx = pooled[:, None, :].expand(-1, pos_x.shape[1], -1)
        sup = self.support(torch.cat([pos_x, ctx], -1)).squeeze(-1)
        sup = sup.masked_fill(~cand, torch.finfo(sup.dtype).min)
        kind = self.kind(pooled)
        kd = (torch.nn.functional.one_hot(force_kind, N_TYPES).float()
              if force_kind is not None else torch.softmax(kind, -1))
        cw = force_carrier if force_carrier is not None else torch.sigmoid(sup)
        z = self.statement(pooled, pos_x, cand, kd, cw)
        return {"kind": kind, "support": sup, "sign": self.sign(z),
                "strength": self.strength(z)}
