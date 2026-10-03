"""The semantic interpreter: a causal pattern in, the name of a cause out.

Localisation is not part of this model. It has no head that scores units for
importance and no loss that asks it to. The lower layer already did that, better
than anything learned here could (`docs/RESULTS.md`, confirmatory run).

Candidates arrive at inference time as fixed-width signatures, so one set of
weights names causes among four candidates in one domain and seven in another,
and can be pointed at a vocabulary it was never trained on.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoding import (D_GLOBAL, D_SIGNATURE, D_UNIT, GLOBAL_FEATURES,
                       GLOBAL_PATTERN_FEATURES, SIGNATURE_FEATURES,
                       SIGNATURE_PATTERN_FEATURES, UNIT_FEATURES)

# The structural features a control keeps when the causal pattern is withheld:
# architecture shape only, nothing about what the model is doing right now.
STRUCTURAL = ("depth", "is_mlp_neuron", "is_attn_head", "log_site_width")
PATTERN_FEATURES = tuple(f for f in UNIT_FEATURES if f not in STRUCTURAL)
# The pattern also leaks through the global context: `log_fo_mass` is a summary
# of the same first-order term.  Masking only the per-unit features leaves a
# control that still reads the causal layer -- the exact failure the audit found
# in the previous generation's internals-blind arm.


class SemanticInterpreter(nn.Module):
    def __init__(self, d_model: int = 96, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.1, use_locus: bool = True,
                 blind: bool = False):
        super().__init__()
        mask = torch.ones(D_UNIT)
        gmask = torch.ones(D_GLOBAL)
        smask = torch.ones(D_SIGNATURE)
        if blind:  # keep architecture and B's own output; drop every trace of the pattern
            for f in PATTERN_FEATURES:
                mask[UNIT_FEATURES.index(f)] = 0.0
            for f in GLOBAL_PATTERN_FEATURES:
                gmask[GLOBAL_FEATURES.index(f)] = 0.0
            # The candidate signatures carry an analytic half describing how the
            # causal pattern moves; leaving it visible made the control not blind.
            for f in SIGNATURE_PATTERN_FEATURES:
                smask[SIGNATURE_FEATURES.index(f)] = 0.0
        self.register_buffer("feat_mask", mask)
        self.register_buffer("global_mask", gmask)
        self.register_buffer("sig_mask_feat", smask)
        self.use_locus, self.blind = use_locus and not blind, blind
        self.d_model = d_model

        self.inp = nn.Sequential(nn.Linear(D_UNIT + D_GLOBAL, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        self.sig_in = nn.Sequential(nn.Linear(D_SIGNATURE, d_model), nn.GELU(),
                                    nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.locus_gain = nn.Parameter(torch.tensor(1.0))
        n_eng = 2 * len(PATTERN_FEATURES)
        self.eng_in = nn.Sequential(nn.Linear(n_eng, d_model), nn.GELU())
        self.name_out = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))

    def forward(self, unit_x, global_x, mask, sig_x, sig_mask, locus=None) -> dict:
        ux = unit_x * self.feat_mask
        g = (global_x * self.global_mask)[:, None, :].expand(-1, ux.shape[1], -1)
        h = self.norm(self.blocks(self.inp(torch.cat([ux, g], -1)),
                                  src_key_padding_mask=~mask))
        s = self.sig_in(sig_x * self.sig_mask_feat)  # (B, K, d)
        att = self.q(s) @ self.k(h).transpose(-2, -1) / math.sqrt(self.d_model)

        eng = torch.zeros(s.shape[0], s.shape[1], 2 * len(PATTERN_FEATURES), device=s.device)
        if self.use_locus and locus is not None:
            p = locus * mask[:, None, :]
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
            att = att + self.locus_gain * torch.log1p(p * mask.sum(1)[:, None, None].clamp(min=1))
            idx = [UNIT_FEATURES.index(f) for f in PATTERN_FEATURES]
            f = (unit_x * self.feat_mask)[..., idx]
            eng = torch.cat([p @ f, p @ f.abs()], dim=-1)

        neg = torch.finfo(att.dtype).min
        att = att.masked_fill(~mask[:, None, :], neg)
        z = torch.cat([torch.softmax(att, -1) @ self.v(h), s, self.eng_in(eng)], -1)
        return {"name": self.name_out(z).squeeze(-1).masked_fill(~sig_mask, neg),
                "attention": att}
