"""A reranker over positions: take the first-order report, fix the boundary."""
from __future__ import annotations

import torch
import torch.nn as nn

# NOTE: the absolute position index is deliberately absent. With it, a control
# that sees nothing else about the model reaches 0.528 exact match against the
# full model's 0.499 -- it memorises which slots of this domain's fixed layout
# belong in the minimal set. That is learning the ontology, which is the one
# thing this project exists to rule out, and it does not transfer to a domain
# laid out differently.
POSITION_FEATURES = ("fo_signed", "fo_abs", "fo_rank", "fo_share", "is_candidate",
                     "fo_sign", "n_candidates")
D_POS = len(POSITION_FEATURES)


class SetProposer(nn.Module):
    """Per-position inclusion logits, permutation-equivariant over positions."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, blind: bool = False):
        super().__init__()
        mask = torch.ones(D_POS)
        if blind:  # keeps only structure: which positions are candidates, and where
            for f in ("fo_signed", "fo_abs", "fo_rank", "fo_share", "fo_sign"):
                mask[POSITION_FEATURES.index(f)] = 0.0
        self.register_buffer("feat_mask", mask)
        self.inp = nn.Sequential(nn.Linear(D_POS, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, 1)

    def forward(self, pos_x: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(pos_x * self.feat_mask))
        neg = torch.finfo(h.dtype).min
        return self.out(h).squeeze(-1).masked_fill(~cand, neg)
