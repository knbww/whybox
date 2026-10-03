"""Assembling semantic problems, with the reference/scored split enforced here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..interpreter.dataset import Bundle
from .encoding import candidate_locus, candidate_signatures, causal_pattern


@dataclass
class SemanticSet:
    unit_x: torch.Tensor  # (M, U_max, D_UNIT)
    global_x: torch.Tensor  # (M, D_GLOBAL)
    mask: torch.Tensor  # (M, U_max)
    sig_x: torch.Tensor  # (M, K_max, D_SIGNATURE)
    sig_mask: torch.Tensor  # (M, K_max)
    locus: torch.Tensor  # (M, K_max, U_max)
    retrieval: torch.Tensor  # (M, K_max) reference vote, assisted condition only
    y: torch.Tensor  # (M,) decisive candidate
    bundle_id: torch.Tensor
    state_id: torch.Tensor
    cell_id: torch.Tensor
    cells: list[str]

    def __len__(self) -> int:
        return int(self.unit_x.shape[0])

    def subset(self, idx) -> "SemanticSet":
        return SemanticSet(*[getattr(self, f)[idx] for f in
                             ("unit_x", "global_x", "mask", "sig_x", "sig_mask", "locus",
                              "retrieval", "y", "bundle_id", "state_id", "cell_id")],
                           cells=self.cells)


def build(bundles: list[Bundle], cells: list[str] | None = None,
          ref_frac: float = 0.375, n_neighbours: int = 8,
          min_effect: float = 0.0) -> SemanticSet:
    """`min_effect` drops states where no candidate moves B: there is no cause to
    name there, and the annotation's argmax over an all-zero vector is an
    artefact of tie-breaking, not a label."""
    cells = cells or sorted({b.cell for b in bundles})
    u_max = max(b.n_units for b in bundles)
    k_max = max(b.gt.factor_total.shape[1] for b in bundles)
    cols = {k: [] for k in ("unit_x", "global_x", "mask", "sig_x", "sig_mask", "locus",
                            "retrieval", "y", "bundle_id", "state_id", "cell_id")}
    for j, b in enumerate(bundles):
        gt = b.gt
        n_all = len(gt.decisive)
        n_ref = max(2, int(round(ref_frac * n_all)))
        ref, keep = np.arange(n_ref), np.arange(n_ref, n_all)
        if min_effect > 0:
            live = np.abs(gt.factor_total[keep, gt.decisive[keep]]) > \
                min_effect * (gt.probe.logit_std + 1e-6)
            keep = keep[live]
            if len(keep) < 4:
                continue
        n, u = len(keep), b.n_units
        pad, kpad = u_max - u, k_max - gt.factor_total.shape[1]

        ux, gx = causal_pattern(gt)
        cols["unit_x"].append(np.pad(ux[keep], ((0, 0), (0, pad), (0, 0))))
        cols["global_x"].append(gx[keep])
        m = np.zeros((n, u_max), bool); m[:, :u] = True
        cols["mask"].append(m)

        sig = candidate_signatures(gt, ref)
        cols["sig_x"].append(np.repeat(np.pad(sig, ((0, kpad), (0, 0)))[None], n, 0))
        sm = np.zeros((n, k_max), bool); sm[:, :sig.shape[0]] = True
        cols["sig_mask"].append(sm)
        loc = candidate_locus(gt, ref)
        cols["locus"].append(np.repeat(np.pad(loc, ((0, kpad), (0, pad)))[None], n, 0))

        # reference vote in causal-pattern space (assisted condition only)
        fo = gt.first_order / (np.linalg.norm(gt.first_order, axis=1, keepdims=True) + 1e-9)
        sim = fo[keep] @ fo[ref].T
        top = np.argsort(-sim, axis=1)[:, :n_neighbours]
        s_top = np.take_along_axis(sim, top, axis=1)
        w = np.exp((s_top - s_top.max(1, keepdims=True)) / (0.5 * s_top.std(1, keepdims=True) + 1e-6))
        w /= w.sum(1, keepdims=True) + 1e-9
        votes = gt.decisive[ref][top]
        post = np.zeros((n, k_max))
        for gidx in range(sig.shape[0]):
            post[:, gidx] = (w * (votes == gidx)).sum(1)
        cols["retrieval"].append(post)

        cols["y"].append(gt.decisive[keep])
        cols["bundle_id"].append(np.full(n, j))
        cols["state_id"].append(keep)
        cols["cell_id"].append(np.full(n, cells.index(b.cell)))

    cat = lambda k, dt: torch.as_tensor(np.concatenate(cols[k], 0), dtype=dt)
    return SemanticSet(
        unit_x=cat("unit_x", torch.float32), global_x=cat("global_x", torch.float32),
        mask=cat("mask", torch.bool), sig_x=cat("sig_x", torch.float32),
        sig_mask=cat("sig_mask", torch.bool), locus=cat("locus", torch.float32),
        retrieval=cat("retrieval", torch.float32), y=cat("y", torch.long),
        bundle_id=cat("bundle_id", torch.long), state_id=cat("state_id", torch.long),
        cell_id=cat("cell_id", torch.long), cells=cells)
