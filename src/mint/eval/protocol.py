"""Executed validation of a proposed explanation.

A score over units is only a *candidate* explanation.  This module runs the
last two steps of the protocol the study is about -- intervention and causal
validation -- by actually editing B and reading what happens:

sufficiency@k       patch only the k top-scored units to the activations they
                    take under the executed world counterfactual
                    do(decisive factor := neutral); how much of B's behavioural
                    response to that counterfactual is reproduced?
comprehensiveness@k patch everything *except* those k units; how much of the
                    response survives without them?

A good explanation has high sufficiency and low comprehensiveness.  Both are
measured against `factor_total`, B's own response to the executed intervention,
so the numbers are comparable across models and domains.
"""
from __future__ import annotations

import numpy as np
import torch

from ..causal.interventions import counterfactual_site_acts, subset_patch_effect
from ..causal.layout import UnitLayout
from ..domains import get_domain
from ..interpreter.dataset import Bundle


def gather_counterfactual(model, domain, raw: np.ndarray, factor: np.ndarray,
                          layout: UnitLayout) -> dict:
    """Component-wise counterfactual activations, one factor per state.

    Each state may be explained by a different candidate, so the replacement
    tensor is assembled row by row from that state's own counterfactual.
    """
    n_f = domain.spec.n_factors
    per_factor = [counterfactual_site_acts(model, domain, raw, f) for f in range(n_f)]
    idx = torch.as_tensor(np.asarray(factor), dtype=torch.long)
    rows = torch.arange(len(idx))
    return {s: torch.stack([per_factor[f][s] for f in range(n_f)], dim=0)[idx, rows]
            for s in layout.sites}


def executed_protocol(bundle: Bundle, scores: np.ndarray, ks=(1, 3, 5),
                      min_effect: float = 0.25, state_idx: np.ndarray | None = None,
                      factor_idx: np.ndarray | None = None,
                      denom_idx: np.ndarray | None = None) -> dict[str, float]:
    """`scores`: (N, U) candidate-explanation scores for the given states.

    `state_idx` selects which of the bundle's focal states the rows refer to.
    `factor_idx` is the factor whose counterfactual gets executed -- the human
    decisive one by default, or the interpreter's *named* factor when we are
    validating a whole explanation rather than a localisation.

    `denom_idx` is the factor the result is scored against.  Setting it to the
    human decisive factor while `factor_idx` is the named one makes the score
    end-to-end and unforgiving: naming an inert cause reproduces none of what
    actually moved B, and the state stays in the denominator instead of being
    filtered out of it.
    """
    model, domain = bundle.trained.model, get_domain(bundle.domain)
    layout = UnitLayout.of(model)
    gt = bundle.gt
    n, u_n = scores.shape
    sel = np.arange(len(gt.decisive)) if state_idx is None else np.asarray(state_idx)
    raw = bundle.focal_raw[sel]

    dec = gt.decisive[sel] if factor_idx is None else np.asarray(factor_idx)
    ref = dec if denom_idx is None else np.asarray(denom_idx)
    target_vals = gather_counterfactual(model, domain, raw, dec, layout)
    total = gt.factor_total[sel, ref]  # (N,)
    keep = np.abs(total) > min_effect * (gt.probe.logit_std + 1e-6)
    if keep.sum() < 4:
        return {}

    x = torch.as_tensor(domain.to_model_input(raw))
    order = np.argsort(-scores, axis=1)
    out: dict[str, float] = {"n_states_scored": float(keep.sum())}
    for k in ks:
        m = np.zeros((n, u_n), dtype=bool)
        rows = np.repeat(np.arange(n)[:, None], k, axis=1)
        m[rows, order[:, :k]] = True
        suf = subset_patch_effect(model, layout, x, target_vals, m, gt.base_logit[sel])
        com = subset_patch_effect(model, layout, x, target_vals, ~m, gt.base_logit[sel])
        out[f"sufficiency@{k}"] = float(np.mean(np.clip(suf[keep] / total[keep], -1.0, 2.0)))
        out[f"comprehensiveness@{k}"] = float(np.mean(np.clip(com[keep] / total[keep], -1.0, 2.0)))
    return out
