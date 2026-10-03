"""Domain-neutral descriptions of a candidate explanation.

The closed vocabulary head -- a softmax over one domain's factor names -- cannot
transfer, by construction.  For an interpreter to *name* the cause in a domain
it has never seen, the candidate explanations have to enter at inference time as
data rather than at training time as label indices.

A candidate factor is therefore presented to the interpreter as a fixed-width
**signature** with no domain content at all: how B's behaviour responds when
that factor is intervened on across a set of reference states, and where inside
B the factor lives.  Nothing here names hypoxia or subject number; a factor from
a clinical model and a factor from a language model produce comparable vectors.

This matches the realistic division of labour.  A human proposes the candidate
hypotheses and supplies a recipe for intervening on each of them; the recipe is
executed on a handful of reference states to build the signature.  What the
interpreter then has to do -- for every new state, without executing anything --
is say which candidate is in play and which internal variables carry it.

Crucially the signature is built on states **disjoint** from the state being
explained.  Otherwise the answer would be sitting in the input.
"""
from __future__ import annotations

import numpy as np

from .model_state import tied_rank

from ..causal.interventions import CausalGroundTruth

FACTOR_FEATURES = (
    # behavioural: what happens to B's output when this factor is removed
    "mean_signed_effect", "mean_abs_effect", "sd_effect", "p90_abs_effect",
    "frac_in_play", "base_rate_decisive", "corr_effect_output", "mean_effect_share",
    # internal locus: where inside B the factor lives
    "top1_ie_share", "log_participation_ratio", "mean_locus_depth", "sd_locus_depth",
    "frac_locus_attention", "mean_single_unit_carry", "locus_consistency",
    "locus_vs_importance_rho",
)
D_FACTOR = len(FACTOR_FEATURES)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ra = tied_rank(a).astype(float)
    rb = tied_rank(b).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def factor_signatures(gt: CausalGroundTruth, ref_idx: np.ndarray,
                      in_play: float = 0.25) -> np.ndarray:
    """(F, D_FACTOR) -- one domain-neutral descriptor per candidate factor."""
    scale = gt.probe.logit_std + 1e-6
    tot = gt.factor_total[ref_idx] / scale  # (R, F)
    ie = gt.unit_ie[ref_idx] / scale  # (R, U, F)
    eff = np.abs(gt.unit_effect[ref_idx])  # (R, U)
    base = (gt.base_logit[ref_idx] - gt.probe.logit_mean) / scale
    depth, kind = gt.probe.depth, gt.probe.kind_id
    n_f = tot.shape[1]

    decisive = np.abs(tot).argmax(1)
    share_den = np.abs(tot).sum(1, keepdims=True) + 1e-9
    out = np.zeros((n_f, D_FACTOR), dtype=np.float64)
    for f in range(n_f):
        t = tot[:, f]
        a = np.abs(ie[:, :, f])  # (R, U)
        row_sum = a.sum(1) + 1e-9
        top = a.argmax(1)
        live = np.abs(t) > in_play  # states where this factor actually moves B
        w = live if live.sum() >= 3 else np.ones(len(t), dtype=bool)

        an = a[w] / (np.linalg.norm(a[w], axis=1, keepdims=True) + 1e-9)
        consistency = float((an @ an.T)[np.triu_indices(len(an), 1)].mean()) if len(an) > 1 else 1.0

        out[f] = [
            t.mean(), np.abs(t).mean(), t.std(), np.percentile(np.abs(t), 90),
            live.mean(), (decisive == f).mean(),
            _rho(t, base), float((np.abs(t) / share_den[:, 0]).mean()),
            float((a.max(1) / row_sum).mean()),
            float(np.log1p((row_sum[w] ** 2 / ((a[w] ** 2).sum(1) + 1e-9)).mean())),
            float(depth[top[w]].mean()), float(depth[top[w]].std()),
            float((kind[top[w]] == 1).mean()),
            float((a[w].max(1) / (np.abs(t[w]) + 1e-6)).mean()),
            consistency,
            _rho(a[w].mean(0), eff[w].mean(0)),
        ]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
