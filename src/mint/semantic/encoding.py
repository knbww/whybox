"""The semantic layer's inputs: a causal pattern, and candidate descriptions.

The lower causal layer is frozen and analytic. `first_order` — the exact
first-order term `sum_j g_j (c_j - a_j)` over a unit's components — locates the
units that matter and gives the sign and size of their effect. The confirmatory
run showed it tracks the true single-unit ablation ranking at Spearman 0.995 on
a language model, so nothing is learned about *where* to look any more.

What it cannot do is be about a *factor*. It is a property of a unit. The
semantic layer's whole job is the step it cannot take:

    causal pattern over units  ->  the name of the cause

**Leakage is the design risk here**, and the shape of the code is what prevents
it. At the state being explained the interpreter sees only the *factual* pattern.
Candidates are described exclusively from a disjoint set of reference states. If
the counterfactual pattern at the scored state were an input, naming would be a
lookup — the decisive factor's own perturbation mass exceeds the mean
alternative's in 83% of states.
"""
from __future__ import annotations

import numpy as np

from ..causal.interventions import CausalGroundTruth
from ..encoding.model_state import tied_rank

UNIT_FEATURES = (
    "depth", "is_mlp_neuron", "is_attn_head", "log_site_width",
    "fo_signed", "fo_abs", "fo_rank", "fo_share", "fo_sign",
)
GLOBAL_FEATURES = (
    "log_n_units", "frac_attn_units", "n_depth_levels",
    "state_prob", "state_confidence", "state_logit_z", "log_probe_logit_std",
    "log_fo_mass",
)
D_UNIT = len(UNIT_FEATURES)
D_GLOBAL = len(GLOBAL_FEATURES)

# Global features that summarise the target's internal-unit causal pattern. Any
# arm that ablates the unit stream must drop these too, or it is not blind.
GLOBAL_PATTERN_FEATURES = ("log_fo_mass",)


# behavioural half: how B's *output* answers do(f) on the reference states
# analytic half: how the *causal pattern* moves under the same counterfactual
SIGNATURE_FEATURES = (
    "mean_signed_effect", "mean_abs_effect", "sd_effect", "p90_abs_effect",
    "frac_in_play", "base_rate_decisive", "corr_effect_output", "mean_effect_share",
    "dfo_mass", "dfo_concentration", "dfo_locus_depth", "dfo_locus_depth_sd",
    "dfo_frac_attention", "dfo_consistency", "dfo_vs_fo_rho",
)
D_SIGNATURE = len(SIGNATURE_FEATURES)

# The analytic half describes how the target's *causal pattern* moves under the
# counterfactual, so it is internals-derived and an internals-blind arm must not
# see it. Measured cost of the omission when it was still visible: 0.0097 on the
# blind arm and 0.0120 on the full one, both inside their seed spread -- a defect
# to close rather than a retraction.
SIGNATURE_PATTERN_FEATURES = tuple(f for f in SIGNATURE_FEATURES if f.startswith("dfo_"))


def _rank(v: np.ndarray) -> np.ndarray:
    return tied_rank(v)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    # tie-averaged, which is both the correct Spearman treatment of ties and the
    # one that cannot encode index order through a tie block
    ra, rb = tied_rank(a).astype(float), tied_rank(b).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def causal_pattern(gt: CausalGroundTruth) -> tuple[np.ndarray, np.ndarray]:
    """(N, U, D_UNIT) factual causal pattern and (N, D_GLOBAL) context.

    Reads `gt.first_order` only. No activation, no weight, no counterfactual.
    """
    fo = gt.first_order
    n, u = fo.shape
    probe = gt.probe
    absfo = np.abs(fo)
    total = absfo.sum(1, keepdims=True) + 1e-9
    scale = np.maximum(absfo.max(1, keepdims=True), 1e-9)

    static = np.stack([probe.depth, (probe.kind_id == 0).astype(float),
                       (probe.kind_id == 1).astype(float),
                       np.log1p(probe.site_width) / 6.0], axis=1)  # (U, 4)
    dyn = np.stack([
        fo / scale, absfo / scale,
        np.stack([_rank(absfo[i]) for i in range(n)]),
        absfo / total, np.sign(fo),
    ], axis=2)  # (N, U, 5)
    unit_x = np.concatenate([np.repeat(static[None], n, axis=0), dyn], axis=2)

    p = 1.0 / (1.0 + np.exp(-np.clip(gt.base_logit, -30, 30)))
    g = np.stack([
        np.full(n, np.log1p(u) / 6.0),
        np.full(n, float((probe.kind_id == 1).mean())),
        np.full(n, len(np.unique(probe.depth)) / 8.0),
        p, np.abs(p - 0.5) * 2.0,
        (gt.base_logit - probe.logit_mean) / (probe.logit_std + 1e-6),
        np.full(n, np.log1p(probe.logit_std)),
        np.log1p(total[:, 0]),
    ], axis=1)
    return (np.nan_to_num(unit_x).astype(np.float32),
            np.nan_to_num(g).astype(np.float32))


def candidate_signatures(gt: CausalGroundTruth, ref_idx: np.ndarray,
                         in_play: float = 0.25) -> np.ndarray:
    """(F, D_SIGNATURE) — each candidate described from reference states only."""
    scale = gt.probe.logit_std + 1e-6
    tot = gt.factor_total[ref_idx] / scale
    fo = gt.first_order[ref_idx]
    dfo = np.abs(gt.first_order_cf[ref_idx] - fo[:, :, None])  # (R, U, F)
    base = (gt.base_logit[ref_idx] - gt.probe.logit_mean) / scale
    depth, kind = gt.probe.depth, gt.probe.kind_id
    decisive = np.abs(tot).argmax(1)
    share_den = np.abs(tot).sum(1, keepdims=True) + 1e-9

    out = np.zeros((tot.shape[1], D_SIGNATURE))
    for f in range(tot.shape[1]):
        t = tot[:, f]
        a = dfo[:, :, f]
        live = np.abs(t) > in_play
        w = live if live.sum() >= 3 else np.ones(len(t), dtype=bool)
        row = a[w].sum(1) + 1e-9
        top = a[w].argmax(1)
        an = a[w] / (np.linalg.norm(a[w], axis=1, keepdims=True) + 1e-9)
        cons = float((an @ an.T)[np.triu_indices(len(an), 1)].mean()) if len(an) > 1 else 1.0
        out[f] = [
            t.mean(), np.abs(t).mean(), t.std(), np.percentile(np.abs(t), 90),
            live.mean(), (decisive == f).mean(), _rho(t, base),
            float((np.abs(t) / share_den[:, 0]).mean()),
            float(np.log1p(row.mean())), float((a[w].max(1) / row).mean()),
            float(depth[top].mean()), float(depth[top].std()),
            float((kind[top] == 1).mean()), cons,
            _rho(a[w].mean(0), np.abs(fo[w]).mean(0)),
        ]
    return np.nan_to_num(out)


def candidate_locus(gt: CausalGroundTruth, ref_idx: np.ndarray) -> np.ndarray:
    """(F, U) where each candidate perturbs the causal pattern, from reference
    states.  Analytic: no activation is ever patched to produce it."""
    fo = gt.first_order[ref_idx]
    dfo = np.abs(gt.first_order_cf[ref_idx] - fo[:, :, None]).mean(0).T  # (F, U)
    return dfo / (dfo.sum(1, keepdims=True) + 1e-9)
