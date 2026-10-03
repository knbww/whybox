"""The interpreter's view of a target model -- deliberately ontology-free.

This module defines what interpreter A is allowed to see.  Every descriptor is
either (a) a property of B's parameters, (b) a statistic of B's activations
over a probe set, or (c) B's state at the focal input.  Nothing here carries
domain semantics:

* no input feature values, names, counts or types;
* no factor names, no labels, no domain identity;
* no layer names -- position is a continuous normalised depth;
* no ablation or patching results (those are the *targets*, not the inputs).

Everything is normalised **within a model** (z-scores and percentile ranks over
that model's own units), so a 48-unit MLP over standardised tabular features
and a 104-unit transformer over token ids produce descriptor matrices on the
same scale.  That within-model normalisation is what makes cross-domain,
cross-architecture transfer even meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..causal.interventions import CausalGroundTruth, ProbeSummary

UNIT_FEATURES = (
    "depth", "is_mlp_neuron", "is_attn_head", "log_site_width",
    "in_norm_z", "out_norm_z", "out_norm_rank",
    "act_mean_z", "log_act_std_z", "act_p10_z", "act_p50_z", "act_p90_z",
    "frac_active", "corr_out", "abs_corr_out", "abs_corr_out_rank",
    "log_grad_absmean_z", "gradact_mean_z", "log_gradact_absmean_z",
    "gradact_absmean_rank", "redundancy",
    "state_act_z", "abs_state_act_z", "state_act_pctile",
    "state_grad_scaled", "state_gradact_scaled", "abs_state_gradact_rank",
    "first_order_ablation", "abs_first_order_ablation_rank", "state_is_active",
)
GLOBAL_FEATURES = (
    "log_n_units", "log_n_params", "frac_attn_units", "n_depth_levels",
    "state_prob", "state_confidence", "state_logit_z", "log_probe_logit_std",
)
D_UNIT = len(UNIT_FEATURES)
D_GLOBAL = len(GLOBAL_FEATURES)


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-8)


def tied_rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1], with equal values given one shared rank.

    `argsort(argsort(v))` breaks ties by index, which smuggles the absolute
    position index back into a representation that forbids it (`INTERFACE.md`,
    and the module notes in `generative/model.py` and `generative/structure_model.py`).
    The leak is not hypothetical: every non-candidate position carries a
    first-order value of exactly 0, so the order *within* that tie block is the
    position order, and an audit recovered the absolute index from `fo_rank`
    alone at 0.36 / 0.33 against chance 0.09 / 0.13.

    `kind="stable"` does not fix it -- it makes the leak deterministic rather
    than absent. Averaging the rank across each run of equal values does.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return np.zeros(0)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    new = np.concatenate(([True], sv[1:] != sv[:-1]))
    first = np.flatnonzero(new)                       # start of each run of equals
    avg = first + (np.diff(np.concatenate((first, [n]))) - 1) / 2.0
    out = np.empty(n, dtype=float)
    out[order] = avg[np.cumsum(new) - 1]
    return out / max(n - 1, 1)


def _rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] across the model's own units."""
    return tied_rank(v)


@dataclass
class EncodedProblem:
    """One (model, batch of states) problem instance, ready for interpreter A."""

    unit_x: np.ndarray  # (N, U, D_UNIT)
    global_x: np.ndarray  # (N, D_GLOBAL)
    n_units: int
    model_tag: str
    domain: str


def encode_units(probe: ProbeSummary, acts: np.ndarray, grads: np.ndarray,
                 base_logit: np.ndarray, n_params: float,
                 first_order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(N, U, D_UNIT) unit descriptors and (N, D_GLOBAL) global context."""
    u_n = len(probe.act_mean)
    n = acts.shape[0]

    static = np.stack([
        probe.depth,
        (probe.kind_id == 0).astype(float),
        (probe.kind_id == 1).astype(float),
        np.log1p(probe.site_width) / 6.0,
        _z(np.log1p(probe.in_norm)),
        _z(np.log1p(probe.out_norm)),
        _rank(probe.out_norm),
        _z(probe.act_mean),
        _z(np.log1p(probe.act_std)),
        _z(probe.act_p10), _z(probe.act_p50), _z(probe.act_p90),
        probe.frac_active,
        probe.corr_out, np.abs(probe.corr_out), _rank(np.abs(probe.corr_out)),
        _z(np.log1p(probe.grad_absmean)),
        _z(probe.gradact_mean),
        _z(np.log1p(probe.gradact_absmean)),
        _rank(probe.gradact_absmean),
        probe.redundancy,
    ], axis=1)  # (U, 21)

    sd = probe.act_std + 1e-6
    act_z = (acts - probe.act_mean[None]) / sd[None]
    pct = np.clip((acts - probe.act_p10[None]) /
                  (probe.act_p90[None] - probe.act_p10[None] + 1e-6), -0.5, 1.5)
    gscale = grads / (probe.logit_std + 1e-6)
    ga = grads * acts / (probe.logit_std + 1e-6)
    # Mean ablation is a replacement, so the first-order term is
    # sum_j g_j (c_j - a_j) over the unit's components.  When the component-exact
    # value is available use it; the scalar surrogate below is equal only if the
    # gradient and the activation are uncorrelated across components, which they
    # are not inside an attention head.
    if first_order is None:
        first_order = (probe.act_mean[None] - acts) * grads
    first_order = first_order / (probe.logit_std + 1e-6)
    dyn = np.stack([
        act_z, np.abs(act_z), pct, gscale, ga,
        np.stack([_rank(np.abs(ga[i])) for i in range(n)]),
        first_order,
        np.stack([_rank(np.abs(first_order[i])) for i in range(n)]),
        (acts > 1e-6).astype(float),
    ], axis=2)  # (N, U, 9)

    unit_x = np.concatenate([np.repeat(static[None], n, axis=0), dyn], axis=2)

    p = 1.0 / (1.0 + np.exp(-np.clip(base_logit, -30, 30)))
    g = np.stack([
        np.full(n, np.log1p(u_n) / 6.0),
        np.full(n, np.log1p(n_params) / 12.0),
        np.full(n, float((probe.kind_id == 1).mean())),
        np.full(n, len(np.unique(probe.depth)) / 8.0),
        p, np.abs(p - 0.5) * 2.0,
        (base_logit - probe.logit_mean) / (probe.logit_std + 1e-6),
        np.full(n, np.log1p(probe.logit_std)),
    ], axis=1)
    return np.nan_to_num(unit_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), \
        np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def encode_problem(gt: CausalGroundTruth, n_params: float, model_tag: str,
                   domain: str) -> EncodedProblem:
    unit_x, global_x = encode_units(gt.probe, gt.acts, gt.grads, gt.base_logit, n_params,
                                    first_order=getattr(gt, "first_order", None))
    return EncodedProblem(unit_x, global_x, unit_x.shape[1], model_tag, domain)
