"""The causal *structure* of a state, extracted by execution.

A derivative at a point reports sensitivity. It cannot report what kind of cause
it is looking at, because that is a property of a neighbourhood: whether two
positions are interchangeable, whether each is separately required, whether a
pair cancels. Those distinctions need joint probing, and they are the same
distinctions across every domain -- which is what makes them a candidate for a
shared language of causality rather than a per-domain label set.

The types below are computed mechanically from the target model, so they serve
both as exact training supervision and as the verifier's ground truth. No human
names an instance.

    NECESSARY_SINGLE  one position carries the effect on its own
    CONJUNCTIVE       two or more positions each carry it alone: every one of
                      them is separately required for the effect to exist
    REDUNDANT         no single position carries it, but a pair does, and each
                      alone does almost nothing -- a disjunction
    CANCELLING        a pair whose individual effects have opposite signs and
                      whose joint effect is smaller than either
    COMPLEX           none of the above at pair order
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

TYPES = ("NECESSARY_SINGLE", "CONJUNCTIVE", "REDUNDANT", "CANCELLING", "COMPLEX")
TAU = 0.9          # share of the full response a set must recover to count
SMALL = 0.30       # below this share, a position "does almost nothing" alone


@dataclass
class CausalStructure:
    kind: np.ndarray          # (N,) index into TYPES
    support: np.ndarray       # (N, T) bool: the minimal sufficient set
    inert: np.ndarray         # (N, T) bool: candidates that alone do almost nothing
    single_effect: np.ndarray  # (N, T) share of the full response each alone recovers
    full_effect: np.ndarray   # (N,)
    candidates: np.ndarray    # (N, T) bool
    live: np.ndarray          # (N,) bool


def _reference(domain, raw: np.ndarray) -> np.ndarray:
    out = raw.copy()
    for f in range(domain.spec.n_factors):
        out = domain.do_neutral(out, f)
    return out


def extract(model, domain, raw: np.ndarray, tau: float = TAU,
            small: float = SMALL, min_effect: float = 0.25) -> CausalStructure:
    ctx = domain.to_model_input(raw)
    ref = domain.to_model_input(_reference(domain, raw))
    with torch.no_grad():
        base = model(torch.as_tensor(ctx))[0].numpy()
        full = model(torch.as_tensor(ref))[0].numpy() - base
    n, t = ctx.shape
    cand = ctx != ref
    kind = np.full(n, TYPES.index("COMPLEX"))
    support = np.zeros((n, t), dtype=bool)
    inert = np.zeros((n, t), dtype=bool)
    single = np.zeros((n, t))
    live = np.abs(full) > min_effect

    def eff(i: int, S) -> float:
        z = ctx[i].copy()
        for p in S:
            z[p] = ref[i, p]
        with torch.no_grad():
            return float(model(torch.as_tensor(z[None]))[0].numpy()[0] - base[i])

    for i in range(n):
        if not live[i]:
            continue
        D = list(np.nonzero(cand[i])[0])
        share = {p: eff(i, [p]) / full[i] for p in D}
        for p in D:
            single[i, p] = share[p]
        solo = [p for p in D if share[p] >= tau]
        inert[i, [p for p in D if abs(share[p]) < small]] = True

        if len(solo) >= 2:
            kind[i] = TYPES.index("CONJUNCTIVE")
            support[i, solo[:1]] = True          # any one of them suffices
        elif len(solo) == 1:
            kind[i] = TYPES.index("NECESSARY_SINGLE")
            support[i, solo] = True
        else:
            pair = None
            for p, q in itertools.combinations(D, 2):
                if eff(i, [p, q]) / full[i] >= tau:
                    pair = (p, q)
                    break
            cancel = [(p, q) for p, q in itertools.combinations(D, 2)
                      if share[p] * share[q] < 0
                      and abs(eff(i, [p, q]) / full[i]) < min(abs(share[p]), abs(share[q]))]
            if pair and abs(share[pair[0]]) < small and abs(share[pair[1]]) < small:
                kind[i] = TYPES.index("REDUNDANT")
                support[i, list(pair)] = True
            elif cancel:
                kind[i] = TYPES.index("CANCELLING")
                support[i, list(cancel[0])] = True
            elif pair:
                kind[i] = TYPES.index("COMPLEX")
                support[i, list(pair)] = True
            else:
                live[i] = False
    return CausalStructure(kind, support, inert, single, full, cand, live)


def heuristic_kind(st: CausalStructure, first_order: np.ndarray,
                   tau: float = TAU, small: float = SMALL) -> np.ndarray:
    """The analytic competitor, declared before any learned model is built.

    Reads only the first-order per-position report and applies the same rules
    the executed extractor applies -- the best a derivative can do at this task
    without probing.
    """
    n, t = first_order.shape
    out = np.full(n, TYPES.index("COMPLEX"))
    for i in range(n):
        if not st.live[i]:
            continue
        D = np.nonzero(st.candidates[i])[0]
        if len(D) == 0:
            continue
        tot = first_order[i, D].sum()
        share = first_order[i, D] / (tot if abs(tot) > 1e-9 else 1e-9)
        solo = (share >= tau).sum()
        if solo >= 2:
            out[i] = TYPES.index("CONJUNCTIVE")
        elif solo == 1:
            out[i] = TYPES.index("NECESSARY_SINGLE")
        elif (np.abs(share) < small).sum() >= 2:
            out[i] = TYPES.index("REDUNDANT")
        elif len(D) >= 2 and (share.min() * share.max() < 0):
            out[i] = TYPES.index("CANCELLING")
    return out
