"""A generated causal explanation, and how to check it.

The analytic layer is an instrument, not a rival. It measures where a model is
sensitive, with a sign and a size, and it does that better than anything learned
here -- so it is an input. What it does not do, and cannot, is *say what the
cause is*: it emits no proposition, and the part of a proposition it could never
supply is the kind of structure it is looking at.

So the interpreter's output is a composition, not a pick from a list:

    type      one of a cross-domain inventory of causal structures
    support   which input positions carry it -- a pointer it generates, never a
              name it selects
    sign      which way the output moves when the cause is removed
    strength  how far

Nothing in the output is a domain label. The human-readable name is recovered
afterwards, by asking which human factor's carriers the generated support
matches -- for reporting, never as a training signal and never as an input.

Every part is checkable by executing the implied intervention on the target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..causal.structure import TYPES, extract
from .task import _contrast as _reference

SIGNS = ("NEGATIVE", "POSITIVE")
STRENGTHS = ("WEAK", "MODERATE", "STRONG")
CUTS = (0.75, 1.75)  # |effect| in units of the model's own logit spread
# Below this the target did not measurably move, so no direction was demonstrated.
# Float32 disagreement between a batched and a single-row forward is ~3e-6, so this
# sits three orders above the noise and far below any real effect.
SIGN_EPS = 1e-3


def bucket(x: np.ndarray) -> np.ndarray:
    return np.digitize(np.abs(x), CUTS)


@dataclass
class Composite:
    kind: np.ndarray       # (N,) index into TYPES
    support: np.ndarray    # (N, T) bool
    sign: np.ndarray       # (N,) index into SIGNS
    strength: np.ndarray   # (N,) index into STRENGTHS
    live: np.ndarray       # (N,)


def truth(model, domain, raw: np.ndarray, logit_sd: float) -> Composite:
    """The strength and sign a claim should make are those of *its own support*.

    Defining them from the full response instead makes the claim refer to a
    quantity it never mentions: the minimal support recovers 90% of the full
    effect by construction, so a bucket boundary can fall between the two and
    even the oracle then scores 0.62 on its own strength clause.
    """
    st = extract(model, domain, raw)
    achieved = support_effect(model, domain, raw, st.support) / max(logit_sd, 1e-6)
    return Composite(st.kind, st.support, (achieved > 0).astype(int), bucket(achieved), st.live)


def support_effect(model, domain, raw: np.ndarray, support: np.ndarray) -> np.ndarray:
    """(N,) change in the target's output when the given support is patched."""
    ctx = domain.to_model_input(raw)
    ref = domain.to_model_input(_reference(domain, raw))
    out = np.zeros(len(ctx))
    for i in range(len(ctx)):
        z = ctx[i].copy()
        z[np.nonzero(support[i])[0]] = ref[i, np.nonzero(support[i])[0]]
        # base is read one row at a time, exactly as the patched forward is: a
        # batched base disagrees with a single-row one by ~3e-6 in float32, which
        # is enough to decide the sign of a no-op patch by kernel dispatch.
        with torch.no_grad():
            b = float(model(torch.as_tensor(ctx[i][None]))[0].numpy()[0])
            out[i] = float(model(torch.as_tensor(z[None]))[0].numpy()[0] - b)
    return out


def verify(model, domain, raw: np.ndarray, comp: Composite, logit_sd: float,
           truth_c: Composite) -> dict[str, float]:
    """Execute the proposed intervention and score every clause of the claim.

    A claim earns its executed clauses only if it actually proposes an
    intervention. An empty support -- or one naming only positions where the
    context already equals the reference -- changes nothing, so `d` is float
    noise and `(d > 0)` is settled by kernel dispatch. Scored naively that hands
    `(support=none, NEGATIVE, WEAK)` a perfect executed record by construction,
    which measured 0.665 / 0.647 executed validity, above the confirmatory
    hybrid in one direction. Such a claim now scores 0 on the executed clauses,
    and a direction counts as shown only if the target actually moved.
    """
    ctx = domain.to_model_input(raw)
    ref = domain.to_model_input(_reference(domain, raw))
    idx = np.nonzero(truth_c.live)[0]
    if len(idx) == 0:
        return {}
    got_sign, got_str, jac, exact = [], [], [], []
    for i in idx:
        S = np.nonzero(comp.support[i])[0]
        z = ctx[i].copy()
        for p in S:
            z[p] = ref[i, p]
        acted = bool(np.any(z != ctx[i]))
        with torch.no_grad():
            b = float(model(torch.as_tensor(ctx[i][None]))[0].numpy()[0])
            d = float(model(torch.as_tensor(z[None]))[0].numpy()[0] - b) / max(logit_sd, 1e-6)
        shown = acted and abs(d) >= SIGN_EPS
        got_sign.append(float(shown and (d > 0) == bool(comp.sign[i])))
        got_str.append(float(acted and bucket(np.array([d]))[0] == comp.strength[i]))
        T = set(np.nonzero(truth_c.support[i])[0])
        jac.append(len(set(S) & T) / max(len(set(S) | T), 1))
        exact.append(float(set(S) == T))
    ok_type = (comp.kind[idx] == truth_c.kind[idx]).astype(float)
    ok_sign = (comp.sign[idx] == truth_c.sign[idx]).astype(float)
    ok_str = (comp.strength[idx] == truth_c.strength[idx]).astype(float)
    return {
        "type": float(ok_type.mean()),
        "support_exact": float(np.mean(exact)),
        "support_jaccard": float(np.mean(jac)),
        "sign": float(ok_sign.mean()),
        "strength": float(ok_str.mean()),
        # the whole claim, not its easiest clause
        "composition": float((ok_type * np.array(exact) * ok_sign * ok_str).mean()),
        # does executing what it proposed do what it said it would?
        "verified_sign": float(np.mean(got_sign)),
        "verified_strength": float(np.mean(got_str)),
        # co-primary: running the claim does *both* of what it said
        "executed_validity": float(np.mean(np.array(got_sign) * np.array(got_str))),
    }


def formula_template(model, domain, raw: np.ndarray, first_order: np.ndarray,
                     candidates: np.ndarray, live: np.ndarray, logit_sd: float,
                     majority_type: int, tau: float = 0.9,
                     execute_sign_strength: bool = True) -> Composite:
    """The instrument used as well as it can be used, on its own.

    Greedy-by-first-order for the support -- which reaches sufficiency in every
    state -- and the type named as the commonest, since it cannot be computed
    from a derivative at all.

    `execute_sign_strength` decides the regime. With it on, sign and size are
    read off by running the patch, and the instrument is trivially right about
    them; the learned generator predicts the same quantities without a single
    probe, so a composition score that mixes the two regimes is not a comparison.
    Off is the matched regime: both estimate from the first-order report.
    """
    ctx = domain.to_model_input(raw)
    ref = domain.to_model_input(_reference(domain, raw))
    with torch.no_grad():
        base = model(torch.as_tensor(ctx))[0].numpy()
        full = model(torch.as_tensor(ref))[0].numpy() - base
    n, t = ctx.shape
    sup = np.zeros((n, t), dtype=bool)
    sign = np.zeros(n, dtype=int)
    strength = np.zeros(n, dtype=int)
    for i in np.nonzero(live)[0]:
        D = list(np.nonzero(candidates[i])[0])
        order = sorted(D, key=lambda p: -abs(first_order[i, p]))
        S, d = [], 0.0
        for p in order:
            S.append(p)
            z = ctx[i].copy()
            for q in S:
                z[q] = ref[i, q]
            with torch.no_grad():
                d = float(model(torch.as_tensor(z[None]))[0].numpy()[0] - base[i])
            if full[i] != 0 and d / full[i] >= tau:
                break
        sup[i, S] = True
        est = d if execute_sign_strength else float(first_order[i, S].sum())
        sign[i] = int(est > 0)
        strength[i] = int(bucket(np.array([est / max(logit_sd, 1e-6)]))[0])
    return Composite(np.full(n, majority_type), sup, sign, strength, live)
