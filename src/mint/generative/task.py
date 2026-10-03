"""The generative task: produce the cause, do not select it.

No candidate list. The answer is a **minimal sufficient set** of input positions:
patching S to the reference recovers at least `tau` of the model's full response,
and no proper subset of S does. Ground truth is computed exactly by search over
subsets -- there is no annotator anywhere in this task.

The design follows the one lesson this project has learned twice. A first-order
report already reaches sufficiency in every state; what it cannot do is stop.
It takes 2.51 positions where 1.92 suffice and recovers the exact minimal set in
59% of states on an interaction-bearing domain, against 73-96% on additive ones.
So the learned model is not asked to compete with the derivative: it is handed
the derivative's output and asked to fix the boundary.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

TAU = 0.9


@dataclass
class GenerativeTruth:
    positions: np.ndarray  # (N, T) bool: candidate positions (differ from reference)
    minimal: np.ndarray  # (N, T) bool: the exact minimal sufficient set
    first_order: np.ndarray  # (N, T) the analytic per-position term
    full_effect: np.ndarray  # (N,)
    live: np.ndarray  # (N,) bool: states where the reference actually moves B


def _contrast(domain, raw: np.ndarray) -> np.ndarray:
    out = raw.copy()
    for f in range(domain.spec.n_factors):
        out = domain.do_neutral(out, f)
    return out


def position_first_order(model, domain, raw: np.ndarray) -> np.ndarray:
    """(N, T) first-order effect of moving each position to the reference."""
    ctx = torch.as_tensor(domain.to_model_input(raw))
    ref = torch.as_tensor(domain.to_model_input(_contrast(domain, raw)))
    emb = model.tok(ctx)
    emb.retain_grad()
    h = model.ln_f(model.trunk(emb + model.pos(torch.arange(ctx.shape[1]))[None], None, None))
    lg = model.lm_head(h[:, -1])
    a, c = model.contrast
    (lg[:, c] - lg[:, a]).sum().backward()
    return (emb.grad.detach() * (model.tok(ref).detach() - emb.detach())).sum(-1).numpy()


def ground_truth(model, domain, raw: np.ndarray, tau: float = TAU,
                 min_effect: float = 0.25) -> GenerativeTruth:
    ctx = domain.to_model_input(raw)
    ref = domain.to_model_input(_contrast(domain, raw))
    with torch.no_grad():
        base = model(torch.as_tensor(ctx))[0].numpy()
        full = model(torch.as_tensor(ref))[0].numpy() - base
    n, t = ctx.shape
    cand = ctx != ref
    minimal = np.zeros((n, t), dtype=bool)
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
        D = np.nonzero(cand[i])[0]
        best = None
        for k in range(1, len(D) + 1):
            hits = [S for S in itertools.combinations(D, k) if eff(i, S) / full[i] >= tau]
            if hits:
                best = max(hits, key=lambda S: eff(i, S) / full[i])
                break
        if best is None:
            live[i] = False
            continue
        minimal[i, list(best)] = True
    return GenerativeTruth(cand, minimal, position_first_order(model, domain, raw), full, live)


def greedy_first_order(model, domain, raw: np.ndarray, gt: GenerativeTruth,
                       tau: float = TAU) -> np.ndarray:
    """The declared competitor: add positions by first-order size until sufficient."""
    ctx = domain.to_model_input(raw)
    ref = domain.to_model_input(_contrast(domain, raw))
    with torch.no_grad():
        base = model(torch.as_tensor(ctx))[0].numpy()
    out = np.zeros_like(gt.minimal)
    for i in np.nonzero(gt.live)[0]:
        D = list(np.nonzero(gt.positions[i])[0])
        order = sorted(D, key=lambda p: -abs(gt.first_order[i, p]))
        S = []
        for p in order:
            S.append(p)
            z = ctx[i].copy()
            for q in S:
                z[q] = ref[i, q]
            with torch.no_grad():
                got = float(model(torch.as_tensor(z[None]))[0].numpy()[0] - base[i])
            if got / gt.full_effect[i] >= tau:
                break
        out[i, S] = True
    return out


def score_sets(pred: np.ndarray, gt: GenerativeTruth, model, domain,
               raw: np.ndarray, tau: float = TAU) -> dict[str, float]:
    """Executed scoring: exact match, redundancy, and achieved sufficiency."""
    ctx = domain.to_model_input(raw)
    ref = domain.to_model_input(_contrast(domain, raw))
    with torch.no_grad():
        base = model(torch.as_tensor(ctx))[0].numpy()
    idx = np.nonzero(gt.live)[0]
    if len(idx) == 0:
        return {}
    exact, suf, size_p, size_t, jac = [], [], [], [], []
    for i in idx:
        P, T = set(np.nonzero(pred[i])[0]), set(np.nonzero(gt.minimal[i])[0])
        exact.append(float(P == T))
        jac.append(len(P & T) / max(len(P | T), 1))
        size_p.append(len(P)); size_t.append(len(T))
        z = ctx[i].copy()
        for q in P:
            z[q] = ref[i, q]
        with torch.no_grad():
            got = float(model(torch.as_tensor(z[None]))[0].numpy()[0] - base[i])
        suf.append(float(got / gt.full_effect[i] >= tau))
    return {"exact_match": float(np.mean(exact)), "jaccard": float(np.mean(jac)),
            "sufficiency": float(np.mean(suf)), "size_pred": float(np.mean(size_p)),
            "size_true": float(np.mean(size_t)),
            "redundancy": float(np.mean(size_p) / max(np.mean(size_t), 1e-9) - 1)}
