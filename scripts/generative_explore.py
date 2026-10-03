#!/usr/bin/env python
"""Exploratory: can a learned reranker fix what the derivative gets wrong?

The derivative reaches sufficiency in every state and cannot stop: it proposes
33% more positions than the minimal sufficient set and recovers that set exactly
in 61% of states. This asks whether a model handed the derivative's own output
can fix the boundary. No candidate list, no annotator: the target is computed by
exact search over subsets.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mint.domains import get_domain
from mint.encoding.model_state import tied_rank
from mint.generative import (D_POS, SetProposer, greedy_first_order, ground_truth,
                             score_sets)
from mint.models import TargetConfig, train_target


def features(gt) -> tuple[np.ndarray, np.ndarray]:
    fo, cand = gt.first_order, gt.positions
    n, t = fo.shape
    a = np.abs(fo)
    scale = np.maximum(a.max(1, keepdims=True), 1e-9)
    share = a / (a.sum(1, keepdims=True) + 1e-9)
    rank = np.stack([tied_rank(a[i]) for i in range(n)])
    x = np.stack([fo / scale, a / scale, rank, share, cand.astype(float),
                  np.sign(fo), np.repeat(cand.sum(1, keepdims=True) / t, t, 1)], -1)
    return np.nan_to_num(x).astype(np.float32), cand


def build(models, domain, n_focal, seed):
    xs, cs, ys, keep, packs = [], [], [], [], []
    for k, m in enumerate(models):
        raw = domain.sample(n_focal, np.random.default_rng(1000 + seed + k)).raw
        gt = ground_truth(m, domain, raw)
        x, c = features(gt)
        xs.append(x); cs.append(c); ys.append(gt.minimal)
        keep.append(gt.live); packs.append((m, raw, gt))
    return (torch.as_tensor(np.concatenate(xs)), torch.as_tensor(np.concatenate(cs)),
            torch.as_tensor(np.concatenate(ys), dtype=torch.float32),
            np.concatenate(keep), packs)


def train_proposer(x, c, y, live, epochs=60, seed=0, blind=False):
    torch.manual_seed(seed)
    m = SetProposer(blind=blind)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    idx = torch.as_tensor(np.nonzero(live)[0])
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            logit = m(x[b], c[b])
            loss = (F.binary_cross_entropy_with_logits(logit, y[b], reduction="none")
                    * c[b]).sum() / c[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def propose(m, x, c) -> np.ndarray:
    p = torch.sigmoid(m(x, c)).numpy()
    out = (p > 0.5) & c.numpy()
    empty = ~out.any(1)
    if empty.any():
        arg = np.where(c.numpy(), p, -1).argmax(1)
        out[empty, arg[empty]] = True
    return out


def main(n_models: int = 12, n_focal: int = 80, seeds: int = 2) -> int:
    d = get_domain("interact")
    t0 = time.time()
    pop = [train_target(d, TargetConfig("interact", {"family": "lm", "d_model": 48,
                                                     "n_layers": 2, "n_heads": 4, "d_ff": 96},
                                        s, 2.0, n_train=6000, epochs=40, lr=3e-3),
                        np.random.default_rng(s)).model for s in range(n_models)]
    fit, held = pop[:n_models // 2], pop[n_models // 2:]
    print(f"{n_models} targets trained ({time.time() - t0:.0f}s)")
    xtr, ctr, ytr, ltr, _ = build(fit, d, n_focal, 0)
    xte, cte, yte, lte, packs = build(held, d, n_focal, 500)
    print(f"train {int(ltr.sum())} live states, held out {int(lte.sum())}")

    rows = {}
    for name, blind in (("A_generative", False), ("A_blind", True)):
        per = []
        for sd in range(seeds):
            m = train_proposer(xtr, ctr, ytr, ltr, seed=sd, blind=blind)
            pred = propose(m, xte, cte)
            off = 0
            for mm, raw, gt in packs:
                sl = slice(off, off + len(raw)); off += len(raw)
                per.append(score_sets(pred[sl], gt, mm, d, raw))
        rows[name] = {k: float(np.mean([p[k] for p in per if k in p])) for k in per[0]}
    off = 0
    gr, orc = [], []
    for mm, raw, gt in packs:
        gr.append(score_sets(greedy_first_order(mm, d, raw, gt), gt, mm, d, raw))
        orc.append(score_sets(gt.minimal, gt, mm, d, raw))
    rows["greedy_first_order"] = {k: float(np.mean([p[k] for p in gr])) for k in gr[0]}
    rows["oracle"] = {k: float(np.mean([p[k] for p in orc])) for k in orc[0]}
    allc, top1 = [], []
    off = 0
    for mm, raw, gt in packs:
        allc.append(score_sets(gt.positions.copy(), gt, mm, d, raw))
        t1 = np.zeros_like(gt.minimal)
        for i in np.nonzero(gt.live)[0]:
            D = np.nonzero(gt.positions[i])[0]
            t1[i, D[np.argmax(np.abs(gt.first_order[i, D]))]] = True
        top1.append(score_sets(t1, gt, mm, d, raw))
    rows["all_candidates"] = {k: float(np.mean([p[k] for p in allc])) for k in allc[0]}
    rows["single_top_fo"] = {k: float(np.mean([p[k] for p in top1])) for k in top1[0]}

    cols = ["exact_match", "jaccard", "sufficiency", "size_pred", "size_true", "redundancy"]
    print(f"\n{'method':22s}" + "".join(c.rjust(14) for c in cols))
    print("-" * (22 + 14 * len(cols)))
    for k in ("oracle", "greedy_first_order", "A_generative", "A_blind",
              "all_candidates", "single_top_fo"):
        print(f"{k:22s}" + "".join(f"{rows[k][c]:14.3f}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
