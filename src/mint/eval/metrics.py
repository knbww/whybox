"""Ranking and discrimination metrics, all computed per (model, state) then averaged."""
from __future__ import annotations

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 3 or np.allclose(pred, pred[0]) or np.allclose(true, true[0]):
        return 0.0
    rp, rt = _rankdata(pred), _rankdata(true)
    rp -= rp.mean(); rt -= rt.mean()
    d = np.sqrt((rp ** 2).sum() * (rt ** 2).sum())
    return float((rp * rt).sum() / d) if d > 0 else 0.0


def precision_at_k(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    k = min(k, len(pred))
    top_p = set(np.argsort(-pred)[:k].tolist())
    top_t = set(np.argsort(-true)[:k].tolist())
    return len(top_p & top_t) / k


def top1_effect_ratio(pred: np.ndarray, true: np.ndarray) -> float:
    """|effect| of the unit the method picks, over |effect| of the true best unit."""
    best = true.max()
    return float(true[int(np.argmax(pred))] / best) if best > 0 else 0.0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = _rankdata(scores) + 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def rank_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray,
                 ks=(1, 3, 5)) -> dict[str, float]:
    """pred/true/mask: (N, U).  Returns metrics averaged over rows."""
    out: dict[str, list[float]] = {"spearman": [], "top1_effect_ratio": []}
    for k in ks:
        out[f"precision@{k}"] = []
    for i in range(pred.shape[0]):
        m = mask[i]
        p, t = pred[i][m], true[i][m]
        out["spearman"].append(spearman(p, t))
        out["top1_effect_ratio"].append(top1_effect_ratio(p, t))
        for k in ks:
            out[f"precision@{k}"].append(precision_at_k(p, t, k))
    return {k: float(np.mean(v)) for k, v in out.items()}
