"""Paired tests over target models, with a multiplicity correction.

The unit of analysis is a target model, and every method is scored on the same
held-out models, so comparisons are paired.  With `m` models the exact two-sided
sign-flip test has `2^m` sign assignments and a smallest attainable p of
`2 / 2^m` -- 0.125 at four models, 0.0078 at eight, 1.2e-7 at twenty-four. Any
design that needs significance has to be sized for it in advance.

Comparisons are declared before the run and corrected as one family
(Holm-Bonferroni), so that "the interpreter beat a baseline somewhere" cannot be
harvested after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


def paired_sign_flip(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
                     seed: int = 0) -> tuple[float, float]:
    """Two-sided exact/Monte-Carlo sign-flip test on the paired differences.

    Returns (mean difference, p).  Exact when 2^m <= n_perm.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    m = len(d)
    if m == 0:
        return float("nan"), float("nan")
    obs = abs(d.mean())
    if 2 ** m <= n_perm:
        signs = np.array(list(product([1, -1], repeat=m)), dtype=float)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([1.0, -1.0], size=(n_perm, m))
    null = np.abs((signs * d).mean(axis=1))
    return float(d.mean()), float((null >= obs - 1e-12).mean())


def min_attainable_p(n_models: int) -> float:
    """The smallest two-sided p any paired sign-flip test can return."""
    return 2.0 / (2 ** n_models) if n_models > 0 else float("nan")


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[dict]:
    """Step-down correction over a declared family of comparisons."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    n = len(order)
    out = [{"p": p, "p_adj": float("nan"), "reject": False} for p in pvals]
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(running, (n - rank) * pvals[i]))
        running = adj
        out[i] = {"p": pvals[i], "p_adj": adj, "reject": bool(adj <= alpha)}
    return out


@dataclass
class Comparison:
    metric: str
    method: str
    reference: str
    cell: str


def run_comparisons(rows: list[dict], declared: list[dict], alpha: float = 0.05,
                    cells: list[str] | None = None) -> list[dict]:
    """Evaluate every declared (metric, method vs reference) on every cell."""
    by = {(r["method"], r["cell"]): r for r in rows}
    cells = cells or sorted({r["cell"] for r in rows})
    results, pvals = [], []
    for d in declared:
        for cell in cells:
            ra, rb = by.get((d["method"], cell)), by.get((d["reference"], cell))
            if ra is None or rb is None:
                continue
            key = d["metric"]
            a = [m.get(key, np.nan) for m in ra["per_model"]]
            b = [m.get(key, np.nan) for m in rb["per_model"]]
            if not a or not b or len(a) != len(b):
                continue
            delta, p = paired_sign_flip(np.array(a), np.array(b))
            results.append({"metric": key, "method": d["method"], "reference": d["reference"],
                            "cell": cell, "n_models": len(a), "delta": delta,
                            "a": float(np.nanmean(a)), "b": float(np.nanmean(b))})
            pvals.append(p)
    for r, adj in zip(results, holm_bonferroni(pvals, alpha)):
        r.update(adj)
    return results


def comparison_table(results: list[dict]) -> str:
    if not results:
        return "(no declared comparisons could be evaluated)"
    w = max(len(f"{r['method']} vs {r['reference']}") for r in results) + 2
    head = ("comparison".ljust(w) + "cell".ljust(26) + "metric".ljust(28)
            + "delta".rjust(9) + "p".rjust(9) + "p_adj".rjust(9) + "  sig")
    lines = [head, "-" * len(head)]
    for r in sorted(results, key=lambda x: x["p_adj"]):
        lines.append(f"{r['method']} vs {r['reference']}".ljust(w)
                     + r["cell"].ljust(26) + r["metric"].ljust(28)
                     + f"{r['delta']:+9.3f}{r['p']:9.3f}{r['p_adj']:9.3f}"
                     + ("   *" if r["reject"] else "    "))
    return "\n".join(lines)
