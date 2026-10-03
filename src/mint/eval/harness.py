"""Comparing candidate-explanation scorers under one protocol.

Every metric is computed **per target model** and then aggregated across the
held-out models with a bootstrap CI, because the unit of generalisation in this
study is a model, not a state: 160 states read off the same network are not 160
independent observations.
"""
from __future__ import annotations

import numpy as np
import torch

from ..interpreter.dataset import Bundle, ProblemSet
from .metrics import auroc, rank_metrics
from .protocol import executed_protocol

Scores = dict[str, np.ndarray]  # {"causal": (M, U), "mediation": (M, U)}


def as_scores(x) -> Scores:
    return x if isinstance(x, dict) else {"causal": x, "mediation": x}


def decoy_rejection(scores: np.ndarray, decoy: np.ndarray, mask: np.ndarray) -> float:
    """1 - AUROC(score ranks decoys above honest units).  0.5 = no discrimination."""
    a = auroc(scores[mask], decoy[mask])
    return float("nan") if np.isnan(a) else 1.0 - a


def _bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> float:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float((np.percentile(means, 97.5) - np.percentile(means, 2.5)) / 2)


def per_model_metrics(ps: ProblemSet, bundles: list[Bundle], scores: Scores,
                      ks=(1, 3, 5), run_executed: bool = True) -> dict[str, list[dict]]:
    """{domain: [metrics for each held-out model]}"""
    mask = ps.mask.numpy()
    y_eff, y_med, y_dec = ps.y_effect.numpy(), ps.y_mediation.numpy(), ps.y_decoy.numpy()
    dom_id, bid, sid = ps.cell_id.numpy(), ps.bundle_id.numpy(), ps.state_id.numpy()
    out: dict[str, list[dict]] = {}
    for d in np.unique(dom_id):
        rows = []
        for k in np.unique(bid[dom_id == d]):
            r = np.nonzero(bid == k)[0]
            r = r[np.argsort(sid[r])]
            b = bundles[int(k)]
            m = {"model": b.tag, "n_states": len(r)}
            for tag, y, key in (("causal", y_eff, "causal"), ("mediation", y_med, "mediation")):
                for mk, mv in rank_metrics(scores[key][r], y[r], mask[r], ks).items():
                    m[f"{tag}.{mk}"] = mv
            m["decoy_rejection"] = decoy_rejection(scores["causal"][r], y_dec[r], mask[r])
            m["decoy_rate"] = float(y_dec[r][mask[r]].mean())
            if run_executed:
                m.update({f"executed.{a}": v for a, v in
                          executed_protocol(b, scores["mediation"][r][:, :b.n_units], ks,
                                            state_idx=sid[r]).items()})
            rows.append(m)
        out[ps.cells[d]] = rows
    return out


def _merge_seeds(details: list[dict[str, list[dict]]]) -> dict[str, list[dict]]:
    """Average each target model's metrics across interpreter seeds.

    Folding seed variance into the per-model estimate keeps the bootstrap unit
    the target model, which is the thing we claim to generalise over, while
    stopping a single initialisation from standing in for the arm.
    """
    out: dict[str, list[dict]] = {}
    for cell, first in details[0].items():
        merged = []
        for i, m0 in enumerate(first):
            m = dict(m0)
            for key in m0:
                if key in ("model", "n_states"):
                    continue
                vals = [d[cell][i][key] for d in details if key in d[cell][i]]
                m[key] = float(np.nanmean(vals)) if vals else float("nan")
                if len(vals) > 1:
                    m[f"seed_sd.{key}"] = float(np.nanstd(vals))
            merged.append(m)
        out[cell] = merged
    return out


def evaluate(ps: ProblemSet, bundles: list[Bundle], scores, method: str,
             ks=(1, 3, 5), run_executed: bool = True) -> list[dict]:
    """One aggregated row per (method, cell), plus the per-model detail.

    `scores` may be a list, one entry per interpreter seed.
    """
    if isinstance(scores, list):
        detail = _merge_seeds([per_model_metrics(ps, bundles, as_scores(x), ks, run_executed)
                               for x in scores])
    else:
        detail = per_model_metrics(ps, bundles, as_scores(scores), ks, run_executed)
    rows = []
    for dom, models in detail.items():
        row: dict = {"method": method, "cell": dom, "domain": dom.split("/")[0],
                     "family": dom.split("/")[-1], "n_models": len(models),
                     "per_model": models}
        keys = sorted({k for m in models for k in m if k not in ("model", "n_states")})
        for k in keys:
            vals = [m[k] for m in models if k in m]
            row[k] = float(np.nanmean(vals)) if vals else float("nan")
            row[k + ".ci"] = _bootstrap_ci(vals)
        rows.append(row)
    return rows


def baseline_scores(ps: ProblemSet, fn) -> np.ndarray:
    s = fn(ps.unit_x, ps.global_x, ps.mask).clone().float()
    s[~ps.mask] = -1e9
    return s.numpy()


def table(rows: list[dict], columns: list[str], sort_by: str | None = None,
          show_ci: bool = True) -> str:
    cols = ["method"] + columns
    present = [c for c in cols if any(c in r for r in rows)]
    if sort_by:
        rows = sorted(rows, key=lambda r: -(r.get(sort_by) if np.isfinite(r.get(sort_by, np.nan))
                                            else -1e9))
    head = [c.replace("causal.", "C.").replace("mediation.", "M.").replace("executed.", "X.")
            .replace("precision@", "p@").replace("sufficiency@", "suff@")
            .replace("comprehensiveness@", "comp@") for c in present]
    w = [max(len(h), 16 if show_ci else 11) for h in head]
    w[0] = max(w[0], max(len(str(r.get("method", ""))) for r in rows) + 1)
    lines = ["  ".join(h.rjust(x) for h, x in zip(head, w)),
             "  ".join("-" * x for x in w)]
    for r in rows:
        cells = []
        for c, x in zip(present, w):
            v = r.get(c, "")
            if isinstance(v, float):
                ci = r.get(c + ".ci", float("nan"))
                txt = f"{v:.3f}" + (f"±{ci:.3f}" if show_ci and np.isfinite(ci) else "")
            else:
                txt = str(v)
            cells.append(txt.rjust(x))
        lines.append("  ".join(cells))
    return "\n".join(lines)
