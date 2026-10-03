#!/usr/bin/env python
"""Applies the read-out rule of docs/PROTOCOL_SCALING.md, mechanically.

The rule was fixed before the run, so nothing here decides what counts as a
trend; it only computes it. Directions are never averaged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mint.encoding.model_state import tied_rank

from mint.eval.stats import holm_bonferroni, paired_sign_flip

METRICS = ["structure", "sign", "magnitude", "executed_validity", "whole_statement"]
LABEL = {"structure": "structure", "sign": "sign", "magnitude": "magnitude",
         "executed_validity": "executed validity", "whole_statement": "whole statement*"}
SRC = Path("results/scaling_study.json")
FIG = Path("results/figures")


def longest_run(v: np.ndarray) -> int:
    """Longest stretch of consecutive rungs over which the metric does not fall."""
    best = run = 1
    for i in range(1, len(v)):
        run = run + 1 if v[i] >= v[i - 1] - 1e-12 else 1
        best = max(best, run)
    return best


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    r = lambda a: tied_rank(a).astype(float)
    a, b = r(x) - r(x).mean(), r(y) - r(y).mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def series(rows, cell, axis):
    learned = [r for r in rows if r["cell"] == cell and not r.get("analytic")]
    if axis == "capacity":
        s = [r for r in learned if r["n_fitted"] == 12]
        return sorted(s, key=lambda r: r["params"]), "params"
    top = max(r["params"] for r in learned)
    return sorted([r for r in learned if r["params"] == top],
                  key=lambda r: r["n_fitted"]), "n_fitted"


def main() -> int:
    doc = json.loads(SRC.read_text())
    rows = doc["rows"]
    cells = sorted({r["cell"] for r in rows})
    base = {r["cell"]: r for r in rows if r.get("analytic")}
    tests, out = [], []

    for cell in cells:
        for axis, need in (("capacity", 3), ("data", 3)):
            s, key = series(rows, cell, axis)
            xs = np.array([r[key] for r in s], float)
            print(f"\n### {cell} -- {axis}"
                  + f"   ({'parameters' if key == 'params' else 'fitted target models'}: "
                  + ", ".join(f"{x:g}" if key == "n_fitted" else f"{x/1e6:.3f}M" for x in xs)
                  + ")")
            print(f"{'metric':22s}" + "".join(f"{x:>10.3g}" if key == "n_fitted"
                                              else f"{x/1e6:>10.3f}" for x in xs)
                  + f"{'analytic':>12s}{'run':>6s}{'rho':>7s}{'top-bot':>9s}{'p':>9s}")
            print("-" * (22 + 10 * len(xs) + 12 + 6 + 7 + 9 + 9))
            for mt in METRICS:
                ys = np.array([r[mt] for r in s], float)
                a = np.array([m[mt] for m in s[-1]["per_model"]])
                b = np.array([m[mt] for m in s[0]["per_model"]])
                delta, p = paired_sign_flip(a, b)
                run, rho = longest_run(ys), spearman(xs, ys)
                tests.append(p)
                out.append({"cell": cell, "axis": axis, "metric": mt, "run": run,
                            "n_rungs": len(ys), "need": need, "rho": rho,
                            "delta": delta, "p": p,
                            "values": ys.tolist(), "x": xs.tolist(),
                            "analytic": base[cell][mt]})
                print(f"{LABEL[mt]:22s}" + "".join(f"{y:>10.3f}" for y in ys)
                      + f"{base[cell][mt]:>12.3f}{run:>4d}/{len(ys)}{rho:>7.2f}"
                      + f"{delta:>+9.3f}{p:>9.4f}")

    for t, h in zip(out, holm_bonferroni([o["p"] for o in out], 0.05)):
        t["p_adj"], t["reject"] = h["p_adj"], h["reject"]
        t["trend"] = bool(t["run"] >= t["need"] and t["reject"] and t["delta"] > 0)

    print(f"\n\n=== read-out, Holm over the declared {len(out)} tests ===")
    print(f"{'cell':22s}{'axis':10s}{'metric':22s}{'run':>7s}{'delta':>9s}"
          f"{'p_adj':>9s}{'trend':>8s}")
    print("-" * 87)
    for t in out:
        print(f"{t['cell']:22s}{t['axis']:10s}{LABEL[t['metric']]:22s}"
              f"{t['run']:>4d}/{t['n_rungs']}{t['delta']:>+9.3f}{t['p_adj']:>9.4f}"
              f"{'YES' if t['trend'] else 'no':>8s}")

    print("\n* whole statement is a downstream diagnostic (docs/INTERFACE.md), not "
          "a metric the hypothesis rests on.")
    print("The analytic column has no capacity or data axis: it is flat by "
          "construction, not by measurement.")

    ev = {(t["cell"], t["axis"]): t for t in out if t["metric"] == "executed_validity"}
    cap = [ev[(c, "capacity")]["trend"] for c in cells]
    dat = [ev[(c, "data")]["trend"] for c in cells]
    verdict = ("CAPACITY-LIMITED - capacity moves executed validity, experience does not"
               if all(cap) and not any(dat) else
               "EXPERIENCE-LIMITED - experience moves it, capacity does not"
               if all(dat) and not any(cap) else
               "BOTH AXES MOVE - capacity and causal experience each help"
               if all(cap) and all(dat) else
               "NEITHER AXIS MOVES - a ceiling at this scale, cause not yet known"
               if not any(cap) and not any(dat) else
               "MIXED / ASYMMETRIC - the axes disagree between transfer directions")
    print(f"\nVERDICT on executed validity: {verdict}")
    print("The 93.4M rung, and any argument for a much larger interpreter, is "
          "motivated only by a trend on at least one axis.")
    Path("results/scaling_readout.json").write_text(
        json.dumps({"provenance": doc["provenance"], "tests": out,
                    "verdict": verdict}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
