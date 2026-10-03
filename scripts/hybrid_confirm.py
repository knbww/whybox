#!/usr/bin/env python
"""Confirmatory. Protocol frozen in docs/PREREG_HYBRID.md before this ran."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mint.domains import get_domain
from mint.eval.stats import comparison_table, min_attainable_p, run_comparisons
from mint.generative import formula_template, verify
from mint.generative.composite import Composite, bucket, support_effect
from scripts.composite_corrected import (corrected_truth, generate_joint, train_joint)
from scripts.composite_explore import DOMAINS, build_one, tensors

PRIMARY, CO = "composition", "executed_validity"
DIAG = ["type", "support_exact", "sign", "strength"]
DECLARED = ([{"metric": PRIMARY, "method": "hybrid", "reference": r}
             for r in ("instrument_aligned", "interpreter_joint", "random_composition")]
            + [{"metric": CO, "method": "hybrid", "reference": "instrument_aligned"}])


def random_composition(pack, rng) -> Composite:
    n, T = len(pack["live"]), pack["pos_x"].shape[1]
    sup = np.zeros((n, T), bool)
    for i in np.nonzero(pack["live"])[0]:
        D = np.nonzero(pack["cand"][i])[0]
        k = rng.integers(1, len(D) + 1)
        sup[i, rng.choice(D, k, replace=False)] = True
    return Composite(rng.integers(0, 5, n), sup, rng.integers(0, 2, n),
                     rng.integers(0, 3, n), pack["live"])


def main(n_models: int = 24, n_focal: int = 80, seeds: int = 3, data_seed: int = 5) -> int:
    t0 = time.time()
    packs = {d: [build_one(d, s, n_focal, data_seed) for s in range(n_models)] for d in DOMAINS}
    u = max(p["unit_x"].shape[1] for d in DOMAINS for p in packs[d])
    q = max(p["pos_x"].shape[1] for d in DOMAINS for p in packs[d])
    half = n_models // 2
    print(f"packs ready in {time.time() - t0:.0f}s", flush=True)

    rows, rng = [], np.random.default_rng(0)
    for src in DOMAINS:
        tgt = [x for x in DOMAINS if x != src][0]
        cell = f"{src}->{tgt}"
        fit, held = packs[src][:half], packs[tgt][half:]
        tr = tensors(fit, u, q)
        maj = int(np.bincount(tr["kind"].numpy()).argmax())
        d = get_domain(tgt)
        per = {k: [[] for _ in held] for k in
               ("instrument_aligned", "interpreter_joint", "hybrid", "random_composition", "oracle")}
        for sd in range(seeds):
            m = train_joint(tr, seed=sd)
            for j, p in enumerate(held):
                gtc = corrected_truth(p, d)
                args = (p["model"], d, p["raw"])
                f = formula_template(p["model"], d, p["raw"], p["fo"], p["cand"], p["live"],
                                     p["logit_sd"], maj, execute_sign_strength=False)
                g = generate_joint(m, p, u, q)
                h = generate_joint(m, p, u, q, carrier=f.support)
                per["interpreter_joint"][j].append(verify(*args, g, p["logit_sd"], gtc))
                per["hybrid"][j].append(verify(
                    *args, Composite(h.kind, f.support, h.sign, h.strength, p["live"]),
                    p["logit_sd"], gtc))
                if sd == 0:
                    per["instrument_aligned"][j].append(verify(*args, f, p["logit_sd"], gtc))
                    per["random_composition"][j].append(
                        verify(*args, random_composition(p, rng), p["logit_sd"], gtc))
                    per["oracle"][j].append(verify(*args, gtc, p["logit_sd"], gtc))
            print(f"  {cell} seed {sd} done ({time.time() - t0:.0f}s)", flush=True)
        for name, models in per.items():
            pm = [{k: float(np.mean([r[k] for r in runs if r])) for k in runs[0]}
                  for runs in models if runs and runs[0]]
            if not pm:
                continue
            row = {"method": name, "cell": cell, "per_model": pm}
            for k in pm[0]:
                row[k] = float(np.mean([m[k] for m in pm]))
            rows.append(row)

    cells = [f"{s}->{[d for d in DOMAINS if d != s][0]}" for s in DOMAINS]
    stats = run_comparisons(rows, DECLARED, 0.05, cells=cells)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    Path("results/hybrid_confirm.json").write_text(json.dumps(
        {"provenance": {"commit": git("rev-parse", "HEAD"),
                        "dirty": bool(git("status", "--porcelain")),
                        "prereg": "docs/PREREG_HYBRID.md", "data_seed": data_seed,
                        "seeds": list(range(seeds)), "n_models": n_models},
         "cells": cells, "rows": rows, "comparisons": stats,
         "wall_seconds": time.time() - t0}, indent=2, default=float))

    for cell in cells:
        print(f"\n### {cell}")
        print(f"{'arm':22s}{'whole-stmt':>14s}{'exec validity':>16s}  |"
              + "".join(c.rjust(14) for c in DIAG))
        print("-" * (22 + 30 + 3 + 14 * len(DIAG)))
        for nm in ("oracle", "hybrid", "instrument_aligned", "interpreter_joint",
                   "random_composition"):
            r = [x for x in rows if x["method"] == nm and x["cell"] == cell]
            if r:
                print(f"{nm:22s}{r[0][PRIMARY]:14.3f}{r[0][CO]:16.3f}  |"
                      + "".join(f"{r[0][c]:14.3f}" for c in DIAG))
    print(f"\ndeclared family of {len(stats)}, Holm; smallest attainable p "
          f"{min_attainable_p(n_models - half):.5f}")
    print(comparison_table(stats))

    key = {x["cell"]: x for x in stats
           if x["metric"] == PRIMARY and x["reference"] == "instrument_aligned"}
    rnd = [x for x in stats if x["reference"] == "random_composition"]
    win = [c for c, x in key.items() if x["reject"] and x["delta"] > 0]
    if any(x["reject"] and x["delta"] > 0 for x in key.values()) and not all(
            x["reject"] and x["delta"] > 0 for x in rnd):
        v = "VOID - the family cannot separate the hybrid from a random claim"
    elif len(win) == len(cells):
        v = "CONFIRMED - the division of labour beats the aligned instrument in both directions"
    elif win:
        v = f"ASYMMETRIC - holds in {win[0]} only; the asymmetry is the finding"
    else:
        v = "FAIL - the exploratory advantage does not replicate"
    print(f"\nGATE: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
