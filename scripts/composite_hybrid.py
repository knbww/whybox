#!/usr/bin/env python
"""Division of labour: the instrument measures, the learned model names the kind.

Not a contest. The derivative is better at measuring the support, the sign and
the size, and that is its job. It cannot compute the *kind* of causal structure
at all, and has to guess the commonest one. This asks whether substituting a
learned kind into an otherwise measured claim improves the claim as a whole.
"""
from __future__ import annotations

import sys

import numpy as np

from mint.domains import get_domain
from mint.generative import formula_template, verify
from mint.generative.composite import Composite
from scripts.composite_explore import DOMAINS, build_one, generate, tensors, train

COLS = ["type", "support_exact", "sign", "strength", "composition"]


def main(n_models: int = 12, n_focal: int = 80, seeds: int = 2, data_seed: int = 4) -> int:
    packs = {d: [build_one(d, s, n_focal, data_seed) for s in range(n_models)] for d in DOMAINS}
    u = max(p["unit_x"].shape[1] for d in DOMAINS for p in packs[d])
    q = max(p["pos_x"].shape[1] for d in DOMAINS for p in packs[d])
    half = n_models // 2
    for src in DOMAINS:
        tgt = [x for x in DOMAINS if x != src][0]
        fit, held = packs[src][:half], packs[tgt][half:]
        tr = tensors(fit, u, q)
        maj = int(np.bincount(tr["kind"].numpy()).argmax())
        d = get_domain(tgt)
        rows = {"instrument": [], "generator": [], "hybrid": []}
        for sd in range(seeds):
            m = train(tr, seed=sd)
            for p in held:
                gtc = Composite(p["kind"], p["support"].astype(bool), p["sign"],
                                p["strength"], p["live"])
                g = generate(m, p, u, q)
                f = formula_template(p["model"], d, p["raw"], p["fo"], p["cand"],
                                     p["live"], p["logit_sd"], maj)
                h = Composite(g.kind, f.support, f.sign, f.strength, p["live"])
                rows["generator"].append(verify(p["model"], d, p["raw"], g, p["logit_sd"], gtc))
                rows["hybrid"].append(verify(p["model"], d, p["raw"], h, p["logit_sd"], gtc))
                if sd == 0:
                    rows["instrument"].append(
                        verify(p["model"], d, p["raw"], f, p["logit_sd"], gtc))
        print(f"\n### {src} -> {tgt}")
        print(f"{'method':14s}" + "".join(c.rjust(17) for c in COLS))
        print("-" * (14 + 17 * len(COLS)))
        for k in ("instrument", "generator", "hybrid"):
            v = [r for r in rows[k] if r]
            print(f"{k:14s}" + "".join(f"{np.mean([r[c] for r in v]):17.3f}" for c in COLS),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
