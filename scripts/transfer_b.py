#!/usr/bin/env python
"""Does the reading transfer to a different ontology?

Protocol frozen in `docs/PREREG_TRANSFER_B.md` before this ran. Follows
`scripts/follows_b.py`, whose in-cell gate returned CONFIRMED, and reuses its
pointer, its label and its margin rule unchanged.

The interpreter is fitted on target models of one domain and applied to target
models of the other. It never learns a cause name, so there is nothing to carry
over except a way of reading a model: it emits one score per input position, and
the name is recovered afterwards from the *target* domain's carrier table.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time

import numpy as np
import torch

from mint.domains import get_domain
from mint.eval.stats import holm_bonferroni, paired_sign_flip
from scripts.follows_b import (MARGIN, load_population, prepare, recover_name, tensors,
                                train_pointer)

PAIRS = [("skirmish", "clinic"), ("clinic", "skirmish")]
FAMILY = [("constant_source", "DISAGREE"), ("random", "DISAGREE"),
          ("constant_source", "ALL")]


CACHES: dict[str, tuple[str, str]] = {}


def load(domain_name: str, n_models: int):
    d = get_domain(domain_name)
    bundles, path, sha = load_population(domain_name, n_models)
    CACHES[domain_name] = (path, sha)
    return d, [prepare(b, d) for b in bundles]


def arms(pack, models, spec, u_max, maj_src, rng) -> dict[str, np.ndarray]:
    """maj_src is a cause *index* from the source domain, applied blindly here."""
    k = pack["margin"] > MARGIN
    n = int(k.sum())
    out = {"world_oracle": pack["world"][k],
           "first_order_pointer": recover_name(np.abs(pack["fo"][k]), spec),
           "constant_source": np.full(n, min(maj_src, spec.n_factors - 1)),
           "random": rng.integers(0, spec.n_factors, n)}
    with torch.no_grad():
        tr = tensors([pack], u_max, spec, MARGIN)
        for i, m in enumerate(models):
            out[f"interpreter_s{i}"] = recover_name(m(*tr[:4]).numpy(), spec)
    return out


def main(n_models: int = 24, seeds: int = 3) -> int:
    torch.set_num_threads(4)
    t0 = time.time()
    doms = {n: load(n, n_models) for n in {d for p in PAIRS for d in p}}
    u_max = max(p["unit_x"].shape[1] for _, packs in doms.values() for p in packs)
    rows, sens = [], []

    for src, tgt in PAIRS:
        cell = f"{src}->{tgt}"
        d_src, packs_src = doms[src]
        d_tgt, packs_tgt = doms[tgt]
        fit, held = packs_src[:12], packs_tgt[12:]
        tr = tensors(fit, u_max, d_src.spec, MARGIN)
        maj_src = int(np.bincount(tr[5], minlength=d_src.spec.n_factors).argmax())
        models = []
        for sd in range(seeds):
            models.append(train_pointer(tr, d_src.spec, seed=sd))
            print(f"    {cell}: seed {sd} обучен ({time.time() - t0:.0f}s)", flush=True)

        rng = np.random.default_rng(0)
        per: dict[tuple[str, str], list[float]] = {}
        n_dis = 0
        for p in held:
            k = p["margin"] > MARGIN
            y, agree = p["y"][k], (p["world"][k] == p["y"][k])
            n_dis += int((~agree).sum())
            a = arms(p, models, d_tgt.spec, u_max, maj_src, rng)
            interp = np.mean([(a[f"interpreter_s{i}"] == y) for i in range(seeds)], axis=0)
            scored = {"interpreter": interp} | {
                nm: (a[nm] == y).astype(float) for nm in
                ("world_oracle", "first_order_pointer", "constant_source", "random")}
            for nm, ok in scored.items():
                for sub, msk in (("ALL", np.ones_like(agree)), ("AGREE", agree),
                                 ("DISAGREE", ~agree)):
                    per.setdefault((nm, sub), []).append(
                        float(ok[msk].mean()) if msk.sum() else np.nan)

        print(f"  {cell}: {n_dis} расходящихся состояний на {len(held)} моделях "
              f"(причин у цели {d_tgt.spec.n_factors}, случайность "
              f"{1 / d_tgt.spec.n_factors:.3f})", flush=True)
        for (nm, sub), vals in sorted(per.items()):
            v = [x for x in vals if np.isfinite(x)]
            rows.append({"method": nm, "cell": f"{cell}|{sub}", "n_models": len(v),
                         "per_model": [{"accuracy": x} for x in v],
                         "accuracy": float(np.mean(v))})
            print(f"     {nm:22s}{sub:10s}{np.mean(v):.4f}")

    by = {(r["method"], r["cell"]): r for r in rows}
    stats, pv = [], []
    for src, tgt in PAIRS:
        for ref, sub in FAMILY:
            key = f"{src}->{tgt}|{sub}"
            a = np.array([m["accuracy"] for m in by[("interpreter", key)]["per_model"]])
            b = np.array([m["accuracy"] for m in by[(ref, key)]["per_model"]])
            delta, p = paired_sign_flip(a, b)
            stats.append({"metric": "accuracy", "method": "interpreter", "reference": ref,
                          "cell": key, "n_models": len(a), "delta": delta,
                          "a": float(a.mean()), "b": float(b.mean())})
            pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
    print(f"\n{'comparison':34s}{'cell':26s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}  sig")
    print("-" * 92)
    for r in stats:
        print(f"interpreter vs {r['reference']:19s}{r['cell']:26s}{r['delta']:>+9.4f}"
              f"{r['p']:>10.5f}{r['p_adj']:>10.5f}  {('+' if r['delta'] > 0 else '-') if r['reject'] else ''}")

    prim = [c for c in stats if c["reference"] == "constant_source"
            and c["cell"].endswith("DISAGREE")]
    pos = [c for c in prim if c["reject"] and c["delta"] > 0]
    void = any(c["reference"] == "random" and c["cell"].endswith("DISAGREE")
               and not c["reject"] for c in stats) and pos
    gate = ("VOID" if void else "CONFIRMED" if len(pos) == 2 else
            "ASYMMETRIC" if len(pos) == 1 else "FAIL")
    print(f"\nGATE: {gate}")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty": bool(git("status", "--porcelain")),
                          "prereg": "docs/PREREG_TRANSFER_B.md", "pairs": PAIRS,
                          "caches": {k: {"path": v[0], "sha256": v[1]} for k, v in CACHES.items()},
                          "margin": MARGIN, "n_models": n_models,
                          "seeds": list(range(seeds)), "torch": torch.__version__},
           "rows": rows, "comparisons": stats, "gate": gate,
           "wall_seconds": time.time() - t0}
    with open("results/transfer_b.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/transfer_b.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
