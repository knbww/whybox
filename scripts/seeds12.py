#!/usr/bin/env python
"""The main claim again, on twelve independent initialisations instead of four.

Why this run exists. Every confirmatory result in this project is scored on
twelve held-out targets that are four independent initialisations at three decoy
strengths. The sign-flip test is computed over the twelve as if they were
independent units; at the level of the thing that is actually independent, the
initialisation, the smallest attainable p is 0.125 and no result can be resolved
below it. That qualification has had to travel with every number.

It does not need new target models. `results/cache/volume/new_models.pt` already
holds 84 targets spanning 28 independent initialisations, built for the volume
study. Twelve of them, one per initialisation, are enough to fit on, and twelve
more, one per initialisation and disjoint from the first set, are enough to score
on. The smallest attainable p then becomes 2/4096.

PROTOCOL, declared here before the run and hashed into the result file
-----------------------------------------------------------------------
Claim.        On targets from twelve independent initialisations, the
              interpreter names the cause the target model itself relies on,
              and does so on the states where that cause differs from the
              world's.
Arms.         interpreter (3 interpreter seeds, averaged per target);
              world_oracle; constant (majority label of the fit set);
              random; first_order_meanabs; first_order_signed.
Metric.       accuracy, per held-out target, on ALL / AGREE / DISAGREE.
Family.       six comparisons of interpreter against a reference, Holm over
              exactly these six:
                (constant, DISAGREE) (world_oracle, DISAGREE) (random, DISAGREE)
                (constant, AGREE)    (random, AGREE)          (constant, ALL)
Read-out.     CONFIRMED if both DISAGREE primaries -- against constant and
              against world_oracle -- reject at 0.05 after Holm with a positive
              delta. VOID if they do but (random, DISAGREE) does not reject.
              FAIL otherwise.
Роль floors.  first_order_meanabs and first_order_signed are reported as floors
              and are not in the family. They are how an artifact is told apart
              from a reading, never the comparison the claim rests on.
Fixed before the run: margin 0.10, 40 epochs, no early stopping, 3 interpreter
seeds, fit = initialisations 8..19, held = initialisations 24..35, one decoy
strength per initialisation rotating 0, 1, 2.
-----------------------------------------------------------------------

Disclosure. These 84 targets were the training population of the volume study.
No interpreter fitted here has seen the held-out twelve, but the population is
not new, and a reader should know that.

Writes `results/seeds12.json`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import torch

from mint.domains import get_domain
from mint.eval.stats import holm_bonferroni, paired_sign_flip

from follows_b import (MARGIN, carriers_in, named, prepare, tensors, train_pointer)

DOMAIN = "skirmish"
POPULATION = "results/cache/volume/new_models.pt"
FIT_SEEDS = tuple(range(8, 20))
HELD_SEEDS = tuple(range(24, 36))
BASE_SEED, N_SPURIOUS = 8, 3
SEEDS = 3
FAMILY = [("constant", "DISAGREE"), ("world_oracle", "DISAGREE"), ("random", "DISAGREE"),
          ("constant", "AGREE"), ("random", "AGREE"), ("constant", "ALL")]


def protocol_hash() -> str:
    """The declaration above, hashed, so the result file carries the text it ran under."""
    doc = (__doc__ or "").encode()
    return hashlib.sha256(doc).hexdigest()[:16]


def pick(bundles, seeds: tuple[int, ...]) -> list[int]:
    """One target per initialisation, rotating the decoy strength 0, 1, 2."""
    out = []
    for i, s in enumerate(seeds):
        idx = (s - BASE_SEED) * N_SPURIOUS + (i % N_SPURIOUS)
        cfg = bundles[idx].trained.cfg
        assert cfg.seed == s, f"index {idx} is initialisation {cfg.seed}, expected {s}"
        out.append(idx)
    return out


def main() -> int:
    torch.set_num_threads(4)
    t0 = time.time()
    domain = get_domain(DOMAIN)
    spec = domain.spec

    h = hashlib.sha256()
    with open(POPULATION, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    pop_sha = h.hexdigest()
    bundles = torch.load(POPULATION, weights_only=False)
    fit_idx, held_idx = pick(bundles, FIT_SEEDS), pick(bundles, HELD_SEEDS)
    assert not set(fit_idx) & set(held_idx)
    print(f"{POPULATION}: {len(bundles)} моделей, обучение на инициализациях "
          f"{FIT_SEEDS[0]}..{FIT_SEEDS[-1]}, проверка на {HELD_SEEDS[0]}..{HELD_SEEDS[-1]}")
    for tag, idx in (("fit", fit_idx), ("held", held_idx)):
        cfgs = [bundles[i].trained.cfg for i in idx]
        print(f"  {tag}: инициализации {[c.seed for c in cfgs]}, "
              f"сила ложного признака {[c.spurious_strength for c in cfgs]}")

    fit = [prepare(bundles[i], domain) for i in fit_idx]
    held = [prepare(bundles[i], domain) for i in held_idx]
    u_max = max(p["unit_x"].shape[1] for p in fit + held)
    tr = tensors(fit, u_max, spec, MARGIN)
    maj = int(np.bincount(tr[5], minlength=spec.n_factors).argmax())
    print(f"пакеты готовы, константа = причина {maj} ({time.time() - t0:.0f}s)", flush=True)

    models = []
    for s in range(SEEDS):
        models.append(train_pointer(tr, spec, seed=s))
        print(f"    seed {s} обучен ({time.time() - t0:.0f}s)", flush=True)

    car = carriers_in(spec, held[0]["pos_x"].shape[1])
    rng = np.random.default_rng(0)
    per: dict[tuple[str, str], list[float]] = {}
    counts = {"ALL": 0, "AGREE": 0, "DISAGREE": 0}
    for p in held:
        k = p["margin"] > MARGIN
        y, agree = p["y"][k], (p["world"][k] == p["y"][k])
        fo = p["fo"][k]
        arms = {
            "interpreter": np.mean([(named(m, p, spec, u_max, MARGIN) == y) for m in models], 0),
            "world_oracle": (p["world"][k] == y).astype(float),
            "constant": (np.full(len(y), maj) == y).astype(float),
            "random": (rng.integers(0, spec.n_factors, len(y)) == y).astype(float),
            "first_order_meanabs": (np.stack([np.abs(fo)[:, c].mean(1) for c in car], 1
                                             ).argmax(1) == y).astype(float),
            "first_order_signed": (np.stack([np.abs(fo[:, c].sum(1)) for c in car], 1
                                            ).argmax(1) == y).astype(float),
        }
        for sub, msk in (("ALL", np.ones_like(agree)), ("AGREE", agree), ("DISAGREE", ~agree)):
            counts[sub] += int(msk.sum())
            for nm, ok in arms.items():
                per.setdefault((nm, sub), []).append(float(ok[msk].mean()) if msk.sum() else np.nan)

    rows = []
    print(f"\nоцениваемых состояний {counts['ALL']}, расходящихся {counts['DISAGREE']}")
    for (nm, sub), vals in sorted(per.items()):
        v = [x for x in vals if np.isfinite(x)]
        rows.append({"method": nm, "cell": sub, "n_models": len(v), "n_states": counts[sub],
                     "per_model": [{"accuracy": x} for x in v], "accuracy": float(np.mean(v))})
    for nm in ("interpreter", "first_order_signed", "first_order_meanabs", "world_oracle",
               "constant", "random"):
        print(f"   {nm:22s}" + "  ".join(
            f"{sub} {np.nanmean(per[(nm, sub)]):.4f}" for sub in ("ALL", "AGREE", "DISAGREE")))

    by = {(r["method"], r["cell"]): r for r in rows}
    stats, pv = [], []
    for ref, sub in FAMILY:
        a = np.array([m["accuracy"] for m in by[("interpreter", sub)]["per_model"]])
        b = np.array([m["accuracy"] for m in by[(ref, sub)]["per_model"]])
        delta, p = paired_sign_flip(a, b)
        stats.append({"metric": "accuracy", "method": "interpreter", "reference": ref,
                      "cell": sub, "n_models": len(a), "delta": delta,
                      "a": float(a.mean()), "b": float(b.mean())})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)

    print(f"\n{'comparison':40s}{'subset':11s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}  sig")
    print("-" * 82)
    for r in stats:
        print(f"interpreter vs {r['reference']:25s}{r['cell']:11s}{r['delta']:>+9.4f}"
              f"{r['p']:>10.5f}{r['p_adj']:>10.5f}  "
              f"{('+' if r['delta'] > 0 else '-') if r['reject'] else ''}")

    prim = [c for c in stats if c["cell"] == "DISAGREE"
            and c["reference"] in ("constant", "world_oracle")]
    void = any(c["cell"] == "DISAGREE" and c["reference"] == "random" and not c["reject"]
               for c in stats)
    ok = len(prim) == 2 and all(c["reject"] and c["delta"] > 0 for c in prim)
    gate = "VOID" if ok and void else "CONFIRMED" if ok else "FAIL"
    print(f"\nGATE: {gate}")
    print(f"наименьшее достижимое p при 12 независимых единицах: {2 / 4096:.6f}")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty": bool(git("status", "--porcelain")),
                          "protocol": "declared in scripts/seeds12.py docstring",
                          "protocol_sha256": protocol_hash(),
                          "domain": DOMAIN, "population": POPULATION,
                          "population_sha256": pop_sha,
                          "fit_initialisations": list(FIT_SEEDS),
                          "held_initialisations": list(HELD_SEEDS),
                          "fit_index": fit_idx, "held_index": held_idx,
                          "margin": MARGIN, "interpreter_seeds": list(range(SEEDS)),
                          "note": "the population was the volume study's training pool; "
                                  "no interpreter fitted here saw the held-out twelve",
                          "torch": torch.__version__},
           "rows": rows, "comparisons": stats, "gate": gate,
           "wall_seconds": time.time() - t0}
    with open("results/seeds12.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/seeds12.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
