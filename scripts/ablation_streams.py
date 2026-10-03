#!/usr/bin/env python
"""Which of the interpreter's input streams carries the reading?

Diagnostic, not confirmatory: no protocol was frozen before this ran, and nothing
here is a claim. It exists because the answer decides an engineering question --
whether an external target model has to be summarised per neuron at all.

The interpreter reads three streams (`follows_b.prepare`):

  unit_x    9 features per internal unit of B: continuous depth, unit kind, site
            width, and the first-order influence term in five forms
  pos_x     5 features per input position of B: the same first-order term, scaled,
            ranked, shared and signed within the state
  global_x  8 scalars describing B and its output at this state

Each arm zeroes one stream for training and for evaluation alike, so the
interpreter never sees it in either phase. Everything else -- margin filter, fit
and held-out split, epochs, seeds -- is the `follows_b` recipe unchanged, and the
`full` arm therefore has to reproduce the committed result exactly. It does; that
is the control that says the harness is wired correctly.

Writes `results/ablation_streams.json`. `--arms no_units --full-from <file>`
re-runs one arm against a stored reference.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch

from mint.domains import get_domain
from mint.eval.stats import holm_bonferroni, paired_sign_flip

from mint.semantic.encoding import GLOBAL_FEATURES, GLOBAL_PATTERN_FEATURES

from follows_b import MARGIN, load_population, named, prepare, tensors, train_pointer

DOMAINS = ("skirmish", "clinic", "entail")
# `no_pos` is not an arm. The Pointer has no positional encoding, so with pos_x
# zeroed every position is the same token, every position gets the same score,
# `recover_name` ties, and argmax names cause 0 whatever the other streams hold.
# Its accuracy is the frequency of cause 0 and tests nothing about the unit or
# global streams. It was run once (results/ablation_streams.json, first version)
# and is dropped here rather than reported as if it were a comparison.
ARMS = ("full", "no_units", "no_globals")
# Global features that summarise the unit stream. An arm that removes the units
# has to remove these too, or it is not blind -- the same rule the semantic layer
# already states, and the exact failure found once before in this project.
_PATTERN_COLS = [GLOBAL_FEATURES.index(f) for f in GLOBAL_PATTERN_FEATURES]
SEEDS = 3
N_MODELS, N_FIT = 24, 12


def strip(packs: list[dict], arm: str) -> list[dict]:
    """A copy of the packs with one stream zeroed, shapes untouched."""
    out = []
    for p in packs:
        q = dict(p)
        if arm == "no_units":
            q["unit_x"] = np.zeros_like(p["unit_x"])
            g = p["global_x"].copy()
            g[:, _PATTERN_COLS] = 0.0
            q["global_x"] = g
        elif arm == "no_globals":
            q["global_x"] = np.zeros_like(p["global_x"])
        elif arm != "full":
            raise KeyError(arm)
        out.append(q)
    return out


def run_arm(packs, spec, u_max, t0, tag) -> dict[str, list[float]]:
    fit, held = packs[:N_FIT], packs[N_FIT:]
    tr = tensors(fit, u_max, spec, MARGIN)
    models = []
    for s in range(SEEDS):
        models.append(train_pointer(tr, spec, seed=s))
        print(f"    {tag}, seed {s} обучен ({time.time() - t0:.0f}s)", flush=True)
    per: dict[str, list[float]] = {}
    for p in held:
        k = p["margin"] > MARGIN
        y, agree = p["y"][k], (p["world"][k] == p["y"][k])
        acc = np.mean([(named(m, p, spec, u_max, MARGIN) == y) for m in models], axis=0)
        for sub, msk in (("ALL", np.ones_like(agree)), ("AGREE", agree), ("DISAGREE", ~agree)):
            per.setdefault(sub, []).append(float(acc[msk].mean()) if msk.sum() else np.nan)
    return per


def stored_full(path: str, domain: str) -> dict[str, list[float]]:
    """The `full` arm's per-model values from an earlier run of this script. The
    audit found them identical to results/follows_b.json model by model, so an arm
    can be re-run against them without retraining the reference."""
    d = json.load(open(path))
    got = {}
    for r in d["rows"]:
        if r["domain"] == domain and r["arm"] == "full":
            got[r["cell"]] = [m["accuracy"] for m in r["per_model"]]
    if set(got) != {"ALL", "AGREE", "DISAGREE"}:
        raise ValueError(f"{path} has no complete full arm for {domain}")
    return got


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--full-from", default=None,
                    help="take the full arm from this earlier result instead of retraining it")
    ap.add_argument("--out", default="results/ablation_streams.json")
    a_ = ap.parse_args(argv)
    arms = tuple(a.strip() for a in a_.arms.split(","))
    torch.set_num_threads(4)
    t0 = time.time()
    rows, comps, prov = [], [], {}

    for name in DOMAINS:
        domain = get_domain(name)
        spec = domain.spec
        bundles, path, sha = load_population(name, N_MODELS)
        prov[name] = {"path": path, "sha256": sha}
        base = [prepare(b, domain) for b in bundles]
        u_max = max(p["unit_x"].shape[1] for p in base)

        got: dict[str, dict[str, list[float]]] = {}
        if "full" not in arms:
            got["full"] = stored_full(a_.full_from, name)
        for arm in arms:
            got[arm] = run_arm(strip(base, arm), spec, u_max, t0, f"{name}/{arm}")
            for sub, vals in got[arm].items():
                v = [x for x in vals if np.isfinite(x)]
                rows.append({"domain": name, "arm": arm, "cell": sub, "n_models": len(v),
                             "per_model": [{"accuracy": x} for x in v],
                             "accuracy": float(np.mean(v))})
            print(f"  {name}/{arm}: " + "  ".join(
                f"{s} {np.nanmean(got[arm][s]):.4f}" for s in ("ALL", "AGREE", "DISAGREE")),
                flush=True)

        pv, block = [], []
        for arm in [a for a in arms if a != "full"]:
            for sub in ("ALL", "DISAGREE"):
                a = np.array([x for x in got[arm][sub] if np.isfinite(x)])
                b = np.array([x for x in got["full"][sub] if np.isfinite(x)])
                delta, p = paired_sign_flip(a, b)
                block.append({"domain": name, "arm": arm, "reference": "full", "cell": sub,
                              "n_models": len(a), "delta": delta,
                              "a": float(a.mean()), "b": float(b.mean())})
                pv.append(p)
        for r, adj in zip(block, holm_bonferroni(pv, 0.05)):
            r.update(adj)
        comps.extend(block)

    print(f"\n{'domain/arm':26s}{'cell':10s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}")
    print("-" * 65)
    for c in comps:
        print(f"{c['domain'] + '/' + c['arm']:26s}{c['cell']:10s}{c['delta']:>+9.4f}"
              f"{c['p']:>10.5f}{c['p_adj']:>10.5f}")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty": bool(git("status", "--porcelain")),
                          "role": "diagnostic, no protocol frozen before the run",
                          "caches": prov, "margin": MARGIN, "arms": list(arms),
                          "full_arm_from": a_.full_from or "trained in this run",
                          "no_units_blinds": ["unit_x", *GLOBAL_PATTERN_FEATURES],
                          "interpreter_seeds": list(range(SEEDS)), "n_models": N_MODELS,
                          "n_fit": N_FIT, "torch": torch.__version__},
           "rows": rows, "comparisons": comps, "wall_seconds": time.time() - t0}
    with open(a_.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a_.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
