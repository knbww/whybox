#!/usr/bin/env python
"""The language-model disagreement cell, split into what it actually contains.

Diagnostic of a committed result. `results/statement_lm.json` reports the
interpreter on the states where the language target B relies on a different
cause than the world. An audit found that a third of those states (160 of 485 on
the held-out models) are states where the world has no cause at all: every one of
the world's factor effects is exactly zero, and the stable argsort in
`mint.domains.base.annotate` then names factor 0 as "the world's cause". Whatever
B relies on there, the state is counted as a disagreement, and it is not one in
the sense the claim needs: there is no world's cause for B to differ from.

This re-runs the committed recipe unchanged (scripts/statement.py, entail/lm,
same pinned population, same seeds), checks that every per-model value it
produces equals the committed file, and only then reports the DISAGREE cell
twice more:

  DISAGREE_GENUINE          the world names a cause and B relies on another
  DISAGREE_NO_WORLD_CAUSE   the world has no cause at all

Chance is reported two ways: uniform over the four causes, and uniform over the
causes actually present in the sentence (those whose removal changes B at all),
which on this world is the honest chance level.

Writes `results/lm_disagree_split.json`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch

from mint.domains import annotate, get_domain
from mint.eval.stats import paired_sign_flip

from follows_b import load_population, prepare
from statement import analytic, score, speak, train_speaker, truth

DOMAIN, SEEDS = "entail", 3
COMMITTED = "results/statement_lm.json"
METRICS = ("cause", "direction", "magnitude", "whole")
CELLS = ("ALL", "DISAGREE", "DISAGREE_GENUINE", "DISAGREE_NO_WORLD_CAUSE")


def main() -> int:
    torch.set_num_threads(4)
    t0 = time.time()
    d = get_domain(DOMAIN); spec = d.spec
    bundles, path, sha = load_population(DOMAIN, 24)
    packs = [prepare(b, d) for b in bundles]
    for p, b in zip(packs, bundles):
        p["raw"], p["sd"] = b.focal_raw, float(b.gt.probe.logit_std)
    models_B = [b.trained.model for b in bundles]
    truths = [truth(p, d, m) for p, m in zip(packs, models_B)]
    u_max = max(p["unit_x"].shape[1] for p in packs)
    fit, held = range(12), range(12, 24)

    speakers = []
    for s in range(SEEDS):
        speakers.append(train_speaker([packs[i] for i in fit], [truths[i] for i in fit],
                                      u_max, spec, seed=s))
        print(f"    seed {s} обучен ({time.time() - t0:.0f}s)", flush=True)

    yc = np.concatenate([truths[i]["cause"] for i in fit])
    ys = np.concatenate([truths[i]["sign"] for i in fit])
    ym = np.concatenate([truths[i]["strength"] for i in fit])
    maj = (int(np.bincount(yc, minlength=spec.n_factors).argmax()),
           int(np.bincount(ys, minlength=2).argmax()),
           int(np.bincount(ym, minlength=3).argmax()))
    rng = np.random.default_rng(0)            # consumed in exactly statement.py's order

    per: dict[tuple[str, str, str], list[float]] = {}
    counts = {c: 0 for c in CELLS}
    for i in held:
        p, t, mB = packs[i], truths[i], models_B[i]
        n = len(t["cause"])
        claims = {
            "constant": {"cause": np.full(n, maj[0]), "sign": np.full(n, maj[1]),
                         "strength": np.full(n, maj[2])},
            "random": {"cause": rng.integers(0, spec.n_factors, n),
                       "sign": rng.integers(0, 2, n), "strength": rng.integers(0, 3, n)},
            "analytic": analytic(p | t, spec, t, False, d, mB),
            "analytic_executed": analytic(p | t, spec, t, True, d, mB),
        }
        sc = {nm: score(c, t, d, mB) for nm, c in claims.items()}
        ip = [score(speak(s, p, u_max, spec), t, d, mB) for s in speakers]
        sc["interpreter"] = {k: np.mean([x[k] for x in ip], axis=0) for k in ip[0]}

        # chance over the causes present in the sentence, as an exact expectation
        ft = bundles[i].gt.factor_total[t["keep"]]
        live = np.abs(ft) > 1e-9
        present = np.where(live[np.arange(n), t["cause"]],
                           1.0 / np.maximum(live.sum(1), 1), 0.0)
        sc["chance_present"] = {"cause": present}
        sc["chance_uniform"] = {"cause": np.full(n, 1.0 / spec.n_factors)}

        none = np.abs(annotate(d, t["raw"]).factor_effect).max(1) == 0
        dis = ~t["agree"]
        masks = {"ALL": np.ones(n, bool), "DISAGREE": dis,
                 "DISAGREE_GENUINE": dis & ~none, "DISAGREE_NO_WORLD_CAUSE": dis & none}
        for c, m in masks.items():
            counts[c] += int(m.sum())
        for nm, s in sc.items():
            for metric in METRICS:
                if metric not in s:
                    continue
                for c, m in masks.items():
                    if m.sum():
                        per.setdefault((nm, metric, c), []).append(
                            float(np.asarray(s[metric], float)[m].mean()))

    # control: the committed per-model values must come out unchanged
    ref = json.load(open(COMMITTED))
    stored = {(r["method"], r["metric"], r["cell"]): [m["accuracy"] for m in r["per_model"]]
              for r in ref["rows"]}
    worst = 0.0
    for key, v in stored.items():
        got = per.get(key)
        if got is None or len(got) != len(v):
            raise RuntimeError(f"control failed: {key} missing or wrong length")
        worst = max(worst, float(np.max(np.abs(np.array(got) - np.array(v)))))
    print(f"контроль: все {len(stored)} строк {COMMITTED} воспроизведены, "
          f"наибольшее расхождение {worst:.2e}", flush=True)
    if worst > 1e-9:
        raise RuntimeError("the committed result does not reproduce; nothing below is valid")

    rows = [{"method": nm, "metric": m, "cell": c, "n_models": len(v),
             "per_model": [{"accuracy": x} for x in v], "accuracy": float(np.mean(v))}
            for (nm, m, c), v in per.items()]
    diag = []
    for refm in ("constant", "chance_present"):
        for c in ("DISAGREE", "DISAGREE_GENUINE"):
            a = per.get(("interpreter", "cause", c)); b = per.get((refm, "cause", c))
            if a and b and len(a) == len(b):
                delta, p = paired_sign_flip(np.array(a), np.array(b))
                diag.append({"metric": "cause", "method": "interpreter", "reference": refm,
                             "cell": c, "n_models": len(a), "delta": delta, "p": p,
                             "note": "diagnostic, not in any pre-registered family; "
                                     "12 models = 4 independent initialisations"})

    print(f"\nсостояний на 12 отложенных моделях: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    print(f"\n{'метод, причина':22s}" + "".join(f"{c:>26s}" for c in CELLS))
    for nm in ("interpreter", "constant", "random", "chance_present", "chance_uniform"):
        print(f"{nm:22s}" + "".join(
            f"{np.mean(per[(nm, 'cause', c)]):>26.4f}" if (nm, "cause", c) in per else f"{'':>26s}"
            for c in CELLS))
    for r in diag:
        print(f"  interpreter vs {r['reference']:15s}{r['cell']:18s}{r['delta']:+.4f}  p {r['p']:.4f}")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no")),
                          "role": "diagnostic split of a committed result",
                          "reproduces": COMMITTED, "max_abs_diff_to_committed": worst,
                          "domain": f"{DOMAIN}/lm", "cache": path, "cache_sha256": sha,
                          "seeds": list(range(SEEDS)), "state_counts": counts,
                          "held_out_note": "models 12-23 = 4 independent initialisations "
                                           "x 3 decoy strengths",
                          "torch": torch.__version__},
           "rows": rows, "diagnostic_comparisons": diag, "wall_seconds": time.time() - t0}
    with open("results/lm_disagree_split.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/lm_disagree_split.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
