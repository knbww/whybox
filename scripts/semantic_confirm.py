#!/usr/bin/env python
"""Confirmatory semantic run. Protocol frozen in docs/PREREG_SEMANTIC.md."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mint.eval.stats import comparison_table, min_attainable_p, run_comparisons
from mint.experiments.transfer import ExperimentConfig, build_or_load, split_bundles
from mint.domains import get_domain
from mint.semantic import SemanticConfig, baselines, build, leakage_checks, predict, score, train

SOURCE = "agreement/lm"
CELLS = [SOURCE, "polarity/lm", "skirmish/mlp"]
ARMS = {"A_semantic": {}, "A_no_locus": {"use_locus": False}, "A_blind": {"blind": True}}
DECLARED = [{"metric": "balanced_accuracy", "method": "A_semantic", "reference": r}
            for r in ("A_blind", "prior", "random", "A_no_locus")]


def main(seeds: int = 3, epochs: int = 20, data_seed: int = 2) -> int:
    t0 = time.time()
    cfg = ExperimentConfig(seeds=(0, 1, 2, 3, 4, 5, 6, 7), spurious=(0.0, 1.0, 2.0),
                           target_n_train=6000, target_epochs=60, n_focal=160,
                           n_probe=512, data_seed=data_seed, holdout_models=12)
    pops, fit, held = {}, {}, {}
    for c in CELLS:
        pops[c] = build_or_load(c.split("/")[0], c.split("/")[1], cfg)
        f, h = split_bundles(pops[c], 12, seed=data_seed)
        fit[c], held[c] = [pops[c][i] for i in f], [pops[c][i] for i in h]

    train_ds = build(fit[SOURCE], cells=[SOURCE])
    evals = {c: build(held[c], cells=[c]) for c in CELLS}
    print(f"train {len(train_ds)} problems from {SOURCE}; "
          + ", ".join(f"{c}:{len(v)}" for c, v in evals.items()))
    for m in leakage_checks(train_ds, fit[SOURCE]):
        print("  leakage check:", m)

    rows = []
    for name, over in ARMS.items():
        per_seed = {c: [] for c in CELLS}
        for sd in range(seeds):
            m = train(train_ds, SemanticConfig(epochs=epochs, seed=sd, **over))
            for c, es in evals.items():
                per_seed[c].append(score(es, predict(m, es).argmax(1), name)[0])
        for c in CELLS:
            merged = dict(per_seed[c][0])
            merged["per_model"] = [
                {k: float(np.mean([r["per_model"][i][k] for r in per_seed[c]]))
                 for k in per_seed[c][0]["per_model"][i]}
                for i in range(len(per_seed[c][0]["per_model"]))]
            for k in ("balanced_accuracy", "accuracy"):
                merged[k] = float(np.mean([r[k] for r in per_seed[c]]))
                merged[k + ".seed_sd"] = float(np.std([r[k] for r in per_seed[c]]))
            rows.append(merged)
        print(f"  {name} done ({time.time() - t0:.0f}s)")

    for sd in range(seeds):
        m = train(build(fit["polarity/lm"], cells=["polarity/lm"]),
                  SemanticConfig(epochs=epochs, seed=sd))
        rows.append(score(evals["polarity/lm"], predict(m, evals["polarity/lm"]).argmax(1),
                          f"ceiling_polarity (seed {sd})")[0])
    for c, es in evals.items():
        for bn, pred in baselines(es, held[c], get_domain(c.split("/")[0])).items():
            rows.append(score(es, pred, bn)[0])
        rows.append(score(es, es.y.numpy(), "oracle")[0])

    stats = run_comparisons(rows, DECLARED, 0.05, cells=CELLS)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                          "prereg": "docs/PREREG_SEMANTIC.md", "condition": "strict",
                          "data_seed": data_seed, "interpreter_seeds": list(range(seeds)),
                          "epochs": epochs},
           "cells": CELLS, "source_cells": [SOURCE], "rows": rows, "comparisons": stats,
           "wall_seconds": time.time() - t0}
    Path("results/semantic_confirm.json").write_text(json.dumps(out, indent=2, default=float))

    for c in CELLS:
        sub = sorted([r for r in rows if r["cell"] == c], key=lambda r: -r["balanced_accuracy"])
        tag = "SOURCE" if c == SOURCE else ("NEGATIVE CONTROL" if c == "skirmish/mlp" else "TRANSFER")
        print(f"\n### {c}  [{tag}]")
        print(f"{'method':30s}{'balanced acc':>20s}{'accuracy':>20s}")
        print("-" * 70)
        for r in sub:
            print(f"{r['method']:30s}{r['balanced_accuracy']:14.3f}±{r['balanced_accuracy.ci']:.3f}"
                  f"{r['accuracy']:14.3f}±{r['accuracy.ci']:.3f}")
    print(f"\ndeclared comparisons (Holm over {len(stats)}, smallest attainable p "
          f"{min_attainable_p(12):.5f})")
    print(comparison_table(stats))

    key = [x for x in stats if x["reference"] == "A_blind"]
    tr = [x for x in key if x["cell"] == "polarity/lm"]
    src = [x for x in key if x["cell"] == SOURCE]
    neg = [x for x in key if x["cell"] == "skirmish/mlp"]
    if neg and neg[0]["reject"] and neg[0]["delta"] > 0:
        v = "VOID — the negative control cell also shows an effect; suspect a leak"
    elif tr and tr[0]["reject"] and tr[0]["delta"] > 0:
        v = "PASS — the mapping transfers to a different logical task"
    elif src and src[0]["reject"] and src[0]["delta"] > 0:
        v = "SOURCE-ONLY — generalises across models of the same task, not across tasks"
    else:
        v = "FAIL — no cell beats the internals-blind control"
    print(f"\nGATE: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
